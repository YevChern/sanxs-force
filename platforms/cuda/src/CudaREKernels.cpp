/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2014 Stanford University and the Authors.           *
 * Authors: Peter Eastman                                                     *
 * Contributors:                                                              *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included in *
 * all copies or substantial portions of the Software.                        *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    *
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      *
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE  *
 * USE OR OTHER DEALINGS IN THE SOFTWARE.                                     *
 * -------------------------------------------------------------------------- */

#include "CudaREKernels.h"
#include "CudaREKernelSources.h"
#include "CudaFFMaps.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/cuda/CudaForceInfo.h"
#include <math.h>

using namespace REPlugin;
using namespace OpenMM;
using namespace std;

const double FOUR_PI_SQUARED = 157.91367041742973790135185599802L;


class CudaREForceInfo : public CudaForceInfo {
public:
    CudaREForceInfo(const REForce& force) : force(force) {
    }

private:
    const REForce& force;
};


CudaCalcREForceKernel::CudaCalcREForceKernel(std::string name, const OpenMM::Platform& platform, OpenMM::CudaContext& cu, const OpenMM::System& system) :
        CalcREForceKernel(name, platform), hasInitializedKernel(false), cu(cu), system(system) {}

CudaCalcREForceKernel::~CudaCalcREForceKernel(){
    cu.setAsCurrent();
    if (write_output) {
        outfile_xray.close();
        outfile_neutron.close();
        outfile_scale_factor_xray.close();
        outfile_scale_factor_neutron.close();
    }
}

double CudaCalcREForceKernel::getXrayStrength(std::string name, float q, std::map<char, vector<double> > map, float charge) {
    std::map<char,vector<double> >::const_iterator constants_itr = map.find(char(name[0]));
    float fraction = 0.0;
    double fi = 0.0;
    vector<double> constants = constants_itr->second;
    for (int i = 0; i < 4; ++i){
        fi += constants.at(i)*exp(-constants.at(i+4)*q*q/FOUR_PI_SQUARED);
        // Add up total number of electrons of the atom
        fraction += constants.at(i);
    }
    fi += constants.at(8);
    fraction += constants.at(8);
    // Determine scaling factor due to partial charge
    fraction /= fraction + charge;
    fi *= fraction;
    return fi;
}

double CudaCalcREForceKernel::getIonXrayStrength(std::string name, float q, std::map<std::string, vector<double> > map, float charge) {
    std::map<string,vector<double> >::const_iterator constants_itr = map.find(name);
    float fraction = 0.0;
    double fi = 0.0;
    vector<double> constants = constants_itr->second;
    for (int i = 0; i < 4; ++i){
        fi += constants.at(i)*exp(-constants.at(i+4)*q*q/FOUR_PI_SQUARED);
        // Add up total number of electrons of the atom
        fraction += constants.at(i);
    }
    fi += constants.at(8);
    fraction += constants.at(8);
    // Determine scaling factor due to partial charge
    fraction /= fraction + charge;
    fi *= fraction;
    return fi;
}

// Charge delocalization-corrected x-ray scattering strength for water
double CudaCalcREForceKernel::getWaterXrayStrength(std::string name, float q, std::map<char, vector<double> > map){
    float delta = 0.22;                     // nm^-1
    if (char(name[0]) == 'H') {
        float alpha_H = -0.48;
        double fi = getXrayStrength("H", q, map) * (1.0 + alpha_H*exp(-q*q/(2*delta*delta)));
        return fi;
    } else if (char(name[0]) == 'O') {
        float alpha_O = 0.12;
        double fi = getXrayStrength("O", q, map) * (1.0 + alpha_O*exp(-q*q/(2*delta*delta)));
        return fi;
    } else {
        throw OpenMMException("CalcREForceKernel::getWaterXrayStrength got the 'name' != 'O' or 'H'");
    }
}

double CudaCalcREForceKernel::getNeutronStrength(std::string name, std::map<char, double> map){
    std::map<char, double>::const_iterator neutronSL_itr = map.find(char(name[0]));
    if (neutronSL_itr==map.end())
        throw OpenMMException("Warning: Atom type " + name + " not found in neutron atomic FF map!");

    double fi = neutronSL_itr->second;
    return fi;
}

// Same as getNeutronStrength, but for ions (as we can't rely on a single char to define a type)
double CudaCalcREForceKernel::getIonNeutronStrength(std::string name, std::map<std::string, double> map){
    std::map<string, double>::const_iterator neutronSL_itr = map.find(name);
    if (neutronSL_itr==map.end())
        throw OpenMMException("Warning: Atom type " + name + " not found in neutron atomic FF map!");

    double fi = neutronSL_itr->second;
    return fi;
}

double CudaCalcREForceKernel::getWaterNeutronStrength(std::string name, float deut, std::map<char, double > map){
    if (char(name[0]) == 'H') {
        double fi = getNeutronStrength("H", map)*(1 - deut) + getNeutronStrength("D", map)*deut;
        return fi;
    } else if (char(name[0]) == 'O') {
        double fi = getNeutronStrength("O", map);
        return fi;
    } else {
        throw OpenMMException("CalcREForceKernel::getWaterNeutronStrength got the 'name' != 'O' or 'H'");
    }
}

void CudaCalcREForceKernel::initialize(const System& system, const REForce& force) {
    cu.setAsCurrent();
    Natoms = system.getNumParticles();
    couple_steps = force.getCoupleSteps();

    double h_T, h_k_xray, h_k_neutron, h_w_dens, h_w_dens_sqr;
    float h_cutoff;
    force.getAllParams(h_T, h_k_xray, h_k_neutron, h_w_dens, h_w_dens_sqr, h_tau, h_cutoff);

    vector<double> h_mass_for_origin;
    h_mass_for_origin.reserve(Natoms);
    for (int i=0; i<Natoms; ++i){
        h_mass_for_origin.push_back(system.getParticleMass(i));
    }
    mass_for_origin = CudaArray::create<double>(cu, Natoms, "mass_for_origin");
    mass_for_origin->upload(h_mass_for_origin);

    vector<string> h_atom_names_tmp;
    force.getNames(h_atom_names_tmp);
    vector<char> h_atom_names;
    for (int i=0; i<h_atom_names_tmp.size(); ++i){
        h_atom_names.push_back(h_atom_names_tmp[i][0]);
    }

    atom_names = CudaArray::create<char>(cu, Natoms, "atom_names");
    atom_names->upload(h_atom_names);

    // We get Fs_exp from force, but it isn't scaled yet. We should call rescaleExpF() before actuall run to get a proper scaling.
    for (int i=0; i<force.getNumExpFF(); ++i){
        formFactor tmp_FF;
        force.getExpFF(i, tmp_FF.F, tmp_FF.delta_F, tmp_FF.q, tmp_FF.is_neutron, tmp_FF.d_part);
        Fs_exp.push_back(tmp_FF);
    }

    std::vector<float> h_q_xray, h_q_neutron, h_d_parts;    
    std::vector<double> h_F_exp_xray, h_delta_F_exp_xray, h_F_exp_neutron, h_delta_F_exp_neutron;
    Nq_xray = 0;
    Nq_neutron = 0;
    int num_ffs_xray = 0;
    int num_ffs_neutron = 0;
    for (int i=0; i<Fs_exp.size(); ++i){
        if (Fs_exp[i].is_neutron) {
            for (int j=0; j<Fs_exp[i].q.size(); ++j){
                Nq_neutron++;
                h_q_neutron.push_back(Fs_exp[i].q[j]);
                h_F_exp_neutron.push_back(Fs_exp[i].F[j]);
                h_delta_F_exp_neutron.push_back(Fs_exp[i].delta_F[j]);
                h_d_parts.push_back(Fs_exp[i].d_part);
            }
            num_ffs_neutron++;
        } else {
            for (int j=0; j<Fs_exp[i].q.size(); ++j){
                Nq_xray++;
                h_q_xray.push_back(Fs_exp[i].q[j]);
                h_F_exp_xray.push_back(Fs_exp[i].F[j]);
                h_delta_F_exp_xray.push_back(Fs_exp[i].delta_F[j]);
            }
            num_ffs_xray++;
        }
    }
    if ((Nq_xray==0) && (Nq_neutron==0)) throw OpenMMException("No X-ray or neutron experimental data is given. Aborting.");
    // We create dummy array with two experimental data points if we have only X-ray or neutron data.
    // That way we dont have to change the rest of the logic if we want to have X-ray or neutron only refinement.
    if (Nq_xray==0) {
        cout << "WARNING: No X-ray data provided. Setting X-ray force constant to 0." << endl;
        Nq_xray=2;
        formFactor tmp_FF;
        tmp_FF.is_neutron=false;
        tmp_FF.d_part=0.0;
        for (int i=0;i<Nq_xray; ++i){
            tmp_FF.F.push_back(1.0);
            tmp_FF.q.push_back(1.0);
            tmp_FF.delta_F.push_back(1.0);
            h_q_xray.push_back(1.0);
            h_F_exp_xray.push_back(1.0);
            h_delta_F_exp_xray.push_back(1.0);
        }
        Fs_exp.push_back(tmp_FF);
        h_k_xray=0.0;                 // Set the X-ray force to 0.0
    }
    if (Nq_neutron==0) {
        cout << "WARNING: No neutron data provided. Setting neutron force constant to 0." << endl;
        Nq_neutron=2;
        formFactor tmp_FF;
        tmp_FF.is_neutron=true;
        tmp_FF.d_part=0.0;
        for (int i=0; i<Nq_neutron; ++i){
            tmp_FF.F.push_back(1.0);
            tmp_FF.q.push_back(1.0);
            tmp_FF.delta_F.push_back(1.0);
            h_q_neutron.push_back(1.0);
            h_F_exp_neutron.push_back(1.0);
            h_delta_F_exp_neutron.push_back(1.0);
            h_d_parts.push_back(0.0);
        }
        Fs_exp.push_back(tmp_FF);
        h_k_neutron=0.0;              // Set the neutron force to 0.0
    }

    F_exp_xray = CudaArray::create<double>(cu, Nq_xray, "F_exp_xray");
    F_exp_xray->upload(h_F_exp_xray);
    delta_F_exp_xray = CudaArray::create<double>(cu, Nq_xray, "delta_F_exp_xray");
    delta_F_exp_xray->upload(h_delta_F_exp_xray);
    F_exp_neutron = CudaArray::create<double>(cu, Nq_neutron, "F_exp_neutron");
    F_exp_neutron->upload(h_F_exp_neutron);
    delta_F_exp_neutron = CudaArray::create<double>(cu, Nq_neutron, "delta_F_exp_neutron");
    delta_F_exp_neutron->upload(h_delta_F_exp_neutron);

    xray_qs = CudaArray::create<float>(cu, Nq_xray, "xray_qs");
    xray_qs->upload(h_q_xray);

    neutron_qs = CudaArray::create<float>(cu, Nq_neutron, "neutron_qs");
    d_parts = CudaArray::create<float>(cu, Nq_neutron, "d_parts");
    neutron_qs->upload(h_q_neutron);
    d_parts->upload(h_d_parts);

    // Set up host x-ray and neutron scattering strength arrays
    std::vector<float> h_xray_strength(Nq_xray*Natoms, 0.0), h_neutron_strength(Nq_neutron*Natoms, 0.0);

    // Setup maps for the helpers getXray/NeutronStrength()
    // Xray constants map
    std::map<char, vector<double> > aff_constants=creat_aff_map();
    // Xray constants map (ions)
    std::map<std::string, vector<double> > aff_constants_ions=creat_aff_ion_map();
    // Neutron constants map
    std::map<char, double> neutronFFmap=creat_nsld_map();
    // Neutron constants map (ions)
    std::map<std::string, double> neutronFFmap_ions=creat_nsld_ion_map();

    // Get atom indices for atoms that are subject of the refinement force, atoms for origin, water, ions, exchangeable hydrogens
    vector<int> h_particles, h_particles_for_origin, particles_water_tmp, particles_ion_tmp, particles_exch_h_tmp;
    force.getAllParticles(h_particles, h_particles_for_origin, particles_water_tmp, particles_ion_tmp, particles_exch_h_tmp);
    particles = CudaArray::create<int>(cu, h_particles.size(), "particles");
    particles_for_origin = CudaArray::create<int>(cu, h_particles_for_origin.size(), "particles_for_origin");
    particles->upload(h_particles);
    particles_for_origin->upload(h_particles_for_origin);

    originNatoms = h_particles_for_origin.size();
    
    total_mass_origin = 0.0;
    for (int i=0; i<originNatoms; ++i){
        total_mass_origin += system.getParticleMass(i);
    }

    // Initialize is_water bool mask
    vector<int> h_is_water;
    h_is_water.reserve(Natoms);
    for (int i=0; i<Natoms; ++i)  h_is_water.push_back(0);
    for (int i=0; i<particles_water_tmp.size(); ++i){
        h_is_water[particles_water_tmp[i]] = 1;
    }

    is_water = CudaArray::create<int>(cu, Natoms, "is_water");
    is_water->upload(h_is_water);
    
    // Ions should have the name which correspond to the naming in the FFXray/FFNeutron Ion maps (e.g. POT -> K+)
    vector<bool> is_ion;
    is_ion.reserve(Natoms);
    for (int i=0; i<Natoms; ++i) is_ion.push_back(false);
    for (int i=0; i<particles_ion_tmp.size(); ++i){
        is_ion[particles_ion_tmp[i]] = true;
    }

    vector<float> h_charges;
    force.getCharges(h_charges);

    for (int j=0; j<Natoms; ++j){
        if (h_is_water[j]==1){      // Water
            for (int i=0; i<Nq_xray; ++i){
                h_xray_strength[j*Nq_xray + i] = getWaterXrayStrength(h_atom_names_tmp[j], h_q_xray[i], aff_constants);
            }
        } else {                    // Not water
            if (is_ion[j]){   	    // Ion
                for (int i=0; i<Nq_xray; ++i){
                    h_xray_strength[j*Nq_xray + i] = getIonXrayStrength(h_atom_names_tmp[j], h_q_xray[i], aff_constants_ions, h_charges[j]);
                }
            } else {        	    // Not ion
                for (int i=0; i<Nq_xray; ++i){
                    h_xray_strength[j*Nq_xray + i] = getXrayStrength(h_atom_names_tmp[j], h_q_xray[i], aff_constants, h_charges[j]);
                }
            }
        }
    }
    
    vector<float> h_xray_o_strength, h_xray_h_strength;
    h_xray_o_strength.reserve(Nq_xray);
    h_xray_h_strength.reserve(Nq_xray);
    for (int i=0; i<Nq_xray; ++i){
        h_xray_o_strength.push_back(getWaterXrayStrength("O", h_q_xray[i], aff_constants));
        h_xray_h_strength.push_back(getWaterXrayStrength("H", h_q_xray[i], aff_constants));
    }

    vector<float> h_neutron_o_strength, h_neutron_h_strength, h_neutron_d_strength;
    h_neutron_o_strength.push_back(getNeutronStrength("O", neutronFFmap));
    h_neutron_h_strength.push_back(getNeutronStrength("H", neutronFFmap));
    h_neutron_d_strength.push_back(getNeutronStrength("D", neutronFFmap));
    for (int j=0; j<Natoms; ++j){
        if (h_is_water[j]==1){          // Water
            for (int i=0; i<Nq_neutron; ++i){
                h_neutron_strength[j*Nq_neutron + i] = getWaterNeutronStrength(h_atom_names_tmp[j], h_d_parts[i], neutronFFmap);
            }
        } else {
            if (is_ion[j]) {            // Ion
                for (int i=0; i<Nq_neutron; ++i){
                    h_neutron_strength[j*Nq_neutron + i] = getIonNeutronStrength(h_atom_names_tmp[j], neutronFFmap_ions);
                }
            } else {                    // Not ion
                for (int i=0; i<Nq_neutron; ++i){
                    h_neutron_strength[j*Nq_neutron + i] = getNeutronStrength(h_atom_names_tmp[j], neutronFFmap);
                }
            }
        }
    }
    // Set neutron scattering length for exchangeable hydrogens
    for (int j=0; j<particles_exch_h_tmp.size(); ++j){
        for (int i=0; i<Nq_neutron; ++i){
            if ((h_atom_names_tmp[particles_exch_h_tmp[j]][0]!='H') || (h_atom_names_tmp[particles_exch_h_tmp[j]][0]!='D')){
                h_neutron_strength[particles_exch_h_tmp[j]*Nq_neutron + i] = getWaterNeutronStrength(h_atom_names_tmp[particles_exch_h_tmp[j]],
                        h_d_parts[i], neutronFFmap);
            } else {
                throw OpenMMException("Atom " + to_string(particles_exch_h_tmp[j]) + "is marked as exchangeable hydrogen, but the name isn't 'H' or 'D' ("
                                      + h_atom_names_tmp[particles_exch_h_tmp[j]] + ")");
            }
        }
    }

    xray_strength = CudaArray::create<float>(cu, Nq_xray*Natoms, "xray_strength");
    neutron_strength = CudaArray::create<float>(cu, Nq_neutron*Natoms, "neutron_strength");
    neutron_o_strength = CudaArray::create<float>(cu, 1, "neutron_o_strength");
    neutron_h_strength = CudaArray::create<float>(cu, 1, "neutron_h_strength");
    neutron_d_strength = CudaArray::create<float>(cu, 1, "neutron_d_strength");
    xray_o_strength = CudaArray::create<float>(cu, Nq_xray, "xray_o_strength");
    xray_h_strength = CudaArray::create<float>(cu, Nq_xray, "xray_h_strength");
    xray_strength->upload(h_xray_strength);
    neutron_strength->upload(h_neutron_strength);
    xray_o_strength->upload(h_xray_o_strength);
    xray_h_strength->upload(h_xray_h_strength);
    neutron_o_strength->upload(h_neutron_o_strength);
    neutron_h_strength->upload(h_neutron_h_strength);
    neutron_d_strength->upload(h_neutron_d_strength);

    vector<double> h_N;
    h_N.reserve(1);
    h_N.push_back(0.0);
    N = CudaArray::create<double>(cu, 1, "N");
    N->upload(h_N);

    vector<double> h_A_real_xray(Nq_xray, 0.0), h_A_real_neutron(Nq_neutron, 0.0), h_A_complex_xray(Nq_xray, 0.0),
            h_A_complex_neutron(Nq_neutron, 0.0), h_B_real_xray(Nq_xray, 0.0), h_B_real_neutron(Nq_neutron, 0.0);

    A_real_xray = CudaArray::create<double>(cu, Nq_xray, "A_real_xray");
    A_complex_xray = CudaArray::create<double>(cu, Nq_xray, "A_complex_xray");
    B_real_xray = CudaArray::create<double>(cu, Nq_xray, "B_real_xray");
    A_real_xray->upload(h_A_real_xray);
    A_complex_xray->upload(h_A_complex_xray);
    B_real_xray->upload(h_B_real_xray);
    F_total_xray = CudaArray::create<double>(cu, Nq_xray, "F_total_xray");
    F_total_xray->upload(h_A_real_xray);

    A_real_neutron = CudaArray::create<double>(cu, Nq_neutron, "A_real_neutron");
    A_complex_neutron = CudaArray::create<double>(cu, Nq_neutron, "A_complex_neutron");
    B_real_neutron = CudaArray::create<double>(cu, Nq_neutron, "B_real_neutron");
    A_real_neutron->upload(h_A_real_neutron);
    A_complex_neutron->upload(h_A_complex_neutron);
    B_real_neutron->upload(h_B_real_neutron);
    F_total_neutron = CudaArray::create<double>(cu, Nq_neutron, "F_total_neutron");
    F_total_neutron->upload(h_A_real_neutron);

    // Output
    write_output = force.getWriteOutFF();
    if (write_output) {
        std::string out_prefix;
        force.getOutPrefix(out_prefix);
        outfile_xray.open(out_prefix + "ff_xray.dat");
        outfile_neutron.open(out_prefix + "ff_neutron.dat");
        outfile_scale_factor_xray.open(out_prefix + "scale_xray.dat");
        outfile_scale_factor_neutron.open(out_prefix + "scale_neutron.dat");
    }
    F_total_xray_out.reserve(Nq_xray);
    F_total_neutron_out.reserve(Nq_neutron);
    
    // Out buffers
    vector<double> h_A_real_xray_out(Nq_xray, 0.0), h_A_real_neutron_out(Nq_neutron, 0.0), h_A_complex_xray_out(Nq_xray, 0.0),
            h_A_sqr_xray_out(Nq_xray, 0.0), h_A_complex_neutron_out(Nq_neutron, 0.0), h_A_sqr_neutron_out(Nq_neutron, 0.0);
    A_real_xray_out = CudaArray::create<double>(cu, Nq_xray, "A_real_xray_out");
    A_complex_xray_out = CudaArray::create<double>(cu, Nq_xray, "A_complex_xray_out");
    A_sqr_xray_out = CudaArray::create<double>(cu, Nq_xray, "A_sqr_xray_out");
    A_real_xray_out->upload(h_A_real_xray_out);
    A_complex_xray_out->upload(h_A_complex_xray_out);
    A_sqr_xray_out->upload(h_A_sqr_xray_out);

    A_real_neutron_out = CudaArray::create<double>(cu, Nq_neutron, "A_real_neutron_out");
    A_complex_neutron_out = CudaArray::create<double>(cu, Nq_neutron, "A_complex_neutron_out");
    A_sqr_neutron_out = CudaArray::create<double>(cu, Nq_neutron, "A_sqr_neutron_out");
    A_real_neutron_out->upload(h_A_real_neutron_out);
    A_complex_neutron_out->upload(h_A_complex_neutron_out);
    A_sqr_neutron_out->upload(h_A_sqr_neutron_out);

    // Buffer for global F_total calculation. Reuse A.._out, so we need B components only
    vector<double> h_B_real_xray_global(Nq_xray, 0.0), h_B_sqr_xray_global(Nq_xray, 0.0),
            h_B_real_neutron_global(Nq_neutron, 0.0), h_B_sqr_neutron_global(Nq_neutron, 0.0);
    B_real_xray_global = CudaArray::create<double>(cu, Nq_xray, "B_real_xray_global");
    B_sqr_xray_global = CudaArray::create<double>(cu, Nq_xray, "B_sqr_xray_global");
    B_real_neutron_global = CudaArray::create<double>(cu, Nq_neutron, "B_real_neutron_global");
    B_sqr_neutron_global = CudaArray::create<double>(cu, Nq_neutron, "B_sqr_neutron_global");
    B_real_xray_global->upload(h_B_real_xray_global);
    B_sqr_xray_global->upload(h_B_sqr_xray_global);
    B_real_neutron_global->upload(h_B_real_neutron_global);
    B_sqr_neutron_global->upload(h_B_sqr_neutron_global);

    // Should reserve the arrays of size not smaller than number of threads used.
    // Number of threads is arbitrary, but we run global reductions in one block in some kernels, so decreasing the number of threads
    // will degrade performance. Modern GPUs should be able to run at least 1024 threads/block.
    int Num_threads_test = 1024;

    vector<double> h_A_real_xray_current(Num_threads_test*Nq_xray, 0.0), h_A_real_neutron_current(Num_threads_test*Nq_neutron, 0.0),
            h_A_complex_xray_current(Num_threads_test*Nq_xray, 0.0), h_A_sqr_xray_current(Num_threads_test*Nq_xray, 0.0),
            h_A_complex_neutron_current(Num_threads_test*Nq_neutron, 0.0), h_A_sqr_neutron_current(Num_threads_test*Nq_neutron, 0.0);
    A_real_xray_current = CudaArray::create<double>(cu, Num_threads_test*Nq_xray, "A_real_xray_current");
    A_complex_xray_current = CudaArray::create<double>(cu, Num_threads_test*Nq_xray, "A_complex_xray_current");
    A_sqr_xray_current = CudaArray::create<double>(cu, Num_threads_test*Nq_xray, "A_sqr_xray_current");
    A_real_xray_current->upload(h_A_real_xray_current);
    A_complex_xray_current->upload(h_A_complex_xray_current);
    A_sqr_xray_current->upload(h_A_sqr_xray_current);

    A_real_neutron_current = CudaArray::create<double>(cu, Num_threads_test*Nq_neutron, "A_real_neutron_current");
    A_complex_neutron_current = CudaArray::create<double>(cu, Num_threads_test*Nq_neutron, "A_complex_neutron_current");
    A_sqr_neutron_current = CudaArray::create<double>(cu, Num_threads_test*Nq_neutron, "A_sqr_neutron_current");
    A_real_neutron_current->upload(h_A_real_neutron_current);
    A_complex_neutron_current->upload(h_A_complex_neutron_current);
    A_sqr_neutron_current->upload(h_A_sqr_neutron_current);

    vector<double> h_B_real_xray_current(Nq_xray, 0.0), h_B_sqr_xray_current(Nq_xray, 0.0),
            h_B_real_neutron_current(Nq_neutron, 0.0), h_B_sqr_neutron_current(Nq_neutron, 0.0);
    B_real_xray_current = CudaArray::create<double>(cu, Nq_xray, "B_real_xray_current");
    B_sqr_xray_current = CudaArray::create<double>(cu, Nq_xray, "B_sqr_xray_current");
    B_real_xray_current->upload(h_B_real_xray_current);
    B_sqr_xray_current->upload(h_B_sqr_xray_current);

    B_real_neutron_current = CudaArray::create<double>(cu, Nq_neutron, "B_real_neutron_current");
    B_sqr_neutron_current = CudaArray::create<double>(cu, Nq_neutron, "B_sqr_neutron_current");
    B_real_neutron_current->upload(h_B_real_neutron_current);
    B_sqr_neutron_current->upload(h_B_sqr_neutron_current);

    alpha = OpenMM::CudaArray::create<float>(cu, 1, "alpha");
    origin = OpenMM::CudaArray::create<double>(cu, 1, "origin");

    box = OpenMM::CudaArray::create<float>(cu, 3, "box");
    vector<double> h_tau_tmp;
    h_tau_tmp.reserve(1);
    h_tau_tmp.push_back(h_tau);
    tau = OpenMM::CudaArray::create<double>(cu, 1, "tau");
    tau->upload(h_tau_tmp);

    vector<double> h_T_tmp;
    h_T_tmp.reserve(1);
    h_T_tmp.push_back(h_T);
    T = OpenMM::CudaArray::create<double>(cu, 1, "T");
    T->upload(h_T_tmp);
    vector<double> h_k_xray_tmp;
    h_k_xray_tmp.reserve(1);
    h_k_xray_tmp.push_back(h_k_xray);
    k_xray = OpenMM::CudaArray::create<double>(cu, 1, "k_xray");
    k_xray->upload(h_k_xray_tmp);
    vector<double> h_k_neutron_tmp;
    h_k_neutron_tmp.reserve(1);
    h_k_neutron_tmp.push_back(h_k_neutron);
    k_neutron = OpenMM::CudaArray::create<double>(cu, 1, "k_neutron");
    k_neutron->upload(h_k_neutron_tmp);
    vector<double> h_w_dens_tmp;
    h_w_dens_tmp.reserve(1);
    h_w_dens_tmp.push_back(h_w_dens);
    w_dens = OpenMM::CudaArray::create<double>(cu, 1, "w_dens");
    w_dens->upload(h_w_dens_tmp);
    vector<double> h_w_dens_sqr_tmp;
    h_w_dens_sqr_tmp.reserve(1);
    h_w_dens_sqr_tmp.push_back(h_w_dens_sqr);
    w_dens_sqr = OpenMM::CudaArray::create<double>(cu, 1, "w_dens_sqr");
    w_dens_sqr->upload(h_w_dens_sqr_tmp);
    vector<float> h_cutoff_tmp;
    h_cutoff_tmp.reserve(1);
    h_cutoff_tmp.push_back(h_cutoff);
    cutoff = OpenMM::CudaArray::create<float>(cu, 1, "cutoff");
    cutoff->upload(vector<float>(1, h_cutoff));
    
    prev_box.reserve(3);
    prev_box.push_back(0.0);
    prev_box.push_back(0.0);
    prev_box.push_back(0.0);

    origin_buffer = OpenMM::CudaArray::create<double>(cu, Num_threads_test, "origin_buffer");
    energy_buffer = OpenMM::CudaArray::create<float>(cu, Num_threads_test, "energy_buffer");

    std::map<std::string, std::string> replacements;
    std::map<std::string, std::string> defines;
    defines["Natoms"] = cu.intToString(Natoms);
    defines["PADDED_NUM_ATOMS"] = cu.intToString(cu.getPaddedNumAtoms());
    defines["ORIGIN_NUM_ATOMS"] = cu.intToString(originNatoms);
    defines["total_mass_origin"] = cu.doubleToString(total_mass_origin);
    defines["particles_size"] = cu.intToString(particles->getSize());
    defines["Nq_xray"] = cu.intToString(Nq_xray);
    defines["Nq_neutron"] = cu.intToString(Nq_neutron);
    CUmodule module = cu.createModule(cu.replaceStrings(CudaREKernelSources::sanxsForce, replacements), defines);

    computeOrigin = cu.getKernel(module, "computeOrigin");
    computePreFFtotal = cu.getKernel(module, "computePreFFtotal");
    computeFFtotal = cu.getKernel(module, "computeFFtotal");
    computePostFFtotal = cu.getKernel(module, "computePostFFtotal");
    computePreForceKernel = cu.getKernel(module, "computePreForce");
    computeForceKernel = cu.getKernel(module, "computeForce");
    computeEnergyKernel = cu.getKernel(module, "computeEnergy");
    computeGlobalFtotal = cu.getKernel(module, "computeGlobalFtotal");
    
    cu.addForce(new CudaREForceInfo(force));
}

double CudaCalcREForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    // Number of threads is arbitrary, but we run global reductions in one block in some kernels, so decreasing the number of threads
    // will degrade performance. Modern GPUs should be able to run at least 1024 threads/block.
    int Num_threads_test = 1024;

    // Update box only if changed from the last step
    Vec3 a, b, c;
    context.getPeriodicBoxVectors(a, b, c);
    vector<float> h_box;
    h_box.push_back(a[0]);
    h_box.push_back(b[1]);
    h_box.push_back(c[2]);
    if ((c[0]!=0.0) || (c[1]!=0.0)){
        throw OpenMMException("Periodic box is not rectangular!");
    }
    if ((prev_box[0]!=h_box[0]) || (prev_box[1]!=h_box[1]) || (prev_box[2]!=h_box[2])){
        box->upload(h_box);
    }
    prev_box[0] = h_box[0];
    prev_box[1] = h_box[1];
    prev_box[2] = h_box[2];

    // We introduce force gradually at the very beginning of the run. That might increase the algorithm stability if we significantly
    // change the force magnitude compared to the previous run.
    // Alpha is the factor that the force is being multiplied with.
    float current_time = context.getTime();
    std::vector<float> h_alpha;
    if (current_time > 2.0*h_tau) {
        h_alpha.push_back(1.0);
    } else if (current_time < h_tau) {
        h_alpha.push_back(0.0);
    } else {
        h_alpha.push_back(0.5*(1.0 - cos(M_PI*(current_time - h_tau)/h_tau)));
    }
    alpha->upload(h_alpha);

    // Kernel params
    void* args_origin[] = {
        &cu.getPosq().getDevicePointer(),
        &origin->getDevicePointer(),
        &origin_buffer->getDevicePointer(),
        &particles_for_origin->getDevicePointer(),
        &mass_for_origin->getDevicePointer()
    };
    void* args_pre_ff_total[] = {
        &A_real_xray_current->getDevicePointer(),
        &A_real_neutron_current->getDevicePointer(),
        &A_complex_xray_current->getDevicePointer(),
        &A_complex_neutron_current->getDevicePointer()
    };
    void* args_ff_total[] = {
        &cu.getPosq().getDevicePointer(),
        &origin->getDevicePointer(),
        &box->getDevicePointer(),
        &is_water->getDevicePointer(),
        &xray_strength->getDevicePointer(),
        &neutron_strength->getDevicePointer(),
        &atom_names->getDevicePointer(),
        &xray_qs->getDevicePointer(),
        &neutron_qs->getDevicePointer(),
        &cutoff->getDevicePointer(),
        &d_parts->getDevicePointer(),                               
        &A_real_xray_current->getDevicePointer(),
        &A_real_neutron_current->getDevicePointer(),
        &A_complex_xray_current->getDevicePointer(),
        &A_complex_neutron_current->getDevicePointer()
    };
    void* args_post_ff_total[] = {
        &A_real_xray_current->getDevicePointer(),
        &A_real_neutron_current->getDevicePointer(),
        &A_complex_xray_current->getDevicePointer(),
        &A_complex_neutron_current->getDevicePointer(),
        &A_real_xray_out->getDevicePointer(),
        &A_real_neutron_out->getDevicePointer(),
        &A_complex_xray_out->getDevicePointer(),
        &A_complex_neutron_out->getDevicePointer(),
        &A_sqr_xray_out->getDevicePointer(),
        &A_sqr_neutron_out->getDevicePointer()
    };
    void* args_pre_force[] = {
        &cu.getPosq().getDevicePointer(),
        &box->getDevicePointer(),
        &xray_h_strength->getDevicePointer(),
        &xray_o_strength->getDevicePointer(),
        &xray_qs->getDevicePointer(),
        &neutron_qs->getDevicePointer(),
        &neutron_o_strength->getDevicePointer(),
        &neutron_h_strength->getDevicePointer(),
        &neutron_d_strength->getDevicePointer(),
        &w_dens->getDevicePointer(),
        &w_dens_sqr->getDevicePointer(),
        &cutoff->getDevicePointer(),
        &d_parts->getDevicePointer(),
        &B_real_xray_current->getDevicePointer(),
        &B_real_neutron_current->getDevicePointer(),
        &B_sqr_xray_current->getDevicePointer(),
        &B_sqr_neutron_current->getDevicePointer()
    };
    void* args_force[] = {
        &cu.getPosq().getDevicePointer(),
        &alpha->getDevicePointer(),
        &origin->getDevicePointer(),
        &k_xray->getDevicePointer(),
        &k_neutron->getDevicePointer(),
        &T->getDevicePointer(),
        &box->getDevicePointer(),
        &is_water->getDevicePointer(),
        &xray_strength->getDevicePointer(),
        &neutron_strength->getDevicePointer(),
        &atom_names->getDevicePointer(),
        &xray_qs->getDevicePointer(),
        &neutron_qs->getDevicePointer(),
        &cutoff->getDevicePointer(),
        &d_parts->getDevicePointer(),
        &B_real_xray_current->getDevicePointer(),
        &B_real_neutron_current->getDevicePointer(),
        &A_real_xray_out->getDevicePointer(),
        &A_real_neutron_out->getDevicePointer(),
        &A_complex_xray_out->getDevicePointer(),
        &A_complex_neutron_out->getDevicePointer(),
        &F_total_xray->getDevicePointer(),
        &F_total_neutron->getDevicePointer(),
        &particles->getDevicePointer(),
        &F_exp_xray->getDevicePointer(),
        &F_exp_neutron->getDevicePointer(),
        &delta_F_exp_xray->getDevicePointer(),
        &delta_F_exp_neutron->getDevicePointer(),
        &cu.getForce().getDevicePointer()
    };
    void* args_energy[] = {
        &alpha->getDevicePointer(),
        &energy_buffer->getDevicePointer(),
        &k_xray->getDevicePointer(),
        &k_neutron->getDevicePointer(),
        &T->getDevicePointer(),
        &F_total_xray->getDevicePointer(),
        &F_total_neutron->getDevicePointer(),
        &F_exp_xray->getDevicePointer(),
        &F_exp_neutron->getDevicePointer(),
        &delta_F_exp_xray->getDevicePointer(),
        &delta_F_exp_neutron->getDevicePointer(),
        &cu.getEnergyBuffer().getDevicePointer()
    };

    cu.executeKernel(computeOrigin, args_origin, 1, Num_threads_test);

    if (cu.getStepCount() % couple_steps == 0){
        // Here we calculate A_current
        cu.executeKernel(computePreFFtotal, args_pre_ff_total, 1, Num_threads_test);
        cu.executeKernel(computeFFtotal, args_ff_total, Natoms, Num_threads_test);
        cu.executeKernel(computePostFFtotal, args_post_ff_total, 1, Num_threads_test);

        // Output
        if (write_output) {
            // F_total output
            F_total_xray->download(F_total_xray_out);
            F_total_neutron->download(F_total_neutron_out);
            for(int i=0; i<F_total_xray_out.size(); ++i){
                outfile_xray << F_total_xray_out[i] << " " ;
            }
            outfile_xray << endl;
            for(int i=0; i<F_total_neutron_out.size(); ++i){
                outfile_neutron << F_total_neutron_out[i] << " " ;
            }
            outfile_neutron << endl;

            // Scaling factors output
            for (int i=0; i<scale_factor_xray.size(); ++i){
                outfile_scale_factor_xray << scale_factor_xray[i] << " ";
            }
            outfile_scale_factor_xray << endl;
            for (int i=0; i<scale_factor_neutron.size(); ++i){
                outfile_scale_factor_neutron << scale_factor_neutron[i] << " ";
            }
            outfile_scale_factor_neutron << endl;
        }
    }

    // Here we calculate force, B_current
    cu.executeKernel(computePreForceKernel, args_pre_force, 1, Num_threads_test);
    cu.executeKernel(computeForceKernel, args_force, particles->getSize(), Num_threads_test);
    cu.executeKernel(computeEnergyKernel, args_energy, 1, Num_threads_test);
    
    return 0.0;
}

template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

void CudaCalcREForceKernel::copyParametersToContext(ContextImpl& context, const REForce& force, bool on_gpu) {
    // Define if we are using CPU or GPU version of the function
    if (!on_gpu) {
        // Does the calculation on the host CPU
        cu.setAsCurrent();
        vector<double> h_A_real_xray, h_A_complex_xray, h_A_real_neutron, h_A_complex_neutron, h_A_sqr_xray, h_A_sqr_neutron,
                       h_B_real_xray, h_B_sqr_xray, h_B_real_neutron, h_B_sqr_neutron;

        // Get averaged components
        force.getAComponents(h_A_real_xray, h_A_complex_xray, h_A_sqr_xray, h_A_real_neutron, h_A_complex_neutron, h_A_sqr_neutron, h_B_real_xray, h_B_sqr_xray, h_B_real_neutron, h_B_sqr_neutron);

        // Compute F_total
        vector<double> h_F_total_xray, h_F_total_neutron;
        h_F_total_xray.reserve(Nq_xray);
        h_F_total_neutron.reserve(Nq_neutron);
        for (int i=0; i<Nq_xray; ++i){
            double intens = h_A_real_xray[i]*h_A_real_xray[i] + h_A_complex_xray[i]*h_A_complex_xray[i]
                                          + h_B_real_xray[i]*h_B_real_xray[i] - 2.0*h_A_real_xray[i]*h_B_real_xray[i]
                                          + h_A_sqr_xray[i] - h_A_real_xray[i]*h_A_real_xray[i] - h_A_complex_xray[i]*h_A_complex_xray[i]
                                          - h_B_sqr_xray[i] + h_B_real_xray[i]*h_B_real_xray[i];

            h_F_total_xray.push_back(sgn(intens)*sqrt(fabs(intens)));
        }
        for (int i=0; i<Nq_neutron; ++i){
            double intens = h_A_real_neutron[i]*h_A_real_neutron[i] + h_A_complex_neutron[i]*h_A_complex_neutron[i]
                                             + h_B_real_neutron[i]*h_B_real_neutron[i] - 2.0*h_A_real_neutron[i]*h_B_real_neutron[i]
                                             + h_A_sqr_neutron[i] - h_A_real_neutron[i]*h_A_real_neutron[i] - h_A_complex_neutron[i]*h_A_complex_neutron[i]
                                             - h_B_sqr_neutron[i] + h_B_real_neutron[i]*h_B_real_neutron[i];
            h_F_total_neutron.push_back(sgn(intens)*sqrt(fabs(intens)));
        }
        F_total_xray->upload(h_F_total_xray);
        F_total_neutron->upload(h_F_total_neutron);
    } else {
        // Does the calculation on GPU. Might be faster to do on host if the number of exp points is not very high.
        cu.setAsCurrent();
        vector<double> h_A_real_xray, h_A_complex_xray, h_A_real_neutron, h_A_complex_neutron, h_A_sqr_xray, h_A_sqr_neutron,
                       h_B_real_xray, h_B_sqr_xray, h_B_real_neutron, h_B_sqr_neutron;

        // Get averaged components
        force.getAComponents(h_A_real_xray, h_A_complex_xray, h_A_sqr_xray, h_A_real_neutron, h_A_complex_neutron, h_A_sqr_neutron, h_B_real_xray, h_B_sqr_xray, h_B_real_neutron, h_B_sqr_neutron);

        // Upload and do calculations on gpu
        A_real_xray_out->upload(h_A_real_xray);
        A_complex_xray_out->upload(h_A_complex_xray);
        A_sqr_xray_out->upload(h_A_sqr_xray);
        A_real_neutron_out->upload(h_A_real_neutron);
        A_complex_neutron_out->upload(h_A_complex_neutron);
        A_sqr_neutron_out->upload(h_A_sqr_neutron);
        B_real_xray_global->upload(h_B_real_xray);
        B_sqr_xray_global->upload(h_B_sqr_xray);
        B_real_neutron_global->upload(h_B_real_neutron);
        B_sqr_neutron_global->upload(h_B_sqr_neutron);

        void* args_GlobalFtotal[] = {
            &A_real_xray_out->getDevicePointer(),
            &A_complex_xray_out->getDevicePointer(),
            &A_sqr_xray_out->getDevicePointer(),
            &A_real_neutron_out->getDevicePointer(),
            &A_complex_neutron_out->getDevicePointer(),
            &A_sqr_neutron_out->getDevicePointer(),
            &B_real_xray_global->getDevicePointer(),
            &B_sqr_xray_global->getDevicePointer(),
            &B_real_neutron_global->getDevicePointer(),
            &B_sqr_neutron_global->getDevicePointer(),
            &F_total_xray->getDevicePointer(),
            &F_total_neutron->getDevicePointer()
        };

        cu.executeKernel(computeGlobalFtotal, args_GlobalFtotal, Nq_xray+Nq_neutron, 128);
    }
}

void CudaCalcREForceKernel::copyParametersFromContext(ContextImpl& context, const REForce& force) {
    REForce& force_editable = const_cast<REForce&>(force);
    vector<double> h_A_real_xray_current, h_A_complex_xray_current, h_A_real_neutron_current, h_A_complex_neutron_current, h_A_sqr_xray_current, h_A_sqr_neutron_current,
                   h_B_real_xray_current, h_B_sqr_xray_current, h_B_real_neutron_current, h_B_sqr_neutron_current;

    h_B_real_xray_current.reserve(Nq_xray);
    h_B_sqr_xray_current.reserve(Nq_xray);
    h_B_real_neutron_current.reserve(Nq_neutron);
    h_B_sqr_neutron_current.reserve(Nq_neutron);
    h_A_real_xray_current.reserve(Nq_xray);
    h_A_complex_xray_current.reserve(Nq_xray);
    h_A_real_neutron_current.reserve(Nq_neutron);
    h_A_complex_neutron_current.reserve(Nq_neutron);

    A_real_xray_out->download(h_A_real_xray_current);
    A_complex_xray_out->download(h_A_complex_xray_current);
    A_sqr_xray_out->download(h_A_sqr_xray_current);
    A_real_neutron_out->download(h_A_real_neutron_current);
    A_complex_neutron_out->download(h_A_complex_neutron_current);
    A_sqr_neutron_out->download(h_A_sqr_neutron_current);
    B_real_xray_current->download(h_B_real_xray_current);
    B_sqr_xray_current->download(h_B_sqr_xray_current);
    B_real_neutron_current->download(h_B_real_neutron_current);
    B_sqr_neutron_current->download(h_B_sqr_neutron_current);

    force_editable.setAComponents(h_A_real_xray_current, h_A_complex_xray_current, h_A_sqr_xray_current, h_A_real_neutron_current, h_A_complex_neutron_current, h_A_sqr_neutron_current,
                                  h_B_real_xray_current, h_B_sqr_xray_current, h_B_real_neutron_current, h_B_sqr_neutron_current);
}

// Does the job of calculating the scaling factor.
double calcScalingFactor(vector<double> F_exp, vector<double> delta_F_exp, vector<double> F_current) {

   int data_size = F_exp.size();

   //calculate the denominator
   double denominator = 0, numerator = 0;

   for(int i = 0; i < data_size; ++i) {
       double exp_val = F_exp[i];
       double sim_val = F_current[i];
       double uncert_squared = pow(delta_F_exp[i],2);
       denominator += pow(exp_val,2) / uncert_squared;
       numerator += fabs(exp_val * sim_val) / uncert_squared;
   }

   double scaling_factor = numerator / denominator;

   return scaling_factor;
}

void CudaCalcREForceKernel::rescaleExpF(OpenMM::ContextImpl& context, const REForce& force){
    // We get current F_exp_xray/neutron, scale it to F_total and upload new exp values. Scale both F and delta_F.
    // Also keep scale factors for the output/check.

    scale_factor_xray.clear();
    scale_factor_neutron.clear();

    // Get current sim values of F
    vector<double> h_F_total_xray, h_F_total_neutron;
    h_F_total_xray.reserve(Nq_xray);
    h_F_total_neutron.reserve(Nq_neutron);
    F_total_xray->download(h_F_total_xray);
    F_total_neutron->download(h_F_total_neutron);

    vector<double> h_F_exp_xray, h_delta_F_exp_xray, h_F_exp_neutron, h_delta_F_exp_neutron;
    int neutron_qs_done = 0;
    int xray_qs_done = 0;
    for (int i=0; i<Fs_exp.size(); ++i){
        if (Fs_exp[i].is_neutron) {
            // Get a scaling factor for a current exp data set
            // We rely on the original order in which F values are in F_total, i.e. that first Fs_exp[0].q.size() values of F_total_neutron correspond to Fs_exp[0].F and so on.

            vector<double> tmp_h_F_total_neutron;   // a part of F_total_neutron which corresponds to the current Fs_exp[i] data set.
            for (int k=0; k<Fs_exp[i].q.size(); ++k){
                tmp_h_F_total_neutron.push_back(h_F_total_neutron[k + neutron_qs_done]);
            }
            neutron_qs_done += Fs_exp[i].q.size();

            double scale_factor = calcScalingFactor(Fs_exp[i].F, Fs_exp[i].delta_F, tmp_h_F_total_neutron);

            // Scale
            for (int j=0; j<Fs_exp[i].q.size(); ++j){
                h_F_exp_neutron.push_back(Fs_exp[i].F[j] * scale_factor);
                h_delta_F_exp_neutron.push_back(Fs_exp[i].delta_F[j] * scale_factor);
            }
            scale_factor_neutron.push_back(scale_factor);
        } else {
            vector<double> tmp_h_F_total_xray;   // a part of F_total_xray which corresponds to the current Fs_exp[i] data set.
            for (int k=0; k<Fs_exp[i].q.size(); ++k){
                tmp_h_F_total_xray.push_back(h_F_total_xray[k + xray_qs_done]);
            }
            xray_qs_done += Fs_exp[i].q.size();

            double scale_factor = calcScalingFactor(Fs_exp[i].F, Fs_exp[i].delta_F, tmp_h_F_total_xray);

            for (int j=0; j<Fs_exp[i].q.size(); ++j){
                h_F_exp_xray.push_back(Fs_exp[i].F[j] * scale_factor);
                h_delta_F_exp_xray.push_back(Fs_exp[i].delta_F[j] * scale_factor);
            }
            scale_factor_xray.push_back(scale_factor);
        }
    }

    // Check the size
    if (h_F_exp_xray.size() != h_F_total_xray.size())
        cout << "WARNING: Wrong size of xray vectors after scaling! (CudaCalcREForceKernel::rescaleExpF())" << endl;
    if (h_F_exp_neutron.size() != h_F_total_neutron.size())
        cout << "WARNING: Wrong size of neutron vectors after scaling! (CudaCalcREForceKernel::rescaleExpF())" << endl;

    // Upload the scaled values
    F_exp_xray->upload(h_F_exp_xray);
    delta_F_exp_xray->upload(h_delta_F_exp_xray);
    F_exp_neutron->upload(h_F_exp_neutron);
    delta_F_exp_neutron->upload(h_delta_F_exp_neutron);
}
