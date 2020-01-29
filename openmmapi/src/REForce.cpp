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

#include "REForce.h"
#include "internal/REForceImpl.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/AssertionUtilities.h"

using namespace REPlugin;
using namespace OpenMM;
using namespace std;

REForce::REForce() {
}

void REForce::addParticle(int i){
    particles.push_back(i);
}

void REForce::addParticles(std::vector<int> particles1){
    for (int i=0; i<particles1.size(); ++i){
        particles.push_back(particles1[i]);
    }
}

void REForce::addParticleOrigin(int i){
    particles_for_origin.push_back(i);
}

void REForce::addParticlesOrigin(std::vector<int> particles1){
    for (int i=0; i<particles1.size(); ++i){
        particles_for_origin.push_back(particles1[i]);
    }
}

void REForce::addParticleWater(int i){
    particles_water.push_back(i);
}

void REForce::addParticlesWater(std::vector<int> particles_water1){
    for (int i=0; i<particles_water1.size(); ++i){
        particles_water.push_back(particles_water1[i]);
    }
}

void REForce::addParticleIon(int i){
    particles_ion.push_back(i);
}

void REForce::addParticlesIon(std::vector<int> particles_ion1){
    for (int i=0; i<particles_ion1.size(); ++i){
        particles_ion.push_back(particles_ion1[i]);
    }
}

void REForce::addParticleExchH(int i){
    particles_exch_h.push_back(i);
}

void REForce::addParticlesExchH(std::vector<int> particles_exch_h1){
    for (int i=0; i<particles_exch_h1.size(); ++i){
        particles_exch_h.push_back(particles_exch_h1[i]);
    }
}

void REForce::addName(string name){
    atom_names.push_back(name);
}

void REForce::addNames(std::vector<string> names){
    for (int i=0; i<names.size(); ++i){
        atom_names.push_back(names[i]);
    }
}

void REForce::addCharge(float charge){
    charges.push_back(charge);
}

void REForce::addCharges(std::vector<float> charge){
    for (int i=0; i<charge.size(); ++i){
        charges.push_back(charge[i]);
    }
}

void REForce::setCoupleSteps(int c_steps){
    coupl_steps = c_steps;
}

void REForce::setK(double k_xray1, double k_neutron1){
    k_xray = k_xray1;
    k_neutron = k_neutron1;
}

void REForce::setTau(double tau1){
    tau = tau1;
}

void REForce::setTemp(double T1){
    T = T1;
}

void REForce::setWdens(double w_dens1){
    w_dens = w_dens1;
}

void REForce::setWdensSqr(double w_dens_sqr1){
    w_dens_sqr = w_dens_sqr1;
}

void REForce::setCutoff(double cutoff1){
    cutoff = cutoff1;
}

void REForce::setAllParams(double T1, double k_xray1, double k_neutron1, double w_dens1, double w_dens_sqr1, double tau1, float cutoff1){
    T = T1;
    k_xray = k_xray1;
    k_neutron = k_neutron1;
    w_dens = w_dens1;
    w_dens_sqr = w_dens_sqr1;
    tau = tau1;
    cutoff = cutoff1;
}

void REForce::getAllParams(double &T1, double& k_xray1, double& k_neutron1, double &w_dens1, double &w_dens_sqr1, double &tau1, float& cutoff1) const {
    T1 = T;
    k_xray1 = k_xray;
    k_neutron1 = k_neutron;
    w_dens1 = w_dens;
    w_dens_sqr1 = w_dens_sqr;
    tau1 = tau;
    cutoff1 = cutoff;
}

int REForce::getCoupleSteps() const{
    return coupl_steps;
}

void REForce::getAllParticles(std::vector<int> &particles1, std::vector<int> &particles_for_origin1, std::vector<int> &particles_water1,
                                   std::vector<int> &particles_ion1, std::vector<int> &particles_exch_h1) const {
    particles1.resize(0);
    particles_for_origin1.resize(0);
    particles_water1.resize(0);
    particles_ion1.resize(0);
    particles_exch_h1.resize(0);
    for (int i=0; i<particles.size(); ++i){
        particles1.push_back(particles[i]);
    }
    for (int i=0; i<particles_for_origin.size(); ++i){
        particles_for_origin1.push_back(particles_for_origin[i]);
    }
    for (int i=0; i<particles_water.size(); ++i){
        particles_water1.push_back(particles_water[i]);
    }
    for (int i=0; i<particles_ion.size(); ++i){
        particles_ion1.push_back(particles_ion[i]);
    }
    for (int i=0; i<particles_exch_h.size(); ++i){
        particles_exch_h1.push_back(particles_exch_h[i]);
    }
}

void REForce::addExpFF(std::vector<double> F, std::vector<double> delta_F, std::vector<double> q, bool neutron, float d_part){
    formFactor tmp;
    for (int i=0; i<F.size(); ++i){
	tmp.F.push_back(F[i]);
	tmp.delta_F.push_back(delta_F[i]);
	tmp.q.push_back(q[i]);
    }
    tmp.is_neutron = neutron;
    tmp.d_part = d_part;
    Fs_exp.push_back(tmp);
}

int REForce::getNumExpFF() const{
    return Fs_exp.size();
}

void REForce::getExpFF(int i, std::vector<double>& F1, std::vector<double>& delta_F1, std::vector<double>& q1, bool& is_neutron1, float& d_part1) const {
    if (Fs_exp.size()>0){
	F1.clear();
	delta_F1.clear();
	q1.clear();
	for (int j=0; j<Fs_exp[i].F.size(); ++j){
    	    F1.push_back(Fs_exp[i].F[j]);
    	    delta_F1.push_back(Fs_exp[i].delta_F[j]);
    	    q1.push_back(Fs_exp[i].q[j]);
        }
        is_neutron1 = Fs_exp[i].is_neutron;
        d_part1 = Fs_exp[i].d_part;
    } else {
        throw OpenMMException("CalcREForceKernel::getExpFF No experimental data were added!");
    }
}

void REForce::getNames(std::vector<std::string> &names) const{
    names.resize(0);
    for (int i=0; i<atom_names.size(); ++i){
        names.push_back(atom_names[i]);
    }
}

void REForce::getCharges(std::vector<float> &charge) const{
    charge.resize(0);
    for (int i=0; i<charges.size(); ++i){
        charge.push_back(charges[i]);
    }
}

void REForce::setWriteOutFF(bool write_output) {
    write_out = write_output;
}

bool REForce::getWriteOutFF() const {
    return write_out;
}

void REForce::setOutPrefix(std::string out_prefix) {
    out_file_prefix = out_prefix;
}
void REForce::getOutPrefix(std::string &out_prefix) const{
    out_prefix = out_file_prefix;
}

void REForce::setAComponents(std::vector<double> A_real_xray, std::vector<double> A_complex_xray, std::vector<double> A_sqr_xray,
                                  std::vector<double> A_real_neutron, std::vector<double> A_complex_neutron, std::vector<double> A_sqr_neutron,
                                  std::vector<double> B_real_xray, std::vector<double> B_sqr_xray, std::vector<double> B_real_neutron, std::vector<double> B_sqr_neutron) {
    A_real_xray_current.clear();
    A_complex_xray_current.clear();
    A_sqr_xray_current.clear();
    A_real_neutron_current.clear();
    A_complex_neutron_current.clear();
    A_sqr_neutron_current.clear();
    B_real_xray_current.clear();
    B_sqr_xray_current.clear();
    B_real_neutron_current.clear();
    B_sqr_neutron_current.clear();
    for (int i=0; i<A_real_xray.size(); ++i){
        A_real_xray_current.push_back(A_real_xray[i]);
        A_complex_xray_current.push_back(A_complex_xray[i]);
        A_sqr_xray_current.push_back(A_sqr_xray[i]);
        B_real_xray_current.push_back(B_real_xray[i]);
        B_sqr_xray_current.push_back(B_sqr_xray[i]);
    }
    for (int i=0; i<A_real_neutron.size(); ++i){
        A_real_neutron_current.push_back(A_real_neutron[i]);
        A_complex_neutron_current.push_back(A_complex_neutron[i]);
        A_sqr_neutron_current.push_back(A_sqr_neutron[i]);
        B_real_neutron_current.push_back(B_real_neutron[i]);
        B_sqr_neutron_current.push_back(B_sqr_neutron[i]);
    }
}

void REForce::getAComponents(std::vector<double>& A_real_xray, std::vector<double>& A_complex_xray, std::vector<double>& A_sqr_xray,
                                  std::vector<double>& A_real_neutron, std::vector<double>& A_complex_neutron, std::vector<double>& A_sqr_neutron,
                                  std::vector<double> &B_real_xray, std::vector<double> &B_sqr_xray, std::vector<double> &B_real_neutron, std::vector<double> &B_sqr_neutron) const {;
    A_real_xray.resize(0);
    A_complex_xray.resize(0);
    A_sqr_xray.resize(0);
    A_real_neutron.resize(0);
    A_complex_neutron.resize(0);
    A_sqr_neutron.resize(0);
    B_real_xray.resize(0);
    B_sqr_xray.resize(0);
    B_real_neutron.resize(0);
    B_sqr_neutron.resize(0);
    for (int i=0; i<A_real_xray_current.size(); ++i){
        A_real_xray.push_back(A_real_xray_current[i]);
        A_complex_xray.push_back(A_complex_xray_current[i]);
        A_sqr_xray.push_back(A_sqr_xray_current[i]);
        B_real_xray.push_back(B_real_xray_current[i]);
        B_sqr_xray.push_back(B_sqr_xray_current[i]);
    }
    for (int i=0; i<A_real_neutron_current.size(); ++i){
        A_real_neutron.push_back(A_real_neutron_current[i]);
        A_complex_neutron.push_back(A_complex_neutron_current[i]);
        A_sqr_neutron.push_back(A_sqr_neutron_current[i]);
        B_real_neutron.push_back(B_real_neutron_current[i]);
        B_sqr_neutron.push_back(B_sqr_neutron_current[i]);
    }
}

ForceImpl* REForce::createImpl() const {
    return new REForceImpl(*this);
}

void REForce::updateParametersInContext(Context& context, bool on_gpu) {
    dynamic_cast<REForceImpl&>(getImplInContext(context)).updateParametersInContext(getContextImpl(context), on_gpu);
}

void REForce::getParametersFromContext(Context& context) {
    dynamic_cast<REForceImpl&>(getImplInContext(context)).getParametersFromContext(getContextImpl(context));
}

void REForce::rescaleExpFInContext(Context& context){
    dynamic_cast<REForceImpl&>(getImplInContext(context)).rescaleExpFInContext(getContextImpl(context));
}
