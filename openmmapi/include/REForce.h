#ifndef OPENMM_REFORCE_H_
#define OPENMM_REFORCE_H_

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

#include "openmm/Context.h"
#include "openmm/Force.h"
#include <vector>
#include "internal/windowsExportRE.h"

namespace REPlugin {

class OPENMM_EXPORT_RE REForce : public OpenMM::Force {
public:
    /**
     * Create an REForce.
     */
    REForce();

    struct formFactor {
        std::vector<double> F;
        std::vector<double> delta_F;
        std::vector<double> q;
        bool is_neutron;
        float d_part;
    };

    void addParticle(int i);
    void addParticles(std::vector<int> particles1);
    void addParticleOrigin(int i);
    void addParticlesOrigin(std::vector<int> particles_for_orign1);
    void addParticleWater(int i);
    void addParticlesWater(std::vector<int> particles_water1);
    void addParticleIon(int i);
    void addParticlesIon(std::vector<int> particles_ion1);
    void addParticleExchH(int i);
    void addParticlesExchH(std::vector<int> particles_exch_h1);
    void addName(std::string name);
    void addNames(std::vector<std::string> names);
    void addCharge(float charge);
    void addCharges(std::vector<float> charge);

    void setCoupleSteps(int c_steps);
    int getCoupleSteps() const;

    /**
     * Set experimental form factor data
     * @param F         the form factor values
     * @param F         the uncertainties of the form factor values
     * @param q         the q values
     * @param d_part    the part of water being deuterated (1.0 - all, 0.0 - none)
     */
    void addExpFF(std::vector<double> F, std::vector<double> delta_F, std::vector<double> q, bool neutron, float d_part);
    /**
     * Set temperature
     * @param T         the temperature
     */
    void setTemp(double T1);
    /**
     * Set the force strength
     * @param k_xray     xray force constant
     * @param k_neutron  neutron force constant
     */
    void setK(double k_xray1, double k_neutron1);
    /**
     * Set tau
     * @param tau       the averaging factor, also defines alpha - switching function
     */
    void setTau(double tau1);
    /**
     * Set water density
     * @param w_dens    average number density of water
     */
    void setWdens(double w_dens1);
    /**
     * Set water density squared
     * @param w_dens_sqr   average sqared number density of water
     */
    void setWdensSqr(double w_dens_sqr1);
    /**
     * Set the distance cutoff
     * @param cutoff    the distance cutoff
     */
    void setCutoff(double cutoff1);

    /**
     * Set all the above parameters at once
     * @param T          the temperature
     * @param k_xray     xray force constant
     * @param k_neutron  neutron force constant
     * @param w_dens     average number density of water
     * @param w_dens_sqr average squared number density of water
     * @param tau        the averaging factor, also defines alpha - switching function
     * @param cutoff     distance cutoff
     */
    void setAllParams(double T1, double k_xray1, double k_neutron1, double w_dens1, double w_dens_sqr1, double tau1, float cutoff1);
    /**
     * Get all the above parameters at once
     * @param T          the temperature
     * @param k_xray     xray force constant
     * @param k_neutron  neutron force constant
     * @param w_dens     average number density of water
     * @param w_dens_sqr average squared number density of water
     * @param tau        the averaging factor, also defines alpha - switching function
     * @param cutoff     distance cutoff
     */
    void getAllParams(double& T1, double& k_xray1, double& k_neutron1, double& w_dens1, double& w_dens_sqr1, double& tau1, float& cutoff1) const;
    /**
     * Get particles on which the force acts, their names and the list of particles for the box centring
     * @param particles              the vector of particle indexes on which the force acts
     * @param names                  the vector of corresponding particle names on which the force acts
     * @param particles_for_origin1  the vector of particle indexes used for the box centring (origin calculations)
     * @param particles_water1       the vector of indexes of water atoms
     * @param particles_ion1         the vector of indexes of ion atoms
     * @param particles_exch_h1      the vector of indexes of exchangeable hydrogens
     */
    void getAllParticles(std::vector<int>& particles1, std::vector<int>& particles_for_origin1, std::vector<int>& particles_water1, std::vector<int>& particles_ion1, std::vector<int>& particles_exch_h1) const;
    int getNumExpFF() const;
    /**
     * Get the data from the experimental data set
     * @param i         the number of the data set
     */
    void getExpFF(int i, std::vector<double>& F1, std::vector<double>& delta_F1, std::vector<double>& q1, bool& is_neutron1, float& d_part1) const;
    void getNames(std::vector<std::string>& names) const;
    void getCharges(std::vector<float>& charge) const;

    // Output
    void setWriteOutFF(bool write_output);
    bool getWriteOutFF() const;
    void setOutPrefix(std::string out_prefix);
    void getOutPrefix(std::string &out_prefix) const;
    
    void setAComponents(std::vector<double> A_real_xray, std::vector<double> A_complex_xray, std::vector<double> A_sqr_xray,
                        std::vector<double> A_real_neutron, std::vector<double> A_complex_neutron, std::vector<double> A_sqr_neutron,
                        std::vector<double> B_real_xray, std::vector<double> B_sqr_xray, std::vector<double> B_real_neutron, std::vector<double> B_sqr_neutron);
    void getAComponents(std::vector<double>& A_real_xray, std::vector<double>& A_complex_xray, std::vector<double>& A_sqr_xray,
                        std::vector<double>& A_real_neutron, std::vector<double>& A_complex_neutron, std::vector<double>& A_sqr_neutron,
                        std::vector<double>& B_real_xray, std::vector<double>& B_sqr_xray, std::vector<double>& B_real_neutron, std::vector<double>& B_sqr_neutron) const;

    void updateParametersInContext(OpenMM::Context& context, bool on_gpu);
    void getParametersFromContext(OpenMM::Context& context);
    void rescaleExpFInContext(OpenMM::Context& context);

    bool usesPeriodicBoundaryConditions() const {
        return true;
    }
protected:
    OpenMM::ForceImpl* createImpl() const;
private:
    int coupl_steps;
    std::vector<int> particles,             // All atoms
                     particles_for_origin,  // Atoms that are used for the origin calculation
                     particles_water,       // Atoms that are water
                     particles_ion,         // Atoms that are ions
                     particles_exch_h;      // Atoms that are exchangeable hydrogens
    std::vector<std::string> atom_names;
    std::vector<float> charges;
    double w_dens,
           w_dens_sqr,
           tau,
           k_xray,
           k_neutron,
           T,
           cutoff;
    std::vector<formFactor> Fs_exp;
    std::vector<double> A_real_xray_current,
                        A_complex_xray_current,
                        A_sqr_xray_current,
                        A_real_neutron_current,
                        A_complex_neutron_current,
                        A_sqr_neutron_current,
                        B_real_xray_current,
                        B_sqr_xray_current,
                        B_real_neutron_current,
                        B_sqr_neutron_current;
    bool write_out;
    std::string out_file_prefix;
};

} // namespace REPlugin

#endif /*OPENMM_REFORCE_H_*/
