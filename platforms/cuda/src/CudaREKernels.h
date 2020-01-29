#ifndef CUDA_RE_KERNELS_H_
#define CUDA_RE_KERNELS_H_

#include "REKernels.h"
#include "openmm/cuda/CudaContext.h"
#include "openmm/cuda/CudaArray.h"

#include <fstream>

namespace REPlugin {

/**
 * This kernel is invoked by REForce to calculate the forces acting on the system and the energy of the system.
 */
class CudaCalcREForceKernel : public CalcREForceKernel {
public:
    CudaCalcREForceKernel(std::string name, const OpenMM::Platform& platform, OpenMM::CudaContext& cu, const OpenMM::System& system);

    ~CudaCalcREForceKernel();

    double getXrayStrength(std::string name, float q, std::map<char, std::vector<double> > map, float charge=0.0);
    double getIonXrayStrength(std::string name, float q, std::map<std::string, std::vector<double> > map, float charge);
    double getWaterXrayStrength(std::string name, float q, std::map<char, std::vector<double> > map);
    double getNeutronStrength(std::string name, std::map<char, double> map);
    double getIonNeutronStrength(std::string name, std::map<std::string, double> map);
    double getWaterNeutronStrength(std::string name, float deut, std::map<char, double> map);

    /**
     * Initialize the kernel.
     * 
     * @param system     the System this kernel will be applied to
     * @param force      the REForce this kernel will be used for
     */
    void initialize(const OpenMM::System& system, const REForce& force);
    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy);
    /**
     * Copy changed parameters over to a context.
     *
     * @param context    the context to copy parameters to
     * @param force      the REForce to copy the parameters from
     */
    void copyParametersToContext(OpenMM::ContextImpl& context, const REForce& force, bool on_gpu);
    /**
     * Copy changed parameters from a context.
     *
     * @param context    the context to copy parameters from
     * @param force      the REForce to copy the parameters to
     */
    void copyParametersFromContext(OpenMM::ContextImpl& context, const REForce& force);
    /**
     * Rescale experimental data to simulated F
     *
     * @param context    the context to copy parameters from
     * @param force      the REForce to copy the parameters to
     */
    void rescaleExpF(OpenMM::ContextImpl& context, const REForce& force);
private:

    CUfunction computeOrigin;
    CUfunction computePreFFtotal;
    CUfunction computeFFtotal;
    CUfunction computePostFFtotal;
    CUfunction computePreForceKernel;
    CUfunction computeForceKernel;
    CUfunction computeEnergyKernel;
    CUfunction computeGlobalFtotal;
    bool hasInitializedKernel;
    OpenMM::CudaContext& cu;
    const OpenMM::System& system;
    
    bool write_output;
    std::ofstream outfile_xray;
    std::ofstream outfile_neutron;
    std::ofstream outfile_scale_factor_xray;
    std::ofstream outfile_scale_factor_neutron;
    std::vector<double> F_total_xray_out, F_total_neutron_out;
    std::vector<double> scale_factor_xray, scale_factor_neutron;

    int couple_steps;
    int Natoms;                             // Total number of atoms in the system
    int Nq_xray;                            // Total number of x-ray data q points
    int Nq_neutron;                         // Total number of neutron data q points
    OpenMM::CudaArray* particles;           // Particles on which the force will act
    OpenMM::CudaArray* particles_for_origin;// Particles which are used for the box centring (their COM -> 0.0 z)
    OpenMM::CudaArray* mass_for_origin;     // Mass of each particle for origin
    double total_mass_origin;               // Total mass of the particles for origin
    int originNatoms;                       // Number of particles in particles_for_origin
    OpenMM::CudaArray* is_water;            // Bool mask to distinguish water and non-water particles
    OpenMM::CudaArray* atom_names;          // Names of all atoms in the system
    OpenMM::CudaArray* N;
    OpenMM::CudaArray* box;
    OpenMM::CudaArray* tau;
    double h_tau;
    std::vector<float> prev_box;
    OpenMM::CudaArray* T;
    OpenMM::CudaArray* k_xray;
    OpenMM::CudaArray* k_neutron;
    OpenMM::CudaArray* w_dens;
    OpenMM::CudaArray* w_dens_sqr;
    OpenMM::CudaArray* cutoff;

    OpenMM::CudaArray* xray_strength;       // Stores x-ray scattering strength for each atom (total Natom types) for each q
    OpenMM::CudaArray* neutron_strength;    // Stores neutron scattering strength for each atom type (total 6 types) for each q
    OpenMM::CudaArray* xray_h_strength;     // Stores x-ray scattering strength of H for each q (used in B calculation)
    OpenMM::CudaArray* xray_o_strength;     // Stores x-ray scattering strength of O for each q (used in B calculation)
    OpenMM::CudaArray* neutron_o_strength;  // Stores neutron scattering strength of O (used in B calculation)
    OpenMM::CudaArray* neutron_h_strength;  // Stores neutron scattering strength of H (used in B calculation)
    OpenMM::CudaArray* neutron_d_strength;  // Stores neutron scattering strength of D (used in B calculation)

    OpenMM::CudaArray* xray_qs;             // q-values for x-ray data
    OpenMM::CudaArray* neutron_qs;          // q-values for neutron data
    OpenMM::CudaArray* d_parts;             // Array of size Nq_neutron with d_part values

    OpenMM::CudaArray* A_real_xray;
    OpenMM::CudaArray* A_real_neutron;
    OpenMM::CudaArray* A_complex_xray;
    OpenMM::CudaArray* A_complex_neutron;
    OpenMM::CudaArray* B_real_xray;
    OpenMM::CudaArray* B_real_neutron;

    // Buffers used in the kernel
    OpenMM::CudaArray* B_real_xray_current;
    OpenMM::CudaArray* B_real_neutron_current;
    OpenMM::CudaArray* B_sqr_xray_current;
    OpenMM::CudaArray* B_sqr_neutron_current;
    OpenMM::CudaArray* A_real_xray_current;
    OpenMM::CudaArray* A_real_neutron_current;
    OpenMM::CudaArray* A_complex_xray_current;
    OpenMM::CudaArray* A_complex_neutron_current;
    OpenMM::CudaArray* A_sqr_xray_current;
    OpenMM::CudaArray* A_sqr_neutron_current;
    OpenMM::CudaArray* F_total_xray;
    OpenMM::CudaArray* F_total_neutron;

    // Buffers for pulling A components out and communicating
    OpenMM::CudaArray* A_real_xray_out;
    OpenMM::CudaArray* A_real_neutron_out;
    OpenMM::CudaArray* A_complex_xray_out;
    OpenMM::CudaArray* A_complex_neutron_out;
    OpenMM::CudaArray* A_sqr_xray_out;
    OpenMM::CudaArray* A_sqr_neutron_out;
    
    // Buffers for global F_total calculation. We need only B as we reuse A.._out buffers
    OpenMM::CudaArray* B_real_xray_global;
    OpenMM::CudaArray* B_sqr_xray_global;
    OpenMM::CudaArray* B_real_neutron_global;
    OpenMM::CudaArray* B_sqr_neutron_global;

    OpenMM::CudaArray* F_exp_xray;
    OpenMM::CudaArray* delta_F_exp_xray;
    OpenMM::CudaArray* F_exp_neutron;
    OpenMM::CudaArray* delta_F_exp_neutron;

    OpenMM::CudaArray* alpha;
    OpenMM::CudaArray* origin;
    
    OpenMM::CudaArray* origin_buffer;
    OpenMM::CudaArray* energy_buffer;
    
    struct formFactor {
        std::vector<double> F;
        std::vector<double> delta_F;
        std::vector<double> q;
        bool is_neutron;
        float d_part;
    };
    std::vector<formFactor> Fs_exp;
    
};


} // namespace REPlugin

#endif /*CUDA_RE_KERNELS_H_*/
