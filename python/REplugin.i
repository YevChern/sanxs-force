%module REplugin

%import(module="simtk.openmm") "swig/OpenMMSwigHeaders.i"
%include "swig/typemaps.i"

/*
 * The following lines are needed to handle std::vector.
 * Similar lines may be needed for vectors of vectors or
 * for other STL types like maps.
 */

%include "std_vector.i"
namespace std {
  %template(vectord) vector<double>;
  %template(vectori) vector<int>;
};

%include "std_string.i"

%{
#include "REForce.h"
#include "OpenMM.h"
#include "OpenMMAmoeba.h"
#include "OpenMMDrude.h"
#include "openmm/RPMDIntegrator.h"
#include "openmm/RPMDMonteCarloBarostat.h"
%}

%pythoncode %{
import simtk.openmm as mm
import simtk.unit as unit
%}

/*
 * Add units to function outputs.
*/

%pythonappend REPlugin::REForce::getAllParams(double& T1, double& k1,
                                                             double& w_dens1, double& w_dens_sqr1, double& tau1, float& cutoff1) const %{
    val[0] = unit.Quantity(val[0], unit.kelvin)
    val[1] = unit.Quantity(val[1], unit.kilojoule_per_mole/(unit.nanometer * unit.nanometer))
    val[2] = unit.Quantity(val[2], 1/unit.nanometer)
    val[3] = unit.Quantity(val[3], 1/(unit.nanometer*unit.nanometer))
    val[4] = unit.Quantity(val[4], unit.picosecond)
    val[5] = unit.Quantity(val[5], unit.nanometer)
%}


namespace REPlugin {

class REForce : public OpenMM::Force {
public:
    REForce();

    void addParticle(int i);

    void addParticles(std::vector<int> particles1);

    void addParticleOrigin(int i);

    void addParticlesOrigin(std::vector<int> particles_for_orign1);

    void addParticleWater(int i);

    void addParticlesWater(std::vector<int> particles_water1);

    void addParticleIon(int i);

    void addParticlesIon(std::vector<int> particles_ion1);

    void addParticleExchH(int i);

    void addParticlesExchH(std::vector<int> particles_particles_exch_h1);

    void addName(std::string name);

    void addNames(std::vector<std::string> names);

    void addCharge(float charge);
    
    void addCharges(std::vector<float> charge);

    void addExpFF(std::vector<double> F, std::vector<double> delta_F, std::vector<double> q, bool neutron, float d_part);
    
    void setTemp(double T1);

    void setK(double k_xray1, double k_neutron1);

    void setTau(double tau1);

    void setWdens(double w_dens1);
    
    void setCoupleSteps(int c_steps);

    void setWdensSqr(double w_dens_sqr1);
    
    void setCutoff(double cutoff1);

    void setAllParams(double T1, double k_xray1, double k_neutron1, double w_dens1, double w_dens_sqr1, double tau1, float cutoff1);

    int getNumExpFF() const;

    void updateParametersInContext(OpenMM::Context& context, bool on_gpu);
    
    void getParametersFromContext(OpenMM::Context& context);

    void rescaleExpFInContext(OpenMM::Context& context);

    void setWriteOutFF(bool write_output);

    void setOutPrefix(std::string out_prefix);

    void setAComponents(std::vector<double> A_real_xray, std::vector<double> A_complex_xray, std::vector<double> A_sqr_xray,
                        std::vector<double> A_real_neutron, std::vector<double> A_complex_neutron, std::vector<double> A_sqr_neutron,
			std::vector<double> B_real_xray, std::vector<double> B_sqr_xray, std::vector<double> B_real_neutron, std::vector<double> B_sqr_neutron);
    
    /*
     * The reference parameters to this function are output values.
     * Marking them as such will cause swig to return a tuple.
    */

    %apply double& OUTPUT {double& T1};
    %apply double& OUTPUT {double& k_xray1};
    %apply double& OUTPUT {double& k_neutron1};
    %apply double& OUTPUT {double& w_dens1};
    %apply double& OUTPUT {double& w_dens_sqr1};
    %apply double& OUTPUT {double& tau1};
    %apply double& OUTPUT {double& cutoff1};
    void getAllParams(double& T1, double& k_xray1, double& k_neutron1, double& w_dens1, double& w_dens_sqr1, double& tau1, float& cutoff1) const;
    %clear double& T1;
    %clear double& k_xray1;
    %clear double& k_neutron1;
    %clear double& w_dens1;
    %clear double& w_dens_sqr1;
    %clear double& tau1;
    %clear double& cutoff1;

    %apply std::vector<int>& OUTPUT {std::vector<int>& particles1};
    %apply std::vector<int>& OUTPUT {std::vector<int>& particles_for_origin1};
    %apply std::vector<int>& OUTPUT {std::vector<int>& particles_water1};
    %apply std::vector<int>& OUTPUT {std::vector<int>& particles_ion1};
    %apply std::vector<int>& OUTPUT {std::vector<int>& particles_exch_h1};
    void getAllParticles(std::vector<int>& particles1, std::vector<int>& particles_for_origin1, std::vector<int>& particles_water1, std::vector<int>& particles_ion1, std::vector<int>& particles_exch_h1) const;
    %clear std::vector<int>& particles1;
    %clear std::vector<int>& particles_for_origin1;
    %clear std::vector<int>& particles_water1;
    %clear std::vector<int>& particles_ion1;
    %clear std::vector<int>& particles_exch_h1;

    %apply std::vector<double>& OUTPUT {std::vector<double>& F1};
    %apply std::vector<double>& OUTPUT {std::vector<double>& delta_F1};
    %apply std::vector<double>& OUTPUT {std::vector<double>& q1};
    %apply bool& OUTPUT {bool& is_neutron1};
    %apply float& OUTPUT {float& d_part1};
    void getExpFF(int i, std::vector<double>& F1, std::vector<double>& delta_F1, std::vector<double>& q1, bool& is_neutron1, float& d_part1) const;
    %clear std::vector<double>& F1;
    %clear std::vector<double>& delta_F1;
    %clear std::vector<double>& q1;
    %clear bool& is_neutron1;
    %clear float& d_part1;

    %apply std::vector<string>& OUTPUT {std::vector<std::string>& names};
    void getNames(std::vector<std::string>& names) const;
    %clear std::vector<string>& names;

    %apply std::vector<double>& OUTPUT {std::vector<double>& A_real_xray};
    %apply std::vector<double>& OUTPUT {std::vector<double>& A_complex_xray};
    %apply std::vector<double>& OUTPUT {std::vector<double>& A_sqr_xray};
    %apply std::vector<double>& OUTPUT {std::vector<double>& A_real_neutron};
    %apply std::vector<double>& OUTPUT {std::vector<double>& A_complex_neutron};
    %apply std::vector<double>& OUTPUT {std::vector<double>& A_sqr_neutron};
    %apply std::vector<double>& OUTPUT {std::vector<double>& B_real_xray};
    %apply std::vector<double>& OUTPUT {std::vector<double>& B_sqr_xray};
    %apply std::vector<double>& OUTPUT {std::vector<double>& B_real_neutron};
    %apply std::vector<double>& OUTPUT {std::vector<double>& B_sqr_neutron};
    void getAComponents(std::vector<double>& A_real_xray, std::vector<double>& A_complex_xray, std::vector<double>& A_sqr_xray,
                        std::vector<double>& A_real_neutron, std::vector<double>& A_complex_neutron, std::vector<double>& A_sqr_neutron,
			std::vector<double>& B_real_xray, std::vector<double>& B_sqr_xray, std::vector<double>& B_real_neutron, std::vector<double>& B_sqr_neutron) const;
    %clear std::vector<int>& A_real_xray;
    %clear std::vector<int>& A_complex_xray;
    %clear std::vector<int>& A_sqr_xray;
    %clear std::vector<int>& A_real_neutron;
    %clear std::vector<int>& A_complex_neutron;
    %clear std::vector<int>& A_sqr_neutron;
    %clear std::vector<int>& B_real_xray;
    %clear std::vector<int>& B_sqr_xray;
    %clear std::vector<int>& B_real_neutron;
    %clear std::vector<int>& B_sqr_neutron;

};

}
