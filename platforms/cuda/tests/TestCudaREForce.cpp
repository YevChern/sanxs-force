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

/**
 * This tests the Reference implementation of REForce.
 */

#include "REForce.h"
#include "openmm/internal/AssertionUtilities.h"
#include "openmm/Context.h"
#include "openmm/Platform.h"
#include "openmm/System.h"
#include "openmm/VerletIntegrator.h"
#include <cmath>
#include <iostream>
#include <vector>

#include "openmm/reference/SimTKOpenMMRealType.h"

using namespace REPlugin;
using namespace OpenMM;
using namespace std;

extern "C" OPENMM_EXPORT void registerRECudaKernelFactories();

const double FOUR_PI_SQUARED = 157.91367041742973790135185599802L;
map<char, vector<double> > aff_constants;
map<char, double> neutronFFmap;

void buildAFFconstantMap() {

    //build the vector to insert, then put in the map, do for each element
    vector<double> hydrogen;
    hydrogen.reserve(9);
    hydrogen.push_back(0.493);
    hydrogen.push_back(0.323);
    hydrogen.push_back(0.14);
    hydrogen.push_back(0.041);
    hydrogen.push_back(10.511);
    hydrogen.push_back(26.126);
    hydrogen.push_back(3.142);
    hydrogen.push_back(57.8);
    hydrogen.push_back(0.003);

    aff_constants['H'] = hydrogen;

    vector<double> deuterium;
    deuterium.reserve(9);
    deuterium.push_back(0.493); //a1
    deuterium.push_back(0.323); //a2
    deuterium.push_back(0.14); //a3
    deuterium.push_back(0.041); //a4
    deuterium.push_back(10.511); //b1
    deuterium.push_back(26.126); //b2
    deuterium.push_back(3.142); //b3
    deuterium.push_back(57.8); //b4
    deuterium.push_back(0.003); //c

    aff_constants['D'] = deuterium;

    vector<double> carbon;
    carbon.reserve(9);
    carbon.push_back(2.31); //a1
    carbon.push_back(1.02); //a2
    carbon.push_back(1.589); //a3
    carbon.push_back(0.865); //a4
    carbon.push_back(20.844); //b1
    carbon.push_back(10.208); //b2
    carbon.push_back(0.569); //b3
    carbon.push_back(51.651); //b4
    carbon.push_back(0.216); //c

    aff_constants['C'] = carbon;

    vector<double> nitrogen;
    nitrogen.reserve(9);
    nitrogen.push_back(12.213); //a1
    nitrogen.push_back(3.132); //a2
    nitrogen.push_back(2.013); //a3
    nitrogen.push_back(1.166); //a4
    nitrogen.push_back(0.006); //b1
    nitrogen.push_back(9.893); //b2
    nitrogen.push_back(28.997); //b3
    nitrogen.push_back(0.583); //b4
    nitrogen.push_back(-11.524); //c

    aff_constants['N'] = nitrogen;

    vector<double> oxygen;
    oxygen.reserve(9);
    oxygen.push_back(3.049); //a1
    oxygen.push_back(2.287); //a2
    oxygen.push_back(1.546); //a3
    oxygen.push_back(0.867); //a4
    oxygen.push_back(13.277); //b1
    oxygen.push_back(5.701); //b2
    oxygen.push_back(0.324); //b3
    oxygen.push_back(32.909); //b4
    oxygen.push_back(0.251); //c

    aff_constants['O'] = oxygen;

    vector<double> phosphorus;
    phosphorus.reserve(9);
    phosphorus.push_back(6.435); //a1
    phosphorus.push_back(4.179); //a2
    phosphorus.push_back(1.78); //a3
    phosphorus.push_back(1.491); //a4
    phosphorus.push_back(1.907); //b1
    phosphorus.push_back(27.157); //b2
    phosphorus.push_back(0.526); //b3
    phosphorus.push_back(68.164); //b4
    phosphorus.push_back(1.115); //c

    aff_constants['P'] = phosphorus;
}

void buildNeutronMap() {
    neutronFFmap.insert(pair<char,double>('H',-3.7409E-5)); //taken from: "Koester, L., Nistier, W.: Z. Phys. A 272 (1975) 189."
    neutronFFmap.insert(pair<char,double>('D',6.67E-5));
    neutronFFmap.insert(pair<char,double>('C',6.6484E-5));  //taken from: "Koester, L., Nistier, W.: Z. Phys. A 272 (1975) 189."
    neutronFFmap.insert(pair<char,double>('N',9.36E-5));    //taken from: "Koester, L., Knopf, K., Waschkowski, W.: Z. Phys. A 277 (1976) 77."
    neutronFFmap.insert(pair<char,double>('O',5.805E-5));   //taken from: "Koester, L., Knopf, K., Waschkowski, W.: Z. Phys. A 292 (1979) 95."
    neutronFFmap.insert(pair<char,double>('P',5.17E-5));
}

double getXrayStrength(string name, float q){
    map<char,vector<double> >::const_iterator constants_itr = aff_constants.find(char(name[0]));
/*
    if (constants_itr==aff_constants.end())
        throw OpenMMException("Warning: Atom type " + name + " not found in X-ray atomic FF map!");
*/
    double fi = 0.0;
    vector<double> constants = constants_itr->second;
    for (int i = 0; i < 4; ++i)
        fi += constants.at(i)*exp(-constants.at(i+4)*q*q/FOUR_PI_SQUARED);

    fi += constants.at(8);

    return fi;
}

double getNeutronStrength(std::string name){
    map<char, double>::const_iterator neutronSL_itr = neutronFFmap.find(char(name[0]));
/*
    if (neutronSL_itr==neutronFFmap.end())
        throw OpenMMException("Warning: Atom type " + name + " not found in neutron atomic FF map!");
*/
    double fi = neutronSL_itr->second;
    return fi;
}

void testForce() {

    // Create a symmetric system (complex part of FF cancels out -> 0.0)
    
    const int numParticles = 23;
    const int numQVal = 678;
    System system;
    Vec3 a(1.0, 0.0, 0.0), b(0.0, 1.0, 0.0), c(0.0, 0.0, (float)numParticles);
    system.setDefaultPeriodicBoxVectors(a, b, c);
    vector<Vec3> positions(numParticles);
    for (int i = 0; i < numParticles; i++) {
        system.addParticle(1.0);
        if (i==0) {
            positions[i] = Vec3(i, 0.1*i, 0.0);
        } else {
            positions[i] = Vec3(i, 0.1*i, pow(-1.0, i));
        }
    }
    REForce* force = new REForce();
    system.addForce(force);
    
    for (int i = 0; i < numParticles; i++){
        force->addParticle(i);
        force->addName(string("H"));
        force->addCharge(0.0);
    }

    force->addParticleOrigin(0);
    force->setCoupleSteps(1);
    double tau = 0.0;
    force->setAllParams(1.0, 1.0, 1.0, 0.0, 0.0, tau, 5000.0);  // T, k_xray, k_neutron, w_dens, w_dens_sqr, tau, cutoff
    vector<double> F, delta_F, q;
 
    buildAFFconstantMap();
    buildNeutronMap();

    // Add expected x-ray values
    for (int i=0; i<numQVal; ++i){
        q.push_back(M_PI*i + M_PI/2.0);
        F.push_back(getXrayStrength("H", q[i]));
        delta_F.push_back(1.0);
    }
    force->addExpFF(F, delta_F, q, false, 0.0);

    // Add expected neutron values
    q.clear();
    F.clear();
    delta_F.clear();
    for (int i=0; i<numQVal; ++i){
        q.push_back(M_PI*i + M_PI/2.0);
        F.push_back(getNeutronStrength("H"));
        delta_F.push_back(1.0);
    }
    force->addExpFF(F, delta_F, q, true, 0.0);

    // Compute the forces and energy.
    VerletIntegrator integ(1.0);
    Platform& platform = Platform::getPlatformByName("CUDA");
    platform.setPropertyDefaultValue("CudaPrecision", "single");
    Context context(system, integ, platform);
    context.setPositions(positions);
    context.setTime(2.0*tau);
    context.getState(State::Energy | State::Forces);
    force->getParametersFromContext(context);
    force->updateParametersInContext(context, false);
    State state = context.getState(State::Energy | State::Forces);

    // See if the energy is correct.
    ASSERT_EQUAL_TOL(0.0, state.getPotentialEnergy(), 1e-5);

    // See if the force is correct.
    for (int i=0; i<numParticles; ++i){
        ASSERT_EQUAL_TOL(0.0, state.getForces()[i][2], 1e-4);
    }
}

int main(int argc, char* argv[]) {

    registerRECudaKernelFactories();
    try {
        if (argc > 1)
	    Platform::getPlatformByName("CUDA").setPropertyDefaultValue("CudaPrecision", string(argv[1]));
        testForce();
    }
    catch(const std::exception& e) {
        std::cout << "Exception: " << e.what() << std::endl;
        return 1;
    }
    std::cout << "Done" << std::endl;
    return 0;
}
