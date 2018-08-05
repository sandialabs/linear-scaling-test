//
////
////////
////////////////
////////////////////////////////
////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// COMPANION SOFTWARE TO: Assessment of localized and randomized algorithms for electronic structure
// USAGE: <executable> <xyz structure file> <chemical potential> <temperature> <solver> <solver parameters ...>

// C99 syntax, OpenMP-based shared memory parallelism, designed to run on a single multi-core supercomputer node
// MPI parallelism is included for PEXSI functionality (solver=2) & all other solvers should use 1 MPI process
// MPI is being used w/ an underlying shared-memory (e.g. single node) system in mind (i.e. nonuniform memory distribution)

// UNITS: energies/temperatures & distances are in electronvolts/Angstroms for input/output & Rydbergs/Bohr internally
//        stress tensor is in gigapascals (GPa)

// Available solvers:
//  # | F-D approx. | trace approx.     | O(N^p) | solver parameters
//----+-------------+-------------------+--------+-------------------
//  0 | none        | none              | 1      | none [pre & post processing only]
//  1 | exact       | exact             | 3      | none
//  2 | rational    | exact (PEXSI)     | 2 (3D) | <#/2 of poles>
//  3 | polynomial  | exact (iterative) | 2      | <# of Cheby.> <res. tol.>
//  4 | rational    | exact (iterative) | 2      | <#/2 of poles> <res. tol.>
//  5 | polynomial  | local             | 1      | <# of Cheby.> <res. tol.> <loc. rad.>
//  6 | rational    | local             | 1      | <#/2 of poles> <res. tol.> <loc. rad.>
//  7 | polynomial  | random            | 1      | <# of Cheby.> <res. tol.> <loc. rad.> <seed> <# of samples>
//  8 | rational    | random            | 1      | <#/2 of poles> <res. tol.> <loc. rad.> <seed> <# of samples>
//  9 | rational    | local (infinite)  | 0      | <#/2 of poles> <res. tol.> <loc. rad.>
// 10 | exact       | k-grid (infinite) | 0      | <# of k-grid pts. per dimension> <loc. rad.>
// Available testers:
// -1 | none        | local (infinite)  | 0      | <pre. shift> <res. tol.> <min. rad.> <max. rad.> <# rad.> [precondition test]

// INPUT KEY:
//  <#/2 of poles> : number of complex-conjugate pole pairs in the rational approximation of the Fermi-Dirac function
//  <# of Cheby.> : number of Chebyshev polynomials used to approximate the Fermi-Dirac function
//  <res. tol.> : residual 2-norm stopping criterion for iterative linear solvers (conjugate gradient & conjugate residual)
//  <loc. rad.> : localization radius that defines the sparsity pattern of local Hamiltonians (solver = 5,6)
//                & the coloring scheme for uncorrelated complex rotors (solver = 7,8)
//  <seed> : integer seed for the pseudo-random number generator
//  <# of samples> : the number of samples drawn from the colored complex rotor multi-vector distribution
//  <# of k-grid pts. per dimension> : the number of points assigned to the k-point grid per reciprocal-space dimension (3)
//  <pre. shift> : imaginary energy shift for the shifted-inverse preconditioner
//  <min. rad.> <max. rad.> <# rad.> : minimum/maximum/number-of radius values for a grid of preconditioner localization radii

// Structure file format (*.xyz) for monoatomic copper clusters:
//  <# of atoms>
//
//  Cu <x coordinate of atom #1> <y coordinate of atom #1> <z coordinate of atom #1>
//  ...
//  Cu <x coordinate of atom #N> <y coordinate of atom #N> <z coordinate of atom #N>
// NOTE: for solver = 9, the positions of the 2nd, 3rd, & 4th atoms relative to the 1st define the crystal lattice vectors

// OUTPUT:
//  Total number of electrons, total energy, & atomic forces to standard output
//  Memory & time usage to standard output (the only output for solver = 0 & -1)
//  Density & response matrix elements in the Hamiltonian sparsity pattern to "debug.out"
//  F-norm for off-diagonal blocks of density & response matrices in "decay.out" (solver = 9 & 10 only)
//  Fermi-smeared electronic density-of-states to "dos.out" (solver = 10 only)

// RECOMMENDED OPENMP SETTINGS:
//  solver = 1 : OMP_NUM_THREADS = 1 & MKL_NUM_THREADS = # of cores , we only utilize threading through LAPACK & BLAS calls
//  solver = 2 : OMP_NUM_THREADS = MKL_NUM_THREADS = 1 , MPI-based parallelism only without any threading
//  otherwise : OMP_NUM_THREADS = # of cores & MKL_NUM_THREADS = 1 , threading in code & BLAS calls only for small matrix blocks

// SOFTWARE ORGANIZATION:
//  1. Fermi-Dirac approximation - fit polynomials & rational functions
//  2. NRL tight-binding model - matrix elements & their derivatives
//  3. Atomic partitioning - sets up neighbors lists for atoms
//  4. Block vector & matrix operations - native linear algebra operations in this software
//  5. Matrix construction & conversion - application-specific construction & conversion to other formats
//  6. Pseudo-random number generation - a standard PRNG generator that is better than C rand()
//  7. Iterative solvers - application-specific implementations of CG & MINRES & Chebyshev recursion
//  8. Solver mains - a main specific to each solver
//  9. Main - global control flow

// EXTERNAL LIBRARIES:
//  - MKL (for BLAS, LAPACK, & FFTW3)
//  - PEXSI 1.0
//    - symPACK post-1.1 [development version that is adapted for PEXSI compatibility]
//      - PT-Scotch 6.0.0
//    - SuperLU_DIST 5.2.1
//      - parMETIS 4.0.3

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <omp.h>
#include <sys/time.h>
#include <sys/resource.h>
#include "mkl.h"
#include "fftw3.h"
#include "c_pexsi_interface.h"

#define A0 0.52917721067 // Bohr radius in Angstroms
#define E0 13.60569253 // Rydberg energy in eV
#define P0 14710.5071 // Ry/Bohr^3 in GPa

#define MIN(x,y) (((x) < (y)) ? (x) : (y))
#define MAX(x,y) (((x) > (y)) ? (x) : (y))

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

// MKL_INT is used as a matrix/vector index for compatibility with both 32-bit & 64-bit versions of MKL
// If MKL is not being used, define MKL_INT locally as the integer type used by BLAS & LAPACK (usually 'int')
//#define MKL_INT int
// Ditto for MKL_Complex16, but changes must also be made to infinite_reciprocal_solver if this is redefined
//#define MKL_Complex16 double complex

// Convenient hard-coded path (relative or absolute) to the rational approximation table
#define RATIONAL_TABLE_PATH "../src/table.txt" // relative path used for all benchmark calculations

// All dense matrices & matrix blocks are stored in Fortran-style column-major order
// All vectors have block structure and are stored as a sequence of memory-contiguous dense blocks (in single-index arrays)
// The block sizes are set by the natural block size for atomic partitioning in our tight-binding model (9),
//  which is not optimal for performance. We choose simplicity over performance here.
#define NBLOCK_MAX 9 // hard-coded maximum block size

// Compressed-column sparsity pattern
struct pattern
{
  int ncol; // number of columns
  int nrow; // number of rows
  int *col; // index of the first element of each column & col[ncol] is the number of nonzero elements [ncol+1]
  int *row; // row of each nonzero matrix element [col[ncol]]
};

// Wrap up memory deallocation for pattern structure
void free_pattern(struct pattern* mat) // sparsity pattern to be set free [1]
{
  free(mat->col);
  free(mat->row);
}

//==============================//
// 1. FERMI-DIRAC APPROXIMATION //
//==============================//

// RETURN: value of the Fermi-Dirac distribution at x
double fermi(double x) // argument of the function
{ return 1.0/(1.0 + exp(x)); }
// RETURN: derivative of the Fermi-Dirac distribution at x
double dfermi_dx(double x) // argument of the function
{ return -0.5/(1.0 + cosh(x)); }

// RETURN: value of the Chebyshev polynomial expansion
double chebyshev(double x, // evaluation point
                 int n, // number of Chebyshev polynomials
                 double *coeff) // coefficients of the Chebyshev polynomials [n]
{
  double T_old = 1.0, T = x, ans = 0.0;
  if(n > 0) { ans += T_old*coeff[0]; }
  if(n > 1) { ans += T*coeff[1]; }
  for(int i=2 ; i<n ; i++)
  {
    double T_new = 2.0*x*T - T_old;
    ans += T_new*coeff[i];
    T_old = T;
    T = T_new;
  }
  return ans;
}

// Chebyshev polynomial approximation of the Fermi-Dirac function
#define CHEBYSHEV_DX 0.1 // grid spacing needed for accurate integrals of the Fermi-Dirac function
#define GOLDEN_RATIO 1.61803398875
#define EXTREMUM_TOLERANCE 1e-12
// RETURN: maximum approximation error
double polynomial_approximation(int n, // number of Chebyshev polynomials
                                double min_energy, // minimum orbital energy of the system
                                double max_energy, // maximum orbital energy of the system
                                double potential, // chemical potential of the system
                                double temperature, // temperature of the system
                                double *coeff) // coefficients for Chebyshev polynomials [n]
{
  // set shifted & scaled domain
  double xmin = (min_energy - potential)/temperature;
  double xmax = (max_energy - potential)/temperature;

  // set quadrature & integrand values
  int npt = MAX((int)ceil((xmax-xmin)/CHEBYSHEV_DX),2*n);
  double *pt = (double*)malloc(sizeof(double)*npt);
  double *val = (double*)malloc(sizeof(double)*npt);
  for(int i=0 ; i<npt ; i++)
  {
    pt[i] = 0.5*(xmin+xmax) + 0.5*(xmin-xmax)*cos(M_PI*((double)i+0.5)/(double)npt);
    val[npt-i-1] = fermi(pt[i])/(double)(2.0*npt); // reversed order & rescaling for FFTW input
  }

  // transform & truncate Chebyshev expansion
  double *coeff_big = (double*)malloc(sizeof(double)*npt);
  fftw_plan p;
  p = fftw_plan_r2r_1d(npt,val,coeff_big,FFTW_REDFT10,FFTW_ESTIMATE);
  fftw_execute(p);
  fftw_destroy_plan(p);
  for(int i=0 ; i<n ; i++) { coeff[i] = 2.0*coeff_big[i]; }
  for(int i=n ; i<npt ; i++) { coeff_big[i] = 0.0; }
  coeff[0] *= 0.5;

  // inverse transform to generate residual grid
  p = fftw_plan_r2r_1d(npt,coeff_big,val,FFTW_REDFT01,FFTW_ESTIMATE);
  fftw_execute(p);
  fftw_destroy_plan(p);
  
  // find grid point with largest residual error
  int ierror = -1;
  double error = 0.0;
  for(int i=0 ; i<npt ; i++)
  {
    if(fabs(val[npt-i-1] - fermi(pt[i])) > error)
    {
      error = fabs(val[npt-i-1] - fermi(pt[i]));
      ierror = i;
    }
  }

  // refine global residual maximum with Golden section search
  double xmin0 = -cos(M_PI*((double)MAX(0,ierror-1)+0.5)/(double)npt);
  double xmax0 = -cos(M_PI*((double)MIN(npt-1,ierror+1)+0.5)/(double)npt);
  double xmin0_new = xmin0 + (xmax0 - xmin0)/GOLDEN_RATIO;
  double xmax0_new = xmax0 - (xmax0 - xmin0)/GOLDEN_RATIO;
  while(fabs(xmax0_new - xmin0_new) > EXTREMUM_TOLERANCE)
  {
    if( fabs(fermi(0.5*(xmin+xmax) - 0.5*(xmin-xmax)*xmin0_new) - chebyshev(xmin0_new,n,coeff)) <
        fabs(fermi(0.5*(xmin+xmax) - 0.5*(xmin-xmax)*xmax0_new) - chebyshev(xmax0_new,n,coeff)) )
    { xmax0 = xmin0_new; }
    else
    { xmin0 = xmax0_new; }

    xmin0_new = xmin0 + (xmax0 - xmin0)/GOLDEN_RATIO;
    xmax0_new = xmax0 - (xmax0 - xmin0)/GOLDEN_RATIO;
  }
  error = fabs(fermi(0.5*(xmin+xmax) - 0.5*(xmin-xmax)*0.5*(xmin0+xmax0)) - chebyshev(0.5*(xmin0+xmax0),n,coeff));

  free(coeff_big);
  free(val);
  free(pt);
  return error;
}

// Find an appropriate rational approximation in the table file
// RETURN: maximum approximation error
double rational_approximation(int n, // number of pole pairs
                              double min_energy, // minimum orbital energy of the system
                              double potential, // chemical potential of the system
                              double temperature, // temperature of the system
                              double complex *w, // approximation residues [n]
                              double complex *z) // poles (ordered by decreasing magnitude of imaginary part) [n]
{
  // open the table of rational approximations
  // NOTE: this version of the table has no header & is ordered by increasing # of poles & increasing error
  FILE *quadrature_table = fopen(RATIONAL_TABLE_PATH,"r");
  if(quadrature_table == NULL)
  { printf("ERROR: rational approximation table not found at %s\n",RATIONAL_TABLE_PATH); MPI_Abort(MPI_COMM_WORLD,0); }
  int num_pole;
  double approximation_error, y, real_part, imag_part;
  double y_target = (potential - min_energy)/temperature;
  double complex *w0 = (double complex*)malloc(sizeof(double complex)*2*n);
  double complex *z0 = (double complex*)malloc(sizeof(double complex)*2*n);

  // loop over entries of the table
  do
  {
    // read the next entry of the input table
    fscanf(quadrature_table,"%d %lf %lf",&num_pole,&approximation_error,&y);
    if(feof(quadrature_table))
    { printf("ERROR: suitable rational approximation was not found in table\n"); MPI_Abort(MPI_COMM_WORLD,0); }

    for(int i=0 ; i<num_pole ; i++)
    {
      fscanf(quadrature_table,"%lf %lf",&real_part,&imag_part);
      w0[i] = real_part + I*imag_part;
      fscanf(quadrature_table,"%lf %lf",&real_part,&imag_part);
      z0[i] = real_part + I*imag_part;
    }
  }while(num_pole != 2*n || y < y_target);
  fclose(quadrature_table);

  // order by magnitude (inefficient bubble sort)
  for(int i=0 ; i<2*n ; i++)
  {
    for(int j=i+1 ; j<2*n ; j++)
    {
      if(cabs(z0[j]) > cabs(z0[i]))
      {
        double complex c;
        c = w0[i]; w0[i] = w0[j]; w0[j] = c;
        c = z0[i]; z0[i] = z0[j]; z0[j] = c;
      }
    }
  }

  // shift & scale the rational approximation
  for(int i=0 ; i<2*n ; i++)
  {
    w0[i] *= temperature;
    z0[i] *= temperature;
    z0[i] += potential;
  }

  // group poles together into conjugate pairs
  for(int i=0 ; i<2*n ; i+=2)
  {
    for(int j=i+2 ; j<2*n ; j++)
    {
      if(cabs(z0[i]-conj(z0[j])) < cabs(z0[i]-conj(z0[i+1])))
      {
        double complex c;
        c = w0[i+1]; w0[i+1] = w0[j]; w0[j] = c;
        c = z0[i+1]; z0[i+1] = z0[j]; z0[j] = c;
      }
      // order positive imaginary part first
      if(cimag(z0[i]) < cimag(z0[i+1]))
      {
        double complex c;
        c = w0[i+1]; w0[i+1] = w0[i]; w0[i] = c;
        c = z0[i+1]; z0[i+1] = z0[i]; z0[i] = c;
      }
    }
  }

  // save only one residue & pole from each pair
  for(int i=0 ; i<n ; i++)
  {
    w[i] = w0[2*i];
    z[i] = z0[2*i];
  }
  free(z0);
  free(w0);

  return approximation_error;
}

//============================//
// 2. NRL TIGHT-BINDING MODEL //
//============================//

// NRL tight-binding model parameters
struct nrl_tb
{
  double Rcut, R0, Rs, lambda; // numerical cutoff radius, screening radius, screening length, & environment decay
  double hs[4], hp[4], hd[4]; // onsite parameters
  double hsss[4], hsps[4], hpps[4], hppp[4], hsds[4], hpds[4], hpdp[4], hdds[4], hddp[4], hddd[4]; // hopping parameters
  double osss[4], osps[4], opps[4], oppp[4], osds[4], opds[4], opdp[4], odds[4], oddp[4], oddd[4]; // overlap parameters
};

// hard-coded parameters for copper
// RETURN: structure filled with copper parameters
struct nrl_tb define_copper()
{
  // C99 syntax for "designated initializers"
  struct nrl_tb cu = {
  .Rcut = 12.5,
  .R0 = 11.25, // RCUT - 5*SCREENL, for some reason not the bare parameter
  .Rs = 0.25,
  .lambda = .145617816949E+01,

  // a, b, c, d
  .hs = {  .291179442078E-01,  .608612040825E+02, -.580815805783E+04,  .225817615341E+06 },
  .hp = {  .344716987246E+00,  .888191059298E+02, -.627796769797E+04,  .175924743450E+06 },
  .hd = { -.290980998425E-02, -.280134504507E+01,  .439691173572E+03, -.133435774471E+05 },

  // e, f, fbar, g
  .hsss = { -.597191735504E+01,  .157276992857E+01, -.447299469804E+00, .968392496859E+00 },
  .hsps = {  .142228825776E+01,  .111328503057E+00,  .209048736613E-01, .816193556611E+00 },
  .hpps = { -.699924962951E+00,  .685983943326E+00, -.283976143863E-01, .766161691504E+00 },
  .hppp = { -.194951465694E-01, -.157553504153E+01,  .301142535846E+00, .943349455985E+00 },
  .hsds = { -.487019125256E+00, -.122729421901E+00, -.282606250674E-01, .925507793241E+00 },
  .hpds = { -.290425374224E+00, -.715797951782E-01,  .137648233927E-02, .743208041114E+00 },
  .hpdp = { -.186619297102E+01,  .827909641955E+00,  .129381300114E+00, .105155367074E+01 },
  .hdds = { -.264216452809E+01,  .612527278745E+00, -.411141233432E-01, .811325004989E+00 },
  .hddp = {  .697425666621E+01, -.173638099984E+01,  .168047875555E+00, .101445807107E+01 },
  .hddd = { -.122136143098E+00, -.106786813791E+00, -.573634877781E-01, .114358651642E+01 },
  .osss = { -.187763110058E+01,  .999745133711E+00,  .294871103015E+00, .963163153997E+00 },
  .osps = {  .349830122695E+02, -.130114254052E+02,  .607050297159E+00, .986803443924E+00 },
  .opps = {  .469831980051E+02, -.150210237460E+02,  .423592218489E+00, .103136127318E+01 },
  .oppp = { -.452858187471E+02,  .212940485258E+02, -.222119065584E+01, .973686678526E+00 },
  .osds = {  .185975554048E+01, -.101721693929E+01,  .361939123784E-01, .113738864025E+01 },
  .opds = {  .151404237752E+01, -.648815291269E+00, -.301781892056E+00, .107714476838E+01 },
  .opdp = { -.824947586413E+01,  .737040055222E+00,  .202806401480E-01, .102268934886E+01 },
  .odds = {  .552906497058E+01,  .859731091202E-01, -.303881382425E+00, .101972266315E+01 },
  .oddp = { -.856025085531E+01,  .413682082679E+00,  .561269698491E+00, .119817640580E+01 },
  .oddd = {  .836929253859E-01, -.307737391082E+00,  .754080691966E-01, .983776299155E+00 } };
  
  return cu;
}

// screening function, F_c(R)
// RETURN: function value
double screen(double R, // distance between a pair of atoms
              struct nrl_tb *param) // tight-binding parameters [1]
{
  if(R > param->Rcut) { return 0.0; }
  else { return 1.0/(1.0 + exp((R-param->R0)/param->Rs)); }
}
// RETURN: function derivative
double dscreen_dR(double R, // distance between a pair of atoms
                  struct nrl_tb *param) // tight-binding parameters [1]
{
  if(R > param->Rcut) { return 0.0; }  
  else { return -0.5/((1.0 + cosh((R-param->R0)/param->Rs))*param->Rs); }
}

// local environment parameter for on-site Hamiltonian, which must be summed over neighbors
// RETURN: function value
double rho(double R, // distance between a pair of atoms
           struct nrl_tb *param) // tight-binding parameters [1]
{
  return exp(-pow(param->lambda,2)*R)*screen(R,param);
}
// RETURN: function derivative
double drho_dR(double R, // distance between a pair of atoms
               struct nrl_tb *param) // tight-binding parameters [1]
{
  return exp(-pow(param->lambda,2)*R)*(dscreen_dR(R,param) - pow(param->lambda,2)*screen(R,param));
}

// on-site tight-binding matrix element
#define RHO0 1e-16 // regularization factor
// RETURN: function value
double onsite(double rho, // total rho value summed over neighbors
              double *abcd) // a, b, c, d from the NRL parameters [4]
{
  return abcd[0] + abcd[1]*pow(RHO0+rho,2.0/3.0) + abcd[2]*pow(RHO0+rho,4.0/3.0) + abcd[3]*pow(RHO0+rho,2);
}
// RETURN: derivative value
double donsite_drho(double rho, // total rho value summed over neighbors
                    double *abcd) // a, b, c, d from the NRL parameters [4]
{
  return (2.0/3.0)*abcd[1]*pow(RHO0+rho,-1.0/3.0) + (4.0/3.0)*abcd[2]*pow(RHO0+rho,1.0/3.0) + 2.0*abcd[3]*(RHO0+rho);
}

// bonding functions used to define hopping matrix elements
// RETURN: function value
double bond(double R, // distance between a pair of atoms
            double *effg, // e, f, fbar, g from the NRL parameters [4]
            struct nrl_tb *param) // tight-binding parameters [1]
{
  return (effg[0] + effg[1]*R + effg[2]*R*R)*exp(-effg[3]*effg[3]*R)*screen(R,param);
}
// RETURN: derivative value
double dbond_dR(double R, // distance between a pair of atoms
                double *effg, // e, f, fbar, g from the NRL parameters [4]
                struct nrl_tb *param) // tight-binding parameters [1]
{
  return (effg[1] + 2.0*effg[2]*R)*exp(-effg[3]*effg[3]*R)*screen(R,param)
         - effg[3]*effg[3]*(effg[0] + effg[1]*R + effg[2]*R*R)*exp(-effg[3]*effg[3]*R)*screen(R,param)
         + (effg[0] + effg[1]*R + effg[2]*R*R)*exp(-effg[3]*effg[3]*R)*dscreen_dR(R,param);
}

// symmetrically fill in a matrix block of an s/p/d Slater-Koster tight-binding model
// NOTE: using notation consistent with [Phys. Rev. 94, 1498 (1954)]
//       & orbitals ordered as: s, p_x, p_y, p_z, d_xy, d_yz, d_zx, d_{x^2-y^2}, d_{3z^2-r^2}
void fill_mat(double l, // x directional cosine
              double m, // y directional cosine
              double n, // z directional cosine
              double sss, // s s sigma term
              double sps, // s p sigma term
              double pps, // p p sigma term
              double ppp, // p p pi term
              double sds, // s d sigma term
              double pds, // p d sigma term
              double pdp, // p d pi term
              double dds, // d d sigma term
              double ddp, // d d pi term
              double ddd, // d d delta term
              double *mat) // 9-by-9 matrix block [81]
{
  // ss terms
  mat[0+0*9] = sss;

  // sp terms
  mat[1+0*9] = -(mat[0+1*9] = l*sps);
  mat[2+0*9] = -(mat[0+2*9] = m*sps);
  mat[3+0*9] = -(mat[0+3*9] = n*sps);

  // pp terms
  mat[1+1*9] = l*l*pps + (1.0 - l*l)*ppp;
  mat[2+2*9] = m*m*pps + (1.0 - m*m)*ppp;
  mat[3+3*9] = n*n*pps + (1.0 - n*n)*ppp;
  mat[2+1*9] = mat[1+2*9] = l*m*pps - l*m*ppp;
  mat[3+1*9] = mat[1+3*9] = l*n*pps - l*n*ppp;
  mat[3+2*9] = mat[2+3*9] = m*n*pps - m*n*ppp;

  // sd terms
  mat[4+0*9] = mat[0+4*9] = sqrt(3.0)*l*m*sds;
  mat[5+0*9] = mat[0+5*9] = sqrt(3.0)*m*n*sds;
  mat[6+0*9] = mat[0+6*9] = sqrt(3.0)*n*l*sds;
  mat[7+0*9] = mat[0+7*9] = 0.5*sqrt(3.0)*(l*l - m*m)*sds;
  mat[8+0*9] = mat[0+8*9] = (n*n - 0.5*(l*l + m*m))*sds;

  // pd terms
  mat[4+1*9] = -(mat[1+4*9] = sqrt(3.0)*l*l*m*pds + m*(1.0 - 2.0*l*l)*pdp);
  mat[5+2*9] = -(mat[2+5*9] = sqrt(3.0)*m*m*n*pds + n*(1.0 - 2.0*m*m)*pdp);
  mat[6+3*9] = -(mat[3+6*9] = sqrt(3.0)*n*n*l*pds + l*(1.0 - 2.0*n*n)*pdp);
  mat[4+3*9] = mat[6+2*9] = mat[5+1*9] = -(mat[1+5*9] = mat[2+6*9] = mat[3+4*9] = sqrt(3.0)*l*m*n*pds - 2.0*l*m*n*pdp);
  mat[6+1*9] = -(mat[1+6*9] = sqrt(3.0)*l*l*n*pds + n*(1.0 - 2.0*l*l)*pdp);
  mat[4+2*9] = -(mat[2+4*9] = sqrt(3.0)*m*m*l*pds + l*(1.0 - 2.0*m*m)*pdp);
  mat[5+3*9] = -(mat[3+5*9] = sqrt(3.0)*n*n*m*pds + m*(1.0 - 2.0*n*n)*pdp);
  mat[7+1*9] = -(mat[1+7*9] = 0.5*sqrt(3.0)*l*(l*l - m*m)*pds + l*(1.0 - l*l + m*m)*pdp);
  mat[7+2*9] = -(mat[2+7*9] = 0.5*sqrt(3.0)*m*(l*l - m*m)*pds - m*(1.0 + l*l - m*m)*pdp);
  mat[7+3*9] = -(mat[3+7*9] = 0.5*sqrt(3.0)*n*(l*l - m*m)*pds - n*(l*l - m*m)*pdp);
  mat[8+1*9] = -(mat[1+8*9] = l*(n*n - 0.5*(l*l + m*m))*pds - sqrt(3.0)*l*n*n*pdp);
  mat[8+2*9] = -(mat[2+8*9] = m*(n*n - 0.5*(l*l + m*m))*pds - sqrt(3.0)*m*n*n*pdp);
  mat[8+3*9] = -(mat[3+8*9] = n*(n*n - 0.5*(l*l + m*m))*pds + sqrt(3.0)*n*(l*l + m*m)*pdp);

  // dd terms
  mat[4+4*9] = 3.0*l*l*m*m*dds + (l*l + m*m - 4.0*l*l*m*m)*ddp + (n*n + l*l*m*m)*ddd;
  mat[5+5*9] = 3.0*m*m*n*n*dds + (m*m + n*n - 4.0*m*m*n*n)*ddp + (l*l + m*m*n*n)*ddd;
  mat[6+6*9] = 3.0*n*n*l*l*dds + (n*n + l*l - 4.0*n*n*l*l)*ddp + (m*m + n*n*l*l)*ddd;
  mat[5+4*9] = mat[4+5*9] = 3.0*l*m*m*n*dds + l*n*(1.0 - 4.0*m*m)*ddp + l*n*(m*m - 1.0)*ddd;
  mat[6+5*9] = mat[5+6*9] = 3.0*m*n*n*l*dds + m*l*(1.0 - 4.0*n*n)*ddp + m*l*(n*n - 1.0)*ddd;
  mat[6+4*9] = mat[4+6*9] = 3.0*n*l*l*m*dds + n*m*(1.0 - 4.0*l*l)*ddp + n*m*(l*l - 1.0)*ddd;
  mat[7+4*9] = mat[4+7*9] = 1.5*l*m*(l*l - m*m)*dds + 2.0*l*m*(m*m - l*l)*ddp + 0.5*l*m*(l*l - m*m)*ddd;
  mat[7+5*9] = mat[5+7*9] = 1.5*m*n*(l*l - m*m)*dds - m*n*(1.0 + 2.0*(l*l - m*m))*ddp + m*n*(1.0 + 0.5*(l*l - m*m))*ddd;
  mat[7+6*9] = mat[6+7*9] = 1.5*n*l*(l*l - m*m)*dds + n*l*(1.0 - 2.0*(l*l - m*m))*ddp - n*l*(1.0 - 0.5*(l*l - m*m))*ddd;
  mat[8+4*9] = mat[4+8*9] = sqrt(3.0)*(l*m*(n*n - 0.5*(l*l + m*m))*dds - 2.0*l*m*n*n*ddp + 0.5*l*m*(1.0 + n*n)*ddd);
  mat[8+5*9] = mat[5+8*9] = sqrt(3.0)*(m*n*(n*n - 0.5*(l*l + m*m))*dds + m*n*(l*l + m*m - n*n)*ddp - 0.5*m*n*(l*l + m*m)*ddd);
  mat[8+6*9] = mat[6+8*9] = sqrt(3.0)*(n*l*(n*n - 0.5*(l*l + m*m))*dds + n*l*(l*l + m*m - n*n)*ddp - 0.5*n*l*(l*l + m*m)*ddd);
  mat[7+7*9] = 0.75*pow(l*l - m*m,2)*dds + (l*l + m*m - pow(l*l - m*m,2))*ddp + (n*n + 0.25*pow(l*l - m*m,2))*ddd;
  mat[8+7*9] = mat[7+8*9] = sqrt(3.0)*(0.5*(l*l - m*m)*(n*n - 0.5*(l*l + m*m))*dds + n*n*(m*m - l*l)*ddp
                                       + 0.25*(1.0 + n*n)*(l*l - m*m)*ddd);
  mat[8+8*9] = pow(n*n - 0.5*(l*l + m*m),2)*dds + 3.0*n*n*(l*l + m*m)*ddp + 0.75*pow(l*l + m*m,2)*ddd;
}

// derivative of the Slater-Koster matrices w.r.t. l/m/n
void fill_dmat(double l, // x directional cosine
               double m, // y directional cosine
               double n, // z directional cosine
               double sss, // s s sigma term
               double sps, // s p sigma term
               double pps, // p p sigma term
               double ppp, // p p pi term
               double sds, // s d sigma term
               double pds, // p d sigma term
               double pdp, // p d pi term
               double dds, // d d sigma term
               double ddp, // d d pi term
               double ddd, // d d delta term
               double *dmat) // array of 3 9-by-9 matrix blocks [243]
{
  // ss terms
  dmat[0+0*9+0*81] = 0.0;
  dmat[0+0*9+1*81] = 0.0;
  dmat[0+0*9+2*81] = 0.0;

  // sp terms
  dmat[1+0*9+0*81] = -(dmat[0+1*9+0*81] = sps);
  dmat[1+0*9+1*81] = -(dmat[0+1*9+1*81] = 0.0);
  dmat[1+0*9+2*81] = -(dmat[0+1*9+2*81] = 0.0);
  dmat[2+0*9+0*81] = -(dmat[0+2*9+0*81] = 0.0);
  dmat[2+0*9+1*81] = -(dmat[0+2*9+1*81] = sps);
  dmat[2+0*9+2*81] = -(dmat[0+2*9+2*81] = 0.0);
  dmat[3+0*9+0*81] = -(dmat[0+3*9+0*81] = 0.0);
  dmat[3+0*9+1*81] = -(dmat[0+3*9+1*81] = 0.0);
  dmat[3+0*9+2*81] = -(dmat[0+3*9+2*81] = sps);

  // pp terms
  dmat[1+1*9+0*81] = 2.0*l*pps - 2.0*l*ppp;
  dmat[1+1*9+1*81] = 0.0;
  dmat[1+1*9+2*81] = 0.0;
  dmat[2+2*9+0*81] = 0.0;
  dmat[2+2*9+1*81] = 2.0*m*pps - 2.0*m*ppp;
  dmat[2+2*9+2*81] = 0.0;  
  dmat[3+3*9+0*81] = 0.0;
  dmat[3+3*9+1*81] = 0.0;
  dmat[3+3*9+2*81] = 2.0*n*pps - 2.0*n*ppp;
  dmat[2+1*9+0*81] = dmat[1+2*9+0*81] = m*pps - m*ppp;
  dmat[2+1*9+1*81] = dmat[1+2*9+1*81] = l*pps - l*ppp;
  dmat[2+1*9+2*81] = dmat[1+2*9+2*81] = 0.0;
  dmat[3+1*9+0*81] = dmat[1+3*9+0*81] = n*pps - n*ppp;
  dmat[3+1*9+1*81] = dmat[1+3*9+1*81] = 0.0;
  dmat[3+1*9+2*81] = dmat[1+3*9+2*81] = l*pps - l*ppp;
  dmat[3+2*9+0*81] = dmat[2+3*9+0*81] = 0.0;
  dmat[3+2*9+1*81] = dmat[2+3*9+1*81] = n*pps - n*ppp;
  dmat[3+2*9+2*81] = dmat[2+3*9+2*81] = m*pps - m*ppp;

  // sd terms
  dmat[4+0*9+0*81] = dmat[0+4*9+0*81] = sqrt(3.0)*m*sds;
  dmat[4+0*9+1*81] = dmat[0+4*9+1*81] = sqrt(3.0)*l*sds;
  dmat[4+0*9+2*81] = dmat[0+4*9+2*81] = 0.0;
  dmat[5+0*9+0*81] = dmat[0+5*9+0*81] = 0.0;
  dmat[5+0*9+1*81] = dmat[0+5*9+1*81] = sqrt(3.0)*n*sds;
  dmat[5+0*9+2*81] = dmat[0+5*9+2*81] = sqrt(3.0)*m*sds;
  dmat[6+0*9+0*81] = dmat[0+6*9+0*81] = sqrt(3.0)*n*sds;
  dmat[6+0*9+1*81] = dmat[0+6*9+1*81] = 0.0;
  dmat[6+0*9+2*81] = dmat[0+6*9+2*81] = sqrt(3.0)*l*sds;
  dmat[7+0*9+0*81] = dmat[0+7*9+0*81] = sqrt(3.0)*l*sds;
  dmat[7+0*9+1*81] = dmat[0+7*9+1*81] = -sqrt(3.0)*m*sds;
  dmat[7+0*9+2*81] = dmat[0+7*9+2*81] = 0.0;
  dmat[8+0*9+0*81] = dmat[0+8*9+0*81] = -l*sds;
  dmat[8+0*9+1*81] = dmat[0+8*9+1*81] = -m*sds;
  dmat[8+0*9+2*81] = dmat[0+8*9+2*81] = 2.0*n*sds;

  // pd terms
  dmat[4+1*9+0*81] = -(dmat[1+4*9+0*81] = 2.0*sqrt(3.0)*l*m*pds - 4.0*m*l*pdp);
  dmat[4+1*9+1*81] = -(dmat[1+4*9+1*81] = sqrt(3.0)*l*l*pds + (1.0 - 2.0*l*l)*pdp);
  dmat[4+1*9+2*81] = -(dmat[1+4*9+2*81] = 0.0);
  dmat[5+2*9+0*81] = -(dmat[2+5*9+0*81] = 0.0);
  dmat[5+2*9+1*81] = -(dmat[2+5*9+1*81] = 2.0*sqrt(3.0)*m*n*pds - 4.0*n*m*pdp);
  dmat[5+2*9+2*81] = -(dmat[2+5*9+2*81] = sqrt(3.0)*m*m*pds + (1.0 - 2.0*m*m)*pdp);
  dmat[6+3*9+0*81] = -(dmat[3+6*9+0*81] = sqrt(3.0)*n*n*pds + (1.0 - 2.0*n*n)*pdp);
  dmat[6+3*9+1*81] = -(dmat[3+6*9+1*81] = 0.0);
  dmat[6+3*9+2*81] = -(dmat[3+6*9+2*81] = 2.0*sqrt(3.0)*n*l*pds - 4.0*l*n*pdp);
  dmat[4+3*9+0*81] = dmat[6+2*9+0*81] = dmat[5+1*9+0*81] 
                   = -(dmat[1+5*9+0*81] = dmat[2+6*9+0*81] = dmat[3+4*9+0*81] = sqrt(3.0)*m*n*pds - 2.0*m*n*pdp);
  dmat[4+3*9+1*81] = dmat[6+2*9+1*81] = dmat[5+1*9+1*81] 
                   = -(dmat[1+5*9+1*81] = dmat[2+6*9+1*81] = dmat[3+4*9+1*81] = sqrt(3.0)*l*n*pds - 2.0*l*n*pdp);
  dmat[4+3*9+2*81] = dmat[6+2*9+2*81] = dmat[5+1*9+2*81] 
                   = -(dmat[1+5*9+2*81] = dmat[2+6*9+2*81] = dmat[3+4*9+2*81] = sqrt(3.0)*l*m*pds - 2.0*l*m*pdp);
  dmat[6+1*9+0*81] = -(dmat[1+6*9+0*81] = 2.0*sqrt(3.0)*l*n*pds - 4.0*n*l*pdp);
  dmat[6+1*9+1*81] = -(dmat[1+6*9+1*81] = 0.0);
  dmat[6+1*9+2*81] = -(dmat[1+6*9+2*81] = sqrt(3.0)*l*l*pds + (1.0 - 2.0*l*l)*pdp);
  dmat[4+2*9+0*81] = -(dmat[2+4*9+0*81] = sqrt(3.0)*m*m*pds + (1.0 - 2.0*m*m)*pdp);
  dmat[4+2*9+1*81] = -(dmat[2+4*9+1*81] = 2.0*sqrt(3.0)*m*l*pds - 4.0*l*m*pdp);
  dmat[4+2*9+2*81] = -(dmat[2+4*9+2*81] = 0.0);
  dmat[5+3*9+0*81] = -(dmat[3+5*9+0*81] = 0.0);
  dmat[5+3*9+1*81] = -(dmat[3+5*9+1*81] = sqrt(3.0)*n*n*pds + (1.0 - 2.0*n*n)*pdp);
  dmat[5+3*9+2*81] = -(dmat[3+5*9+2*81] = 2.0*sqrt(3.0)*n*m*pds - 4.0*m*n*pdp);
  dmat[7+1*9+0*81] = -(dmat[1+7*9+0*81] = 0.5*sqrt(3.0)*(3.0*l*l - m*m)*pds + (1.0 - 3.0*l*l + m*m)*pdp);
  dmat[7+1*9+1*81] = -(dmat[1+7*9+1*81] = -sqrt(3.0)*l*m*pds + 2.0*l*m*pdp);
  dmat[7+1*9+2*81] = -(dmat[1+7*9+2*81] = 0.0);
  dmat[7+2*9+0*81] = -(dmat[2+7*9+0*81] = sqrt(3.0)*m*l*pds - 2.0*m*l*pdp);
  dmat[7+2*9+1*81] = -(dmat[2+7*9+1*81] = 0.5*sqrt(3.0)*(l*l - 3.0*m*m)*pds - (1.0 + l*l - 3.0*m*m)*pdp);
  dmat[7+2*9+2*81] = -(dmat[2+7*9+2*81] = 0.0);
  dmat[7+3*9+0*81] = -(dmat[3+7*9+0*81] = sqrt(3.0)*n*l*pds - 2.0*n*l*pdp);
  dmat[7+3*9+1*81] = -(dmat[3+7*9+1*81] = -sqrt(3.0)*n*m*pds + 2.0*n*m*pdp);
  dmat[7+3*9+2*81] = -(dmat[3+7*9+2*81] = 0.5*sqrt(3.0)*(l*l - m*m)*pds - (l*l - m*m)*pdp);
  dmat[8+1*9+0*81] = -(dmat[1+8*9+0*81] = (n*n - 0.5*(3.0*l*l + m*m))*pds - sqrt(3.0)*n*n*pdp);
  dmat[8+1*9+1*81] = -(dmat[1+8*9+1*81] = -l*m*pds);
  dmat[8+1*9+2*81] = -(dmat[1+8*9+2*81] = 2.0*l*n*pds - 2.0*sqrt(3.0)*l*n*pdp);
  dmat[8+2*9+0*81] = -(dmat[2+8*9+0*81] = -m*l*pds);
  dmat[8+2*9+1*81] = -(dmat[2+8*9+1*81] = (n*n - 0.5*(l*l + 3.0*m*m))*pds - sqrt(3.0)*n*n*pdp);
  dmat[8+2*9+2*81] = -(dmat[2+8*9+2*81] = 2.0*m*n*pds - 2.0*sqrt(3.0)*m*n*pdp);
  dmat[8+3*9+0*81] = -(dmat[3+8*9+0*81] = -n*l*pds + 2.0*sqrt(3.0)*n*l*pdp);
  dmat[8+3*9+1*81] = -(dmat[3+8*9+1*81] = -n*m*pds + 2.0*sqrt(3.0)*n*m*pdp);
  dmat[8+3*9+2*81] = -(dmat[3+8*9+2*81] = (3.0*n*n - 0.5*(l*l + m*m))*pds + sqrt(3.0)*(l*l + m*m)*pdp);

  // dd terms
  dmat[4+4*9+0*81] = 6.0*l*m*m*dds + (2.0*l - 8.0*l*m*m)*ddp + 2.0*l*m*m*ddd;
  dmat[4+4*9+1*81] = 6.0*l*l*m*dds + (2.0*m - 8.0*l*l*m)*ddp + 2.0*l*l*m*ddd;
  dmat[4+4*9+2*81] = 2.0*n*ddd;
  dmat[5+5*9+0*81] = 2.0*l*ddd;
  dmat[5+5*9+1*81] = 6.0*m*n*n*dds + (2.0*m - 8.0*m*n*n)*ddp + 2.0*m*n*n*ddd;
  dmat[5+5*9+2*81] = 6.0*m*m*n*dds + (2.0*n - 8.0*m*m*n)*ddp + 2.0*m*m*n*ddd;
  dmat[6+6*9+0*81] = 6.0*n*n*l*dds + (2.0*l - 8.0*n*n*l)*ddp + 2.0*n*n*l*ddd;
  dmat[6+6*9+1*81] = 2.0*m*ddd;
  dmat[6+6*9+2*81] = 6.0*n*l*l*dds + (2.0*n - 8.0*n*l*l)*ddp + 2.0*n*l*l*ddd;
  dmat[5+4*9+0*81] = dmat[4+5*9+0*81] = 3.0*m*m*n*dds + n*(1.0 - 4.0*m*m)*ddp + n*(m*m - 1.0)*ddd;
  dmat[5+4*9+1*81] = dmat[4+5*9+1*81] = 6.0*l*m*n*dds - 8.0*l*n*m*ddp + 2.0*l*n*m*ddd;
  dmat[5+4*9+2*81] = dmat[4+5*9+2*81] = 3.0*l*m*m*dds + l*(1.0 - 4.0*m*m)*ddp + l*(m*m - 1.0)*ddd;
  dmat[6+5*9+0*81] = dmat[5+6*9+0*81] = 3.0*m*n*n*dds + m*(1.0 - 4.0*n*n)*ddp + m*(n*n - 1.0)*ddd;
  dmat[6+5*9+1*81] = dmat[5+6*9+1*81] = 3.0*n*n*l*dds + l*(1.0 - 4.0*n*n)*ddp + l*(n*n - 1.0)*ddd;
  dmat[6+5*9+2*81] = dmat[5+6*9+2*81] = 6.0*m*n*l*dds - 8.0*m*l*n*ddp + 2.0*m*l*n*ddd;
  dmat[6+4*9+0*81] = dmat[4+6*9+0*81] = 6.0*n*l*m*dds - 8.0*n*m*l*ddp + 2.0*n*m*l*ddd;
  dmat[6+4*9+1*81] = dmat[4+6*9+1*81] = 3.0*n*l*l*dds + n*(1.0 - 4.0*l*l)*ddp + n*(l*l - 1.0)*ddd;
  dmat[6+4*9+2*81] = dmat[4+6*9+2*81] = 3.0*l*l*m*dds + m*(1.0 - 4.0*l*l)*ddp + m*(l*l - 1.0)*ddd;
  dmat[7+4*9+0*81] = dmat[4+7*9+0*81] = 1.5*m*(3.0*l*l - m*m)*dds + 2.0*m*(m*m - 3.0*l*l)*ddp + 0.5*m*(3.0*l*l - m*m)*ddd;
  dmat[7+4*9+1*81] = dmat[4+7*9+1*81] = 1.5*l*(l*l - 3.0*m*m)*dds + 2.0*l*(3.0*m*m - l*l)*ddp + 0.5*l*(l*l - 3.0*m*m)*ddd;
  dmat[7+4*9+2*81] = dmat[4+7*9+2*81] = 0.0;
  dmat[7+5*9+0*81] = dmat[5+7*9+0*81] = 3.0*m*n*l*dds - 4.0*m*n*l*ddp + m*n*l*ddd;
  dmat[7+5*9+1*81] = dmat[5+7*9+1*81] = 1.5*n*(l*l - 3.0*m*m)*dds - n*(1.0 + 2.0*(l*l - 3.0*m*m))*ddp
                                        + n*(1.0 + 0.5*(l*l - 3.0*m*m))*ddd;
  dmat[7+5*9+2*81] = dmat[5+7*9+2*81] = 1.5*m*(l*l - m*m)*dds - m*(1.0 + 2.0*(l*l - m*m))*ddp + m*(1.0 + 0.5*(l*l - m*m))*ddd;
  dmat[7+6*9+0*81] = dmat[6+7*9+0*81] = 1.5*n*(3.0*l*l - m*m)*dds + n*(1.0 - 2.0*(3.0*l*l - m*m))*ddp
                                        - n*(1.0 - 0.5*(3.0*l*l - m*m))*ddd;
  dmat[7+6*9+1*81] = dmat[6+7*9+1*81] = -3.0*n*l*m*dds + 4.0*n*l*m*ddp - n*l*m*ddd;
  dmat[7+6*9+2*81] = dmat[6+7*9+2*81] = 1.5*l*(l*l - m*m)*dds + l*(1.0 - 2.0*(l*l - m*m))*ddp - l*(1.0 - 0.5*(l*l - m*m))*ddd;
  dmat[8+4*9+0*81] = dmat[4+8*9+0*81] = sqrt(3.0)*(m*(n*n - 0.5*(3.0*l*l + m*m))*dds - 2.0*m*n*n*ddp + 0.5*m*(1.0 + n*n)*ddd);
  dmat[8+4*9+1*81] = dmat[4+8*9+1*81] = sqrt(3.0)*(l*(n*n - 0.5*(l*l + 3.0*m*m))*dds - 2.0*l*n*n*ddp + 0.5*l*(1.0 + n*n)*ddd);
  dmat[8+4*9+2*81] = dmat[4+8*9+2*81] = sqrt(3.0)*(2.0*l*m*n*dds - 4.0*l*m*n*ddp + l*m*n*ddd);
  dmat[8+5*9+0*81] = dmat[5+8*9+0*81] = sqrt(3.0)*(-m*n*l*dds + 2.0*m*n*l*ddp - m*n*l*ddd);
  dmat[8+5*9+1*81] = dmat[5+8*9+1*81] = sqrt(3.0)*(n*(n*n - 0.5*(l*l + 3.0*m*m))*dds + n*(l*l + 3.0*m*m - n*n)*ddp
                                        - 0.5*n*(l*l + 3.0*m*m)*ddd);
  dmat[8+5*9+2*81] = dmat[5+8*9+2*81] = sqrt(3.0)*(m*(3.0*n*n - 0.5*(l*l + m*m))*dds + m*(l*l + m*m - 3.0*n*n)*ddp
                                        - 0.5*m*(l*l + m*m)*ddd);
  dmat[8+6*9+0*81] = dmat[6+8*9+0*81] = sqrt(3.0)*(n*(n*n - 0.5*(3.0*l*l + m*m))*dds + n*(3.0*l*l + m*m - n*n)*ddp
                                        - 0.5*n*(3.0*l*l + m*m)*ddd);
  dmat[8+6*9+1*81] = dmat[6+8*9+1*81] = sqrt(3.0)*(-n*l*m*dds + 2.0*n*l*m*ddp - n*l*m*ddd);
  dmat[8+6*9+2*81] = dmat[6+8*9+2*81] = sqrt(3.0)*(l*(3.0*n*n - 0.5*(l*l + m*m))*dds + l*(l*l + m*m - 3.0*n*n)*ddp
                                        - 0.5*l*(l*l + m*m)*ddd);
  dmat[7+7*9+0*81] = 3.0*l*(l*l - m*m)*dds + (2.0*l - 4.0*l*(l*l - m*m))*ddp + l*(l*l - m*m)*ddd;
  dmat[7+7*9+1*81] = -3.0*m*(l*l - m*m)*dds + (2.0*m + 4.0*m*(l*l - m*m))*ddp - m*(l*l - m*m)*ddd;
  dmat[7+7*9+2*81] = 2.0*n*ddd;
  dmat[8+7*9+0*81] = dmat[7+8*9+0*81] = sqrt(3.0)*(l*(n*n - 0.5*(l*l + m*m))*dds - 0.5*(l*l - m*m)*l*dds - 2.0*n*n*l*ddp
                                        + 0.5*(1.0 + n*n)*l*ddd);
  dmat[8+7*9+1*81] = dmat[7+8*9+1*81] = sqrt(3.0)*(-m*(n*n - 0.5*(l*l + m*m))*dds - 0.5*(l*l - m*m)*m*dds + 2.0*n*n*m*ddp
                                        - 0.5*(1.0 + n*n)*m*ddd);
  dmat[8+7*9+2*81] = dmat[7+8*9+2*81] = sqrt(3.0)*((l*l - m*m)*n*dds + 2.0*n*(m*m - l*l)*ddp + 0.5*n*(l*l - m*m)*ddd);
  dmat[8+8*9+0*81] = -2.0*l*(n*n - 0.5*(l*l + m*m))*dds + 6.0*n*n*l*ddp + 3.0*l*(l*l + m*m)*ddd;
  dmat[8+8*9+1*81] = -2.0*m*(n*n - 0.5*(l*l + m*m))*dds + 6.0*n*n*m*ddp + 3.0*m*(l*l + m*m)*ddd;
  dmat[8+8*9+2*81] = 4.0*n*(n*n - 0.5*(l*l + m*m))*dds + 6.0*n*(l*l + m*m)*ddp;
}

// distance between a pair of atoms
double distance(double *atom1, // coordinate of 1st atom [3]
                double *atom2) // coordinate of 2nd atom [3]
{
  double d2 = 0.0;
  for(size_t i=0 ; i<3 ; i++) { d2 += pow(atom1[i] - atom2[i],2); }
  return sqrt(d2);
}

// calculate a diagonal atomic matrix block of the tight-binding model
void tb_diagonal(int iatom, // atom index
                 int natom, // number of atoms
                 double *atom, // atomic coordinates [3*natom]
                 int nneighbor, // number of neighbors coupled to iatom
                 int *neighbor, // neighbor list of iatom [nneighbor]
                 struct nrl_tb *param, // tight-binding parameters [1]
                 double *hblock, // Hamiltonian matrix elements [81]
                 double *oblock) // overlap matrix elements [81]
{
  // calculate rho for iatom
  double rho0 = 0.0;
  for(int i=0 ; i<nneighbor ; i++)
  {
    if(iatom != neighbor[i])
    { rho0 += rho(distance(&(atom[3*iatom]),&(atom[3*neighbor[i]])),param); }
  }

  // calculate the matrix elements
  for(int i=0 ; i<81 ; i++) { hblock[i] = oblock[i] = 0.0; }
  hblock[0+0*9] = onsite(rho0,param->hs);
  hblock[1+1*9] = hblock[2+2*9] = hblock[3+3*9] = onsite(rho0,param->hp);
  hblock[4+4*9] = hblock[5+5*9] = hblock[6+6*9] = hblock[7+7*9] = hblock[8+8*9] = onsite(rho0,param->hd);
  for(int i=0 ; i<9 ; i++) { oblock[i+i*9] = 1.0; }
}

// calculate atomic response of a diagonal atomic matrix block of the tight-binding model
void tb_diagonal_force(int iatom, // atom index of matrix elements
                       int jatom, // atom index of perturbed atom
                       int natom, // number of atoms
                       double *atom, // atomic coordinates [3*natom]
                       int nneighbor, // number of neighbors coupled to iatom
                       int *neighbor, // neighbor list of iatom [nneighbor]
                       struct nrl_tb *param, // tight-binding parameters [1]
                       double *hblock_force) // array of 3 Hamiltonian matrix elements [243]
{
  // calculate rho for iatom
  double rho0 = 0.0, rho0_force[3] = { 0.0, 0.0, 0.0 };
  for(int i=0 ; i<nneighbor ; i++)
  {
    if(iatom != neighbor[i])
    { rho0 += rho(distance(&(atom[3*iatom]),&(atom[3*neighbor[i]])),param); }
  }

  // when iatom == jatom, the entire sum over neighbors contributes
  if(iatom == jatom)
  {
    for(int i=0 ; i<nneighbor ; i++)
    {
      if(iatom == neighbor[i]) { continue; }
      double R = distance(&(atom[3*iatom]),&(atom[3*neighbor[i]]));
      double drho_dR0 = drho_dR(R,param);
      for(int j=0 ; j<3 ; j++)
      { rho0_force[j] += drho_dR0*(atom[j+iatom*3]-atom[j+neighbor[i]*3])/R; }
    }
  }
  else // when iatom != jatom, only a single term in the rho sum is perturbed
  {
    double R = distance(&(atom[3*iatom]),&(atom[3*jatom]));
    double drho_dR0 = drho_dR(R,param);
    for(int j=0 ; j<3 ; j++)
    { rho0_force[j] += drho_dR0*(atom[j+jatom*3]-atom[j+iatom*3])/R; }
  }

  for(int i=0 ; i<243 ; i++) { hblock_force[i] = 0.0; }
  for(int i=0 ; i<3 ; i++)
  {
    hblock_force[0+0*9+i*81] = -donsite_drho(rho0,param->hs)*rho0_force[i];
    hblock_force[1+1*9+i*81] = hblock_force[2+2*9+i*81] = hblock_force[3+3*9+i*81]
                             = -donsite_drho(rho0,param->hp)*rho0_force[i];
    hblock_force[4+4*9+i*81] = hblock_force[5+5*9+i*81] = hblock_force[6+6*9+i*81] = hblock_force[7+7*9+i*81]
                             = hblock_force[8+8*9+i*81] = -donsite_drho(rho0,param->hd)*rho0_force[i];
  }
}

// calculate an offdiagonal atomic matrix block of the tight-binding model
void tb_offdiagonal(int iatom, // 1st atom index
                    int jatom, // 2nd atom index
                    int natom, // number of atoms
                    double *atom, // atomic coordinates [3*natom]
                    struct nrl_tb *param, // tight-binding parameters [1]
                    double *hblock, // Hamiltonian matrix elements [81]
                    double *oblock) // overlap matrix elements [81]
{
  // calculate distance between atoms and directional cosines
  double R = distance(&(atom[3*iatom]),&(atom[3*jatom]));
  double l = (atom[0+iatom*3]-atom[0+jatom*3])/R;
  double m = (atom[1+iatom*3]-atom[1+jatom*3])/R;
  double n = (atom[2+iatom*3]-atom[2+jatom*3])/R;

  fill_mat(l,m,n,bond(R,param->hsss,param),bond(R,param->hsps,param),bond(R,param->hpps,param),bond(R,param->hppp,param),
                 bond(R,param->hsds,param),bond(R,param->hpds,param),bond(R,param->hpdp,param),bond(R,param->hdds,param),
                 bond(R,param->hddp,param),bond(R,param->hddd,param),hblock);
  fill_mat(l,m,n,bond(R,param->osss,param),bond(R,param->osps,param),bond(R,param->opps,param),bond(R,param->oppp,param),
                 bond(R,param->osds,param),bond(R,param->opds,param),bond(R,param->opdp,param),bond(R,param->odds,param),
                 bond(R,param->oddp,param),bond(R,param->oddd,param),oblock);
}

// calculate atomic response of an offdiagonal atomic matrix block of the tight-binding model
void tb_offdiagonal_force(int iatom, // 1st atom index & perturbed atom
                          int jatom, // 2nd atom index
                          int natom, // number of atoms
                          double *atom, // atomic coordinates [3*natom]
                          struct nrl_tb *param, // tight-binding parameters [1]
                          double *hblock_force, // array of 3 Hamiltonian matrix elements [243]
                          double *oblock_force) // array of 3 overlap matrix elements [243]
{
  // calculate distance between atoms and directional cosines
  double R = distance(&(atom[3*iatom]),&(atom[3*jatom]));
  double l = (atom[0+iatom*3] - atom[0+jatom*3])/R;
  double m = (atom[1+iatom*3] - atom[1+jatom*3])/R;
  double n = (atom[2+iatom*3] - atom[2+jatom*3])/R;

  // derivative of the bond functions
  double dhblock_dR[81], doblock_dR[81];
  fill_mat(l,m,n,dbond_dR(R,param->hsss,param),dbond_dR(R,param->hsps,param),dbond_dR(R,param->hpps,param),
                 dbond_dR(R,param->hppp,param),dbond_dR(R,param->hsds,param),dbond_dR(R,param->hpds,param),
                 dbond_dR(R,param->hpdp,param),dbond_dR(R,param->hdds,param),dbond_dR(R,param->hddp,param),
                 dbond_dR(R,param->hddd,param),dhblock_dR);
  fill_mat(l,m,n,dbond_dR(R,param->osss,param),dbond_dR(R,param->osps,param),dbond_dR(R,param->opps,param),
                 dbond_dR(R,param->oppp,param),dbond_dR(R,param->osds,param),dbond_dR(R,param->opds,param),
                 dbond_dR(R,param->opdp,param),dbond_dR(R,param->odds,param),dbond_dR(R,param->oddp,param),
                 dbond_dR(R,param->oddd,param),doblock_dR);

  // derivative of l/m/n
  double dhblock_dlmn[243], doblock_dlmn[243];
  fill_dmat(l,m,n,bond(R,param->hsss,param),bond(R,param->hsps,param),bond(R,param->hpps,param),bond(R,param->hppp,param),
                  bond(R,param->hsds,param),bond(R,param->hpds,param),bond(R,param->hpdp,param),bond(R,param->hdds,param),
                  bond(R,param->hddp,param),bond(R,param->hddd,param),dhblock_dlmn);
  fill_dmat(l,m,n,bond(R,param->osss,param),bond(R,param->osps,param),bond(R,param->opps,param),bond(R,param->oppp,param),
                  bond(R,param->osds,param),bond(R,param->opds,param),bond(R,param->opdp,param),bond(R,param->odds,param),
                  bond(R,param->oddp,param),bond(R,param->oddd,param),doblock_dlmn);

  for(int i=0 ; i<81 ; i++)
  {
    double dhblock0 = dhblock_dlmn[i+0*81]*l + dhblock_dlmn[i+1*81]*m + dhblock_dlmn[i+2*81]*n;
    hblock_force[i+0*81] = -dhblock_dR[i]*l - dhblock_dlmn[i+0*81]/R + dhblock0*l/R;
    hblock_force[i+1*81] = -dhblock_dR[i]*m - dhblock_dlmn[i+1*81]/R + dhblock0*m/R;
    hblock_force[i+2*81] = -dhblock_dR[i]*n - dhblock_dlmn[i+2*81]/R + dhblock0*n/R;
    double doblock0 = doblock_dlmn[i+0*81]*l + doblock_dlmn[i+1*81]*m + doblock_dlmn[i+2*81]*n;
    oblock_force[i+0*81] = -doblock_dR[i]*l - doblock_dlmn[i+0*81]/R + doblock0*l/R;
    oblock_force[i+1*81] = -doblock_dR[i]*m - doblock_dlmn[i+1*81]/R + doblock0*m/R;
    oblock_force[i+2*81] = -doblock_dR[i]*n - doblock_dlmn[i+2*81]/R + doblock0*n/R;
  }
}

//========================//
// 3. ATOMIC PARTITIONING //
//========================//

// NOTE: This version does not support additional blocking of atoms, which would improve performance but complicate the code
//       The atoms would be reordered so that atoms within a block are contiguous & a neighbor list of blocks would be computed
//       in addition to the neighbor list of atoms to define the block-sparse density matrix structure

// grid of boxes structure that partition the atoms
struct grid
{
  int nx[3]; // number of boxes in each direction
  double x0[3]; // minimum coordinate in each direction
  double dx[3]; // width of boxes in each direction
  int *to_atom; // index of locations in atom_index for the first atom in each box [nx[0]*nx[1]*nx[2]+1]
                // NOTE: this list is ordered & to_atom[nx[0]*nx[1]*nx[2]] is the number of atoms
  int *atom_index; // list of atom indices contained in each box [to_atom[nx*ny*nz]]
};

// find the box that contains a given atom
void box_index(double *atom, // target atom [3]
               struct grid *partition, // specification of the grid for partitioning atoms [1]
               int *box) // output box index [3]
{
  for(int i=0 ; i<3 ; i++)
  { box[i] = (int)((atom[i] - partition->x0[i])/partition->dx[i]); }
}

// Find the grid index of a box
int grid_index(int* box, // target box [3]
               struct grid* partition) // specification of the grid for partitioning atoms [1]
{
  return (box[0] + partition->nx[0]*(box[1] + partition->nx[1]*box[2]));
}

// comparison function for sorting atoms by box using the C qsort function in stdlib.h
// RETURN: 1 if a goes after b, -1 if a goes before b, 0 if they are equal
int list_compare(const void *a, const void *b)
{
  // sort by grid index first ...
  if( ((int*)a)[1] > ((int*)b)[1] ) return 1;
  if( ((int*)a)[1] < ((int*)b)[1] ) return -1;
  // ... and atom index second
  if( ((int*)a)[0] > ((int*)b)[0] ) return 1;
  if( ((int*)a)[0] < ((int*)b)[0] ) return -1;
  return 0;
}

// construct the grid structure for a list of atoms and a box width
// RETURN: grid structure with allocated memory
struct grid construct_grid(int natom, // number of atoms
                           double *atom, // atomic coordinates [3*natom]
                           double width) // box width that defines the uniform grid of boxes
{
  struct grid partition;

  // define the grid coordinates
  for(int i=0 ; i<3 ; i++)
  {
    double xmin = atom[i], xmax = atom[i];
    for(int j=1 ; j<natom ; j++)
    {
      if(atom[i+j*3] < xmin) { xmin = atom[i+j*3]; }
      if(atom[i+j*3] > xmax) { xmax = atom[i+j*3]; }
    }
    partition.dx[i] = width; // uniform boxes
    partition.nx[i] = (int)ceil((xmax - xmin)/width) + 1; // pad to prevent atoms near grid boundaries
    partition.x0[i] = 0.5*(xmin + xmax - partition.nx[i]*width);
  }

  // memory allocation
  int ngrid = partition.nx[0]*partition.nx[1]*partition.nx[2];
  int *sort_list = (int*)malloc(sizeof(int)*2*natom);
  partition.atom_index = (int*)malloc(sizeof(int)*natom); // not locally deallocated
  partition.to_atom = (int*)malloc(sizeof(int)*(ngrid+1)); // not locally deallocated

  // assign each atom to a box in the grid
  for(int i=0 ; i<natom ; i++)
  {
    sort_list[2*i] = i;
    int box[3];
    box_index(&(atom[3*i]),&partition,box);
    sort_list[1+2*i] = grid_index(box,&partition);
  }
  
  // sort atoms by box
  qsort(sort_list,natom,sizeof(int)*2,list_compare);

  // move sorted list into atom_index & construct to_atom
  for(int i=0 ; i<ngrid ; i++)
  { partition.to_atom[i] = natom + 1; } // (natom + 1) indicates that a box has not been set yet
  partition.to_atom[ngrid] = natom; // last entry is the number of atoms
  for(int i=0 ; i<natom ; i++)
  {
    partition.atom_index[i] = sort_list[2*i];
    if(partition.to_atom[sort_list[1+2*i]] == (natom + 1))
    { partition.to_atom[sort_list[1+2*i]] = i; }
  }
  for(int i=ngrid ; i>=1 ; i--)
  {
    if(partition.to_atom[i-1] == (natom + 1))
    { partition.to_atom[i-1] = partition.to_atom[i]; }
  }
  
  // memory deallocation
  free(sort_list);

  return partition;
}

// comparison function for sorting neighbor lists using the C qsort function in stdlib.h
// RETURN: 1 if a goes after b, -1 if a goes before b, 0 if they are equal
int neighbor_compare(const void *a, const void *b)
{
  if( *((int*)a) > *((int*)b) ) return 1;
  if( *((int*)a) < *((int*)b) ) return -1;
  return 0;
}

// create a list of neighboring atoms for each atom (including self) as a sparsity pattern in CRS format
void neighbor_list(int natom, // number of atoms
                   double *atom, // atomic coordinates [3*natom]
                   double radius, // cutoff radius used to define the neighbor list
                   struct pattern *neighbor) // neighbor list defined by matrix sparsity pattern (no matrix elements) [1]
{
  // determine a minimum radius value to avoid memory problems
  double xmin[3], xmax[3];
  for(int i=0 ; i<3 ; i++)
  {
    xmin[i] = xmax[i] = atom[i];
    for(int j=1 ; j<natom ; j++)
    {
      if(atom[i+j*3] < xmin[i]) { xmin[i] = atom[i+j*3]; }
      if(atom[i+j*3] > xmax[i]) { xmax[i] = atom[i+j*3]; }
    }
  }
  double radius0 = pow((xmax[0] - xmin[0])*(xmax[1] - xmin[1])*(xmax[2] - xmin[2])/(double)(natom*pow(NBLOCK_MAX,2)),1.0/3.0);

  // create a grid with allocated memory
  struct grid partition;
  if(radius > radius0) { partition = construct_grid(natom,atom,radius); }
  else { partition = construct_grid(natom,atom,radius0); }

  // allocate column list in neighbor matrix to store # of nearest neighbors
  neighbor->ncol = neighbor->nrow = natom;
  neighbor->col = (int*)malloc(sizeof(int)*(natom+1));
  neighbor->col[0] = 0;

  // perform work 1 box at a time (1st pass to count neighbors in neighbor->row)
  for(int i=0 ; i<partition.nx[0] ; i++)
  for(int j=0 ; j<partition.nx[1] ; j++)
  for(int k=0 ; k<partition.nx[2] ; k++)
  {
    int box1[3] = { i, j, k };

    // range of neighboring boxes
    int xmin = 0, ymin = 0, zmin = 0, xmax = partition.nx[0]-1, ymax = partition.nx[1]-1, zmax = partition.nx[2]-1;
    if(i > 0) { xmin = i-1; }
    if(j > 0) { ymin = j-1; }
    if(k > 0) { zmin = k-1; }
    if(i < partition.nx[0]-1) { xmax = i+1; }
    if(j < partition.nx[1]-1) { ymax = j+1; }
    if(k < partition.nx[2]-1) { zmax = k+1; }

    // find neighbors for each atom in the box
    int iatom_min = partition.to_atom[grid_index(box1,&partition)];
    int iatom_max = partition.to_atom[grid_index(box1,&partition)+1];
    for(int iatom=iatom_min; iatom<iatom_max ; iatom++)
    {
      neighbor->col[partition.atom_index[iatom]+1] = 0;

      // count the neighbors
      for(int x=xmin ; x<=xmax ; x++)
      for(int y=ymin ; y<=ymax ; y++)
      for(int z=zmin ; z<=zmax ; z++)
      {
        int box2[3] = { x, y, z };
        int jatom_min = partition.to_atom[grid_index(box2,&partition)];
        int jatom_max = partition.to_atom[grid_index(box2,&partition)+1];
        for(int jatom=jatom_min; jatom<jatom_max ; jatom++)
        {
          if(distance(&(atom[3*partition.atom_index[iatom]]),&(atom[3*partition.atom_index[jatom]])) <= radius)
          { neighbor->col[partition.atom_index[iatom]+1]++; }
        }
      }
    }
  }

  // convert from # of neighbors to column offsets
  for(int i=0 ; i<natom ; i++)
  { neighbor->col[i+1] += neighbor->col[i]; }
  neighbor->row = (int*)malloc(sizeof(int)*neighbor->col[neighbor->ncol]);

  // perform work 1 box at a time (2nd pass to assign neighbors in neighbor->col)
  for(int i=0 ; i<partition.nx[0] ; i++)
  for(int j=0 ; j<partition.nx[1] ; j++)
  for(int k=0 ; k<partition.nx[2] ; k++)
  {
    int box1[3] = { i, j, k };

    // range of neighboring boxes
    int xmin = 0, ymin = 0, zmin = 0, xmax = partition.nx[0]-1, ymax = partition.nx[1]-1, zmax = partition.nx[2]-1;
    if(i > 0) { xmin = i-1; }
    if(j > 0) { ymin = j-1; }
    if(k > 0) { zmin = k-1; }
    if(i < partition.nx[0]-1) { xmax = i+1; }
    if(j < partition.nx[1]-1) { ymax = j+1; }
    if(k < partition.nx[2]-1) { zmax = k+1; }

    // find neighbors for each atom in the box
    int iatom_min = partition.to_atom[grid_index(box1,&partition)];
    int iatom_max = partition.to_atom[grid_index(box1,&partition)+1];
    for(int iatom=iatom_min; iatom<iatom_max ; iatom++)
    {
      // store the neighbors
      int ineighbor = 0;
      for(int x=xmin ; x<=xmax ; x++)
      for(int y=ymin ; y<=ymax ; y++)
      for(int z=zmin ; z<=zmax ; z++)
      {
        int box2[3] = { x, y, z };
        int jatom_min = partition.to_atom[grid_index(box2,&partition)];
        int jatom_max = partition.to_atom[grid_index(box2,&partition)+1];
        for(int jatom=jatom_min; jatom<jatom_max ; jatom++)
        {
          if(distance(&(atom[3*partition.atom_index[iatom]]),&(atom[3*partition.atom_index[jatom]])) <= radius)
          { neighbor->row[neighbor->col[partition.atom_index[iatom]]+(ineighbor++)] = partition.atom_index[jatom]; }
        }
      }
    }
  }

  // order the neighbor lists by atomic index
  for(int i=0 ; i<natom ; i++)
  { qsort(&(neighbor->row[neighbor->col[i]]),neighbor->col[i+1]-neighbor->col[i],sizeof(int),neighbor_compare); }

  // free memory used by the grid
  free(partition.to_atom);
  free(partition.atom_index);
}

// Welsh-Powell greedy graph coloring algorithm (ncolor <= maximum vertex degree + 1)
void color_graph(struct pattern *graph, // adjacency matrix of graph, assumed symmetric [1]
                 int *ncolor, // number of colors used to color the graph [1]
                 int **color, // index of the first entry of each color [1]
                 int **vertex_ptr) // index of vertices, sorted by color [1]
{
  // create & sort a list of vertex degrees
  int *degree = (int*)malloc(sizeof(int)*2*graph->ncol);
  for(int i=0 ; i<graph->ncol ; i++)
  {
    degree[2*i] = graph->col[i+1]-graph->col[i]; // degree of vertex
    degree[2*i+1] = i; // index of vertex
  }
  qsort(degree,graph->ncol,2*sizeof(int),neighbor_compare);

  // temporary inverse list of vertex colors
  *ncolor = 1;
  int *vertex_color = (int*)malloc(sizeof(int)*graph->ncol);
  for(int i=0 ; i<graph->ncol ; i++)
  { vertex_color[i] = -1; }

  // color using the inverse list
  int num_uncolored = graph->ncol;
  *ncolor = 0;
  while(num_uncolored > 0)
  {
    // loop over uncolored vertices
    int offset = 0; // pruning offset
    for(int i=0 ; i<num_uncolored ; i++)
    {
      // copy degree list w/ pruning offset
      degree[2*(i-offset)] = degree[2*i];
      degree[2*(i-offset)+1] = degree[2*i+1];

      // check if it is connected to a vertex of the active color
      int collision = 0;
      for(int j=graph->col[degree[2*i+1]] ; j<graph->col[degree[2*i+1]+1] ; j++)
      { if(vertex_color[graph->row[j]] == *ncolor) { collision = 1; } }

      // if it isn't connected, color & offset for pruning
      if(collision == 0)
      {
        vertex_color[degree[2*i+1]] = *ncolor;
        offset++;
      }
    }
    num_uncolored -= offset;
    (*ncolor)++;
  }

  // allocate memory for the coloring
  *color = (int*)malloc(sizeof(int)*(*ncolor+1));
  *vertex_ptr = (int*)malloc(sizeof(int)*graph->ncol);

  // count the number of vertices per color & properly offset
  for(int i=0 ; i<=*ncolor ; i++) { (*color)[i] = 0; }
  for(int i=0 ; i<graph->ncol ; i++) { ((*color)[vertex_color[i]+1])++; }
  for(int i=0 ; i<*ncolor ; i++) { (*color)[i+1] += (*color)[i]; }

  // invert the vertex color list
  for(int i=0 ; i<graph->ncol ; i++)
  { (*vertex_ptr)[((*color)[vertex_color[i]])++] = i; }

  // re-count the number of vertices per color & properly offset
  for(int i=0 ; i<=*ncolor ; i++) { (*color)[i] = 0; }
  for(int i=0 ; i<graph->ncol ; i++) { ((*color)[vertex_color[i]+1])++; }
  for(int i=0 ; i<*ncolor ; i++) { (*color)[i+1] += (*color)[i]; }

  // deallocate temporary memory
  free(vertex_color);
  free(degree);
}

// comparison function for sorting lattice vector lists using the C qsort function in stdlib.h
// RETURN: 1 if a goes after b, -1 if a goes before b, 0 if they are equal
int latvec_compare(const void *a, const void *b)
{
  if( ((unsigned int*)a)[0] > ((unsigned int*)b)[0] ) return 1;
  if( ((unsigned int*)a)[0] < ((unsigned int*)b)[0] ) return -1;
  if( ((unsigned int*)a)[1] > ((unsigned int*)b)[1] ) return 1;
  if( ((unsigned int*)a)[1] < ((unsigned int*)b)[1] ) return -1;
  if( ((unsigned int*)a)[2] > ((unsigned int*)b)[2] ) return 1;
  if( ((unsigned int*)a)[2] < ((unsigned int*)b)[2] ) return -1;
  return 0;
}

// calculate the volume of a unit cell whose lattice vectors are defined by 4 atomic coordinates (1st is central atom)
double cell_volume(double *atom) // list of atomic coordinates [12]
{
  // calculate lattice vectors
  double latvec[3][3];
  for(int i=0 ; i<3 ; i++)
  for(int j=0 ; j<3 ; j++)
  { latvec[i][j] = atom[j+(1+i)*3] - atom[j]; }

  return fabs(latvec[0][0]*(latvec[1][1]*latvec[2][2] - latvec[1][2]*latvec[2][1])
            + latvec[0][1]*(latvec[1][2]*latvec[2][0] - latvec[1][0]*latvec[2][2])
            + latvec[0][2]*(latvec[1][0]*latvec[2][1] - latvec[1][1]*latvec[2][0]));
}

// construct a list of lattice vectors within a localization radius
// RETURN: number of lattice vectors (nlatvec)
int latvec_list(double local_radius, // localization radius for truncation
                int **list, // (allocated) list of lattice vectors on output [1][3*nlatvec]
                double **atom) // (allocated) equivalent list of atomic coordinates [1][3*nlatvec]
{
  // store lattice vectors
  double latvec[9];
  for(int i=0 ; i<3 ; i++)
  for(int j=0 ; j<3 ; j++)
  { latvec[j+i*3] = (*atom)[j+(1+i)*3] - (*atom)[j]; }
  free(*atom);

  // identify lattice vector bounds
  int max_index[3];
  for(int i=0 ; i<3 ; i++)
  { max_index[i] = MAX(max_index[i],ceil(local_radius/sqrt(pow(latvec[3*i],2)+pow(latvec[3*i+1],2)+pow(latvec[3*i+2],2)))); }

  // count the number of active lattice vectors
  int nlist = 0;
  double atom0[3];
  for(int i=-max_index[0] ; i<=max_index[0] ; i++)
  for(int j=-max_index[1] ; j<=max_index[1] ; j++)
  for(int k=-max_index[2] ; k<=max_index[2] ; k++)
  {
    for(int l=0 ; l<3 ; l++)
    { atom0[l] = i*latvec[l] + j*latvec[l+3] + k*latvec[l+6]; }
    if(sqrt(pow(atom0[0],2)+pow(atom0[1],2)+pow(atom0[2],2)) <= local_radius) { nlist++; }
  }

  // assign the active lattice vectors
  *list = (int*)malloc(sizeof(int)*3*nlist);
  nlist = 0;
  for(int i=-max_index[0] ; i<=max_index[0] ; i++)
  for(int j=-max_index[1] ; j<=max_index[1] ; j++)
  for(int k=-max_index[2] ; k<=max_index[2] ; k++)
  {
    for(int l=0 ; l<3 ; l++)
    { atom0[l] = i*latvec[l] + j*latvec[l+3] + k*latvec[l+6]; }
    if(sqrt(pow(atom0[0],2)+pow(atom0[1],2)+pow(atom0[2],2)) <= local_radius)
    { (*list)[3*nlist] = i; (*list)[1+3*nlist] = j; (*list)[2+3*nlist] = k; nlist++; }
  }

  // sort the active lattice vectors
  qsort(*list,nlist,3*sizeof(int),latvec_compare);

  // construct the sorted atom list
  *atom = (double*)malloc(sizeof(double)*3*nlist);
  for(int i=0 ; i<nlist ; i++)
  {
    for(int j=0 ; j<3 ; j++)
    { (*atom)[j+3*i] = (*list)[3*i]*latvec[j] + (*list)[1+3*i]*latvec[j+3] + (*list)[2+3*i]*latvec[j+6]; }
  }

  return nlist;
}

//=====================================//
// 4. BLOCK VECTOR & MATRIX OPERATIONS //
//=====================================//

// zero the entries of a block vector
void zero_vec(int nblock, // block size
              int nvec, // dimension of vector (# of blocks)
              double *vec) // vector elements [nblock*nblock*nvec]
{
  int ndata = nblock*nblock*nvec;

#pragma omp parallel for
  for(int i=0 ; i<ndata ; i++)
  { vec[i] = 0.0; }
}

// zero the entries of a block-sparse matrix
void zero_mat(int nblock, // block size
              struct pattern *sparsity, // contains the sparsity pattern & dimensions of the matrix [1]
              double **mat) // matrix elements of the sparse matrix [sparsity->col[sparsity->ncol]][nblock*nblock]
{
#pragma omp parallel for collapse(2)
  for(int i=0 ; i<sparsity->col[sparsity->ncol] ; i++)
  for(int j=0 ; j<nblock*nblock ; j++)
  { mat[i][j] = 0.0; }
}

// copy a block vector
void copy_vec(int nblock, // block size
              int nvec, // dimension of vectors (# of blocks)
              double *src, // source vector [nblock*nblock*nvec]
              double *dst) // destination vector [nblock*nblock*nvec]
{
  int ndata = nblock*nblock*nvec;

#pragma omp parallel for
  for(int i=0 ; i<ndata ; i++)
  { dst[i] = src[i]; }
}

// copy a block-sparse matrix
void copy_mat(int nblock, // block size
              struct pattern *sparsity, // contains the sparsity pattern & dimensions of the matrix [1]
              double **src, // matrix elements of the source sparse matrix [sparsity->col[sparsity->ncol]][nblock*nblock]
              double **dst) // matrix elements of the target sparse matrix [sparsity->col[sparsity->ncol]][nblock*nblock]
{
#pragma omp parallel for collapse(2)
  for(int i=0 ; i<sparsity->col[sparsity->ncol] ; i++)
  for(int j=0 ; j<nblock*nblock ; j++)
  { dst[i][j] = src[i][j]; }
}

// rescale a block vector: dst = alpha*dst
// NOTE: each column has a different weight
void scale_vec(int nblock, // block size
               int nvec, // dimension of vectors (# of blocks)
               double *alpha, // scale factors [nblock]
               double *vec) // vector elements [nblock*nblock*nvec]
{
#pragma omp parallel for collapse(3)
  for(int i=0 ; i<nvec ; i++)
  for(int j=0 ; j<nblock ; j++)
  for(int k=0 ; k<nblock ; k++)
  { vec[k+(j+i*nblock)*nblock] *= alpha[j]; }
}

// rescale a block-sparse matrix: dst = alpha*dst
void scale_mat(int nblock, // block size
               struct pattern *sparsity, // contains the sparsity pattern & dimensions of the matrix [1]
               double alpha,
               double **mat) // matrix elements of the sparse matrix [sparsity->col[sparsity->ncol]][nblock*nblock]
{
#pragma omp parallel for collapse(2)
  for(int i=0 ; i<sparsity->col[sparsity->ncol] ; i++)
  for(int j=0 ; j<nblock*nblock ; j++)
  { mat[i][j] *= alpha; }
}

// add two block vectors in BLAS ?AXPY form: dst = alpha*src + dst
// NOTE: each column has a different weight
void add_vec(int nblock, // block size
             int nvec, // dimension of vectors (# of blocks)
             double *alpha, // scale factors on src [nblock]
             double *src, // source vector [nblock*nblock*nvec]
             double *dst) // destination vector [nblock*nblock*nvec]
{
#pragma omp parallel for collapse(3)
  for(int i=0 ; i<nvec ; i++)
  for(int j=0 ; j<nblock ; j++)
  for(int k=0 ; k<nblock ; k++)
  { dst[k+(j+i*nblock)*nblock] += alpha[j]*src[k+(j+i*nblock)*nblock]; }
}

// add two block-sparse matrices in BLAS ?AXPY form: dst = alpha*src + dst
void add_mat(int nblock, // block size
             struct pattern *sparsity, // contains the sparsity pattern & dimensions of the matrix [1]
             double alpha,
             double **src, // matrix elements of the source sparse matrix [sparsity->col[sparsity->ncol]][nblock*nblock]
             double **dst) // matrix elements of the target sparse matrix [sparsity->col[sparsity->ncol]][nblock*nblock]
{
#pragma omp parallel for collapse(2)
  for(int i=0 ; i<sparsity->col[sparsity->ncol] ; i++)
  for(int j=0 ; j<nblock*nblock ; j++)
  { dst[i][j] += alpha*src[i][j]; }
}

// inner products between columns of two block vectors
void dot_vec(int nblock, // block size
             int nvec, // dimension of vectors (# of blocks)
             double *vec1, // source vector [nblock*nblock*nvec]
             double *vec2, // destination vector [nblock*nblock*nvec]
             double *dot) // accumulated dot products on output [nblock]
{
  for(int i=0 ; i<nblock ; i++) { dot[i] = 0.0; }

#pragma omp parallel
// begin openmp block
{
  double local_dot[NBLOCK_MAX];
  for(int i=0 ; i<nblock ; i++) { local_dot[i] = 0.0; }

#pragma omp for collapse(3)
  for(int i=0 ; i<nvec ; i++)
  for(int j=0 ; j<nblock ; j++)
  for(int k=0 ; k<nblock ; k++)
  { local_dot[j] += vec1[k+(j+i*nblock)*nblock]*vec2[k+(j+i*nblock)*nblock]; }

  for(int i=0 ; i<nblock ; i++)
  {
#pragma omp atomic
    dot[i] += local_dot[i];
  }
}
// end openmp block
}

// inner product (trace) between two sparse matrices of the same pattern
// RETURN: value of the inner product
double dot_mat(int nblock, // block size
               struct pattern *sparsity, // contains the sparsity pattern & dimensions of the matrix [1]
               double **mat1, // matrix elements of the source sparse matrix [sparsity->col[sparsity->ncol]][nblock*nblock]
               double **mat2) // matrix elements of the target sparse matrix [sparsity->col[sparsity->ncol]][nblock*nblock]
{
  double ans = 0.0;

#pragma omp parallel for collapse(2) reduction(+:ans)
  for(int i=0 ; i<sparsity->col[sparsity->ncol] ; i++)
  for(int j=0 ; j<nblock*nblock ; j++)
  { ans += mat1[i][j]*mat2[i][j]; }

  return ans;
}

// block-sparse matrix-vector multiplication in BLAS ?GEMV form, vec_out = alpha*mat^T*vec_in + beta*vec_out
void mat_vec(int nblock, // matrix & vector block size
             struct pattern *sparsity, // contains the sparsity pattern & dimensions of the matrix [1]
             double alpha, // scale factor on vec_in*mat
             double beta, // scale factor on vec_out
             double **mat, // matrix elements of the sparse matrix [sparsity->col[sparsity->ncol]][nblock*nblock]
             double *vec_in, // input block vector [sparsity->nrow*nblock*nblock]
             double *vec_out) // output block vector [sparsity->ncol*nblock*nblock]
{
  // loop over block entries of vec_out
#pragma omp parallel for
  for(int i=0 ; i<sparsity->ncol ; i++)
  {
    // rescale entries of vec_out by beta
    for(int j=0 ; j<nblock*nblock ; j++) { vec_out[j+i*nblock*nblock] *= beta; }

    // loop over nonzero blocks in the column
    for(int j=sparsity->col[i] ; j<sparsity->col[i+1] ; j++)
    {
      // accumulate blocks of the solution (BLAS call)
      char transa = 'T', transb = 'N';
      double one = 1.0;
      MKL_INT n = nblock;
      dgemm(&transa,&transb,&n,&n,&n,&alpha,mat[j],&n,&(vec_in[sparsity->row[j]*nblock*nblock]),
            &n,&one,&(vec_out[i*nblock*nblock]),&n);
    }
  }
}

// complex wrapper for vec_out = (mat_base + shift*mat_shift)^T*vec_in
void zmat_zvec(int nblock, // matrix & vector block size
               struct pattern *sparsity, // contains the sparsity pattern & dimensions of the matrix [1]
               double complex shift, // complex shift applied to mat2
               double **mat_base, // base matrix in mat-vec operation [sparsity->col[sparsity->ncol]][nblock*nblock]
               double **mat_shift, // shifted matrix in mat-vec operation [sparsity->col[sparsity->ncol]][nblock*nblock]
               double *vec_in, // input block vector, contiguous real & imaginary parts [2*sparsity->nrow*nblock*nblock]
               double *vec_out) // output block vector, contiguous real & imaginary parts [2*sparsity->ncol*nblock*nblock]
{
  int nvec_in = sparsity->nrow*nblock*nblock;
  int nvec_out = sparsity->ncol*nblock*nblock;

  // vec_out = mat_shift^T*vec_in
  mat_vec(nblock,sparsity,1.0,0.0,mat_shift,vec_in,vec_out);
  mat_vec(nblock,sparsity,1.0,0.0,mat_shift,&(vec_in[nvec_in]),&(vec_out[nvec_out]));

  // vec_out <- shift*vec_out
#pragma omp parallel for
  for(int i=0 ; i<nvec_out ; i++)
  {
    double complex work = shift*(vec_out[i] + I*vec_out[i+nvec_out]);
    vec_out[i] = creal(work);
    vec_out[i+nvec_out] = cimag(work);
  }

  // include the base part of the matrix: vec_out <- vec_out + mat_base^T*vec_in
  mat_vec(nblock,sparsity,1.0,1.0,mat_base,vec_in,vec_out);
  mat_vec(nblock,sparsity,1.0,1.0,mat_base,&(vec_in[nvec_in]),&(vec_out[nvec_out]));
}

// add a block vector to the column of a block-sparse matrix within its sparsity pattern
void add_col(int nblock, // matrix & vector block size
             struct pattern *sparsity, // contains the sparsity pattern & dimensions of the matrix [1]
             int icol, // column index to update
             double wt, // weight to add the vector with
             double *vec, // dense block vector to add [sparsity->nrow*nblock*nblock]
             double **mat) // matrix elements of the sparse matrix [sparsity->col[sparsity->ncol]][nblock*nblock]
{
#pragma omp parallel for collapse(2)
  for(int i=sparsity->col[icol] ; i<sparsity->col[icol+1] ; i++)
  {
    for(int j=0 ; j<nblock*nblock ; j++)
    { mat[i][j] += wt*vec[j+sparsity->row[i]*nblock*nblock]; }
  }
}

// comparison function for finding rows using the C bsearch function in stdlib.h
// RETURN: 1 if a goes after b, -1 if a goes before b, 0 if they are equal
int row_compare(const void *a, const void *b)
{
  if( ((int*)a)[0] > ((int*)b)[0] ) return 1;
  if( ((int*)a)[0] < ((int*)b)[0] ) return -1;
  return 0;
}

// add a block vector to the row of a block-sparse matrix within its sparsity pattern
// NOTE: the loop over all columns could be restricted with the promise of a symmetric sparsity pattern
void add_row(int nblock, // matrix & vector block size
             struct pattern *sparsity, // contains the sparsity pattern & dimensions of the matrix [1]
             int irow, // row index to update
             double wt, // weight to add the vector with
             double *vec, // dense block vector to add [sparsity->ncol*nblock*nblock]
             double **mat) // matrix elements of the sparse matrix [sparsity->col[sparsity->ncol]][nblock*nblock]
{
#pragma omp parallel for
  for(int i=0 ; i<sparsity->ncol ; i++)
  {
    // search for the row index inside the column
    int nnz_col = sparsity->col[i+1] - sparsity->col[i];
    int *row_ptr = (int*)bsearch(&irow,&(sparsity->row[sparsity->col[i]]),nnz_col,sizeof(int),row_compare);

    // add the matrix element block
    if(row_ptr != NULL)
    {
      int ielem = (int)(row_ptr - sparsity->row); // pointer arithmetic
      for(int j=0 ; j<nblock ; j++)
      for(int k=0 ; k<nblock ; k++)
      { mat[ielem][j+k*nblock] += wt*vec[k+(j+i*nblock)*nblock]; }
    }
  }
}

//=====================================//
// 5. MATRIX CONSTRUCTION & CONVERSION //
//=====================================//

// block-sparse construction of hamiltonian & overlap (each block is a 9-by-9 atomic subspace)
void tb_matrix(int natom, // number of atoms
               double *atom, // atomic coordinates [3*natom]
               struct nrl_tb *param, // tight-binding parameters [1]
               struct pattern *sparsity, // contains the sparsity pattern & dimensions of the matrix [1]
               double **hamiltonian, // Hamiltonian matrix elements [sparsity->col[sparsity->ncol]][nblock*nblock]
               double **overlap) // overlap matrix elements [sparsity->col[sparsity->ncol]][nblock*nblock]
{
  // fill in the sparse matrix
#pragma omp parallel for
  for(int i=0 ; i<sparsity->ncol ; i++)
  {
    for(int j=sparsity->col[i] ; j<sparsity->col[i+1] ; j++)
    {      
      // calculate block matrix elements
      if(i == sparsity->row[j])
      {
        tb_diagonal(i,natom,atom,sparsity->col[i+1]-sparsity->col[i],&(sparsity->row[sparsity->col[i]]),
                    param,hamiltonian[j],overlap[j]);
      }
      else
      { tb_offdiagonal(i,sparsity->row[j],natom,atom,param,hamiltonian[j],overlap[j]); }
    }
  }
}

// block-sparse to dense matrix embedding
// NOTE: some arrays here can be larger than the maximum value of "int" and need "size_t" indices
void embed_mat(int nblock, // matrix block size
               struct pattern *sparsity, // contains the sparsity pattern & dimensions of the matrix [1]
               double **smat, // matrix elements of the sparse matrix [sparsity->col[sparsity->ncol]][nblock*nblock]
               double *dmat) // dense matrix [sparsity->nrow*sparsity->ncol*nblock*nblock]
{
  // fill dense matrix with zeros
  size_t nrow = sparsity->nrow*nblock, ndata = nrow*sparsity->ncol*nblock;
#pragma omp parallel for
  for(size_t i=0 ; i<ndata ; i++)
  { dmat[i] = 0.0; }

  // block-by-block transfer of block-sparse matrix
#pragma omp parallel for
  for(size_t i=0 ; i<sparsity->ncol ; i++)
  for(size_t j=sparsity->col[i] ; j<sparsity->col[i+1] ; j++)
  {
    // copy a block
    for(size_t k=0 ; k<nblock ; k++)
    for(size_t l=0 ; l<nblock ; l++)
    { dmat[(l+sparsity->row[j]*nblock)+(k+i*nblock)*nrow] = smat[j][l+k*nblock]; }
  }
}

// block-sparse matrix restriction of a block vector outer product (accumulate solution)
void restrict_outvec(int nblock, // matrix & vector block size
                     struct pattern *sparsity, // contains the sparsity pattern & dimensions of the matrix [1]
                     double *leftvec, // left vector [sparsity->nrow*nblock*nblock]
                     double *rightvec, // right vector [sparsity->ncol*nblock*nblock]
                     double **mat) // matrix elements of the sparse matrix [sparsity->col[sparsity->ncol]][nblock*nblock]
{
  // loop over nonzero blocks of the sparsity matrix
#pragma omp parallel for
  for(int i=0 ; i<sparsity->ncol ; i++)
  for(int j=sparsity->col[i] ; j<sparsity->col[i+1] ; j++)
  {
    // calculate block outer product between leftmat & rightmat (BLAS call)
    char transa = 'N', transb = 'T';
    double one = 1.0;
    MKL_INT n = nblock;
    dgemm(&transa,&transb,&n,&n,&n,&one,&(leftvec[sparsity->row[j]*nblock*nblock]),&n,
          &(rightvec[i*nblock*nblock]),&n,&one,mat[j],&n);
  }
}

// block-sparse matrix restriction of a matrix outer product (accumulate solution)
void restrict_outmat(int nblock, // matrix & vector block size
                     int nouter, // inner matrix dimension between left & right matrices
                     struct pattern *sparsity, // contains the sparsity pattern & dimensions of the matrix [1]
                     double *leftmat, // left matrix [sparsity->nrow*nblock*nouter]
                     double *rightmat, // right matrix [sparsity->ncol*nblock*nouter]
                     double **mat) // matrix elements of the sparse matrix [sparsity->col[sparsity->ncol]][nblock*nblock]
{
  // loop over nonzero blocks of bmat
  for(int i=0 ; i<sparsity->ncol ; i++)
  for(int j=sparsity->col[i] ; j<sparsity->col[i+1] ; j++)
  {
    // calculate block outer product between leftmat & rightmat (BLAS call)
    char transa = 'N', transb = 'T';
    double one = 1.0;
    MKL_INT n = nblock, m = nouter, lda = sparsity->nrow*nblock, ldb = sparsity->ncol*nblock;
    dgemm(&transa,&transb,&n,&n,&m,&one,&(leftmat[sparsity->row[j]*nblock]),&lda,
          &(rightmat[i*nblock]),&ldb,&one,mat[j],&n);
  }
}

// conversion between block-sparse & ordered-pair sparse matrix formats
void block2sparse(int nblock, // matrix block size
                  struct pattern *b_sparsity, // contains the sparsity pattern & dimensions of the block-sparse matrix [1]
                  struct pattern *s_sparsity, // contains the sparsity pattern & dimensions of the sparse matrix [1]
                  double **bmat1, // first input block matrix [sparsity_in->col[sparsity_in->ncol]][nblock*nblock]
                  double **bmat2, // second input block matrix [sparsity_in->col[sparsity_in->ncol]][nblock*nblock]
                  double *smat12) // output sparse matrix [2*nblock*nblock*sparsity_in->col[sparsity_in->ncol]]
{
  // allocate memory for the ordered-pair sparsity pattern
  s_sparsity->ncol = b_sparsity->ncol*nblock;
  s_sparsity->nrow = b_sparsity->nrow*nblock;
  s_sparsity->col = (int*)malloc(sizeof(int)*(s_sparsity->ncol+1));
  s_sparsity->row = (int*)malloc(sizeof(int)*nblock*nblock*b_sparsity->col[b_sparsity->ncol]);

  // loop over elements of the block-sparse matrix
#pragma omp parallel for collapse(2)
  for(int i=0 ; i<b_sparsity->ncol ; i++)
  for(int j=0 ; j<nblock ; j++)
  {
    s_sparsity->col[j+i*nblock] = b_sparsity->col[i]*nblock*nblock + j*(b_sparsity->col[i+1] - b_sparsity->col[i])*nblock;
    for(int k=0 ; k<b_sparsity->col[i+1]-b_sparsity->col[i] ; k++)
    for(int l=0 ; l<nblock ; l++)
    {
      s_sparsity->row[l+k*nblock+s_sparsity->col[j+i*nblock]] = l + b_sparsity->row[b_sparsity->col[i]+k]*nblock;
      smat12[2*(l+k*nblock+s_sparsity->col[j+i*nblock])] = bmat1[b_sparsity->col[i]+k][l+j*nblock];
      smat12[2*(l+k*nblock+s_sparsity->col[j+i*nblock])+1] = bmat2[b_sparsity->col[i]+k][l+j*nblock];
    }
  }
  s_sparsity->col[s_sparsity->ncol] = b_sparsity->col[b_sparsity->ncol]*nblock*nblock;
}

// conversion between ordered-pair sparse & block-sparse matrix formats (reverse of previous operation)
void sparse2block(int nblock, // matrix block size
                  struct pattern *s_sparsity, // contains the sparsity pattern & dimensions of the sparse matrix [1]
                  struct pattern *b_sparsity, // contains the sparsity pattern & dimensions of the block-sparse matrix [1]
                  double *smat12, // input sparse matrix [2*nblock*nblock*sparsity_out->col[sparsity_out->ncol]]
                  double **bmat1, // first output block matrix [sparsity_out->col[sparsity_out->ncol]][nblock*nblock]
                  double **bmat2) // second output block matrix [sparsity_out->col[sparsity_out->ncol]][nblock*nblock]
{
  // loop over elements of the block-sparse matrix
#pragma omp parallel for collapse(2)
  for(int i=0 ; i<b_sparsity->ncol ; i++)
  for(int j=0 ; j<nblock ; j++)
  for(int k=0 ; k<b_sparsity->col[i+1]-b_sparsity->col[i] ; k++)
  for(int l=0 ; l<nblock ; l++)
  {
    bmat1[b_sparsity->col[i]+k][l+j*nblock] = smat12[2*(l+k*nblock+s_sparsity->col[j+i*nblock])];
    bmat2[b_sparsity->col[i]+k][l+j*nblock] = smat12[2*(l+k*nblock+s_sparsity->col[j+i*nblock])+1];
  }

  // free memory for the ordered-pair sparsity pattern
  free_pattern(s_sparsity);
}

// add src to dst with different sparsity patterns: dst = alpha*src + dst
void sparse2sparse(int nblock, // matrix block size
                   struct pattern *sparsity_in, // contains the sparsity pattern of the input matrix [1]
                   struct pattern *sparsity_out, // contains the sparsity pattern of the output matrix [1]
                   double alpha, // coefficient in matrix addition
                   double **src, // input matrix [sparsity_in->col[sparsity_in->ncol]][nblock*nblock]
                   double **dst) // output matrix [sparsity_out->col[sparsity_out->ncol]][nblock*nblock]
{
  if(sparsity_in->nrow != sparsity_out->nrow || sparsity_in->ncol != sparsity_out->ncol)
  {
    printf("ERROR: sparse-to-sparse addition of matrices w/ incompatible dimension\n");
    MPI_Abort(MPI_COMM_WORLD,0);
  }

  int is_subset = 1;
  for(int i=0 ; i<sparsity_in->ncol ; i++)
  {
    int k = sparsity_out->col[i];
    for(int j=sparsity_in->col[i] ; j<sparsity_in->col[i+1] ; j++)
    {
      while(sparsity_out->row[k] < sparsity_in->row[j] && k<sparsity_out->col[i+1])
      { k++; }

      if(sparsity_out->row[k] == sparsity_in->row[j] && k<sparsity_out->col[i+1])
      { add_vec(1,nblock*nblock,&alpha,src[j],dst[k++]); }
    }
  }
}

// distributes a sparsity pattern from mpirank == 0 to all the other MPI processes by splitting uniformly over columns
// RETURN: global number of nonzero matrix elements, g_sparsity->col[g_sparsity->ncol]
int split_pattern(int mpirank, // rank of this MPI process
                  int mpisize, // total number of MPI processes
                  struct pattern *g_sparsity, // contains the sparsity pattern & dimensions of the global matrix [1]
                  struct pattern *l_sparsity) // contains the sparsity pattern & dimensions of the local matrix [1]
{
  int nnz;
  if(mpirank == 0)
  {
    nnz = g_sparsity->col[g_sparsity->ncol];
    MPI_Bcast(&(g_sparsity->nrow),1,MPI_INT,0,MPI_COMM_WORLD);
    
    for(int i=0 ; i<mpisize ; i++)
    {
      int ncol_local = g_sparsity->ncol/mpisize, icol_head = i*ncol_local;
      if(i == mpisize-1) { ncol_local = g_sparsity->ncol - (mpisize-1)*ncol_local; } // last MPI process gets more columns
      int nnz_local = g_sparsity->col[icol_head+ncol_local] - g_sparsity->col[icol_head];
      int irow_head = g_sparsity->col[icol_head];

      if(i == 0)
      {
        l_sparsity->ncol = ncol_local;
        l_sparsity->nrow = g_sparsity->nrow;
        l_sparsity->col = (int*)malloc(sizeof(int)*(l_sparsity->ncol+1));
        l_sparsity->row = (int*)malloc(sizeof(int)*nnz_local);

        for(int j=0 ; j<=l_sparsity->ncol ; j++)
        { l_sparsity->col[j] = g_sparsity->col[icol_head+j] - g_sparsity->col[icol_head]; }

        for(int j=0 ; j<nnz_local ; j++)
        { l_sparsity->row[j] = g_sparsity->row[irow_head+j]; }
      }
      else
      {
        MPI_Send(&ncol_local,1,MPI_INT,i,0,MPI_COMM_WORLD);
        MPI_Send(&nnz_local,1,MPI_INT,i,0,MPI_COMM_WORLD);

        MPI_Send(&(g_sparsity->col[icol_head]),ncol_local+1,MPI_INT,i,0,MPI_COMM_WORLD);
        MPI_Send(&(g_sparsity->row[irow_head]),nnz_local,MPI_INT,i,0,MPI_COMM_WORLD);
      }
    }
  }
  else
  {
    int nnz_local;
    MPI_Bcast(&(l_sparsity->nrow),1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Recv(&(l_sparsity->ncol),1,MPI_INT,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    MPI_Recv(&nnz_local,1,MPI_INT,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

    l_sparsity->col = (int*)malloc(sizeof(int)*(l_sparsity->ncol+1));
    l_sparsity->row = (int*)malloc(sizeof(int)*nnz_local);

    MPI_Recv(l_sparsity->col,l_sparsity->ncol+1,MPI_INT,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    MPI_Recv(l_sparsity->row,nnz_local,MPI_INT,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

    // adjust the offsets in column indices
    int offset = l_sparsity->col[0];
    for(int i=0 ; i<=l_sparsity->ncol ; i++)
    { l_sparsity->col[i] -= offset; }
  }

  // broadcast global number of nonzeros
  MPI_Bcast(&nnz,1,MPI_INT,0,MPI_COMM_WORLD);
  return nnz;
}

// build a sparsity pattern in the local restriction of one column of a localization pattern
void localize_pattern(int nlocal, // number of local rows / columns
                      int *local, // ordered list of local rows / columns [nlocal]
                      struct pattern *sparsity, // sparsity pattern that is being restricted [1]
                      struct pattern *local_sparsity) // localized sparsity pattern [1]
{
  local_sparsity->ncol = local_sparsity->nrow = nlocal;
  local_sparsity->col = (int*)malloc(sizeof(int)*(nlocal+1));

  // count number of non-zero entries
  int local_inz = 0;
  local_sparsity->col[0] = 0;
  for(int i=0 ; i<nlocal ; i++)
  {
    int ilocal = 0;
    for(int j=sparsity->col[local[i]] ; j<sparsity->col[local[i]+1] ; j++)
    {
      while(local[ilocal] < sparsity->row[j] && ilocal < nlocal) { ilocal++; }
      if(ilocal == nlocal) { break; }
      if(local[ilocal] == sparsity->row[j]) { local_inz++; }
    }
    local_sparsity->col[i+1] = local_inz;
  }
  local_sparsity->row = (int*)malloc(sizeof(int)*local_sparsity->col[local_sparsity->ncol]);

  // fill in non-zero entries
  local_inz = 0;
  for(int i=0 ; i<nlocal ; i++)
  {
    int ilocal = 0;
    for(int j=sparsity->col[local[i]] ; j<sparsity->col[local[i]+1] ; j++)
    {
      while(local[ilocal] < sparsity->row[j] && ilocal < nlocal) { ilocal++; }
      if(ilocal == nlocal) { break; }
      if(local[ilocal] == sparsity->row[j]) { local_sparsity->row[local_inz++] = ilocal; }
    }
  }
}

// build a localized block-sparse matrix that points to the data of the original block-sparse matrix
void localize_mat(int nlocal, // number of local rows / columns
                  int *local, // ordered list of local rows / columns [nlocal]
                  struct pattern *sparsity, // original sparsity pattern [1]
                  double **mat, // original sparse matrix [sparsity->ncol[sparsity[col]][*]
                  double **local_mat) // localized sparse matrix [sparsity->ncol[sparsity[col]][*]
{
  int local_inz = 0;
  for(int i=0 ; i<nlocal ; i++)
  {
    int ilocal = 0;
    for(int j=sparsity->col[local[i]] ; j<sparsity->col[local[i]+1] ; j++)
    {
      while(local[ilocal] < sparsity->row[j] && ilocal < nlocal) { ilocal++; }
      if(ilocal == nlocal) { break; }
      if(local[ilocal] == sparsity->row[j]) { local_mat[local_inz++] = mat[j]; }
    }
  }
}

// allocate memory for a periodic symmetric sparse matrix where all elements are pointers to the first row & column
void crystal_malloc(int nblock, // size of matrix blocks
                    struct pattern *sparsity, // sparsity pattern of the matrix [1]
                    int *latvec, // ordered list of lattice vectors [3*sparsity->ncol]
                    double **mat) // matrix element pointers for the sparse matrix [sparsity->col[sparsity->ncol]][1]
{
  // allocate the non-redundant memory
  int nnz = sparsity->col[1], elem00 = 1;
  if(sparsity->row[0] != 0) { elem00 = 0; }
  mat[0] = (double*)malloc(sizeof(double)*(2*nnz-elem00)*nblock*nblock);
  for(int i=1 ; i<nnz ; i++)
  { mat[i] = &(mat[i-1][nblock*nblock]); }

  // assign memory to the first row
  mat[sparsity->col[sparsity->row[elem00]]] = &(mat[nnz-1][nblock*nblock]);
  for(int i=1+elem00 ; i<nnz ; i++)
  { mat[sparsity->col[sparsity->row[i]]] = &(mat[sparsity->col[sparsity->row[i-1]]][nblock*nblock]); }

  // loop over columns, excluding the first
  for(int i=1 ; i<sparsity->ncol ; i++)
  {
    int latvec0[3];
    for(int j=0 ; j<3 ; j++) { latvec0[j] = latvec[j+3*i]; }

    // loop over matrix elements in the first column
    for(int j=0 ; j<sparsity->col[1] ; j++)
    {
      // shift lattice vector of matrix element & search for it in the list
      int latvec1[3];
      for(int k=0 ; k<3 ; k++) { latvec1[k] = latvec0[k] + latvec[k+3*sparsity->row[j]]; }
      int *latvec_ptr = (int*)bsearch(latvec1,latvec,sparsity->ncol,3*sizeof(int),latvec_compare);

      // search for index in the sparsity pattern & assign the pointer
      if(latvec_ptr != NULL)
      {
        int irow = (int)(latvec_ptr - latvec)/3;
        int nnz_col = sparsity->col[i+1] - sparsity->col[i];
        int *row_ptr = (int*)bsearch(&irow,&(sparsity->row[sparsity->col[i]]),nnz_col,sizeof(int),row_compare);

        if(irow >= i && row_ptr != NULL)
        {
          int ielem = (int)(row_ptr - sparsity->row);
          mat[ielem] = mat[j];
        }
      }
      
      // search for conjugate matrix element
      for(int k=0 ; k<3 ; k++) { latvec1[k] = latvec0[k] - latvec[k+3*sparsity->row[j]]; }
      latvec_ptr = (int*)bsearch(latvec1,latvec,sparsity->ncol,3*sizeof(int),latvec_compare);

      // search for index in the sparsity pattern & assign the pointer
      if(latvec_ptr != NULL)
      {
        int irow = (int)(latvec_ptr - latvec)/3;
        int nnz_col = sparsity->col[i+1] - sparsity->col[i];
        int *row_ptr = (int*)bsearch(&irow,&(sparsity->row[sparsity->col[i]]),nnz_col,sizeof(int),row_compare);

        if(irow < i && irow > 0 && row_ptr != NULL)
        {
          int ielem = (int)(row_ptr - sparsity->row);
          mat[ielem] = mat[sparsity->col[sparsity->row[j]]];
        }
      }
    }
  }
}

//===================================//
// 6. PSEUDORANDOM NUMBER GENERATION //
//===================================//

// C rand() & srand() are not guaranteed to be reproducible
// & many of their default implementations are considered bad
// Here we define a simple pseudorandom number generator 

// PRNG that passes BigCrush empirical randomness tests, xorshift1024star() from [http://en.wikipedia.org/wiki/Xorshift]
uint64_t random64(const uint32_t seed) // 0 for normal use, nonzero seed value to reseed
{
  static uint64_t s[16];
  static uint8_t p;

  // seed & "warm up" the PRNG
  if(seed != 0)
  {
    p = 0;
    uint32_t i;
    for(i=0 ; i<16 ; i++) s[i] = seed + i;
    for(i=0 ; i<16384 ; i++) random64(0);
  }

  uint64_t s0 = s[p];
  p = (p + 1) & 15;
  uint64_t s1 = s[p];

  s1 ^= s1 << 31; // a
  s1 ^= s1 >> 11; // b
  s0 ^= s0 >> 30; // c
  s[p] = s0 ^ s1;

  return s[p] * 1181783497276652981ULL;
}

// pseudorandom uniform distribution over (0,1]
double random_uniform()
{
  // reduce from 64 random bits to 53 random bits that span the representable unpadded integers using a double
  return (double)((random64(0) >> 11) + 1)/9007199254740992.0;
}

// pseudorandom block-sparse complex rotor vector
void random_vec(int nblock, // size of blocks
                int nvec, // number of blocks
                int nnz, // number of nonzero rotors
                int *index, // block index of rotors [nnz]
                double *vec) // random vector, contiguous real & imaginary parts [2*nblock*nblock*nvec]
{
  zero_vec(nblock,2*nvec,vec);
  double *vec_real = vec, *vec_imag = &(vec[nblock*nblock*nvec]);
  
  for(int i=0 ; i<nnz ; i++)
  {
    for(int j=0 ; j<nblock ; j++)
    {
      double phase = 2.0*M_PI*random_uniform();
      vec_real[j+(j+index[i]*nblock)*nblock] = cos(phase);
      vec_imag[j+(j+index[i]*nblock)*nblock] = sin(phase);
    }
  }
}

//======================//
// 7. ITERATIVE SOLVERS //
//======================//

// Operational condition number estimate: ||r_n|| <= 2*sqrt(K)*[(sqrt(K) - 1)/(sqrt(K) + 1)]^n * ||b||
double condition_number(double epsilon, // CG iteration tolerance
                        int niter) // average number of iterations
{
  // simple iteration for x = sqrt(K)
  double sqrtK0, sqrtK = 1.0;
  do
  {
    sqrtK0 = sqrtK;
    double dtemp = pow(0.5*epsilon/sqrtK0,1.0/(double)niter);
    sqrtK = (1.0 + dtemp)/(1.0 - dtemp);
  }while(fabs(sqrtK0 - sqrtK) > 1e-8*sqrtK);

  // solve x = sqrt(K) for K
  return pow(sqrtK,2);
}

// Apply the inverse of a symmetric positive definite matrix to a vector using the standard conjugate gradient algorithm
// NOTE: using Table 2.1 pseudocode from [D. C.-L. Fong and M. Saunders, SQU Journal for Science 17, 44-62 (2012)]
int spd_inv(int nblock, // matrix & vector block size
            struct pattern *sparsity, // contains the sparsity pattern & dimensions of the matrix [1]
            double res_tol, // target residual error
            double **mat, // matrix elements of the SPD sparse matrix [sparsity->col[sparsity->ncol]][nblock*nblock]
            double *vec, // vector with overlap matrix inverse applied to it on output [sparsity->nrow*nblock*nblock]
            double *work) // pre-allocated workspace [3*sparsity->nrow*nblock*nblock]
{
  int num_iter = 0, nrow = sparsity->nrow;
  double res_tol2 = res_tol*res_tol;
  double alpha[NBLOCK_MAX], beta[NBLOCK_MAX], rho0[NBLOCK_MAX], rho_old[NBLOCK_MAX], rho[NBLOCK_MAX];

  // memory allocation (p, q, & r)
  int ndata = sparsity->nrow*nblock*nblock;
  double *p = work;
  double *q = &(work[ndata]);
  double *r = &(work[2*ndata]);
  double *x = vec; // memory reuse

  // r = b
  copy_vec(nblock,nrow,vec,r);

  // x = 0
  zero_vec(nblock,nrow,x);

  // p = r
  copy_vec(nblock,nrow,r,p);

  // rho = r^T*r
  dot_vec(nblock,nrow,r,r,rho);
  for(int i=0 ; i<nblock ; i++) { rho0[i] = rho[i]; }

  // repeat until convergence
  int not_converged = 1;
  while(not_converged)
  {
    num_iter++;

    // q = A*p
    mat_vec(nblock,sparsity,1.0,0.0,mat,p,q);

    // alpha = rho / p^T*q
    dot_vec(nblock,nrow,p,q,alpha);
    for(int i=0 ; i<nblock ; i++)
    { if(alpha[i] != 0.0) { alpha[i] = rho[i]/alpha[i]; } }

    // x <- x + alpha*p
    add_vec(nblock,nrow,alpha,p,x);

    // r <- r - alpha*q
    for(int i=0 ; i<nblock ; i++) { alpha[i] = -alpha[i]; }
    add_vec(nblock,nrow,alpha,q,r);

    // rho_old = rho
    for(int i=0 ; i<nblock ; i++) { rho_old[i] = rho[i]; }
    
    // rho = r^T*r
    dot_vec(nblock,nrow,r,r,rho);

    // beta = rho / rho_old
    for(int i=0 ; i<nblock ; i++)
    { if(rho_old[i] != 0.0) { beta[i] = rho[i]/rho_old[i]; } }

    // p <- r + beta*p
    for(int i=0 ; i<nblock ; i++) { alpha[i] = 1.0; }
    scale_vec(nblock,nrow,beta,p);
    add_vec(nblock,nrow,alpha,r,p);

    // check for convergence of all nblock CG solves
    not_converged = 0;
    for(int i=0 ; i<nblock ; i++)
    { if(rho[i] > res_tol2*rho0[i]) { not_converged = 1; } }
  }
  return num_iter;
}

// Chebyshev approximation of matrix functions applied to an input vector
int chebyshev_mat(int nblock, // matrix block size
                  int ncoeff, // number of Chebyshev polynomials
                  double res_tol, // desired residual tolerance for convergence
                  double hwt, // Hamiltonian coefficient for scaled Hamiltonian matrix
                  double owt, // overlap coefficient for scaled Hamiltonian matrix
                  double *coeff, // density coefficients for Chebyshev polynomials [ncoeff]
                  struct pattern *sparsity, // contains the sparsity pattern & dimensions of the matrices [1]
                  double **scale_hamiltonian, // scaled Hamiltonian matrix [sparsity->col[sparsity->ncol]][nblock*nblock]
                  double **overlap, // overlap matrix [sparsity->col[sparsity->ncol]][nblock*nblock]
                  double *vec, // input vector [sparsity->nrow*nblock*nblock]
                  double *func_vec, // output vector [sparsity->nrow*nblock*nblock]
                  double *func2_vec, // output vector #2 [sparsity->nrow*nblock*nblock]
                  double *work) // pre-allocated workspace [6*sparsity->nrow*nblock*nblock]
{
  int num_matvec = 0;

  // allocate memory for block vectors
  int ndata = sparsity->nrow*nblock*nblock;
  double *T_old = &(work[3*ndata]);
  double *T = &(work[4*ndata]);
  double *T_new = &(work[5*ndata]);

  // block weights for add_vec
  double minus_one[NBLOCK_MAX], block_wt[NBLOCK_MAX];
  for(int i=0 ; i<nblock ; i++) { minus_one[i] = -1.0; }

  // clear the output vectors
  zero_vec(nblock,sparsity->nrow,func_vec);

  // setup 1st Chebyshev vector (T0 = 1 -> T_old)
  copy_vec(nblock,sparsity->nrow,vec,T_old);

  // setup 2nd Chebyshev vector (T1 = H*S^{-1} -> T)
  copy_vec(nblock,sparsity->nrow,vec,T_new);
  num_matvec += spd_inv(nblock,sparsity,res_tol,overlap,T_new,work) + 1;
  mat_vec(nblock,sparsity,1.0,0.0,scale_hamiltonian,T_new,T);

  // retain term for the block-sparse density & response matrices
  for(int i=0 ; i<nblock ; i++) { block_wt[i] = coeff[0]; }
  add_vec(nblock,sparsity->nrow,block_wt,T_old,func_vec);
  if(ncoeff > 1)
  {
    for(int i=0 ; i<nblock ; i++) { block_wt[i] = coeff[1]; }
    add_vec(nblock,sparsity->nrow,block_wt,T,func_vec);
  }

  // loop to build Chebyshev polynomial expansion
  for(int i=2 ; i<ncoeff ; i++)
  {
    // prepare new Chebyshev vector (2*H*S^{-1}*T - T_old -> T_new
    copy_vec(nblock,sparsity->nrow,T,T_new);
    num_matvec += spd_inv(nblock,sparsity,res_tol,overlap,T_new,work) + 1;
    copy_vec(nblock,sparsity->nrow,T_new,work);
    mat_vec(nblock,sparsity,2.0,0.0,scale_hamiltonian,work,T_new);
    add_vec(nblock,sparsity->nrow,minus_one,T_old,T_new);

    // retain term for the block-sparse density & response matrices
    for(int j=0 ; j<nblock ; j++) { block_wt[j] = coeff[i]; }
    add_vec(nblock,sparsity->nrow,block_wt,T_new,func_vec);

    // pointer swapping to avoid copying vectors
    double* ptr = T_old;
    T_old = T;
    T = T_new;
    T_new = ptr;
  }

  // final application of S^{-1} for both density & response matrices
  num_matvec += spd_inv(nblock,sparsity,res_tol,overlap,func_vec,work) + 1;

  // final additional application of -S^{-1}*H(unscaled) for response matrix
  // -S^{-1}*H(unscaled) = -S^{-1}*H/hwt + (owt/hwt)*I
  copy_vec(nblock,sparsity->nrow,func_vec,func2_vec);
  mat_vec(nblock,sparsity,-1.0/hwt,0.0,scale_hamiltonian,func_vec,func2_vec);
  num_matvec += spd_inv(nblock,sparsity,res_tol,overlap,func2_vec,work) + 1;
  for(int i=0 ; i<nblock ; i++) { block_wt[i] = owt/hwt; }
  add_vec(nblock,sparsity->nrow,block_wt,func_vec,func2_vec);

  return num_matvec;
}

// Pre-conditioned complex symmetric linear system solver using CGLS [A*x = b -> P*A*x = P*b]
// NOTE: using CGLS pseudocode from [C. C. Paige and M. A. Saunders, TOMS 8, 43-71 (1982)]
int cgls_inv(int nblock, // matrix & vector block size
             struct pattern *mat_sparsity, // sparsity pattern of the sparse matrix [1]
             struct pattern *pre_sparsity, // sparsity pattern of the preconditioner (NULL if none) [1]
             double complex shift, // complex shift applied to mat_shift: mat_base + shift*mat_shift
             double res_tol, // target residual error
             double **mat_base, // base part of the sparse matrix [mat_sparsity->col[mat_sparsity->ncol]][nblock*nblock]
             double **mat_shift, // shifted part of the sparse matrix [mat_sparsity->col[mat_sparsity->ncol]][nblock*nblock]
             double **pre_real, // real part of the preconditioner [pre_sparsity->col[pre_sparsity->ncol]][nblock*nblock]
             double **pre_imag, // imaginary part of the preconditioner [pre_sparsity->col[pre_sparsity->ncol]][nblock*nblock]
             double *rhs_real, // real part of right-hand-side vector [sparsity->nrow*nblock*nblock]
             double *rhs_imag, // imaginary part of right-hand-side vector [sparsity->nrow*nblock*nblock]
             double *x, // solution vector (input a guess), contiguous real & imaginary vectors [2*sparsity->nrow*nblock*nblock]
             double *work) // pre-allocated workspace [8*sparsity->nrow*nblock*nblock]
{
  int num_iter = 0, nrow = 2*mat_sparsity->nrow;
  double res_tol2 = res_tol*res_tol;
  double alpha[NBLOCK_MAX], beta[NBLOCK_MAX], rho_old[NBLOCK_MAX], rho[NBLOCK_MAX], res0[NBLOCK_MAX], res[NBLOCK_MAX];

  // memory allocation (p, q, r, & s)
  int ndata = mat_sparsity->nrow*nblock*nblock;
  double *p = work;
  double *q = &(work[2*ndata]);
  double *r = &(work[4*ndata]);
  double *s = &(work[6*ndata]);

  // compute reference norms for convergence tests (b^H*b)
  dot_vec(nblock,nrow/2,rhs_real,rhs_real,res0);
  if(rhs_imag != NULL)
  {
    dot_vec(nblock,nrow/2,rhs_imag,rhs_imag,res);
    for(int i=0 ; i<nblock ; i++) { res0[i] += res[i]; }
  }

  // r = P*(b - A*x)
  copy_vec(nblock,nrow/2,rhs_real,r);
  if(rhs_imag != NULL) { copy_vec(nblock,nrow/2,rhs_imag,&(r[ndata])); }
  else { zero_vec(nblock,nrow/2,&(r[ndata])); }
  zmat_zvec(nblock,mat_sparsity,shift,mat_base,mat_shift,x,s);
  for(int i=0 ; i<nblock ; i++) { alpha[i] = -1.0; }
  add_vec(nblock,nrow,alpha,s,r);
  if(pre_sparsity != NULL)
  {
    copy_vec(nblock,nrow,r,s);
    zmat_zvec(nblock,pre_sparsity,I,pre_real,pre_imag,s,r);
  }

  // p = A^H*P^H*r
  if(pre_sparsity != NULL)
  {
    zmat_zvec(nblock,pre_sparsity,-I,pre_real,pre_imag,r,s);
    zmat_zvec(nblock,mat_sparsity,conj(shift),mat_base,mat_shift,s,p);
  }
  else
  { zmat_zvec(nblock,mat_sparsity,conj(shift),mat_base,mat_shift,r,p); }

  // rho = p^H*p
  dot_vec(nblock,nrow,p,p,rho);

  // repeat until convergence
  int not_converged = 1;
  while(not_converged)
  {
    num_iter++;

    // q = P*A*p
    if(pre_sparsity != NULL)
    {
      zmat_zvec(nblock,mat_sparsity,shift,mat_base,mat_shift,p,s);
      zmat_zvec(nblock,pre_sparsity,I,pre_real,pre_imag,s,q);
    }
    else
    { zmat_zvec(nblock,mat_sparsity,shift,mat_base,mat_shift,p,q); }

    // alpha = rho / q^H*q
    dot_vec(nblock,nrow,q,q,alpha);
    for(int i=0 ; i<nblock ; i++)
    { if(alpha[i] != 0.0) { alpha[i] = rho[i]/alpha[i]; } }

    // x <- x + alpha*p
    add_vec(nblock,nrow,alpha,p,x);

    // r <- r - alpha*q
    for(int i=0 ; i<nblock ; i++) { alpha[i] = -alpha[i]; }
    add_vec(nblock,nrow,alpha,q,r);

    // q = A^H*P^H*r
    if(pre_sparsity != NULL)
    {
      zmat_zvec(nblock,pre_sparsity,-I,pre_real,pre_imag,r,s);
      zmat_zvec(nblock,mat_sparsity,conj(shift),mat_base,mat_shift,s,q);
    }
    else
    { zmat_zvec(nblock,mat_sparsity,conj(shift),mat_base,mat_shift,r,q); }

    // rho_old = rho
    for(int i=0 ; i<nblock ; i++) { rho_old[i] = rho[i]; }
    
    // rho = q^H*q
    dot_vec(nblock,nrow,q,q,rho);

    // beta = rho / rho_old
    for(int i=0 ; i<nblock ; i++)
    { if(rho_old[i] != 0.0) { beta[i] = rho[i]/rho_old[i]; } }

    // p <- q + beta*p
    for(int i=0 ; i<nblock ; i++) { alpha[i] = 1.0; }
    scale_vec(nblock,nrow,beta,p);
    add_vec(nblock,nrow,alpha,q,p);

    // check for convergence of all nblock CG solves
    dot_vec(nblock,nrow,r,r,res);
    not_converged = 0;
    for(int i=0 ; i<nblock ; i++)
    { if(res[i] > res_tol2*res0[i]) { not_converged = 1; } }
  }

  return num_iter;
}

// rational approximation of matrix functions applied to a real input vector
int rational_mat(int nblock, // matrix block size
                 int npole, // number of pole pairs in the rational approximation
                 double res_tol, // desired residual tolerance for convergence
                 double complex *w, // weights for rational approximation [npole]
                 double complex *z, // poles for rational approximation [npole]
                 struct pattern *sparsity, // contains the sparsity pattern & dimensions of the matrices [1]
                 double **hamiltonian, // Hamiltonian matrix [sparsity->col[sparsity->ncol]][nblock*nblock]
                 double **overlap, // overlap matrix [sparsity->col[sparsity->ncol]][nblock*nblock]
                 double *vec, // input vector [sparsity->nrow*nblock*nblock]
                 double *density_vec, // density vector [sparsity->nrow*nblock*nblock]
                 double *response_vec, // response vector [sparsity->nrow*nblock*nblock]
                 double *work) // pre-allocated workspace [10*sparsity->nrow*nblock*nblock]
{
  int num_matvec = 0;

  // clear locations for solution vectors
  zero_vec(nblock,sparsity->nrow,density_vec);
  zero_vec(nblock,sparsity->nrow,response_vec);

  // allocate memory
  int ndata = sparsity->nrow*nblock*nblock;
  double *inverse_vec = &(work[8*ndata]);
  double *inverse_vec_real = inverse_vec, *inverse_vec_imag = &(inverse_vec[ndata]);

  // accumulate wt0 = -2.0*sum_i w_i
  double wt0 = 0.0;
  for(int i=0 ; i<npole ; i++) { wt0 -= 2.0*creal(w[i]); }

  // response correction with wt0*S^{-1}
  copy_vec(nblock,sparsity->nrow,vec,inverse_vec_real);
  zero_vec(nblock,sparsity->nrow,inverse_vec_imag);
  num_matvec += spd_inv(nblock,sparsity,res_tol,overlap,inverse_vec_real,work);
  double wt[NBLOCK_MAX];
  for(int i=0 ; i<nblock ; i++) { wt[i] = wt0; }
  add_vec(nblock,sparsity->nrow,wt,inverse_vec_real,response_vec);

  // loop over poles & solve for shifted inverses
  for(int i=0 ; i<npole ; i++)
  {
    // each complex mat-vec operation is equivalent to 4 real mat-vec operations
    num_matvec += 8 + 8*cgls_inv(nblock,sparsity,NULL,-z[i],res_tol,hamiltonian,overlap,NULL,NULL,vec,NULL,inverse_vec,work);

    for(int j=0 ; j<nblock ; j++) { wt[j] = 2.0*creal(w[i]); }
    add_vec(nblock,sparsity->nrow,wt,inverse_vec_real,density_vec);
    for(int j=0 ; j<nblock ; j++) { wt[j] = -2.0*cimag(w[i]); }
    add_vec(nblock,sparsity->nrow,wt,inverse_vec_imag,density_vec);
    for(int j=0 ; j<nblock ; j++) { wt[j] = -2.0*creal(z[i]*w[i]); }
    add_vec(nblock,sparsity->nrow,wt,inverse_vec_real,response_vec);
    for(int j=0 ; j<nblock ; j++) { wt[j] = 2.0*cimag(z[i]*w[i]); }
    add_vec(nblock,sparsity->nrow,wt,inverse_vec_imag,response_vec);
  }

  return num_matvec;
}

//=================//
// 8. SOLVER MAINS //
//=================//

// reference solver: embed the sparse matrix into a dense matrix, diagonalize, & restrict the density matrix to a sparse matrix
// NOTE: some arrays here can be larger than the maximum value of "int" and need "size_t" to function
void dense_solver(int nblock, // matrix block size
                  double potential, // chemical potential of the system
                  double temperature, // temperature of the system
                  double min_energy, // assumed minimum energy (to be checked)
                  double max_energy, // assumed maximum energy (to be checked)
                  struct pattern *sparsity, // contains the sparsity pattern & dimensions of the matrices [1]
                  double **hamiltonian, // Hamiltonian matrix [sparsity->col[sparsity->ncol]][nblock*nblock]
                  double **overlap, // overlap matrix [sparsity->col[sparsity->ncol]][nblock*nblock]
                  double **density, // restricted density matrix [sparsity->col[sparsity->ncol]][nblock*nblock]
                  double **response) // restricted response matrix [sparsity->col[sparsity->ncol]][nblock*nblock]
{
  // embed sparse matrices into dense matrices for LAPACK call
  size_t n = sparsity->nrow*nblock;
  double *dense_hamiltonian = (double*)malloc(sizeof(double)*n*n);
  double *dense_overlap = (double*)malloc(sizeof(double)*n*n);
  embed_mat(nblock,sparsity,hamiltonian,dense_hamiltonian);
  embed_mat(nblock,sparsity,overlap,dense_overlap);

  // zero density & response block-sparse matrices
  zero_mat(nblock,sparsity,density);
  zero_mat(nblock,sparsity,response);

  // diagonalize the Hamiltonian (LAPACK call)
  char jobz = 'V', uplo = 'U';
  MKL_INT size = n, itype = 1, lwork = -1, info;
  double *eigenvalue = (double*)malloc(sizeof(double)*n);
  double work0;
  dsygv(&itype,&jobz,&uplo,&size,dense_hamiltonian,&size,dense_overlap,&size,eigenvalue,&work0,&lwork,&info);
  if(info != 0) { printf("ERROR: LAPACK dsygv (memory query) returned an error (%d)\n",info); MPI_Abort(MPI_COMM_WORLD,0); }   
  lwork = (int)work0;
  double *work = (double*)malloc(sizeof(double)*lwork);
  dsygv(&itype,&jobz,&uplo,&size,dense_hamiltonian,&size,dense_overlap,&size,eigenvalue,work,&lwork,&info);
  if(info != 0) { printf("ERROR: LAPACK dsygv returned an error (%d)\n",info); MPI_Abort(MPI_COMM_WORLD,0); }

  // energy bounds check
  if(eigenvalue[0] < min_energy || eigenvalue[n-1] > max_energy)
  {
    printf("ERROR: energy bounds check failed, [%e,%e] not in [%e,%e]\n",eigenvalue[0],eigenvalue[n-1],min_energy,max_energy);
    MPI_Abort(MPI_COMM_WORLD,0);
  }

  // sparsify the output density matrix (to avoid an unnecessary cubic-scaling step)
  for(size_t i=0 ; i<n ; i++) // fill dense_overlap up with eigenvectors times fermi_dirac(eigenvalues)
  {
    double func = 2.0*fermi((eigenvalue[i] - potential)/temperature);
    copy_vec(1,n,&(dense_hamiltonian[i*n]),&(dense_overlap[i*n]));
    scale_vec(1,n,&func,&(dense_overlap[i*n]));
  }
  restrict_outmat(nblock,n,sparsity,dense_hamiltonian,dense_overlap,density);

  // sparsify the output overlap-response matrix (to avoid an unnecessary cubic-scaling step)
  for(size_t i=0 ; i<n ; i++) // fill dense_overlap up with eigenvectors times response(eigenvalues)
  {
    double func = -2.0*eigenvalue[i]*fermi((eigenvalue[i] - potential)/temperature);
    copy_vec(1,n,&(dense_hamiltonian[i*n]),&(dense_overlap[i*n]));
    scale_vec(1,n,&func,&(dense_overlap[i*n]));
  }
  restrict_outmat(nblock,n,sparsity,dense_hamiltonian,dense_overlap,response);

  // deallocate memory
  free(work);
  free(eigenvalue);
  free(dense_overlap);
  free(dense_hamiltonian);
}

// PEXSI solver: convert input to PEXSI's native sparse format & convert output back to a block-sparse format
void PEXSI_solver(int mpirank, // rank of this MPI process
                  int mpisize, // total number of MPI processes
                  int nblock, // matrix block size
                  int npole, // number of pole pairs in the rational approximation
                  double complex *w, // rational approximation residues [npole]
                  double complex *z, // rational approximation poles [npole]
                  struct pattern *sparsity, // contains the sparsity pattern & dimensions of the matrices [1]
                  double **hamiltonian, // Hamiltonian matrix [sparsity->col[sparsity->ncol]][nblock*nblock]
                  double **overlap, // overlap matrix [sparsity->col[sparsity->ncol]][nblock*nblock]
                  double **density, // restricted density matrix [sparsity->col[sparsity->ncol]][nblock*nblock]
                  double **response) // restricted response matrix [sparsity->col[sparsity->ncol]][nblock*nblock]
{
  // send non-MPI input parameters to mpirank != 0
  MPI_Bcast(&nblock,1,MPI_INT,0,MPI_COMM_WORLD);
  MPI_Bcast(&npole,1,MPI_INT,0,MPI_COMM_WORLD);
  if(mpirank != 0)
  {
    w = (double complex*)malloc(sizeof(double complex)*npole);
    z = (double complex*)malloc(sizeof(double complex)*npole);
  }
  MPI_Bcast(w,npole,MPI_C_DOUBLE_COMPLEX,0,MPI_COMM_WORLD);
  MPI_Bcast(z,npole,MPI_C_DOUBLE_COMPLEX,0,MPI_COMM_WORLD);

  // convert block-sparse hamiltonian & overlap matrices to one sparse matrix with ordered-pair elements
  double *hamiltonian_overlap;
  struct pattern sparsity2;
  if(mpirank == 0)
  {
    hamiltonian_overlap = (double*)malloc(sizeof(double)*2*nblock*nblock*sparsity->col[sparsity->ncol]);
    block2sparse(nblock,sparsity,&sparsity2,hamiltonian,overlap,hamiltonian_overlap);
  }

  // distribute the sparsity pattern over MPI processes from mpirank == 0
  struct pattern local_sparsity;
  int nnz = split_pattern(mpirank,mpisize,&sparsity2,&local_sparsity);
  int nnz_local = local_sparsity.col[local_sparsity.ncol];

  // distribute the local sparse matrices
  double *local_hamiltonian_overlap = (double*)malloc(sizeof(double)*2*nnz_local);
  if(mpirank == 0)
  {
    copy_vec(1,2*nnz_local,hamiltonian_overlap,local_hamiltonian_overlap);
    int inz = nnz_local;

    for(int i=1 ; i<mpisize ; i++)
    {
      int nnz_local2;
      MPI_Recv(&nnz_local2,1,MPI_INT,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
      MPI_Send(&(hamiltonian_overlap[2*inz]),2*nnz_local2,MPI_DOUBLE,i,0,MPI_COMM_WORLD);
      inz += nnz_local2;
    }
    free(hamiltonian_overlap);
  }
  else
  {
    MPI_Send(&nnz_local,1,MPI_INT,0,0,MPI_COMM_WORLD);
    MPI_Recv(local_hamiltonian_overlap,2*nnz_local,MPI_DOUBLE,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
  }

  // setup the matrix storing outputs
  double *local_density_response = (double*)malloc(sizeof(double)*2*nnz_local);
  zero_vec(1,2*nnz_local,local_density_response);

  // switch to fortran-style indexing for PEXSI compatibility (1-based instead of 0-based)
  for(int i=0 ; i<=local_sparsity.ncol ; i++)
  { local_sparsity.col[i]++; }
  for(int i=0 ; i<nnz_local ; i++)
  { local_sparsity.row[i]++; }

  // set nprow to largest factor of mpisize that is less than sqrt(mpisize) for naive best load balancing
  int nprow, nprow_max = (int)sqrt((double)mpisize) + 1;
  for(int i=1 ; i<=nprow_max ; i++)
  { if(mpisize%i == 0) { nprow = i; } }

  // setup PEXSI plan & options
  int info;
  PPEXSIPlan plan = PPEXSIPlanInitialize(MPI_COMM_WORLD,nprow,mpisize/nprow,-1,&info);
  PPEXSIOptions options;
  PPEXSISetDefaultOptions(&options);
  // change comments to switch between SuperLU & symPACK solvers:
  options.solver = 0; options.ordering = 0; options.npSymbFact = 1; // SuperLU & ParMETIS
//  options.solver = 1; options.ordering = 0; options.npSymbFact = mpisize; // symPACK & PT-Scotch
  // NOTE: symPACK should be better because it is specific to symmetric matrices, but it is still early in development
  // NOTE: npSymbFact > 1 is not stable for SuperLU
  options.verbosity = 0; // 0 disables PEXSI outputs (1 or 2 for debug info)

  // load the sparsity pattern
  PPEXSILoadRealHSMatrix(plan,options,local_sparsity.nrow,nnz,nnz_local,local_sparsity.ncol,local_sparsity.col,
                         local_sparsity.row,local_hamiltonian_overlap,1,NULL,&info);

  // perform a 1-time symbolic matrix factorization
  PPEXSISymbolicFactorizeComplexSymmetricMatrix(plan,options,&info);

  // loop over poles
  double *local_inverse = (double*)malloc(sizeof(double)*2*nnz_local);
  for(int i=0 ; i<npole ; i++)
  {
    // form the complex-shifted matrix to be inverted
    for(int j=0 ; j<nnz_local ; j++)
    {
      // introduce new scaling & shifting (PEXSI sees the ordered pairs as the real/imaginary parts of the shifted Hamiltonian)
      local_hamiltonian_overlap[2*j] -= creal(z[i])*local_hamiltonian_overlap[2*j+1];
      local_hamiltonian_overlap[2*j+1] *= -cimag(z[i]);
    }

    // selected inversion
    PPEXSISelInvComplexSymmetricMatrix(plan,options,local_hamiltonian_overlap,local_inverse,&info);

    // accumulate density & response matrices
    for(int j=0 ; j<nnz_local ; j++)
    {
      local_density_response[2*j] += 2.0*creal(w[i]*(local_inverse[2*j] + I*local_inverse[2*j+1]));
      local_density_response[2*j+1] -= 2.0*creal(z[i]*w[i]*(local_inverse[2*j] + I*local_inverse[2*j+1]));
    }

    // undo the shifting of the matrix
    for(int j=0 ; j<nnz_local ; j++)
    {
      local_hamiltonian_overlap[2*j+1] /= -cimag(z[i]);
      local_hamiltonian_overlap[2*j] += creal(z[i])*local_hamiltonian_overlap[2*j+1];
    }
  }

  // response correction with -wt0*S^{-1}
  double wt0 = 0.0;
  for(int i=0 ; i<npole ; i++) { wt0 += 2.0*creal(w[i]); }

  // replace the shifted Hamiltonian with the overlap matrix
  for(int i=0 ; i<nnz_local ; i++)
  {
    local_hamiltonian_overlap[2*i] = local_hamiltonian_overlap[2*i+1];
    local_hamiltonian_overlap[2*i+1] = 0.0;
  }
  PPEXSISelInvComplexSymmetricMatrix(plan,options,local_hamiltonian_overlap,local_inverse,&info);
  for(int i=0 ; i<nnz_local ; i++)
  { local_density_response[2*i+1] -= wt0*local_inverse[2*i]; }
  PPEXSIPlanFinalize(plan,&info);

  // switch back to C-style indexing (0-based instead of 1-based)
  for(int i=0 ; i<=local_sparsity.ncol ; i++)
  { local_sparsity.col[i]--; }
  for(int i=0 ; i<nnz_local ; i++)
  { local_sparsity.row[i]--; }

  // move the full output matrix back to mpirank == 0
  if(mpirank == 0)
  {
    double *density_response = (double*)malloc(sizeof(double)*2*nblock*nblock*sparsity->col[sparsity->ncol]);

    copy_vec(1,2*nnz_local,local_density_response,density_response);
    int inz = nnz_local;

    for(int i=1 ; i<mpisize ; i++)
    {
      int nnz_local2;
      MPI_Recv(&nnz_local2,1,MPI_INT,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
      MPI_Recv(&(density_response[2*inz]),2*nnz_local2,MPI_DOUBLE,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
      inz += nnz_local2;
    }
    sparse2block(nblock,&sparsity2,sparsity,density_response,density,response);
    free(density_response);
  }
  else
  {
    MPI_Send(&nnz_local,1,MPI_INT,0,0,MPI_COMM_WORLD);
    MPI_Send(local_density_response,2*nnz_local,MPI_DOUBLE,0,0,MPI_COMM_WORLD);
  }

  // deallocate local sparse matrices
  free(local_inverse);
  free(local_density_response);
  free(local_hamiltonian_overlap);
  free_pattern(&local_sparsity);
  if(mpirank != 0)
  {
    free(z);
    free(w);
  }
}

// Quadratic-scaling solver based on polynomial approximation of the Fermi-Dirac distribution
void quad_poly_solver(int nblock, // matrix block size
                      int ncoeff, // number of Chebyshev polynomials
                      double res_tol, // desired residual tolerance for convergence
                      double hwt, // Hamiltonian coefficient for scaled Hamiltonian matrix
                      double owt, // overlap coefficient for scaled Hamiltonian matrix
                      double *coeff, // density coefficients for Chebyshev polynomials [ncoeff]
                      struct pattern *sparsity, // contains the sparsity pattern & dimensions of the matrices [1]
                      double **hamiltonian, // Hamiltonian matrix [sparsity->col[sparsity->ncol]][nblock*nblock]
                      double **overlap, // overlap matrix [sparsity->col[sparsity->ncol]][nblock*nblock]
                      double **density, // restricted density matrix [sparsity->col[sparsity->ncol]][nblock*nblock]
                      double **response) // restricted response matrix [sparsity->col[sparsity->ncol]][nblock*nblock]
{
  int num_matvec = 0;

  // zero density & response block-sparse matrices
  zero_mat(nblock,sparsity,density);
  zero_mat(nblock,sparsity,response);

  // allocate memory for block vectors
  int ndata = sparsity->nrow*nblock*nblock;
  double *src_vec = (double*)malloc(sizeof(double)*ndata);
  double *density_vec = (double*)malloc(sizeof(double)*ndata);
  double *response_vec = (double*)malloc(sizeof(double)*ndata);
  double *work = (double*)malloc(sizeof(double)*6*ndata);

  // shift & scale the Hamiltonian to bound its spectrum within [-1,1]
  scale_mat(nblock,sparsity,hwt,hamiltonian);
  add_mat(nblock,sparsity,owt,overlap,hamiltonian);

  // loop over block columns of the density & response matrices being constructed
  for(int i=0 ; i<sparsity->ncol ; i++)
  {
    // setup a block column of basis vectors
    zero_vec(nblock,sparsity->nrow,src_vec);
    for(int j=0 ; j<nblock ; j++) { src_vec[j+(j+i*nblock)*nblock] = 1.0; }

    // construct a block column of the density & response matrices
    num_matvec += chebyshev_mat(nblock,ncoeff,res_tol,hwt,owt,coeff,sparsity,hamiltonian,overlap,src_vec,density_vec,
                                response_vec,work);

    // retain terms for the block-sparse density & response matrices
    add_col(nblock,sparsity,i,0.5,density_vec,density);
    add_row(nblock,sparsity,i,0.5,density_vec,density);
    add_col(nblock,sparsity,i,0.5,response_vec,response);
    add_row(nblock,sparsity,i,0.5,response_vec,response);
  }

  // unshift & unscale the Hamiltonian
  add_mat(nblock,sparsity,-owt,overlap,hamiltonian);
  scale_mat(nblock,sparsity,1.0/hwt,hamiltonian);

  printf("> # of mat-vecs = %d\n",num_matvec);

  // deallocate memory
  free(work);
  free(response_vec);
  free(density_vec);
  free(src_vec);
}

// Quadratic-scaling solver based on rational approximation of the Fermi-Dirac distribution
void quad_rational_solver(int nblock, // matrix block size
                          int npole, // number of pole pairs in the rational approximation
                          double res_tol, // desired residual tolerance for convergence
                          double complex *w, // rational approximation residues [npole]
                          double complex *z, // rational approximation poles [npole]
                          struct pattern *sparsity, // contains the sparsity pattern of the matrices [1]
                          double **hamiltonian, // Hamiltonian matrix [sparsity->col[sparsity->ncol]][nblock*nblock]
                          double **overlap, // overlap matrix [sparsity->col[sparsity->ncol]][nblock*nblock]
                          double **density, // restricted density matrix [sparsity->col[sparsity->ncol]][nblock*nblock]
                          double **response) // restricted response matrix [sparsity->col[sparsity->ncol]][nblock*nblock]
{
  int num_matvec = 0;

  // zero density & response block-sparse matrices
  zero_mat(nblock,sparsity,density);
  zero_mat(nblock,sparsity,response);

  // allocate memory for block vectors
  int ndata = sparsity->nrow*nblock*nblock;
  double *src_vec = (double*)malloc(sizeof(double)*ndata);
  double *density_vec = (double*)malloc(sizeof(double)*ndata);
  double *response_vec = (double*)malloc(sizeof(double)*ndata);
  double *work = (double*)malloc(sizeof(double)*10*ndata);

  // loop over block columns of the density & response matrices being constructed
  for(int i=0 ; i<sparsity->ncol ; i++)
  {
    // setup a block column of basis vectors
    zero_vec(nblock,sparsity->nrow,src_vec);
    for(int j=0 ; j<nblock ; j++) { src_vec[j+(j+i*nblock)*nblock] = 1.0; }

    // construct a block column of the density & response matrices
    num_matvec += rational_mat(nblock,npole,res_tol,w,z,sparsity,hamiltonian,overlap,src_vec,density_vec,response_vec,work);

    // retain terms for the block-sparse density & response matrices
    add_col(nblock,sparsity,i,0.5,density_vec,density);
    add_row(nblock,sparsity,i,0.5,density_vec,density);
    add_col(nblock,sparsity,i,0.5,response_vec,response);
    add_row(nblock,sparsity,i,0.5,response_vec,response);
  }

  printf("> # of mat-vecs = %d\n",num_matvec);

  // deallocate memory
  free(work);
  free(response_vec);
  free(density_vec);
  free(src_vec);
}

// Localized solver based on polynomial approximation of the Fermi-Dirac distribution
void local_poly_solver(int nblock, // matrix block size
                       int ncoeff, // number of Chebyshev polynomials
                       double res_tol, // desired residual tolerance for convergence
                       double hwt, // Hamiltonian coefficient for scaled Hamiltonian matrix
                       double owt, // overlap coefficient for scaled Hamiltonian matrix
                       double *coeff, // density coefficients for Chebyshev polynomials [ncoeff]
                       struct pattern *locality, // contains the localization pattern [1]
                       struct pattern *sparsity, // contains the sparsity pattern & dimensions of the matrices [1]
                       double **hamiltonian, // Hamiltonian matrix [sparsity->col[sparsity->ncol]][nblock*nblock]
                       double **overlap, // overlap matrix [sparsity->col[sparsity->ncol]][nblock*nblock]
                       double **density, // restricted density matrix [sparsity->col[sparsity->ncol]][nblock*nblock]
                       double **response) // restricted response matrix [sparsity->col[sparsity->ncol]][nblock*nblock]
{
  int num_matvec = 0;

  // zero density & response block-sparse matrices
  zero_mat(nblock,sparsity,density);
  zero_mat(nblock,sparsity,response);

  // calculate largest local problem dimension
  int nlocal_max = 0;
  for(int i=0 ; i<locality->ncol ; i++)
  { nlocal_max = MAX(nlocal_max,(locality->col[i+1]-locality->col[i])); }

  // allocate memory for block vectors
  int ndata = nlocal_max*nblock*nblock;
  double *src_vec = (double*)malloc(sizeof(double)*ndata);
  double *density_vec = (double*)malloc(sizeof(double)*ndata);
  double *response_vec = (double*)malloc(sizeof(double)*ndata);
  double *work = (double*)malloc(sizeof(double)*6*ndata);

  // allocate memory for localized matrices
  double **local_hamiltonian = (double**)malloc(sizeof(double*)*sparsity->col[sparsity->ncol]);
  double **local_overlap = (double**)malloc(sizeof(double*)*sparsity->col[sparsity->ncol]);
  double **local_density = (double**)malloc(sizeof(double*)*sparsity->col[sparsity->ncol]);
  double **local_response = (double**)malloc(sizeof(double*)*sparsity->col[sparsity->ncol]);

  // shift & scale the Hamiltonian to bound its spectrum within [-1,1]
  scale_mat(nblock,sparsity,hwt,hamiltonian);
  add_mat(nblock,sparsity,owt,overlap,hamiltonian);

  // loop over block columns of the density & response matrices being constructed
  double ave_sparsity = 0.0;
  for(int i=0 ; i<sparsity->ncol ; i++)
  {
    // create local versions of all relevant matrices
    struct pattern local_sparsity;
    int nlocal = locality->col[i+1]-locality->col[i];
    int *local = &(locality->row[locality->col[i]]);
    localize_pattern(nlocal,local,sparsity,&local_sparsity);
    localize_mat(nlocal,local,sparsity,hamiltonian,local_hamiltonian);
    localize_mat(nlocal,local,sparsity,overlap,local_overlap);
    localize_mat(nlocal,local,sparsity,density,local_density);
    localize_mat(nlocal,local,sparsity,response,local_response);
    int local_i = -1;
    for(int j=0 ; j<nlocal ; j++)
    { if(locality->row[j+locality->col[i]] == i) { local_i = j; } }
    ave_sparsity += (double)local_sparsity.col[local_sparsity.ncol]/(double)local_sparsity.ncol;

    // setup source vector
    zero_vec(nblock,local_sparsity.nrow,src_vec);
    for(int j=0 ; j<nblock ; j++) { src_vec[j+(j+local_i*nblock)*nblock] = 1.0; }

    // construct a block column of the density & response matrices
    num_matvec += chebyshev_mat(nblock,ncoeff,res_tol,hwt,owt,coeff,&local_sparsity,local_hamiltonian,local_overlap,src_vec,
                                density_vec,response_vec,work);

    // retain terms for the block-sparse density & response matrices
    add_col(nblock,&local_sparsity,local_i,0.5,density_vec,local_density);
    add_row(nblock,&local_sparsity,local_i,0.5,density_vec,local_density);
    add_col(nblock,&local_sparsity,local_i,0.5,response_vec,local_response);
    add_row(nblock,&local_sparsity,local_i,0.5,response_vec,local_response);

    // deallocate the temporary local sparsity pattern
    free_pattern(&local_sparsity);
  }

  // unshift & unscale the Hamiltonian
  add_mat(nblock,sparsity,-owt,overlap,hamiltonian);
  scale_mat(nblock,sparsity,1.0/hwt,hamiltonian);

  printf("average local H/S sparsity = %lf\n",(double)ave_sparsity/(double)sparsity->ncol);
  printf("# of mat-vecs = %d\n",num_matvec);

  // deallocate memory
  free(local_response);
  free(local_density);
  free(local_overlap);
  free(local_hamiltonian);
  free(work);
  free(response_vec);
  free(density_vec);
  free(src_vec);
}

// Localized solver based on polynomial approximation of the Fermi-Dirac distribution
void local_rational_solver(int nblock, // matrix block size
                           int npole, // number of Chebyshev polynomials
                           double res_tol, // desired residual tolerance for convergence
                           double complex *w, // rational approximation residues [npole]
                           double complex *z, // rational approximation poles [npole]
                           struct pattern *locality, // contains the localization pattern [1]
                           struct pattern *sparsity, // contains the sparsity pattern & dimensions of the matrices [1]
                           double **hamiltonian, // Hamiltonian matrix [sparsity->col[sparsity->ncol]][nblock*nblock]
                           double **overlap, // overlap matrix [sparsity->col[sparsity->ncol]][nblock*nblock]
                           double **density, // restricted density matrix [sparsity->col[sparsity->ncol]][nblock*nblock]
                           double **response) // restricted response matrix [sparsity->col[sparsity->ncol]][nblock*nblock]
{
  int num_matvec = 0;

  // zero density & response block-sparse matrices
  zero_mat(nblock,sparsity,density);
  zero_mat(nblock,sparsity,response);

  // calculate largest local problem dimension
  int nlocal_max = 0;
  for(int i=0 ; i<locality->ncol ; i++)
  { nlocal_max = MAX(nlocal_max,(locality->col[i+1]-locality->col[i])); }

  // allocate memory for block vectors
  int ndata = nlocal_max*nblock*nblock;
  double *src_vec = (double*)malloc(sizeof(double)*ndata);
  double *density_vec = (double*)malloc(sizeof(double)*ndata);
  double *response_vec = (double*)malloc(sizeof(double)*ndata);
  double *work = (double*)malloc(sizeof(double)*10*ndata);

  // allocate memory for localized matrices
  double **local_hamiltonian = (double**)malloc(sizeof(double*)*sparsity->col[sparsity->ncol]);
  double **local_overlap = (double**)malloc(sizeof(double*)*sparsity->col[sparsity->ncol]);
  double **local_density = (double**)malloc(sizeof(double*)*sparsity->col[sparsity->ncol]);
  double **local_response = (double**)malloc(sizeof(double*)*sparsity->col[sparsity->ncol]);

  // loop over block columns of the density & response matrices being constructed
  double ave_sparsity = 0.0;
  for(int i=0 ; i<sparsity->ncol ; i++)
  {
    // create local versions of all relevant matrices
    struct pattern local_sparsity;
    int nlocal = locality->col[i+1]-locality->col[i];
    int *local = &(locality->row[locality->col[i]]);
    localize_pattern(nlocal,local,sparsity,&local_sparsity);
    localize_mat(nlocal,local,sparsity,hamiltonian,local_hamiltonian);
    localize_mat(nlocal,local,sparsity,overlap,local_overlap);
    localize_mat(nlocal,local,sparsity,density,local_density);
    localize_mat(nlocal,local,sparsity,response,local_response);
    int local_i = -1;
    for(int j=0 ; j<nlocal ; j++)
    { if(locality->row[j+locality->col[i]] == i) { local_i = j; } }
    ave_sparsity += (double)local_sparsity.col[local_sparsity.ncol]/(double)local_sparsity.ncol;

    // setup source vector
    zero_vec(nblock,local_sparsity.nrow,src_vec);
    for(int j=0 ; j<nblock ; j++) { src_vec[j+(j+local_i*nblock)*nblock] = 1.0; }

    // construct a block column of the density & response matrices
    num_matvec += rational_mat(nblock,npole,res_tol,w,z,&local_sparsity,local_hamiltonian,local_overlap,src_vec,density_vec,
                               response_vec,work);

    // retain terms for the block-sparse density & response matrices
    add_col(nblock,&local_sparsity,local_i,0.5,density_vec,local_density);
    add_row(nblock,&local_sparsity,local_i,0.5,density_vec,local_density);
    add_col(nblock,&local_sparsity,local_i,0.5,response_vec,local_response);
    add_row(nblock,&local_sparsity,local_i,0.5,response_vec,local_response);

    // deallocate the temporary local sparsity pattern
    free_pattern(&local_sparsity);
  }

  printf("average local H/S sparsity = %lf\n",(double)ave_sparsity/(double)sparsity->ncol);
  printf("# of mat-vecs = %d\n",num_matvec);

  // deallocate memory
  free(local_response);
  free(local_density);
  free(local_overlap);
  free(local_hamiltonian);
  free(work);
  free(response_vec);
  free(density_vec);
  free(src_vec);
}

// Random solver based on polynomial approximation of the Fermi-Dirac distribution
void random_poly_solver(int nblock, // matrix block size
                        int ncoeff, // number of Chebyshev polynomials
                        int ncolor, // number of atom colors
                        int nsample, // number of random samples
                        int seed, // PRNG seed
                        double res_tol, // desired residual tolerance for convergence
                        double hwt, // Hamiltonian coefficient for scaled Hamiltonian matrix
                        double owt, // overlap coefficient for scaled Hamiltonian matrix
                        double *coeff, // density coefficients for Chebyshev polynomials [ncoeff]
                        int *color, // list of color offsets for the atom_ptr list [ncolor+1]
                        int *atom_ptr, // list of atoms of each color [color[ncolor]]
                        struct pattern *sparsity, // contains the sparsity pattern & dimensions of the matrices [1]
                        double **hamiltonian, // Hamiltonian matrix [sparsity->col[sparsity->ncol]][nblock*nblock]
                        double **overlap, // overlap matrix [sparsity->col[sparsity->ncol]][nblock*nblock]
                        double **density, // restricted density matrix [sparsity->col[sparsity->ncol]][nblock*nblock]
                        double **response) // restricted response matrix [sparsity->col[sparsity->ncol]][nblock*nblock]
{
  int num_matvec = 0;

  // seed the solver on entry for deterministic performance
  random64(seed);

  // zero density & response block-sparse matrices
  zero_mat(nblock,sparsity,density);
  zero_mat(nblock,sparsity,response);

  // allocate memory for block vectors
  int ndata = sparsity->nrow*nblock*nblock;
  double *rng_vec = (double*)malloc(sizeof(double)*2*ndata);
  double *rng_vec_real = rng_vec, *rng_vec_imag = &(rng_vec[ndata]);
  double *density_vec = (double*)malloc(sizeof(double)*ndata);
  double *response_vec = (double*)malloc(sizeof(double)*ndata);
  double *work = (double*)malloc(sizeof(double)*6*ndata);

  // shift & scale the Hamiltonian to bound its spectrum within [-1,1]
  scale_mat(nblock,sparsity,hwt,hamiltonian);
  add_mat(nblock,sparsity,owt,overlap,hamiltonian);

  // loop over block columns of the density & response matrices being constructed
  for(int i=0 ; i<nsample ; i++)
  {
    for(int j=0 ; j<ncolor ; j++)
    {
      // construct a random block source vector
      random_vec(nblock,sparsity->nrow,color[j+1]-color[j],&(atom_ptr[color[j]]),rng_vec);

      // construct a block column of the density & response matrices for the real part
      num_matvec += chebyshev_mat(nblock,ncoeff,res_tol,hwt,owt,coeff,sparsity,hamiltonian,overlap,rng_vec_real,density_vec,
                                  response_vec,work);

      // add contributions to the (symmetric) density and response matrices
      restrict_outvec(nblock,sparsity,density_vec,rng_vec_real,density);
      restrict_outvec(nblock,sparsity,rng_vec_real,density_vec,density);
      restrict_outvec(nblock,sparsity,response_vec,rng_vec_real,response);
      restrict_outvec(nblock,sparsity,rng_vec_real,response_vec,response);

      // construct a block column of the density & response matrices for the imaginary part
      num_matvec += chebyshev_mat(nblock,ncoeff,res_tol,hwt,owt,coeff,sparsity,hamiltonian,overlap,rng_vec_imag,density_vec,
                                  response_vec,work);

      // add contributions to the (symmetric) density and response matrices
      restrict_outvec(nblock,sparsity,density_vec,rng_vec_imag,density);
      restrict_outvec(nblock,sparsity,rng_vec_imag,density_vec,density);
      restrict_outvec(nblock,sparsity,response_vec,rng_vec_imag,response);
      restrict_outvec(nblock,sparsity,rng_vec_imag,response_vec,response);
    }
  }

  // unshift & unscale the Hamiltonian
  add_mat(nblock,sparsity,-owt,overlap,hamiltonian);
  scale_mat(nblock,sparsity,1.0/hwt,hamiltonian);

  // average the density and response matrices (0.5 factor averages the symmetrization)
  scale_mat(nblock,sparsity,0.5/(double)nsample,density);
  scale_mat(nblock,sparsity,0.5/(double)nsample,response);

  printf("# of mat-vecs = %d\n",num_matvec);

  // deallocate memory
  free(work);
  free(response_vec);
  free(density_vec);
  free(rng_vec);
}

// Random solver based on polynomial approximation of the Fermi-Dirac distribution
void random_rational_solver(int nblock, // matrix block size
                            int npole, // number of Chebyshev polynomials
                            int ncolor, // number of atom colors
                            int nsample, // number of random samples
                            int seed, // PRNG seed
                            double res_tol, // desired residual tolerance for convergence
                            double complex *w, // rational approximation residues [npole]
                            double complex *z, // rational approximation poles [npole]
                            int *color, // list of color offsets for the atom_ptr list [ncolor+1]
                            int *atom_ptr, // list of atoms of each color [color[ncolor]]
                            struct pattern *sparsity, // contains the sparsity pattern & dimensions of the matrices [1]
                            double **hamiltonian, // Hamiltonian matrix [sparsity->col[sparsity->ncol]][nblock*nblock]
                            double **overlap, // overlap matrix [sparsity->col[sparsity->ncol]][nblock*nblock]
                            double **density, // restricted density matrix [sparsity->col[sparsity->ncol]][nblock*nblock]
                            double **response) // restricted response matrix [sparsity->col[sparsity->ncol]][nblock*nblock]
{
  int num_matvec = 0;

  // seed the solver on entry for deterministic performance
  random64(seed);

  // zero density & response block-sparse matrices
  zero_mat(nblock,sparsity,density);
  zero_mat(nblock,sparsity,response);

  // allocate memory for block vectors
  int ndata = sparsity->nrow*nblock*nblock;
  double *rng_vec = (double*)malloc(sizeof(double)*2*ndata);
  double *rng_vec_real = rng_vec, *rng_vec_imag = &(rng_vec[ndata]);
  double *density_vec = (double*)malloc(sizeof(double)*ndata);
  double *response_vec = (double*)malloc(sizeof(double)*ndata);
  double *work = (double*)malloc(sizeof(double)*10*ndata);

  // loop over block columns of the density & response matrices being constructed
  for(int i=0 ; i<nsample ; i++)
  {
    for(int j=0 ; j<ncolor ; j++)
    {
      // construct a random block source vector
      random_vec(nblock,sparsity->nrow,color[j+1]-color[j],&(atom_ptr[color[j]]),rng_vec);

      // construct a block column of the density & response matrices for the real part
      num_matvec += rational_mat(nblock,npole,res_tol,w,z,sparsity,hamiltonian,overlap,rng_vec_real,density_vec,response_vec,
                    work);

      // add contributions to the (symmetric) density and response matrices
      restrict_outvec(nblock,sparsity,density_vec,rng_vec_real,density);
      restrict_outvec(nblock,sparsity,rng_vec_real,density_vec,density);
      restrict_outvec(nblock,sparsity,response_vec,rng_vec_real,response);
      restrict_outvec(nblock,sparsity,rng_vec_real,response_vec,response);

      // construct a block column of the density & response matrices for the imaginary part
      num_matvec += rational_mat(nblock,npole,res_tol,w,z,sparsity,hamiltonian,overlap,rng_vec_imag,density_vec,response_vec,
                    work);

      // add contributions to the (symmetric) density and response matrices
      restrict_outvec(nblock,sparsity,density_vec,rng_vec_imag,density);
      restrict_outvec(nblock,sparsity,rng_vec_imag,density_vec,density);
      restrict_outvec(nblock,sparsity,response_vec,rng_vec_imag,response);
      restrict_outvec(nblock,sparsity,rng_vec_imag,response_vec,response);
    }
  }

  // average the density and response matrices (0.5 factor averages the symmetrization)
  scale_mat(nblock,sparsity,0.5/(double)nsample,density);
  scale_mat(nblock,sparsity,0.5/(double)nsample,response);

  printf("# of mat-vecs = %d\n",num_matvec);

  // deallocate memory
  free(work);
  free(response_vec);
  free(density_vec);
  free(rng_vec);
}

// Infinite-crystal solver based on rational approximation of the Fermi-Dirac distribution
void infinite_rational_solver(int nblock, // matrix block size
                              int npole, // number of pole pairs in the rational approximation
                              double res_tol, // desired residual tolerance for convergence
                              double *atom, // atomic coordinates [3*sparsity->nrow]
                              double complex *w, // rational approximation residues [npole]
                              double complex *z, // rational approximation poles [npole]
                              struct pattern *sparsity, // contains the sparsity pattern of the matrices [1]
                              double **hamiltonian, // Hamiltonian matrix [sparsity->col[sparsity->ncol]][nblock*nblock]
                              double **overlap, // overlap matrix [sparsity->col[sparsity->ncol]][nblock*nblock]
                              double **density, // restricted density matrix [sparsity->col[sparsity->ncol]][nblock*nblock]
                              double **response) // restricted response matrix [sparsity->col[sparsity->ncol]][nblock*nblock]
{
  int num_matvec = 0;

  // zero density & response block-sparse matrices
  zero_mat(nblock,sparsity,density);
  zero_mat(nblock,sparsity,response);

  // allocate memory for block vectors
  int ndata = sparsity->nrow*nblock*nblock;
  double *src_vec = (double*)malloc(sizeof(double)*2*ndata);
  double *src_vec_real = src_vec, *src_vec_imag = &(src_vec[ndata]);
  double *density_vec = (double*)malloc(sizeof(double)*2*ndata);
  double *response_vec = (double*)malloc(sizeof(double)*2*ndata);
  double *work = (double*)malloc(sizeof(double)*10*ndata);

  // setup the block column of basis vectors
  zero_vec(nblock,2*sparsity->nrow,src_vec);
  for(int i=0 ; i<nblock ; i++) { src_vec[i+i*nblock] = 1.0; }

  // construct the single independent block column of the density & response matrices
  num_matvec += rational_mat(nblock,npole,res_tol,w,z,sparsity,hamiltonian,overlap,src_vec,density_vec,response_vec,work);

  // multiply the (0,0) block by 0.5 to avoid double-counting
  for(int i=0 ; i<nblock*nblock ; i++) { density_vec[i] *= 0.5; response_vec[i] *= 0.5; }

  // retain terms for the block-sparse density & response matrices
  add_col(nblock,sparsity,0,1.0,density_vec,density);
  add_row(nblock,sparsity,0,1.0,density_vec,density);
  add_col(nblock,sparsity,0,1.0,response_vec,response);
  add_row(nblock,sparsity,0,1.0,response_vec,response);

  printf("> # of mat-vecs = %d\n",num_matvec);

  // calculate the Frobenius-norm of matrix blocks for density & response matrices
  FILE *decay_file = fopen("decay.out","w");
  fprintf(decay_file,"%d\n",sparsity->nrow);
  for(int i=0 ; i<sparsity->nrow ; i++)
  {
    double dist = A0*distance(&(atom[0]),&(atom[3*i])), density_norm, response_norm;
    dot_vec(1,nblock*nblock,&(density_vec[i*nblock*nblock]),&(density_vec[i*nblock*nblock]),&density_norm);
    dot_vec(1,nblock*nblock,&(response_vec[i*nblock*nblock]),&(response_vec[i*nblock*nblock]),&response_norm);
    fprintf(decay_file,"%e %e %e\n",dist,sqrt(density_norm),sqrt(response_norm));
  }
  fclose(decay_file);

  // deallocate memory
  free(work);
  free(response_vec);
  free(density_vec);
  free(src_vec);
}

// Infinite-crystal solver based on reciprocal-space decomposition of the eigenvalue problem (band structure)
// NOTE: this function must be modified if MKL_Complex16 differs from its MKL specification
// NOTE: this function does not have any threading & it is not meant to have high performance
void infinite_reciprocal_solver(int nblock, // matrix block size
                                double potential, // chemical potential of the system
                                double temperature, // temperature of the system
                                double min_energy, // minimum energy for density-of-states plot
                                double max_energy, // maximum energy for density-of-states plot
                                int ngrid, // number of k-space grid points per dimension
                                int *latvec, // list of lattice vectors [3*sparsity->nrow]
                                double *atom, // atomic coordinates [3*sparsity->nrow]
                                struct pattern *sparsity, // contains the sparsity pattern of the matrices [1]
                                double **hamiltonian, // Hamiltonian matrix [sparsity->col[sparsity->ncol]][nblock*nblock]
                                double **overlap, // overlap matrix [sparsity->col[sparsity->ncol]][nblock*nblock]
                                double **density, // restricted density matrix [sparsity->col[sparsity->ncol]][nblock*nblock]
                                double **response) // restricted response matrix [sparsity->col[sparsity->ncol]][nblock*nblock]
{
  // zero density & response block-sparse matrices
  zero_mat(nblock,sparsity,density);
  zero_mat(nblock,sparsity,response);

  // allocate memory for LAPACK solver
  MKL_INT itype = 1, size = nblock, lwork = -1, info;
  char jobz = 'V', uplo = 'U', transa = 'N', transb = 'C';
  double *eigenvalue = (double*)malloc(sizeof(double)*size);
  double *rwork = (double*)malloc(sizeof(double)*(3*size-2));
  MKL_Complex16 *hamiltonian_k = (MKL_Complex16*)malloc(sizeof(MKL_Complex16)*size*size);
  MKL_Complex16 *overlap_k = (MKL_Complex16*)malloc(sizeof(MKL_Complex16)*size*size);
  MKL_Complex16 *density0 = (MKL_Complex16*)malloc(sizeof(MKL_Complex16)*size*size);
  MKL_Complex16 *response0 = (MKL_Complex16*)malloc(sizeof(MKL_Complex16)*size*size);
  MKL_Complex16 work0, one, zero;
  zhegv(&itype,&jobz,&uplo,&size,hamiltonian_k,&size,overlap_k,&size,eigenvalue,&work0,&lwork,rwork,&info);
  if(info != 0) { printf("ERROR: LAPACK zhegv (memory query) returned an error (%d)\n",info); MPI_Abort(MPI_COMM_WORLD,0); }
  lwork = (int)(work0.real); // depends on details of MKL_Complex16
  one.real = 1.0; one.imag = 0.0; // depends on details of MKL_Complex16
  zero.real = 0.0; zero.imag = 0.0; // depends on details of MKL_Complex16
  MKL_Complex16 *work = (MKL_Complex16*)malloc(sizeof(MKL_Complex16)*lwork);

  // setup density-of-states information
  int ndos = (int)(4.0*(max_energy-min_energy)/temperature);
  double denergy = (max_energy-min_energy)/(double)(ndos-1);
  double *dos = (double*)malloc(sizeof(double)*ndos);
  double *dos_int = (double*)malloc(sizeof(double)*ndos);
  for(int i=0 ; i<ndos ; i++) { dos[i] = dos_int[i] = 0.0; }

  // allocate memory for block vectors
  int ndata = sparsity->nrow*nblock*nblock;
  double *density_vec = (double*)malloc(sizeof(double)*2*ndata);
  double *response_vec = (double*)malloc(sizeof(double)*2*ndata);

  // loop over k-points in each dimension
  double wt = 1.0/(double)(ngrid*ngrid*ngrid);
  double complex phase[3];
  for(int i=0 ; i<ngrid ; i++)
  {
    phase[0] = I*2.0*M_PI*(double)i/(double)ngrid;
    for(int j=0 ; j<ngrid ; j++)
    {
      phase[1] = I*2.0*M_PI*(double)j/(double)ngrid;
      for(int k=0 ; k<ngrid ; k++)
      {
        phase[2] = I*2.0*M_PI*(double)k/(double)ngrid;

        // construct reciprocal-space Hamiltonian & overlap matrices
        for(int l=0 ; l<nblock*nblock ; l++)
        { hamiltonian_k[l].real = hamiltonian_k[l].imag = overlap_k[l].real = overlap_k[l].imag = 0.0; }
        for(int l=0 ; l<sparsity->col[1] ; l++)
        {
          int ilat = sparsity->row[l];
          double complex phase0 = phase[0]*latvec[3*ilat] + phase[1]*latvec[3*ilat+1] + phase[2]*latvec[3*ilat+2];
          double complex exp_phase0 = cexp(phase0);
          for(int m=0 ; m<nblock*nblock ; m++)
          {
            hamiltonian_k[m].real += creal(exp_phase0)*hamiltonian[l][m];
            hamiltonian_k[m].imag += cimag(exp_phase0)*hamiltonian[l][m];
            overlap_k[m].real += creal(exp_phase0)*overlap[l][m];
            overlap_k[m].imag += cimag(exp_phase0)*overlap[l][m];
          }
        }

        // diagonalize the complex-Hermitian Hamiltonian (LAPACK call)
        zhegv(&itype,&jobz,&uplo,&size,hamiltonian_k,&size,overlap_k,&size,eigenvalue,work,&lwork,rwork,&info);
        if(info != 0) { printf("ERROR: LAPACK zhegv returned an error (%d)\n",info); MPI_Abort(MPI_COMM_WORLD,0); }

        // sparsely accumulate DOS contributions
        for(int l=0 ; l<size ; l++)
        {
          int min_dos = MAX(0,(int)((eigenvalue[l] - min_energy - 20.0*temperature)/denergy));
          int max_dos = MIN(ndos-1,(int)((eigenvalue[l] - min_energy + 20.0*temperature)/denergy));
          for(int m=min_dos ; m<=max_dos ; m++)
          {
            double dos_energy = min_energy + (double)m*denergy;
            dos_int[m] += wt*2.0*(fermi((eigenvalue[l] - dos_energy)/temperature)
                                - fermi((eigenvalue[l] - dos_energy + denergy)/temperature));
            dos[m] -= wt*2.0*dfermi_dx((eigenvalue[l] - dos_energy)/temperature)/temperature;
          }
        }

        // construct k-point contribution to real-space density matrix
        for(int l=0 ; l<nblock ; l++)
        {
          double func = 2.0*fermi((eigenvalue[l] - potential)/temperature);
          for(int m=0 ; m<nblock ; m++)
          {
            overlap_k[m+l*nblock].real = func*hamiltonian_k[m+l*nblock].real;
            overlap_k[m+l*nblock].imag = func*hamiltonian_k[m+l*nblock].imag;
          }
        }
        zgemm(&transa,&transb,&size,&size,&size,&one,hamiltonian_k,&size,overlap_k,&size,&zero,density0,&size);

        // construct k-point contribution to real-space response matrix
        for(int l=0 ; l<nblock ; l++)
        {
          double func = -2.0*eigenvalue[l]*fermi((eigenvalue[l] - potential)/temperature);
          for(int m=0 ; m<nblock ; m++)
          {
            overlap_k[m+l*nblock].real = func*hamiltonian_k[m+l*nblock].real;
            overlap_k[m+l*nblock].imag = func*hamiltonian_k[m+l*nblock].imag;
          }
        }
        zgemm(&transa,&transb,&size,&size,&size,&one,hamiltonian_k,&size,overlap_k,&size,&zero,response0,&size);

        // accumulate contributions to real-space density & response matrices
        for(int l=0 ; l<nblock*nblock ; l++)
        {
          density[0][l] += wt*density0[l].real;
          response[0][l] += wt*response0[l].real;
        }
        for(int l=1 ; l<sparsity->col[1] ; l++)
        {
          int ilat = sparsity->row[l];
          double complex phase0 = phase[0]*latvec[3*ilat] + phase[1]*latvec[3*ilat+1] + phase[2]*latvec[3*ilat+2];
          double complex exp_phase0 = cexp(-phase0);
          for(int m=0 ; m<nblock*nblock ; m++)
          {
            double complex density00 = density0[m].real + I*density0[m].imag;
            double complex response00 = response0[m].real + I*response0[m].imag;
            density[l][m] += wt*creal(exp_phase0*density00);
            response[l][m] += wt*creal(exp_phase0*response00);
            density[sparsity->col[ilat]][m] += wt*creal(conj(exp_phase0)*density00);
            response[sparsity->col[ilat]][m] += wt*creal(conj(exp_phase0)*response00);
          }
        }

        // accumulate contributions to extended real-space density & response matrices (likely bottleneck for intended use)
#pragma omp parallel for
        for(int l=0 ; l<sparsity->nrow ; l++)
        {
          double complex phase0 = phase[0]*latvec[3*l] + phase[1]*latvec[3*l+1] + phase[2]*latvec[3*l+2];
          double complex exp_phase0 = cexp(-phase0);
          int offset = l*nblock*nblock;
          for(int m=0 ; m<nblock*nblock ; m++)
          {
            double complex density00 = density0[m].real + I*density0[m].imag;
            double complex response00 = response0[m].real + I*response0[m].imag;
            density_vec[offset+m] += wt*creal(exp_phase0*density00);
            response_vec[offset+m] += wt*creal(exp_phase0*response00);
          }
        }
      }
    }
  }

  // print both the DOS and accumulated DOS
  FILE* dos_file = fopen("dos.out","w");
  double inc_energy = (double)(ndos-1)/((max_energy-min_energy)*E0);
  double acc_dos = 0.0;
  for(int i=0 ; i<ndos ; i++)
  {
    acc_dos += dos_int[i];
    double dos_energy = min_energy + (max_energy-min_energy)*(double)i/(double)(ndos-1);
    fprintf(dos_file,"%e %e %e\n",dos_energy*E0,dos[i]/E0,acc_dos);
  }
  fclose(dos_file);

  // calculate the Frobenius-norm of matrix blocks for density & response matrices
  FILE *decay_file = fopen("decay.out","w");
  fprintf(decay_file,"%d\n",sparsity->nrow);
  for(int i=0 ; i<sparsity->nrow ; i++)
  {
    double dist = A0*distance(&(atom[0]),&(atom[3*i])), density_norm, response_norm;
    dot_vec(1,nblock*nblock,&(density_vec[i*nblock*nblock]),&(density_vec[i*nblock*nblock]),&density_norm);
    dot_vec(1,nblock*nblock,&(response_vec[i*nblock*nblock]),&(response_vec[i*nblock*nblock]),&response_norm);
    fprintf(decay_file,"%e %e %e\n",dist,sqrt(density_norm),sqrt(response_norm));
  }
  fclose(decay_file);

  // deallocate memory
  free(response_vec);
  free(density_vec);
  free(dos_int);
  free(dos);
  free(work);
  free(response0);
  free(density0);
  free(overlap_k);
  free(hamiltonian_k);
  free(rwork);
  free(eigenvalue);
}

// test out the effects of localization on Green's function accuracy and CG condition numbers
void infinite_pre_tester(int nblock, // matrix block size
                         int nradius, // number of localization radii to test
                         double min_radius, // minimum localization radius
                         double max_radius, // maximum localization radius
                         double res_tol, // desired residual tolerance for convergence
                         double complex z0, // complex energy shift for the iterative solve
                         double complex z1, // complex energy shift for the preconditioner
                         struct pattern *sparsity, // contains the sparsity pattern of the matrices [1]
                         double *atom, // atomic coordinates to define new sparsity patterns [3*sparsity->nrow]
                         int *latvec, // ordered lattice vector list [3*sparsity->nrow]
                         double **hamiltonian, // Hamiltonian matrix [sparsity->col[sparsity->ncol]][nblock*nblock]
                         double **overlap) // overlap matrix [sparsity->col[sparsity->ncol]][nblock*nblock]
{
  // allocate memory for block vectors
  int ndata = sparsity->nrow*nblock*nblock;
  double *rhs = (double*)malloc(sizeof(double)*2*ndata);
  double *rhs_real = rhs, *rhs_imag = &(rhs[ndata]);
  double *x0 = (double*)malloc(sizeof(double)*2*ndata);
  double *x0_real = x0, *x0_imag = &(x0[ndata]);
  double *x1 = (double*)malloc(sizeof(double)*2*ndata);
  double *x1_real = x1, *x1_imag = &(x1[ndata]);
  double *x2 = (double*)malloc(sizeof(double)*2*ndata);
  double *work = (double*)malloc(sizeof(double)*8*ndata);
  zero_vec(nblock,2*sparsity->nrow,rhs);
  for(int i=0 ; i<nblock ; i++) { rhs[i+i*nblock] = 1.0; }

  // benchmark time for an overlap matrix solve
  printf("> iterative overlap inversion\n");
  double time_before = omp_get_wtime();
  copy_vec(nblock,sparsity->nrow,rhs,x0);
  int niter = spd_inv(nblock,sparsity,res_tol,overlap,x0,work);
  double time_after = omp_get_wtime();

  printf(">> number of iterations = %d\n",niter);
  printf(">> time usage = %e s\n",time_after-time_before);

  // prepare the solutions for use in the sparse approximate inverse (avoid double-counting (0,0) matrix block)
  for(int i=0 ; i<nblock*nblock ; i++) { x0[i] *= 0.5; }

  // for each radius, construct a sparse approximate inverse overlap matrix
  for(int i=0 ; i<nradius ; i++)
  {
    double x = (double)i/(double)(nradius-1);
    double radius = (1.0 - x)*min_radius + x*max_radius;
    printf("> inverse sparsity radius = %lf\n",A0*radius);

    // allocate memory for block-sparse inverse overlap matrix
    struct pattern inv_sparsity;
    neighbor_list(sparsity->nrow,atom,radius,&inv_sparsity);
    int nnz = inv_sparsity.col[inv_sparsity.ncol];
    double **inverse = (double**)malloc(sizeof(double*)*nnz);
    crystal_malloc(nblock,&inv_sparsity,latvec,inverse);

    // setup inverse matrix elements
    zero_mat(nblock,&inv_sparsity,inverse);
    add_col(nblock,&inv_sparsity,0,1.0,x0,inverse);
    add_row(nblock,&inv_sparsity,0,1.0,x0,inverse);

    // benchmark the time of applying the inverse
    time_before = omp_get_wtime();
    mat_vec(nblock,&inv_sparsity,1.0,0.0,inverse,rhs,x1);
    time_after = omp_get_wtime();
    printf(">> time usage = %e s\n",time_after-time_before);

    // calculate the residual of the inverse
    copy_vec(nblock,sparsity->nrow,rhs,x2);
    mat_vec(nblock,sparsity,-1.0,1.0,overlap,x1,x2);
    double res[NBLOCK_MAX], res_max;
    dot_vec(nblock,sparsity->nrow,x2,x2,res);
    res_max = res[0];
    for(int j=1 ; j<nblock ; j++)
    { if(res[j] > res_max) { res_max = res[j]; } }
    printf(">> residual error = %e\n",sqrt(res_max));

    // deallocate loop memory
    free(inverse[0]);
    free(inverse);
    free_pattern(&inv_sparsity);
  }

  // solve the larger imaginary shift first
  printf("> preconditioner construction\n");
  time_before = omp_get_wtime();
  copy_vec(nblock,2*sparsity->nrow,rhs,x1);
  niter = cgls_inv(nblock,sparsity,NULL,z1,res_tol,hamiltonian,overlap,NULL,NULL,rhs_real,rhs_imag,x1,work);
  time_after = omp_get_wtime();

  printf(">> number of iterations = %d\n",niter);
  printf(">> time usage = %e s\n",time_after-time_before);
  printf(">> operational condition number = %lf\n",condition_number(res_tol,niter));  

  // solve the small imaginary shift
  printf("> unpreconditioned solve\n");
  time_before = omp_get_wtime();
  copy_vec(nblock,2*sparsity->nrow,rhs,x0);
  niter = cgls_inv(nblock,sparsity,NULL,z0,res_tol,hamiltonian,overlap,NULL,NULL,rhs_real,rhs_imag,x0,work);
  time_after = omp_get_wtime();

  printf(">> number of iterations = %d\n",niter);
  printf(">> time usage = %e s\n",time_after-time_before);
  printf(">> operational condition number = %lf\n",condition_number(res_tol,niter));  

  // prepare the solutions for use in the preconditioner (avoid double-counting (0,0) matrix block)
  for(int i=0 ; i<nblock*nblock ; i++) { x0_real[i] *= 0.5; x0_imag[i] *= 0.5; x1_real[i] *= 0.5; x1_imag[i] *= 0.5; }

  // for each radius, construct a preconditioner from each solution
  for(int i=0 ; i<nradius ; i++)
  {
    double x = (double)i/(double)(nradius-1);
    double radius = (1.0 - x)*min_radius + x*max_radius;
    printf("> preconditioner radius = %lf\n",A0*radius);

    // allocate memory for block-sparse shifted inverse matrix
    struct pattern pre_sparsity;
    neighbor_list(sparsity->nrow,atom,radius,&pre_sparsity);
    int nnz = pre_sparsity.col[pre_sparsity.ncol];
    double **inverse_real = (double**)malloc(sizeof(double*)*nnz);
    double **inverse_imag = (double**)malloc(sizeof(double*)*nnz);
    crystal_malloc(nblock,&pre_sparsity,latvec,inverse_real);
    crystal_malloc(nblock,&pre_sparsity,latvec,inverse_imag);

    // setup shifted preconditioner
    zero_mat(nblock,&pre_sparsity,inverse_real);
    zero_mat(nblock,&pre_sparsity,inverse_imag);
    add_col(nblock,&pre_sparsity,0,1.0,x1_real,inverse_real);
    add_row(nblock,&pre_sparsity,0,1.0,x1_real,inverse_real);
    add_col(nblock,&pre_sparsity,0,1.0,x1_imag,inverse_imag);
    add_row(nblock,&pre_sparsity,0,1.0,x1_imag,inverse_imag);

    time_before = omp_get_wtime();
    copy_vec(nblock,2*sparsity->nrow,rhs,x2);
    niter = cgls_inv(nblock,sparsity,&pre_sparsity,z0,res_tol,hamiltonian,overlap,inverse_real,inverse_imag,rhs_real,rhs_imag,
                     x2,work);
    time_after = omp_get_wtime();

    printf(">> number of iterations (shifted) = %d\n",niter);
    printf(">> time usage (shifted) = %e s\n",time_after-time_before);
    printf(">> operational condition number (shifted) = %lf\n",condition_number(res_tol,niter)); 

    // setup self preconditioner
    zero_mat(nblock,&pre_sparsity,inverse_real);
    zero_mat(nblock,&pre_sparsity,inverse_imag);
    add_col(nblock,&pre_sparsity,0,1.0,x0_real,inverse_real);
    add_row(nblock,&pre_sparsity,0,1.0,x0_real,inverse_real);
    add_col(nblock,&pre_sparsity,0,1.0,x0_imag,inverse_imag);
    add_row(nblock,&pre_sparsity,0,1.0,x0_imag,inverse_imag);

    time_before = omp_get_wtime();
    copy_vec(nblock,2*sparsity->nrow,rhs,x2);
    niter = cgls_inv(nblock,sparsity,&pre_sparsity,z0,res_tol,hamiltonian,overlap,inverse_real,inverse_imag,rhs_real,rhs_imag,
                     x2,work);
    time_after = omp_get_wtime();

    printf(">> number of iterations (self) = %d\n",niter);
    printf(">> time usage (self) = %e s\n",time_after-time_before);
    printf(">> operational condition number (self) = %lf\n",condition_number(res_tol,niter)); 

    // deallocate loop memory
    free(inverse_imag[0]);
    free(inverse_real[0]);
    free(inverse_imag);
    free(inverse_real);
    free_pattern(&pre_sparsity);
  }

  // deallocate local memory
  free(work);
  free(x2);
  free(x1);
  free(x0);
  free(rhs);
}

//=========//
// 9. MAIN //
//=========//

int main(int argc, char** argv)
{
  // MPI initialization
  int mpirank, mpisize;
  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&mpirank);
  MPI_Comm_size(MPI_COMM_WORLD,&mpisize);

  // the vast majority of the program is performed by mpirank == 0 only
  if(mpirank == 0)
  {
    // initial timing point
    double time1 = omp_get_wtime();

    // parse command-line input
    int solver, natom, napprox, nsample, seed;
    double temperature, potential, res_tol, pre_shift = 0.0, pre_radius = 0.0, local_radius = 0.0;

    // check for an appropriate number of command-line arguments
    if(argc < 5)
    {
      printf("USAGE: <executable> <structure file> <chemical potential> <temperature> <solver> <solver parameters ...>\n");
      MPI_Abort(MPI_COMM_WORLD,0);
    }

    // read the solver-independent input variables
    sscanf(argv[2],"%lf",&potential);
    sscanf(argv[3],"%lf",&temperature);
    sscanf(argv[4],"%d",&solver);

    // parse the atomic structure file
    FILE *structure_file = fopen(argv[1],"r");
    if(structure_file == NULL)
    { printf("ERROR: %s structure file not found\n",argv[1]); MPI_Abort(MPI_COMM_WORLD,0); }
    fscanf(structure_file,"%d",&natom);
    double *atom = (double*)malloc(sizeof(double)*3*natom);
    for(int i=0 ; i<natom ; i++)
    {
      char element[16];
      fscanf(structure_file,"%s",element);
      if(strcmp(element,"Cu"))
      { printf("ERROR: Only element available is Cu (%s != Cu)\n",element); MPI_Abort(MPI_COMM_WORLD,0); }
      for(int j=0 ; j<3 ; j++) { fscanf(structure_file,"%lf",&(atom[j+i*3])); }
      if(feof(structure_file))
      { printf("ERROR: Not enough atoms in %s\n",argv[1]); MPI_Abort(MPI_COMM_WORLD,0); }
    }
    fclose(structure_file);
    if((solver == 9 || solver == 10 || solver - 1) && natom < 4)
    { printf("ERROR: solver = 9 needs 4 atomic coordinates to define crystal lattice vectors\n"); MPI_Abort(MPI_COMM_WORLD,0); }

    // only PEXSI actually uses multiple MPI processes, everything else should have one
    if(solver != 2 && mpisize > 1)
    { printf("ERROR: only one MPI process should be used for solver != 2\n"); MPI_Abort(MPI_COMM_WORLD,0); }

    // read the solver-dependent input variables
    switch(solver)
    {
      case 0:
      case 1:
      {
        // no solver parameters
      } break;

      case 2:
      {
        if(argc < 6)
        {
          printf("USAGE: <executable> <structure file> <fermi energy> <temperature> <solver> <#/2 of poles>\n");
          MPI_Abort(MPI_COMM_WORLD,0);
        }
        sscanf(argv[5],"%d",&napprox);
      } break;

      case 3:
      {
        if(argc < 7)
        {
          printf("USAGE: <executable> <structure file> <fermi energy> <temperature> <solver> <# of Cheby.> <res. tol.>\n");
          MPI_Abort(MPI_COMM_WORLD,0);
        }
        sscanf(argv[5],"%d",&napprox);
        sscanf(argv[6],"%lf",&res_tol);
      } break;

      case 4:
      {
        if(argc < 7)
        {
          printf("USAGE: <executable> <structure file> <fermi energy> <temperature> <solver> <#/2 of poles> <res. tol.>\n");
          MPI_Abort(MPI_COMM_WORLD,0);
        }
        sscanf(argv[5],"%d",&napprox);
        sscanf(argv[6],"%lf",&res_tol);
      } break;

      case 5:
      {
        if(argc < 8)
        {
          printf("USAGE: <executable> <structure file> <fermi energy> <temperature> <solver> <# of Cheby.> <res. tol.> "
                 "<loc. rad.>\n");
          MPI_Abort(MPI_COMM_WORLD,0);
        }
        sscanf(argv[5],"%d",&napprox);
        sscanf(argv[6],"%lf",&res_tol);
        sscanf(argv[7],"%lf",&local_radius);
      } break;

      case 6:
      case 9:
      {
        if(argc < 8)
        {
          printf("USAGE: <executable> <structure file> <fermi energy> <temperature> <solver> <#/2 of poles> <res. tol.> "
                 "<loc. rad.>\n");
          MPI_Abort(MPI_COMM_WORLD,0);
        }
        sscanf(argv[5],"%d",&napprox);
        sscanf(argv[6],"%lf",&res_tol);
        sscanf(argv[7],"%lf",&local_radius);
      } break;

      case 7:
      {
        if(argc < 10)
        {
          printf("USAGE: <executable> <structure file> <fermi energy> <temperature> <solver> <# of Cheby.> <res. tol.> "
                 "<loc. rad.> <seed> <# of samples>\n");
          MPI_Abort(MPI_COMM_WORLD,0);
        }
        sscanf(argv[5],"%d",&napprox);
        sscanf(argv[6],"%lf",&res_tol);
        sscanf(argv[7],"%lf",&local_radius);
        sscanf(argv[8],"%d",&seed);
        sscanf(argv[9],"%d",&nsample);
        if(seed <= 0)
        { printf("ERROR: PRNG seed must have positive nonzero value\n"); MPI_Abort(MPI_COMM_WORLD,0); }
      } break;

      case 8:
      {
        if(argc < 10)
        {
          printf("USAGE: <executable> <structure file> <fermi energy> <temperature> <solver> <#/2 of poles> <res. tol.> "
                 "<loc. rad.> <seed> <# of samples>\n");
          MPI_Abort(MPI_COMM_WORLD,0);
        }
        sscanf(argv[5],"%d",&napprox);
        sscanf(argv[6],"%lf",&res_tol);
        sscanf(argv[7],"%lf",&local_radius);
        sscanf(argv[8],"%d",&seed);
        sscanf(argv[9],"%d",&nsample);
        if(seed <= 0)
        { printf("ERROR: PRNG seed must have positive nonzero value\n"); MPI_Abort(MPI_COMM_WORLD,0); }
      } break;
      
      case 10:
      {
        if(argc < 7)
        {
          printf("USAGE: <executable> <structure file> <fermi energy> <temperature> <solver> "
                 "<# of k-grid pts. per dimension> <loc. rad.>\n");
          MPI_Abort(MPI_COMM_WORLD,0);
        }
        sscanf(argv[5],"%d",&napprox);
        sscanf(argv[6],"%lf",&local_radius);
      } break;

      case -1:
      {
        if(argc < 10)
        {
          printf("USAGE: <executable> <structure file> <fermi energy> <temperature> <solver> <pre. shift> <res. tol.> "
                 "<min. rad.> <max. rad.> <# rad.>\n");
          MPI_Abort(MPI_COMM_WORLD,0);
        }
        sscanf(argv[5],"%lf",&pre_shift);
        sscanf(argv[6],"%lf",&res_tol);
        sscanf(argv[7],"%lf",&pre_radius);
        sscanf(argv[8],"%lf",&local_radius);
        sscanf(argv[9],"%d",&nsample);
      } break;

      default:
      {
        printf("ERROR: unknown solver, %d is not contained in { -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 }\n",solver);
        MPI_Abort(MPI_COMM_WORLD,0);
      }
    }

    // convert from eV/Ang to Ry/Bohr
    potential /= E0;
    temperature /= E0;
    for(int i=0 ; i<3*natom ; i++) { atom[i] /= A0; }
    pre_shift /= E0;
    pre_radius /= A0;
    local_radius /= A0;

    // hard-coded tight-binding parameters for copper
    struct nrl_tb param = define_copper();

    // setup the ordered list of relevant lattice vectors
    int *latvec;
    double volume;
    if(solver == 9 || solver == 10 || solver == -1)
    {
      if(local_radius < param.Rcut)
      {
        printf("WARNING: local radius is too small & truncates central-cell Hamiltonian (increased to %lf)\n",param.Rcut*A0);
        local_radius = param.Rcut;
      }
      volume = cell_volume(atom);
      natom = latvec_list(local_radius,&latvec,&atom);
      printf("# of active atoms in crystal = %d\n",natom);
    }
    else
    { printf("# of atoms = %d\n",natom); }

    // setup the block-sparse Hamiltonian, overlap, density, & response matrices
    struct pattern sparsity;
    neighbor_list(natom,atom,param.Rcut,&sparsity);
    int nblock = 9, nnz = sparsity.col[sparsity.ncol];
    double **hamiltonian = (double**)malloc(sizeof(double*)*nnz);
    double **overlap = (double**)malloc(sizeof(double*)*nnz);
    double **density = (double**)malloc(sizeof(double*)*nnz);
    double **response = (double**)malloc(sizeof(double*)*nnz);
    if( solver == 9 || solver == 10 || solver == -1 )
    {
      crystal_malloc(nblock,&sparsity,latvec,hamiltonian);
      crystal_malloc(nblock,&sparsity,latvec,overlap);
      crystal_malloc(nblock,&sparsity,latvec,density);
      crystal_malloc(nblock,&sparsity,latvec,response);

      // fill in only the first column explicitly (to avoid recomputing redundant matrix elements)
      sparsity.ncol = 1;
      tb_matrix(natom,atom,&param,&sparsity,hamiltonian,overlap);
      sparsity.ncol = natom;

      // transposed copy to the first row, stored in memory after the first column
      int nmem = nblock*nblock*(sparsity.col[1]-1);
      for(int i=1 ; i<sparsity.col[1] ; i++)
      {
        for(int j=0 ; j<nblock ; j++)
        for(int k=0 ; k<nblock ; k++)
        {
          hamiltonian[i][nmem+k+j*nblock] = hamiltonian[i][j+k*nblock];
          overlap[i][nmem+k+j*nblock] = overlap[i][j+k*nblock];
        }
      }
    }
    else
    {
      hamiltonian[0] = (double*)malloc(sizeof(double)*nnz*nblock*nblock);
      overlap[0] = (double*)malloc(sizeof(double)*nnz*nblock*nblock);
      density[0] = (double*)malloc(sizeof(double)*nnz*nblock*nblock);
      response[0] = (double*)malloc(sizeof(double)*nnz*nblock*nblock);
      for(int i=1 ; i<nnz ; i++)
      {
        hamiltonian[i] = &(hamiltonian[i-1][nblock*nblock]);
        overlap[i] = &(overlap[i-1][nblock*nblock]);
        density[i] = &(density[i-1][nblock*nblock]);
        response[i] = &(response[i-1][nblock*nblock]);
      }
      tb_matrix(natom,atom,&param,&sparsity,hamiltonian,overlap);
    }
    printf("H & S sparsity = %lf\n",(double)sparsity.col[sparsity.ncol]/(double)sparsity.ncol);

    // setup the sparsity pattern for a local region
    struct pattern locality;
    if(solver == 5 || solver == 6)
    {
      neighbor_list(natom,atom,local_radius,&locality);
      printf("local sparsity = %lf\n",(double)locality.col[locality.ncol]/(double)locality.ncol);
    }

    // greedy coloring of adjacency matrix to define uncorrelated complex rotor ensemble
    int ncolor, *color, *atom_ptr;
    if(solver == 7 || solver == 8)
    {
      neighbor_list(natom,atom,local_radius,&locality);
      color_graph(&locality,&ncolor,&color,&atom_ptr);
      printf("total number of atom colors = %d\n",ncolor);
      free_pattern(&locality);
    }

    // reasonable energy interval based on bulk copper calculations in the Julich tight-binding code
    double min_energy = -10.0/E0, max_energy = 32.0/E0, approximation_error = 0.0;
    double *pcoeff;
    double complex *w, *z;

    // fit the Fermi-Dirac polynomial approximation (Chebyshev interpolation then truncation)
    double hwt = 2.0/(max_energy - min_energy);
    double owt = -(max_energy + min_energy)/(max_energy - min_energy);
    if(solver == 3 || solver == 5 || solver == 7)
    {
      pcoeff = (double*)malloc(sizeof(double)*napprox);
      approximation_error = polynomial_approximation(napprox,min_energy,max_energy,potential,temperature,pcoeff);
      for(int i=0 ; i<napprox ; i++) { pcoeff[i] *= 2.0; } // spin degeneracy factor
      printf("approximation error (%d Chebyshev polynomials) = %e\n",napprox,2.0*approximation_error);
    }

    // parse the Fermi-Dirac rational approximation table (read from a precomputed table of approximations)
    if(solver == 2 || solver == 4 || solver == 6 || solver == 8 || solver == 9)
    {
      w = (double complex*)malloc(sizeof(double complex)*napprox);
      z = (double complex*)malloc(sizeof(double complex)*napprox);
      approximation_error = rational_approximation(napprox,min_energy,potential,temperature,w,z);
      for(int i=0 ; i<napprox ; i++) { w[i] *= 2.0; } // spin degeneracy factor
      printf("approximation error (%d pole pairs) = %e\n",napprox,2.0*approximation_error);
    }

    // solver-dependent inner loop
    switch(solver)
    {
      // no solver: fill density & response w/ overlap    
      case 0:
      {
        printf("no solver (pre-processing & post-processing only)\n");
        copy_mat(nblock,&sparsity,overlap,density);
        copy_mat(nblock,&sparsity,overlap,response);
      } break;

      // reference solver: dense matrix diagonalization
      case 1:
      {
        printf("LAPACK solver\n");
        dense_solver(nblock,potential,temperature,min_energy,max_energy,&sparsity,hamiltonian,overlap,density,response);
      } break;

      // PEXSI-based rational-approximation solver (quadratic scaling in 3D)
      // NOTE: only mpirank == 0 enters the PEXSI solver here, the other ranks enter near the end of main
      case 2:
      {
        printf("PEXSI solver\n");
        PEXSI_solver(mpirank,mpisize,nblock,napprox,w,z,&sparsity,hamiltonian,overlap,density,response);
      } break;

      // quadratic-scaling polynomial-approximation solver
      case 3:
      {
        printf("polynomial solver\n");
        quad_poly_solver(nblock,napprox,res_tol,hwt,owt,pcoeff,&sparsity,hamiltonian,overlap,density,response);
      } break;

      // quadratic-scaling rational-approximation solver
      case 4:
      {
        printf("rational solver\n");
        quad_rational_solver(nblock,napprox,res_tol,w,z,&sparsity,hamiltonian,overlap,density,response);
      } break;

      // local polynomial-approximation solver
      case 5:
      {
        printf("localized polynomial solver\n");
        local_poly_solver(nblock,napprox,res_tol,hwt,owt,pcoeff,&locality,&sparsity,hamiltonian,overlap,density,response);
      } break;

      // local rational-approximation solver
      case 6:
      {
        printf("localized rational solver\n");
        local_rational_solver(nblock,napprox,res_tol,w,z,&locality,&sparsity,hamiltonian,overlap,density,response);
      } break;

      // random polynomial-approximation solver
      case 7:
      {
        printf("randomized polynomial solver\n");
        random_poly_solver(nblock,napprox,ncolor,nsample,seed,res_tol,hwt,owt,pcoeff,color,atom_ptr,&sparsity,hamiltonian,
                           overlap,density,response);
      } break;

      // random rational-approximation solver
      case 8:
      {
        printf("randomized rational solver\n");
        random_rational_solver(nblock,napprox,ncolor,nsample,seed,res_tol,w,z,color,atom_ptr,&sparsity,hamiltonian,overlap,
                               density,response);
      } break;

      // infinite rational-approximation solver
      case 9:
      {
        printf("infinite rational solver\n");
        infinite_rational_solver(nblock,napprox,res_tol,atom,w,z,&sparsity,hamiltonian,overlap,density,response);
      } break;

      // infinite reciprocal-space solver
      case 10:
      {
        printf("infinite k-space solver\n");
        infinite_reciprocal_solver(nblock,potential,temperature,min_energy,max_energy,napprox,latvec,atom,&sparsity,hamiltonian,
                                   overlap,density,response);
      } break;

      // infinite localization tester
      case -1:
      {
        printf("infinite preconditioning tester\n");
        double complex z0 = potential + I*M_PI*temperature;
        double complex z1 = potential + I*pre_shift;
        infinite_pre_tester(nblock,nsample,pre_radius,local_radius,res_tol,z0,z1,&sparsity,atom,latvec,hamiltonian,overlap);
      } break;

      default:
      {
        printf("ERROR: unknown solver, %d is not contained in { -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 }\n",solver);
        MPI_Abort(MPI_COMM_WORLD,0);
      }
    }

    // solver-independent energy, number, & force calculations on block density & response matrices
    if(solver == 9 || solver == 10) { sparsity.ncol = 1; } // observable contributions only from the central cell of crystals
    double number = dot_mat(nblock,&sparsity,density,overlap);
    double energy = dot_mat(nblock,&sparsity,density,hamiltonian);
    double *force = (double*)malloc(sizeof(double)*3*natom);
    double *hblock_force = (double*)malloc(sizeof(double)*3*nblock*nblock);
    double *oblock_force = (double*)malloc(sizeof(double)*3*nblock*nblock);
    for(int i=0 ; i<3*natom ; i++) { force[i] = 0.0; }
    for(int i=0 ; i<sparsity.ncol ; i++)
    for(int j=sparsity.col[i] ; j<sparsity.col[i+1] ; j++)
    {
      // diagonal force contributions
      if(i == sparsity.row[j])
      {
        // contributes to all terms that are neighbors of atom i
        for(int k=sparsity.col[i] ; k<sparsity.col[i+1] ; k++)
        {
          tb_diagonal_force(i,sparsity.row[k],natom,atom,sparsity.col[i+1]-sparsity.col[i],
                            &(sparsity.row[sparsity.col[i]]),&param,hblock_force);
          for(int l=0 ; l<3 ; l++)
          for(int m=0 ; m<nblock*nblock ; m++)
          { force[l+sparsity.row[k]*3] += density[j][m]*hblock_force[m+l*nblock*nblock]; }
        }
      }
      else // off-diagonal force contributions (general case not assuming numerical symmetry of density & response matrices)
      {
        tb_offdiagonal_force(i,sparsity.row[j],natom,atom,&param,hblock_force,oblock_force);
        for(int k=0 ; k<3 ; k++)
        for(int l=0 ; l<nblock*nblock ; l++)
        {
          force[k+i*3] += density[j][l]*hblock_force[l+k*nblock*nblock] + response[j][l]*oblock_force[l+k*nblock*nblock];
        }

        tb_offdiagonal_force(sparsity.row[j],i,natom,atom,&param,hblock_force,oblock_force);
        for(int k=0 ; k<3 ; k++)
        for(int l=0 ; l<nblock ; l++)
        for(int m=0 ; m<nblock ; m++)
        {
          force[k+sparsity.row[j]*3] += density[j][m+l*nblock]*hblock_force[l+(m+k*nblock)*nblock]
                                       + response[j][m+l*nblock]*oblock_force[l+(m+k*nblock)*nblock];
        }
      }
    }

    // physical outputs
    if(solver != 0 && solver != -1)
    {
      printf("number = %16.16e\n",number);
      printf("energy = %16.16e\n",energy*E0);
    }
    if(solver == 9 || solver == 10)
    {
      double force0[3], stress[9];
      for(int i=0 ; i<3 ; i++) { force0[i] = 0.0; }
      for(int i=0 ; i<9 ; i++) { stress[i] = 0.0; }
      for(int i=0 ; i<natom ; i++)
      {
        for(int j=0 ; j<3 ; j++)
        { force0[j] += force[j+i*3]; }
        for(int j=0 ; j<3 ; j++)
        for(int k=0 ; k<3 ; k++)
        { stress[k+j*3] += atom[k+i*3]*force[j+i*3]; }
      }
      for(int i=0 ; i<9 ; i++) { stress[i] /= volume; }
      printf("force = { %16.16e , %16.16e , %16.16e }\n",force0[0]*E0/A0,force0[1]*E0/A0,force0[2]*E0/A0);
      printf("stress = { %16.16e , %16.16e , %16.16e }\n",stress[0]*P0,stress[1]*P0,stress[2]*P0);
      printf("         { %16.16e , %16.16e , %16.16e }\n",stress[3]*P0,stress[4]*P0,stress[5]*P0);
      printf("         { %16.16e , %16.16e , %16.16e }\n",stress[6]*P0,stress[7]*P0,stress[8]*P0);
    }
    else if(solver != 0 && solver != -1)
    {
      for(int i=0 ; i<natom ; i++)
      { printf("force[%d] = { %16.16e , %16.16e , %16.16e }\n",i,force[0+3*i]*E0/A0,force[1+3*i]*E0/A0,force[2+3*i]*E0/A0); }
    }

    // final timing point
    double time2 = omp_get_wtime();
    printf("total time usage = %e s\n",time2-time1);

    // print density & response matrices to a debug file (1st block column only for periodic systems)
    if(solver != 0 && solver != -1)
    {
      FILE *debug_file = fopen("debug.out","w");
      fprintf(debug_file,"%d\n",sparsity.col[sparsity.ncol]*nblock*nblock);
      int index = 0;
      for(int i=0 ; i<sparsity.col[sparsity.ncol] ; i++)
      {
        for(int j=0 ; j<nblock*nblock ; j++)
        {
          fprintf(debug_file,"%d %16.16e %16.16e\n",index++,density[i][j],response[i][j]);
        }
      }
      fclose(debug_file);
    }

    // deallocate remaining memory
    free(oblock_force);
    free(hblock_force);
    free(force);
    if(solver == 2 || solver == 4 || solver == 6 || solver == 8)
    { free(w); free(z); }
    if(solver == 3 || solver == 5 || solver == 7)
    { free(pcoeff); }
    if(solver == 7 || solver == 8)
    { free(color); free(atom_ptr); }
    if(solver == 5 || solver == 6)
    { free_pattern(&locality); }
    free(response[0]); free(response);
    free(density[0]); free(density);
    free(overlap[0]); free(overlap);
    free(hamiltonian[0]); free(hamiltonian);
    free_pattern(&sparsity);
    if(solver == 9 || solver == 10 || solver == -1)
    { free(latvec); }
    free(atom);
  }
  else // mpirank != 0 branch of the main program
  {
    // PEXSI-based rational-approximation solver (quadratic scaling in 3D)
    // NOTE: mpirank != 0 enter the PEXSI solver here, mpirank == 0 enters above
    PEXSI_solver(mpirank,mpisize,0,0,NULL,NULL,NULL,NULL,NULL,NULL,NULL);
  }

  // total system resource usage statistics
  struct rusage my_usage;
  long int global_memory;
  getrusage(RUSAGE_SELF,&my_usage);
  MPI_Reduce(&(my_usage.ru_maxrss),&global_memory,1,MPI_LONG,MPI_SUM,0,MPI_COMM_WORLD);
  if(mpirank == 0)
  { printf("total memory usage = %ld kb\n",global_memory); }

  // normal MPI termination
  MPI_Finalize();

  return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////
////////////////////////////////
////////////////
////////
////
//
