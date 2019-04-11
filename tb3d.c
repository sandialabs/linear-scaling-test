// Tests of new linear-scaling electronic structure algorithms on a simple-cubic tight-binding model

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <complex.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

// There are three sets of analytical indices,
// a position index, (x,y,z), centered around (0,0,0) for geometric bookkeeping
// an intermediate index, (s,c,r), in terms of "shell", "cut", and "ring" sub-indices
// and a single shell-ordered global index, g, for matrix bookkeeping

// position -> intermediate transformation
void xyz2scr(int x, int y, int z, int *s, int *c, int *r)
{
  // s = |x| + |y| + |z|, s in [0,s_max]
  *s = abs(x) + abs(y) + abs(z);

  // c = z + s, c in [0,2*s]
  *c = z + *s;

  // w := s - |c-s|, "width" of the ring is a useful intermediate
  int w = *s - abs(*c-*s);

  // r = w + y if x <= 0, r in [0,2*w]
  // r = 3*w - y if x > 0, r in [2*w+1,4*w-1]
  if(x <= 0)
  { *r = w + y; }
  else
  { *r = 3*w - y; }
}

// intermediate -> position transformation
void scr2xyz(int s, int c, int r, int *x, int *y, int *z)
{
  // z = c - s
  *z = c - s;

  // conditional inversion of r formula
  int w = s - abs(c-s);
  if(r <= 2*w)
  { *y = r - w; }
  else
  { *y = 3*w - r; }

  // |x| = s - |y| - |z|
  // set sign of x from r value
  *x = (s - abs(*y) - abs(*z));
  if(r <= 2*w)
  { *x = -*x; }
}

// convenient & fast way to update shell & global indices
void shell_iterator(int *g, int *s, int *c, int *r)
{
  (*g)++;
  int w = *s - abs(*c-*s);
  if(*r < 4*w-1)
  { (*r)++; }
  else
  {
    (*r) = 0;
    if(*c < 2*(*s))
    { (*c)++; }
    else
    {
      (*c) = 0;
      (*s)++;
    }
  }
}

// sum up # of sites in completed cuts
int c_sum(int c, int s)
{
  if(c > 0 && c <= s)
  { return 1 + 2*c*(c-1); }
  if(c > s)
  { return 1 + 4*s*s - 2*(2*s-c)*(2*s-c+1); }
  return 0;
}

// invert s_sum w/ bisection search
int c_invert(int g, int s)
{
  int c_min = 0, c_max = 2*s;
  int g_min = c_sum(c_min,s), g_max = c_sum(c_max,s);
  while(c_max - c_min > 1)
  {
    int c_new = (c_min + c_max)/2;
    int g_new = c_sum(c_new,s);
    if(g_new == g)
    { return c_new; }
    if(g_new < g)
    { c_min = c_new; g_min = g_new; }
    if(g_new > g)
    { c_max = c_new; g_max = g_new; }
  }
  if(g_max == g)
  { return c_max; }
  return c_min;
}

// sum up # of sites in completed shells
int s_sum(int s)
{
  if(s > 0)
  { return 2*s-1 + (2*s*(s-1)*(2*s-1))/3; }
  return 0;
}

// invert s_sum w/ bisection search
int s_invert(int g)
{
  int s_min = 0, s_max = 1;
  int g_min = s_sum(s_min), g_max = s_sum(s_max);
  while(g_max <= g)
  {
    s_min = s_max;
    g_min = g_max;
    s_max *= 2;
    g_max = s_sum(s_max);
  }
  if(g_min == g)
  { return s_min; }
  while(s_max - s_min > 1)
  {
    int s_new = (s_min + s_max)/2;
    int g_new = s_sum(s_new);
    if(g_new == g)
    { return s_new; }
    if(g_new < g)
    { s_min = s_new; g_min = g_new; }
    if(g_new > g)
    { s_max = s_new; g_max = g_new; }
  }
  return s_min;
}

// intermediate -> global transformation
void scr2g(int s, int c, int r, int *g)
{
  *g = s_sum(s) + c_sum(c,s) + r;
}

// global -> intermediate transformation
// NOTE: this is slow because of the index inversion step
void g2scr(int g, int *s, int *c, int *r)
{
  // extract shell
  *s = s_invert(g);
  g -= s_sum(*s);

  // extract cut
  *c = c_invert(g,*s);
  g -= c_sum(*c,*s);

  // only ring index remains
  *r = g;
}

// combined indices
void xyz2g(int x, int y, int z, int *g)
{
  int s, c, r;
  xyz2scr(x,y,z,&s,&c,&r);
  scr2g(s,c,r,g);
}
void g2xyz(int g, int *x, int *y, int *z)
{
  int s, c, r;
  g2scr(g,&s,&c,&r);
  scr2xyz(s,c,r,x,y,z);
}

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

// pseudorandom Gaussian random variable
double complex random_gaussian()
{
  double r1 = random_uniform(), r2 = random_uniform();
  return sqrt(fabs(log(r1)))*(cos(2.0*M_PI*r2) + I*sin(2.0*M_PI*r2));
}

// calculate the 2-norm distance per shell between two Green's functions
void norm2(int nshell, // number of shells
           double complex *green1, // 1st Green's function [s_sum(nshell)]
           double complex *green2, // 2nd Green's function [s_sum(nshell)]
           double *shell_norm) // 2-norm distance per shell & overall [nshell]
{
  for(int i=0 ; i<nshell ; i++)
  {
    shell_norm[i] = 0.0;
    int shell_min = s_sum(i), shell_max = s_sum(i+1);
    for(int j=shell_min ; j<shell_max ; j++)
    {
      shell_norm[i] += pow(cabs(green1[j]-green2[j]),2);
    }
  }

  for(int i=0 ; i<nshell ; i++) { shell_norm[i] = sqrt(shell_norm[i]); }
}

// Brillouin-zone calculation of the Green's function (efficient, high-accuracy reference)
void reciprocal_green(double complex shift, // chemical potential (real part) & temperature (imaginary part)
                      int nkpt, // number of k-points to sum over in each dimension
                      int nshell, // number of shells
                      double complex *green) // Green's function [s_sum(nshell)]
{
  int nsite = s_sum(nshell);
  for(int i=0 ; i<nsite ; i++)
  { green[i] = 0.0; }

  // Brillouin zone summation
  double wt = 1.0/(double)pow(nkpt,3);
  for(int i=0 ; i<nkpt ; i++)
  {
    double xphase = 2.0*M_PI*(double)i/(double)nkpt;
    for(int j=0 ; j<nkpt ; j++)
    {
      double yphase = 2.0*M_PI*(double)j/(double)nkpt;
      for(int k=0 ; k<nkpt ; k++)
      {
        double zphase = 2.0*M_PI*(double)k/(double)nkpt;

        // energy denominator
        double energy = -2.0*(cos(xphase) + cos(yphase) + cos(zphase));
        double complex green0 = wt/(energy - shift);

        // lattice summation
        int l=0, s=0, c=0, r=0;
        while(l<nsite)
        {
          int x, y, z;
          scr2xyz(s,c,r,&x,&y,&z);
          green[l] += green0*cexp(I*(x*xphase + y*yphase + z*zphase));
          shell_iterator(&l,&s,&c,&r);
        }
      }
    }
  }
}

// localized banded-matrix calculation of the Green's function
void local_green(double complex shift, // chemical potential (real part) & temperature (imaginary part)
                 int nshell, // number of shells
                 double complex *green) // Green's function [s_sum(nshell)]
{
  // allocate memory for banded matrix
  int dim = s_sum(nshell), bandwidth = s_sum(nshell) - s_sum(nshell-1);
  int nrhs = 1, ld = 3*bandwidth+1, info;
  int *ipiv = (int*)malloc(sizeof(int)*dim);
  double complex *hamiltonian = (double complex*)malloc(sizeof(double complex)*dim*ld);

  // set the right-hand-side vector
  green[0] = 1.0;
  for(int i=1 ; i<dim ; i++)
  { green[i] = 0.0; }

  // construct banded matrix w/ LAPACK indexing: AB(KL+KU+1+i-j,j) = A(i,j) for max(1,j-KU)<=i<=min$
  for(int j=0 ; j<dim ; j++)
  {
    int x1, y1, z1;
    g2xyz(j,&x1,&y1,&z1);
    for(int i=0 ; i<ld ; i++)
    {
      int irow = i - 2*bandwidth + j, x2, y2, z2;
      g2xyz(irow,&x2,&y2,&z2);
      hamiltonian[i+j*ld] = 0.0;
      if(irow >= 0 && irow < dim)
      {
        int distance = abs(x1-x2) + abs(y1-y2) + abs(z1-z2);
        if(distance == 0) { hamiltonian[i+j*ld] = -shift; }
        if(distance == 1) { hamiltonian[i+j*ld] = -1.0; }
      }
    }
  }

  // call LAPACK
  zgbsv(&dim,&bandwidth,&bandwidth,&nrhs,hamiltonian,&ld,ipiv,green,&dim,&info);

  // deallocate memory
  free(hamiltonian);
  free(ipiv);
}

// least-squares banded-matrix calculation of the Green's function using normal equations
void square_green(double complex shift, // chemical potential (real part) & temperature (imaginary part)
                  int nshell, // number of shells
                  double complex *green) // Green's function [s_sum(nshell)]
{
  // allocate memory for banded matrix
  int dim = s_sum(nshell), bandwidth = s_sum(nshell) - s_sum(nshell-2);
  int nrhs = 1, ld = 3*bandwidth+1, info;
  int *ipiv = (int*)malloc(sizeof(int)*dim);
  double complex *hamiltonian = (double complex*)malloc(sizeof(double complex)*dim*ld);

  // set the right-hand-side vector for the Green's function
  green[0] = -conj(shift);
  green[1] = green[2] = green[3] = green[4] = green[5] = green[6] = -1.0;
  for(int i=7 ; i<dim ; i++)
  { green[i] = 0.0; }

  // construct banded Hamiltonian matrix w/ LAPACK indexing: AB(KL+KU+1+i-j,j) = A(i,j) for max(1,j-KU)<=i<=min$
  for(int j=0 ; j<dim ; j++)
  {
    int x1, y1, z1;
    g2xyz(j,&x1,&y1,&z1);
    for(int i=0 ; i<ld ; i++)
    {
      int irow = i - 2*bandwidth + j, x2, y2, z2;
      g2xyz(irow,&x2,&y2,&z2);
      hamiltonian[i+j*ld] = 0.0;
      if(irow >= 0 && irow < dim)
      {
        int distance = abs(x1-x2) + abs(y1-y2) + abs(z1-z2);
        if(distance == 0) { hamiltonian[i+j*ld] = pow(cabs(shift),2) + 6.0; }
        if(distance == 1) { hamiltonian[i+j*ld] = 2.0*creal(shift); }
        if(distance == 2)
        {
          // crude hack for # of double-hopping paths between sites
          if(abs(x1-x2) == 2 || abs(y1-y2) == 2 || abs(z1-z2) == 2)
          { hamiltonian[i+j*ld] = 1.0; }
          else
          { hamiltonian[i+j*ld] = 2.0; }
        }
      }
    }
  }

  // call LAPACK
  zgbsv(&dim,&bandwidth,&bandwidth,&nrhs,hamiltonian,&ld,ipiv,green,&dim,&info);

  // deallocate memory
  free(hamiltonian);
  free(ipiv);
}

// randomly coarse-grained banded-matrix calculation of the Green's function
void random_green(double complex shift, // chemical potential (real part) & temperature (imaginary part)
                  int nshell, // number of shells
                  int nshell2, // number of coarse-grained shells
                  double complex *green) // Green's function [s_sum(nshell)]
{
  // allocate memory for banded matrix
  int dim = s_sum(nshell), nblock = s_sum(nshell) - s_sum(nshell-1), bandwidth = 2*nblock;
  dim += nblock*(nshell2-nshell);
  int nrhs = 1, ld = 3*bandwidth+1, info;
  int *ipiv = (int*)malloc(sizeof(int)*dim);
  double complex *hamiltonian = (double complex*)malloc(sizeof(double complex)*dim*ld);
  double complex *green2 = (double complex*)malloc(sizeof(double complex)*dim);
  double complex *coarse = (double complex*)malloc(sizeof(double complex)*bandwidth*(s_sum(nshell2) - s_sum(nshell2-1)));
  double complex *coarse_old = (double complex*)malloc(sizeof(double complex)*bandwidth*(s_sum(nshell2) - s_sum(nshell2-1)));
  double complex *coarse_expand = (double complex*)malloc(sizeof(double complex)*bandwidth*(s_sum(nshell2) - s_sum(nshell2-1)));

  // clear the Hamiltonian matrix
  for(int i=0 ; i<dim*ld ; i++)
  { hamiltonian[i] = 0.0; }

  // set the right-hand-side vector
  green2[0] = 1.0;
  for(int i=1 ; i<dim ; i++)
  { green2[i] = 0.0; }

  // set the deterministic block of the Hamiltonian matrix
  int dim0 = s_sum(nshell);
  for(int j=0 ; j<dim0 ; j++)
  {
    int x1, y1, z1;
    g2xyz(j,&x1,&y1,&z1);
    for(int i=0 ; i<ld ; i++)
    {
      int irow = i - 4*nblock + j, x2, y2, z2;
      if(irow >= 0)
      {
        g2xyz(irow,&x2,&y2,&z2);
        if(irow < dim0)
          {
          int distance = abs(x1-x2) + abs(y1-y2) + abs(z1-z2);
          if(distance == 0) { hamiltonian[i+j*ld] = -shift; }
          if(distance == 1) { hamiltonian[i+j*ld] = -1.0; }
        }
      }
    }
  }

  // set the initial shell basis
  for(int i=0 ; i<nblock*nblock ; i++)
  { coarse_old[i] = 0.0; }
  for(int i=0 ; i<nblock ; i++)
  { coarse_old[i+i*nblock] = 1.0; }

  // build coarsened Hamiltonian, shell-by-shell
  for(int i=nshell ; i<nshell2 ; i++)
  {
    int col_offset = s_sum(nshell) + (i-nshell)*nblock;

    // construct the new coarse basis
    int offset = s_sum(i);
    int size = s_sum(i+1) - s_sum(i);
    double complex wt = 1.0/sqrt(nblock);
    for(int j=0 ; j<nblock*size ; j++)
    {
      coarse[j] = wt*random_gaussian();
      coarse_expand[j] = 0.0;
    }

    // project off-diagonal block
    int old_offset = s_sum(i-1);
    int old_size = s_sum(i) - s_sum(i-1);
    for(int j=0 ; j<old_size ; j++) // applying hopping matrix to "expand" previous shell basis
    {
      int x1, y1, z1;
      g2xyz(old_offset+j,&x1,&y1,&z1);
      for(int k=0 ; k<size ; k++)
      {
        int x2, y2, z2;
        g2xyz(offset+k,&x2,&y2,&z2);
        if( (abs(x1-x2) + abs(y1-y2) + abs(z1-z2)) == 1)
        {
          for(int l=0 ; l<bandwidth ; l++)
          { coarse_expand[k+l*size] += coarse_old[j+l*old_size]; }
        }
      }
    }
    char transa = 'C', transb = 'N';
    double complex wt2 = 0.0;
    wt = -1.0;
    zgemm(&transa,&transb,&nblock,&nblock,&size,&wt,coarse_expand,&size,coarse,&size,&wt2,coarse_old,&nblock); // construct off-diagonal block, coarse_old is now a workspace
    for(int j=0 ; j<nblock ; j++) // insert off-diagonal block into the banded matrix format
    {
      for(int k=0 ; k<nblock ; k++)
      {
        hamiltonian[2*bandwidth - nblock + k-j + ld*(j+col_offset)] = coarse_old[k+j*nblock];
        hamiltonian[2*bandwidth + nblock + j-k + ld*(k+col_offset-nblock)] = conj(coarse_old[k+j*nblock]);
      }
    }

    // project & invert diagonal block
    wt = -1.0/shift;
    zgemm(&transa,&transb,&nblock,&nblock,&size,&wt,coarse,&size,coarse,&size,&wt2,coarse_expand,&nblock);
    for(int j=0 ; j<nblock*nblock ; j++)
    { coarse_old[j] = 0.0; }
    for(int j=0 ; j<nblock ; j++)
    { coarse_old[j+j*nblock] = 1.0; }
    zgesv(&nblock,&nblock,coarse_expand,&nblock,ipiv,coarse_old,&nblock,&info);
    for(int j=0 ; j<nblock ; j++) // insert off-diagonal block into the banded matrix format
    {
      for(int k=0 ; k<nblock ; k++)
      {
        hamiltonian[2*bandwidth + k-j + ld*(j+col_offset)] = coarse_old[k+j*nblock];
      }
    }

    // swap new & old coarse bases
    double complex *ptr_temp = coarse_old; coarse_old = coarse; coarse = ptr_temp;
  }

  // call LAPACK
  zgbsv(&dim,&bandwidth,&bandwidth,&nrhs,hamiltonian,&ld,ipiv,green2,&dim,&info);

  // deallocate memory
  free(hamiltonian);
  free(ipiv);

  // copy solution to green
  int nsite = s_sum(nshell);
  for(int i=0 ; i<nsite ; i++)
  { green[i] = green2[i]; }
  free(green2); free(coarse); free(coarse_old); free(coarse_expand);
}

// calculate the 1st-order perturbative Green's function correction: G - G*(H - shift*I)*G
void perturb_green(double complex shift, // chemical potential (real part) & temperature (imaginary part)
                   int nshell, // number of shells
                   double complex *green, // Green's function [s_sum(nshell)]
                   double complex *dgreen) // Green's function perturbative correction [s_sum(nshell)]
{
  int nsite = s_sum(nshell);

  // dG[i,0] = G[i,0]
  for(int i=0 ; i<nsite ; i++)
  { dgreen[i] = green[i]; }

  // dG[i,0] -= G[i,k]*(H - shift*I)[k,j]*G[j,0]
  int i=0, s1=0, c1=0, r1=0;
  while(i < nsite)
  {
    int x1, y1, z1;
    scr2xyz(s1,c1,r1,&x1,&y1,&z1);

    int j=0, s2=0, c2=0, r2=0;
    while(j < nsite)
    {
      int x2, y2, z2;
      scr2xyz(s2,c2,r2,&x2,&y2,&z2);

      int k;
      xyz2g(x1-x2,y1-y2,z1-z2,&k);
      if(k < nsite) { dgreen[i] += shift*green[k]*green[j]; }

      xyz2g(x1-x2+1,y1-y2,z1-z2,&k);
      if(k < nsite) { dgreen[i] += green[k]*green[j]; }

      xyz2g(x1-x2-1,y1-y2,z1-z2,&k);
      if(k < nsite) { dgreen[i] += green[k]*green[j]; }

      xyz2g(x1-x2,y1-y2+1,z1-z2,&k);
      if(k < nsite) { dgreen[i] += green[k]*green[j]; }

      xyz2g(x1-x2,y1-y2-1,z1-z2,&k);
      if(k < nsite) { dgreen[i] += green[k]*green[j]; }

      xyz2g(x1-x2,y1-y2,z1-z2+1,&k);
      if(k < nsite) { dgreen[i] += green[k]*green[j]; }

      xyz2g(x1-x2,y1-y2,z1-z2-1,&k);
      if(k < nsite) { dgreen[i] += green[k]*green[j]; }

      shell_iterator(&j,&s2,&c2,&r2);
    }

    shell_iterator(&i,&s1,&c1,&r1);
  }
}

// calculate the residual of a trial Green's function
double residual(double complex shift, // chemical potential (real part) & temperature (imaginary part)
                int nshell, // number of shells
                double complex *green, // Green's function [s_sum(nshell)]
                double complex *dgreen) // workspace for residual matrix [s_sum(nshell)]
{
  int nsite = s_sum(nshell);
  perturb_green(shift,nshell,green,dgreen);
  double res = 0.0;
  for(int i=0 ; i<nsite ; i++)
  { res += pow(cabs(dgreen[i]),2); }
  return sqrt(res);
}

// analytical derivatives of the residual function w/ Green's function components
void dresidual(double complex shift, // chemical potential (real part) & temperature (imaginary part)
               int nshell, // number of shells
               double complex *green, // Green's function [s_sum(nshell)]
               double complex *dgreen, // residual matrix [s_sum(nshell)]
               double complex *dres) // residual derivatives [s_sum(nshell)]
{
  int nsite = s_sum(nshell);

  // dR[i,0] = dG*[i,0]
  for(int i=0 ; i<nsite ; i++)
  { dres[i] = conj(dgreen[i]); }

  // dG[i,0] -= dG*[i,k]*(H - shift*I)[k,j]*G[j,0] + G*[i,k]*(H - shift*I)[k,j]*(dG*)[j,0]
  int i=0, s1=0, c1=0, r1=0;
  while(i < nsite)
  {
    int x1, y1, z1;
    scr2xyz(s1,c1,r1,&x1,&y1,&z1);

    int j=0, s2=0, c2=0, r2=0;
    while(j < nsite)
    {
      int x2, y2, z2;
      scr2xyz(s2,c2,r2,&x2,&y2,&z2);

      int k;
      xyz2g(x1-x2,y1-y2,z1-z2,&k);
      if(k < nsite) { dres[i] += shift*(conj(dgreen[k])*green[j] + green[k]*conj(dgreen[j])); }

      xyz2g(x1-x2+1,y1-y2,z1-z2,&k);
      if(k < nsite) { dres[i] += conj(dgreen[k])*green[j] + green[k]*conj(dgreen[j]); }

      xyz2g(x1-x2-1,y1-y2,z1-z2,&k);
      if(k < nsite) { dres[i] += conj(dgreen[k])*green[j] + green[k]*conj(dgreen[j]); }

      xyz2g(x1-x2,y1-y2+1,z1-z2,&k);
      if(k < nsite) { dres[i] += conj(dgreen[k])*green[j] + green[k]*conj(dgreen[j]); }

      xyz2g(x1-x2,y1-y2-1,z1-z2,&k);
      if(k < nsite) { dres[i] += conj(dgreen[k])*green[j] + green[k]*conj(dgreen[j]); }

      xyz2g(x1-x2,y1-y2,z1-z2+1,&k);
      if(k < nsite) { dres[i] += conj(dgreen[k])*green[j] + green[k]*conj(dgreen[j]); }

      xyz2g(x1-x2,y1-y2,z1-z2-1,&k);
      if(k < nsite) { dres[i] += conj(dgreen[k])*green[j] + green[k]*conj(dgreen[j]); }

      shell_iterator(&j,&s2,&c2,&r2);
    }

    shell_iterator(&i,&s1,&c1,&r1);
  }

  // final reweighting & sign change
  double res = 0.0;
  for(int i=0 ; i<nsite ; i++)
  { res += pow(cabs(dgreen[i]),2); }
  res = -1.0/sqrt(res);
  for(int i=0 ; i<nsite ; i++)
  { dres[i] *= res; }

  // conjugate
  for(int i=0 ; i<nsite ; i++)
  { dres[i] = conj(dres[i]); }
}

// Lagrange polynomial of the residual function from 5 function evaluations
double res_func(double x, double *res)
{
  double func = 0.0;
  for(int i=0 ; i<5 ; i++)
  {
    double lagrange = res[i];
    for(int j=0 ; j<5 ; j++)
    {
      if(i != j)
      { lagrange *= (x - 0.25*j)/(0.25*i - 0.25*j); }
    }
    func += lagrange;
  }
  return func;
}

// line minimization of the Green's function residual
double line_min(double complex shift, // chemical potential (real part) & temperature (imaginary part)
                int nshell, // number of shells
                double complex *green, // Green's function [s_sum(nshell)]
                double complex *dgreen, // residual matrix [s_sum(nshell)]
                double complex *dres) // residual derivatives [s_sum(nshell)]
{
  int nsite = s_sum(nshell);

  // calculate residual @ 5 points for polynomial
  double res[5];
  for(int i=0 ; i<5 ; i++)
  {
    res[i] = pow(residual(shift,nshell,green,dgreen),2);
    if(i != 4)
    {
      for(int j=0 ; j<nsite ; j++)
      { green[j] += 0.25*dres[j]; }
    }
  }

  // bracket a minimum
  double xleft = 0.0, xmid = 0.5, xright = 1.0;
  double rleft = res_func(xleft,res), rmid = res_func(xmid,res), rright = res_func(xright,res);
  while(rmid > rleft)
  {
    xmid *= 0.5;
    rmid = res_func(xmid,res);
  }
  while(rright < rleft)
  {
    xright *= 2.0;
    rright = res_func(xright,res);
  }
  // find the minimum (bisection of brackets)
  do
  {
    double xnew = (xleft + xmid)/2.0;
    double rnew = res_func(xnew,res);
    if(rnew < rmid)
    { xright = xmid; rright = rmid; xmid = xnew; rmid = rnew; }
    else
    { xleft = xnew; rleft = rnew; }

    xnew = (xmid + xright)/2.0;
    rnew = res_func(xnew,res);
    if(rnew < rmid)
    { xleft = xmid; rleft = rmid; xmid = xnew; rmid = rnew; }
    else
    { xright = xnew; rright = rnew; }
  }while(fabs(xleft-xright) > xmid*1e-14);

  // set green to the minimizer
  for(int i=0 ; i<nsite ; i++)
  { green[i] += (xmid - 1.0)*dres[i]; }
  return residual(shift,nshell,green,dgreen);
}

// calculate local Green's function satisfying G = localized[G*(H-shift*I)*G] w/ nonlinear conjugate gradients
// NOTE: the input green vector is assumed to be a good starting guess
#define MIN_TOL 1e-5
void self_green(double complex shift, // chemical potential (real part) & temperature (imaginary part)
                int nshell, // number of shells
                double complex *green) // Green's function [s_sum(nshell)]
{
  int nsite = s_sum(nshell);
  double complex *dgreen = (double complex*)malloc(sizeof(double complex)*nsite);
  double complex *dres = (double complex*)malloc(sizeof(double complex)*nsite);
  double complex *dres_old = (double complex*)malloc(sizeof(double complex)*nsite);
  double complex *conj_grad = (double complex*)malloc(sizeof(double complex)*nsite);

  for(int i=0 ; i<nsite ; i++)
  { conj_grad[i] = dres_old[i] = 0.0; }

  double dres_norm = 1.0, dres_norm_old, dres_inner, res_norm;
  do
  {
    // calculate steepest descent direction (dres)
    perturb_green(shift,nshell,green,dgreen);
    dresidual(shift,nshell,green,dgreen,dres);
    dres_norm_old = dres_norm;
    dres_norm = dres_inner = 0.0;
    for(int i=0 ; i<nsite ; i++)
    {
      dres_norm += pow(cabs(dres[i]),2);
      dres_inner += creal(dres[i])*creal(dres_old[i]) + cimag(dres[i])*cimag(dres_old[i]);
    }
    for(int i=0 ; i<nsite ; i++)
    { dres_old[i] = dres[i]; }

    // update conjugate direction
    double beta = (dres_norm - dres_inner)/dres_norm_old;
    if(beta < 0.0) { beta = 0.0; }
    for(int i=0 ; i<nsite ; i++)
    { conj_grad[i] = dres[i] + beta*conj_grad[i]; }

    // perform line search
    res_norm = line_min(shift,nshell,green,dgreen,conj_grad);
  }while(res_norm > MIN_TOL);

  free(conj_grad); free(dres_old); free(dres); free(dgreen);
}

// main program
int main(int argc, char **argv)
{
  random64(1);
  int nshell;
  double ishift;
  sscanf(argv[1],"%d",&nshell);
  sscanf(argv[2],"%lf",&ishift);
  int nsite = s_sum(nshell);
  double complex *green = (double complex*)malloc(sizeof(double complex)*s_sum(nshell));
  double complex shift = 0.0 + ishift*I;
  local_green(shift, nshell, green);

  int nkpt = 50;
  double complex *green0 = (double complex*)malloc(sizeof(double complex)*s_sum(nshell));
  reciprocal_green(shift,nkpt,nshell,green0);

  double complex *green2 = (double complex*)malloc(sizeof(double complex)*s_sum(nshell));
  square_green(shift, nshell, green2);

  double complex *green3 = (double complex*)malloc(sizeof(double complex)*s_sum(nshell));
  random_green(shift, nshell, 40, green3);

  double complex *dgreen = (double complex*)malloc(sizeof(double complex)*s_sum(nshell));
  double complex *green4 = (double complex*)malloc(sizeof(double complex)*s_sum(nshell));
  perturb_green(shift, nshell, green, dgreen);
  for(int i=0 ; i<nsite ; i++)
  { green4[i] = green[i] + dgreen[i]; }

  double complex *green5 = (double complex*)malloc(sizeof(double complex)*s_sum(nshell));
  for(int i=0 ; i<nsite ; i++)
  { green5[i] = green0[i]; }
  self_green(shift, nshell, green5);

  double complex *green6 = (double complex*)malloc(sizeof(double complex)*s_sum(nshell));
  perturb_green(shift, nshell, green2, dgreen);
  for(int i=0 ; i<nsite ; i++)
  { green6[i] = green2[i] + dgreen[i]; }

  double *shell_norm = (double*)malloc(sizeof(double)*nshell);
  double *shell_norm2 = (double*)malloc(sizeof(double)*nshell);
  double *shell_norm3 = (double*)malloc(sizeof(double)*nshell);
  double *shell_norm4 = (double*)malloc(sizeof(double)*nshell);
  double *shell_norm5 = (double*)malloc(sizeof(double)*nshell);
  double *shell_norm6 = (double*)malloc(sizeof(double)*nshell);
  norm2(nshell,green0,green,shell_norm);
  norm2(nshell,green0,green2,shell_norm2);
  norm2(nshell,green0,green3,shell_norm3);
  norm2(nshell,green0,green4,shell_norm4);
  norm2(nshell,green0,green5,shell_norm5);
  norm2(nshell,green0,green6,shell_norm6);
  for(int i=0 ; i<nshell ; i++)
  { printf("%d %e %e %e %e %e %e\n",i,shell_norm[i],shell_norm2[i],shell_norm3[i],shell_norm4[i],shell_norm5[i],shell_norm6[i]); }

  return 0;
}
