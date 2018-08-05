// Generate an input file for a rhombicuboctahedral copper cluster of a given width carved from a perfect fcc lattice

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// TB lattice constant of copper (in Angstroms)
#define A0 3.52

// test whether or not a point is inside a rhombicuboctahedron
int poly_test(double x, double y, double z, double width)
{
  // 100 faces
  double d = 0.5*width;
  if(fabs(x) > d || fabs(y) > d || fabs(z) > d) { return 0; }

  // 110 faces;
  d = 0.5*width*sqrt(2.0);
  if((fabs(x)+fabs(y)) > d || (fabs(x)+fabs(z)) > d || (fabs(y)+fabs(z)) > d) { return 0; }

  // 111 faces
  d = 0.5*width*sqrt(3.0)*sqrt( (11.0 + 6.0*sqrt(2.0))/3.0 )/( 1.0 + sqrt(2.0) );
  if((fabs(x)+fabs(y)+fabs(z)) > d) { return 0; }

  return 1;
}

int main(int argc, char** argv)
{
  // input width
  double width;
  int shift;
  sscanf(argv[1],"%lf",&width);
  sscanf(argv[2],"%d",&shift);

  // sum over an encompassing grid of atoms
  int n = 2.0*width/A0;
  int count = 0;
  for(int i=-n ; i<=n ; i++)
  for(int j=-n ; j<=n ; j++)
  for(int k=-n ; k<=n ; k++)
  {
    double x = 0.5*A0*(shift + i + j);
    double y = 0.5*A0*(shift + i + k);
    double z = 0.5*A0*(shift + j + k);
    if(poly_test(x,y,z,width)) count++;
  }

  printf("%d\n\n",count);
  for(int i=-n ; i<=n ; i++)
  for(int j=-n ; j<=n ; j++)
  for(int k=-n ; k<=n ; k++)
  {
    double x = 0.5*A0*(shift + i + j);
    double y = 0.5*A0*(shift + i + k);
    double z = 0.5*A0*(shift + j + k);
    if(poly_test(x,y,z,width))
    { printf("Cu %e %e %e\n",x,y,z); }
  }
  return 1;
}
