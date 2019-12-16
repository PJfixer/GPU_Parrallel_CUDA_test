#include <stdio.h>
#include <math.h>
#include <stdlib.h>
// #include "omp_repair.h"
// #include <omp.h>

 /*
 This program outputs a pretty fractal image in julia.ppm
 Compile with: cc -o julia julia.c -lm
 Then run with simply: julia
 
 Here is what the program does.  For every (x,y) point within
 a rectangular grid, the point is treated as complex number z.
 The following very simple equation is interated for up to
 itermax times:
 z=z^2 +c
 Depending on whether that iteration converges or diverges,
 a color is assigned to the (x,y) point.  By diverge ("breakout"), it is
 meant that |z|^2 becomes greater than the number breakout
 with itermax iterations.
 */
 #define MAX_COLOR 142

 // Tableaux de couleurs (142 couleurs, tirées d'un nuancier trouvé sur le web)
 const unsigned int COLOR_TABLE[] = {
     0xf7df, 0xff5a, 0x07ff, 0x7ffa, 0xf7ff, 0xf7bb, 0xff38, 0xff59, 0x001f, 0x895c,
     0xa145, 0xddd0, 0x5cf4, 0x7fe0, 0xd343, 0xfbea, 0x64bd, 0xffdb, 0xd8a7, 0x07ff,
     0x0011, 0x0451, 0xbc21, 0xad55, 0x0320, 0xbdad, 0x8811, 0x5345, 0xfc60, 0x9999,
     0x8800, 0xecaf, 0x8df1, 0x49f1, 0x2a69, 0x067a, 0x901a, 0xf8b2, 0x05ff, 0x6b4d,
     0x1c9f, 0xd48e, 0xb104, 0xffde, 0x2444, 0xf81f, 0xdefb, 0xffdf, 0xfea0, 0xdd24,
     0x8410, 0x0400, 0xafe5, 0xf7fe, 0xfb56, 0xcaeb, 0x4810, 0xfffe, 0xf731, 0xe73f,
     0xff9e, 0x7fe0, 0xffd9, 0xaedc, 0xf410, 0xe7ff, 0xffda, 0xd69a, 0x9772, 0xfdb8,
     0xfd0f, 0x2595, 0x867f, 0x839f, 0x7453, 0xb63b, 0xfffc, 0x07e0, 0x3666, 0xff9c,
     0xf81f, 0x8000, 0x6675, 0x0019, 0xbaba, 0x939b, 0x3d8e, 0x7b5d, 0x07d3, 0x4e99,
     0xc0b0, 0x18ce, 0xf7ff, 0xff3c, 0xff36, 0xfef5, 0x0010, 0xffbc, 0x8400, 0x6c64,
     0xfd20, 0xfa20, 0xdb9a, 0xef55, 0x9fd3, 0xaf7d, 0xdb92, 0xff7a, 0xfed7, 0xcc27,
     0xfe19, 0xdd1b, 0xb71c, 0x8010, 0xf800, 0xbc71, 0x435c, 0x8a22, 0xfc0e, 0xf52c,
     0x2c4a, 0xffbd, 0xa285, 0xc618, 0x867d, 0x6ad9, 0x7412, 0xffdf, 0x07ef, 0x4416,
     0xd5b1, 0x0410, 0xddfb, 0xfb08, 0x471a, 0xec1d, 0xd112, 0xf6f6, 0xffff, 0xf7be,
     0xffe0, 0x9e66, 0x0000
 };

 int main(int argc, char *argv[])
 {
  int itermax = 300;    /* maximum iters to do*/
  double breakout=512.;  /* |z|^2 greater than this is a breakout */
  double zoom=10;    /* 10 is standard magnification */
  double cr=-.7492; /*real part of c in z=z^2 +c */
  double ci=.1; /*imaginary part of c in z=z^2 +c */
  double x0=.09950; /*center of picture */
  double y0=-.00062;/*center of picture */
  int hxres = 1000;    /* horizontal resolution */
  int hyres = 1000;    /* vertical resolution */
  double x,xx,y,xl,yl,zsq,zm;
  int iter,hx,hy;
  int red,green,blue;
  FILE *ofp;



  ofp=fopen("julia2.ppm","w");
  /* write header for PPM output: */
  fprintf(ofp,"P6\n# zoom=%lf itermax=%d\n",zoom,itermax);
  fprintf(ofp,"%d %d\n255\n",hyres,hxres);
  
  for (hy=1;hy<=hyres;hy++)  
  {
    for (hx=1;hx<=hxres;hx++)  
    {
      y = 4*((hyres+1-hy-.5)/hyres-0.5)/zoom+y0;
      x = 4*((hx-.5)/hxres-0.5)/zoom+x0;
      zm = 0;
      for (iter=1;iter<itermax;iter++)  
      {
        xl=x;
        yl=y;
        xx = x*x-y*y+cr;
        y = 2.0*x*y+ci;
        x = xx;
        zsq=x*x+y*y;
        if (zsq > zm) zm=zsq;
        if (zsq>breakout) break;
       }
       if (iter>=itermax)
       {  /*if no "breakout" occured, color by maximum |z|^2 achieved*/
         red=0;
         green=255.*zm/breakout;
         blue=255.*zm/breakout;
       }
       else
       {
        // Dessine le pixel (avec conversion RGB565 -> RGB888)
        unsigned int color = COLOR_TABLE[iter % MAX_COLOR];
        red = ((color >> 11) & 0x1F) << 3;
        green = ((color >> 5) & 0x3F) << 2;
        blue = (color & 0x1F) << 3;
       }
       /* write out the color triplet in binary ppm format, one byte per integer*/
       fputc((char)red,ofp);
       fputc((char)green,ofp);
       fputc((char)blue,ofp);
    }
  }
  fclose(ofp);
  system("display julia2.ppm");
}
