
#ifndef _QP_H
#define _QP_H

#include <gsl/gsl_cblas.h>

/**
 * This function implements the algorithm of Goldfarb and Idnani 
 * for the solution of a (convex) Quadratic Programming problem
 * by means of a dual method.
 *	 
 * The problem is in the form:
 *
 * minimize f(x) = a^T x + 0.5 x^{T} G x
 * s.t.
 *   C  x - b >= 0
 *   CE x - be = 0
 *	 
 * The matrix and vectors dimensions are as follows:
 *    G: n * n
 *	  a: n
 *				
 *    C: m * n
 *    b: m
 *
 *   CE: me * n
 *   be: me
 *				
 *    x: n
**/
int
qp_solve_with_chol (const enum CBLAS_ORDER order,
                    const enum CBLAS_UPLO uplo,
                    const enum CBLAS_TRANSPOSE TransC, 
                    const enum CBLAS_TRANSPOSE TransCE,
                    const int N, const int M, const int ME,
                    const double *cholG, const int ldg,
                    const double *a,     const int incA,
                    const double *C,     const int ldc,
                    const double *b,     const int incB,
                    const double *CE,    const int ldce,
                    const double *be,    const int incBE,
                    double *x,           const int incX,
                    double *f,
                    double *work);
                    
                    

size_t 
qp_work_size (size_t n);

#endif // _QP_H 
