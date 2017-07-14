
#ifndef _GSL_QP_H
#define _GSL_QP_H

#include <gsl/gsl_blas.h>
#include <gsl/gsl_cblas.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>

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
gsl_qp_solve (CBLAS_TRANSPOSE_t TransC, CBLAS_TRANSPOSE_t TransCE,
              gsl_matrix *G,             const gsl_vector *a, 
              const gsl_matrix *C,       const gsl_vector *b, 
              const gsl_matrix *CE,      const gsl_vector *be,
              gsl_vector *x,             double *f,
              void *work);

int
gsl_qp_solve_with_chol (CBLAS_UPLO_t uplo,
                        CBLAS_TRANSPOSE_t TransC, CBLAS_TRANSPOSE_t TransCE,
                        const gsl_matrix *cholG,  const gsl_vector *a, 
                        const gsl_matrix *C,      const gsl_vector *b, 
                        const gsl_matrix *CE,     const gsl_vector *be,
                        gsl_vector *x,            double *f,
                        void *work);

size_t 
gsl_qp_work_size (size_t n);

#endif // _GSL_QP_H 
