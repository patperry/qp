
#include <assert.h>
#include <float.h>
#include <string.h>

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>

#include "qp.h"
#include "gsl_qp.h"


static enum CBLAS_TRANSPOSE
trans_complement (enum CBLAS_TRANSPOSE Trans)
{
    enum CBLAS_TRANSPOSE result;
    
    if (Trans == CblasNoTrans)
        result = CblasTrans;
    else
        result = CblasNoTrans;
        
    return result;
}


static enum CBLAS_UPLO
uplo_complement (enum CBLAS_UPLO uplo)
{
    enum CBLAS_UPLO result;
    
    if (uplo == CblasUpper)
        result = CblasLower;
    else
        result = CblasUpper;
        
    return result;
}


static enum CBLAS_TRANSPOSE
get_trans (enum CBLAS_ORDER order, enum CBLAS_TRANSPOSE Trans)
{
    enum CBLAS_TRANSPOSE result;
    
    if (order == CblasRowMajor)
        result = Trans;
    else
        result = trans_complement (Trans);
        
    return result;
}


static enum CBLAS_UPLO
get_uplo (enum CBLAS_ORDER order, enum CBLAS_UPLO uplo)
{
    enum CBLAS_UPLO result;
    
    if (order == CblasRowMajor)
        result = uplo;
    else
        result = uplo_complement (uplo);
        
    return result;
}

gsl_vector *
get_gsl_vector (double *x, int incx, int n, gsl_vector_view *view)
{
    gsl_vector *result = NULL;
    
    if (n > 0)
    {
        *view  = gsl_vector_view_array_with_stride (x, incx, n);
        result = &(view->vector);
    }
    
    return result;
}

gsl_matrix *
get_gsl_matrix (const enum CBLAS_ORDER order,
                double *a, const int m, const int n, const int tda, 
                gsl_matrix_view *view)
{
    gsl_matrix *result = NULL;
    
    if (m > 0 && n > 0)
    {
        if (order == CblasRowMajor)
            *view  = gsl_matrix_view_array_with_tda (a, m, n, tda);
        else
            *view  = gsl_matrix_view_array_with_tda (a, n, m, tda);
            
        result = &(view->matrix);
    }
    
    return result;
}

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
                    double *work)
{
    int result;

    enum CBLAS_UPLO      uploG   = get_uplo  (order, uplo);
    enum CBLAS_TRANSPOSE transC  = get_trans (order, TransC);
    enum CBLAS_TRANSPOSE transCE = get_trans (order, TransCE);

    gsl_matrix_view cholG_view; const gsl_matrix *gsl_cholG;
    gsl_matrix_view C_view;     const gsl_matrix *gsl_C; 
    gsl_matrix_view CE_view;    const gsl_matrix *gsl_CE;
    gsl_vector_view a_view;     const gsl_vector *gsl_a;
    gsl_vector_view b_view;     const gsl_vector *gsl_b;
    gsl_vector_view be_view;    const gsl_vector *gsl_be;
    gsl_vector_view x_view;           gsl_vector *gsl_x;

    gsl_cholG = get_gsl_matrix (order, (double *) cholG, N,  N, ldg,  &cholG_view);
    gsl_C     = get_gsl_matrix (order, (double *) C,     M,  N, ldc,  &C_view);
    gsl_CE    = get_gsl_matrix (order, (double *) CE,    ME, N, ldce, &CE_view);
    
    gsl_a  = get_gsl_vector ((double *) a,  incA,  N,  &a_view);
    gsl_b  = get_gsl_vector ((double *) b,  incB,  M,  &b_view);
    gsl_be = get_gsl_vector ((double *) be, incBE, ME, &be_view);
    gsl_x  = get_gsl_vector ((double *) x,  incX,  N,  &x_view);
    
    
    result = gsl_qp_solve_with_chol (
                uploG, transC, transCE, 
                gsl_cholG, gsl_a,
                gsl_C,     gsl_b,
                gsl_CE,    gsl_be,
                gsl_x,     f,
                work);
    
    return result;
}       
                    
                    

size_t 
qp_work_size (size_t n)
{
    return gsl_qp_work_size (n);
}
