
#include <assert.h>
#include <math.h>
#include <stdio.h>

#include <gsl/gsl_blas.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_sort_vector_double.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#include "gsl_qp.h"
#include "check.h"

static double
condition_number (gsl_matrix *matrix)
{
    const size_t n = matrix->size1;
    gsl_eigen_symm_workspace *work = gsl_eigen_symm_alloc (n);
    double eval_data[n];
    gsl_vector_view eval = gsl_vector_view_array (eval_data, n);
    double diag_data[n];
    gsl_vector_view diag_copy = gsl_vector_view_array (diag_data, n);
    gsl_vector_view diag_view = gsl_matrix_diagonal (matrix);
    double result;
    size_t i, j;
    
    // save the matrix diagonal
    gsl_vector_memcpy (&diag_copy.vector, &diag_view.vector);

    // compute the eigenvalues and condition number.  this destroys the
    // diagonal and lower triangular part of the matrix
    gsl_eigen_symm (matrix, &eval.vector, work);
    gsl_sort_vector (&eval.vector);
    result = sqrt (gsl_vector_get (&eval.vector, n - 1)
                   / gsl_vector_get (&eval.vector, 0));
    
    // restore the diagonal and lower triangular part of the matrix
    gsl_vector_memcpy (&diag_view.vector, &diag_copy.vector);
    for (i = 0; i < n; i++)
    {
        for (j = i + 1; j < n; j++)
        {
            gsl_matrix_set (matrix, j, i, gsl_matrix_get (matrix, i, j));
        }
    }
    
    gsl_eigen_symm_free (work);
    
    return result;
}

static void
ran_pos_def (const gsl_rng *r, int ill_cond, gsl_matrix *matrix, double *cond)
{
    size_t i, j, n;
    double extra_inc = 1.0;
    
    assert (r != NULL);
    assert (matrix != NULL);
    assert (matrix->size1 == matrix->size2);
    
    n = matrix->size1;
    
    for (j = 0; j < n; j++)
    {
        gsl_matrix_set (matrix, j, j, 0.0);

        for (i = j + 1; i < n; i++)
        {
            double value = gsl_ran_flat (r, -1.0, 1.0);
            gsl_matrix_set (matrix, i, j, value);
            gsl_matrix_set (matrix, j, i, value);
        }        
    }

    for (j = 0; j < n; j++)
    {
        gsl_vector_const_view column = gsl_matrix_const_column (matrix, j);
        double sum = gsl_blas_dasum (&column.vector);
        double value = sum + extra_inc + gsl_ran_flat (r, 0.0, 1.0);
        
        gsl_matrix_set (matrix, j, j, value);

        if (ill_cond)
            extra_inc += sum;
    }
    
    *cond = condition_number (matrix);
}


static void
ran_matrix (const gsl_rng *r, gsl_matrix *matrix, double min, double max)
{
    size_t i, j, m, n;

    assert (r != NULL);
    assert (min < max);

    if (matrix)
    {
        m = matrix->size1;
        n = matrix->size2;
    
        for (j = 0; j < n; j++)
        {
            for (i = 0; i < m; i++)
            {
                double value = gsl_ran_flat (r, min, max);
            
                gsl_matrix_set (matrix, i, j, value);
            }
        }
    }
}


static void
ran_vector (const gsl_rng *r, gsl_vector *vector, double min, double max)
{
    size_t i, n;
    
    assert (r != NULL);
    assert (min < max);
    
    if (vector)
    {
        n = vector->size;
    
        for (i = 0; i < n; i++)
        {
            double value = gsl_ran_flat (r, min, max);
            gsl_vector_set (vector, i, value);
        }
    }
}

static void
normalize_columns (gsl_matrix *matrix)
{
    size_t j;
    
    if (matrix)
    {
        for (j = 0; j < matrix->size2; j++)
        {
            gsl_vector_view column_j = gsl_matrix_column (matrix, j);
            double norm = gsl_blas_dnrm2 (&column_j.vector);
            gsl_blas_dscal (1.0 / norm, &column_j.vector);
        }
    }
}

static void
ran_qp (const gsl_rng *r,  size_t q, int ill_cond, gsl_matrix *G, double *G_cond,
        gsl_vector *a, gsl_matrix *CE, gsl_vector *be, gsl_matrix *C, 
        gsl_vector *s, gsl_vector *b, gsl_vector *ue, gsl_vector *u, 
        gsl_vector *x, double *f)
{
    gsl_vector *Gx;
    double xGx;
    
    ran_pos_def (r, ill_cond, G, G_cond);
    
    ran_matrix (r, CE, -1.0, 1.0);
    normalize_columns (CE);
    ran_matrix (r, C, -1.0, 1.0);
    normalize_columns (C);
    
    ran_vector (r, x, -5.0, 5.0);
    ran_vector (r, ue, 0.0, 30.0);
    ran_vector (r, s,  0.0, 1.0);
    ran_vector (r, u, 0.0, 30.0);
    
    if (s)
    {
        size_t i;

        for (i = 0; i < q; i++)
        {
            gsl_vector_set (s, i, 0);
        }

        for (i = q; i < u->size; i++)
        {
            gsl_vector_set (u, i, 0);
        }
    }
    
    // be = CE^T x
    if (CE) gsl_blas_dgemv (CblasTrans, 1.0, CE, x, 0.0, be);
    
    // b = C^T x - s
    if (C) 
    {
        gsl_blas_dgemv (CblasTrans, 1.0, C, x, 0.0, b);
        gsl_blas_daxpy (-1.0, s, b);
    }
    
    // a = C u + CE ue - G x
    gsl_blas_dgemv (CblasNoTrans, -1.0, G, x, 0.0, a);
    if (CE) gsl_blas_dgemv (CblasNoTrans, 1.0, CE, ue, 1.0, a);
    if (C) gsl_blas_dgemv (CblasNoTrans, 1.0, C, u, 1.0, a);
    
    // f = 0.5 x^T G x + x^T a
    Gx = gsl_vector_alloc (x->size);
    gsl_blas_dgemv (CblasNoTrans, 1.0, G, x, 0.0, Gx);
    gsl_blas_ddot (x, Gx, &xGx);
    gsl_blas_ddot (x, a, f);
    *f += 0.5 * xGx;
    gsl_vector_free (Gx);
}


static int
qp_trial (const gsl_rng *r, size_t n, size_t m, size_t m_e, size_t q, 
          int ill_cond, size_t replicates)
{
    size_t rep;
    int failed, failures = 0;
    
    gsl_matrix *G = gsl_matrix_alloc (n, n);
    gsl_matrix *G_copy = gsl_matrix_alloc (n, n);
    double G_cond;
    gsl_vector *a = gsl_vector_alloc (n);

    gsl_matrix *CE = m_e ? gsl_matrix_alloc (n, m_e) : NULL;
    gsl_vector *be = m_e ? gsl_vector_alloc (m_e) : NULL;
    gsl_vector *ue = m_e ? gsl_vector_alloc (m_e) : NULL;

    gsl_matrix *C = m ? gsl_matrix_alloc (n, m) : NULL;
    gsl_vector *s = m ? gsl_vector_alloc (m) : NULL;
    gsl_vector *b = m ? gsl_vector_alloc (m) : NULL;
    gsl_vector *u = m ? gsl_vector_alloc (m) : NULL;

    gsl_vector *x = gsl_vector_alloc (n);
    gsl_vector *x_expected = gsl_vector_alloc (n);
    double f, f_expected;
    void *work = malloc (gsl_qp_work_size (n));

    for (rep = 0; rep < replicates; rep++)
    {
        ran_qp (r, q, ill_cond, G, &G_cond, a, CE, be, C, s, b, ue, u, 
                x_expected, &f_expected);
        gsl_matrix_memcpy (G_copy, G);

        fprintf (stderr, ".");
        fflush (stderr);

        gsl_qp_solve (CblasTrans, CblasTrans, G, a, C, b, CE, be, x, &f, work);
        failed = -check_f_x_with_cond (f_expected, f, x_expected, x, G_cond);
        failures += failed;

        if (failed && 0)
        {
            printf ("\nG =\n");
            gsl_matrix_fprintf (stdout, G_copy, "%9f");
            printf ("\na =\n");
            gsl_vector_fprintf (stdout, a, "%9f");
            if (CE)
            {
                printf ("\nCE =\n");
                gsl_matrix_fprintf (stdout, CE, "%9f");
                printf ("\nbe =\n");
                gsl_vector_fprintf (stdout, be, "%9f");
            }
            if (C)
            {
                printf ("\nC =\n");
                gsl_matrix_fprintf (stdout, C, "%9f");
                printf ("\nb =\n");
                gsl_vector_fprintf (stdout, b, "%9f");
            }
            printf ("\nf = %9f\n", f_expected);
            printf ("\nx = \n");
            gsl_vector_fprintf (stdout, x_expected, "%9f");
            printf ("\n");
        }
    }

    gsl_matrix_free (G);
    gsl_matrix_free (G_copy);
    gsl_vector_free (a);
    if (CE) gsl_matrix_free (CE);
    if (be) gsl_vector_free (be);
    if (C) gsl_matrix_free (C);
    if (s) gsl_vector_free (s);
    if (b) gsl_vector_free (b);
    if (ue) gsl_vector_free (ue);
    if (u) gsl_vector_free (u);
    gsl_vector_free (x);
    free (work);

    return failures;
}


int
main (int argc, char **argv)
{
    int failures = 0;
    size_t num_replicates = 40;
    
    gsl_rng *r = gsl_rng_alloc (gsl_rng_mt19937);
    gsl_rng_set (r, 0);
    
                         /*   n,   m, m_e,  q, ill */
    fprintf (stderr, "W\n"); fflush (stderr);
    failures += qp_trial (r,  9,   9,   0,  1, 0, num_replicates);
    failures += qp_trial (r, 27,  27,   0,  3, 0, num_replicates);
    failures += qp_trial (r, 81,  81,   0,  9, 0, num_replicates);
    failures += qp_trial (r,  9,   9,   0,  3, 0, num_replicates);
    failures += qp_trial (r, 27,  27,   0,  9, 0, num_replicates);
    failures += qp_trial (r, 81,  81,   0, 27, 0, num_replicates);
    failures += qp_trial (r,  9,  27,   0,  1, 0, num_replicates);
    failures += qp_trial (r, 27,  81,   0,  3, 0, num_replicates);
    failures += qp_trial (r, 81, 243,   0,  9, 0, num_replicates);
    failures += qp_trial (r,  9,  27,   0,  3, 0, num_replicates);
    failures += qp_trial (r, 27,  81,   0,  9, 0, num_replicates);
    failures += qp_trial (r, 81, 243,   0, 27, 0, num_replicates);
    
    fprintf (stderr, "\nI\n"); fflush (stderr);
    failures += qp_trial (r,  9,   9,   0,  1, 1, num_replicates);
    failures += qp_trial (r, 27,  27,   0,  3, 1, num_replicates);
    failures += qp_trial (r, 81,  81,   0,  9, 1, num_replicates);
    failures += qp_trial (r,  9,   9,   0,  3, 1, num_replicates);
    failures += qp_trial (r, 27,  27,   0,  9, 1, num_replicates);
    failures += qp_trial (r, 81,  81,   0, 27, 1, num_replicates);
    failures += qp_trial (r,  9,  27,   0,  1, 1, num_replicates);
    failures += qp_trial (r, 27,  81,   0,  3, 1, num_replicates);
    failures += qp_trial (r, 81, 243,   0,  9, 1, num_replicates);
    failures += qp_trial (r,  9,  27,   0,  3, 1, num_replicates);
    failures += qp_trial (r, 27,  81,   0,  9, 1, num_replicates);
    failures += qp_trial (r, 81, 243,   0, 27, 1, num_replicates);

    fprintf (stderr, "\nB\n"); fflush (stderr);
    failures += qp_trial (r, 200, 400,   10, 400, 0, num_replicates);    
    failures += qp_trial (r, 200, 400,   10, 400, 1, num_replicates);    
    
    if (!failures)
        printf ("\nSuccess!\n");
    else
        printf ("\n%d Failures\n", failures);
        
    gsl_rng_free (r);

    return failures;
}
    
