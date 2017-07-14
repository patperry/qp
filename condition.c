static double
vector_amax (const gsl_vector *v)
{
    double min, max;
    
    gsl_vector_minmax (v, &min, &max);
    
    return GSL_MAX_DBL (fabs (min), fabs (max));
}

static double
matrix_amax (const gsl_matrix *m)
{
    double min, max;
    
    gsl_matrix_minmax (m, &min, &max);
    
    return GSL_MAX_DBL (fabs (min), fabs (max));
}


// condition number estimate of an upper-triangular matrix
// from Algorithm 3.5.1 of Golub & Van Loan, 3rd Ed.
static void
utri_condition_estimate (const gsl_matrix *T, gsl_vector *y)
{
    size_t k, n = T->size1;
    double p_plus_data[n];
    double p_minus_data[n];    

    gsl_vector_view p_plus = gsl_vector_view_array (p_plus_data, n);    
    gsl_vector_view p_minus = gsl_vector_view_array (p_minus_data, n);

    gsl_vector_set_zero (&p_plus.vector);
    gsl_vector_set_zero (&p_minus.vector);

    for (k = n - 1; k >= 0; k--)
    {
        double p_k = gsl_vector_get (&p_plus.vector, k);
        double T_kk = gsl_matrix_get (T, k, k);
        double y_plus_k = (1.0 - p_k) / T_kk;
        double y_minus_k = (-1.0 - p_k) / T_kk;
        
        if (k > 0)
        {
            gsl_vector_view p_plus_k = gsl_vector_subvector (&p_plus.vector, 0, k);
            gsl_vector_view p_minus_k = gsl_vector_subvector (&p_minus.vector, 0, k);
            gsl_vector_const_view T_k = gsl_matrix_const_column (T, k);
            gsl_vector_const_view T_sub_k = gsl_vector_const_subvector (&T_k.vector, 0, k);
    
            gsl_blas_daxpy (y_plus_k, &T_sub_k.vector, &p_plus_k.vector);
            gsl_blas_daxpy (y_minus_k, &T_sub_k.vector, &p_minus_k.vector);
    
            if ((fabs (y_plus_k) + gsl_blas_dasum (&p_plus_k.vector)) >=
                (fabs (y_minus_k) + gsl_blas_dasum (&p_minus_k.vector)))
            {
                gsl_vector_set (y, k, y_plus_k);
                gsl_blas_dcopy (&p_plus_k.vector, &p_minus_k.vector);
            }
            else
            {
                gsl_vector_set (y, k, y_minus_k);
                gsl_blas_dcopy (&p_minus_k.vector, &p_plus_k.vector);
            }
        }
        else
        {
            if (fabs (y_plus_k) >= fabs (y_minus_k))
            {
                gsl_vector_set (y, k, y_plus_k);
            }
            else
            {
                gsl_vector_set (y, k, y_minus_k);
            }
            
            break;
        }
    }

    gsl_blas_dscal (1.0 / vector_amax (y), y);
}


static double
chol_condition_estimate (const gsl_matrix *cholesky)
{
    double kappa = 1.0;
    size_t n = cholesky->size1;
    double y_data[n];
    double r_norm;
    double z_norm;
    
    gsl_vector_view y = gsl_vector_view_array (y_data, n);

    utri_condition_estimate (cholesky, &y.vector);
    
    // L^T r = y
    gsl_blas_dtrsv (CblasUpper, CblasNoTrans, CblasNonUnit, cholesky, &y.vector);
    r_norm = vector_amax (&y.vector);
    
    // L L^T z = r
    gsl_linalg_cholesky_svx (cholesky, &y.vector);
    z_norm = vector_amax (&y.vector);
    
    kappa = z_norm / r_norm;
    
    return kappa;
}

static double
matrix_trace (const gsl_matrix *a)
{
    double result = 0.0;
    size_t i, n;
    
    assert (a != NULL);
    assert (a->size1 == a->size2);
    
    n = a->size1;
    for (i = 0; i < n; i++)
    {
        result += gsl_matrix_get (a, i, i);
    }
    
    return result;
}

