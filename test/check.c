
#include <assert.h>
#include <stdio.h>
#include <gsl/gsl_math.h>
#include "check.h"

int
check_f_x (double f_expected, double f, 
           const gsl_vector *x_expected, const gsl_vector *x)
{
    return check_f_x_with_cond (f_expected, f, x_expected, x, 1.0);
}
                     

int
check_f_x_with_cond (double f_expected, double f, 
                     const gsl_vector *x_expected, const gsl_vector *x,
                     double cond)
{
    int failures = 0;
    
    failures += check_f_with_cond (f_expected, f, cond);
    failures += check_x_with_cond (x_expected, x, cond);

    return failures;
}


int
check_f (double f_expected, double f)
{
    return check_f_with_cond (f_expected, f, 1.0);
}


int
check_f_with_cond (double f_expected, double f, double cond)
{
    int result = 0;
    
    if (gsl_fcmp (f_expected, f, 5e-10) != 0)
    {
        printf ("\n** Expected f to be '%g', but was '%g' instead.\n",
                f_expected, f);
        printf ("   (condition number was '%g')\n", cond);                
        result = -1;
    }
    
    return result;
}


int
check_x (const gsl_vector *x_expected, const gsl_vector *x)
{
    return check_x_with_cond (x_expected, x, 1.0);
}

int
check_x_with_cond (const gsl_vector *x_expected, const gsl_vector *x, double cond)
{
    int result = 0;
    size_t i;
    
    assert (x_expected != NULL);
    assert (x != NULL);
    
    if (x_expected->size != x->size)
    {
        printf ("\n** Expected x to have length '%u', but was '%u' instead.\n",
                (unsigned int) (x_expected->size), 
                (unsigned int) (x->size));
        result = -1;
    }
    else
    {
        for (i = 0; i < x->size; i++)
        {
            double xe_i = gsl_vector_get (x_expected, i);
            double x_i = gsl_vector_get (x, i);
            if (gsl_fcmp (xe_i + (cond - 1.0) * GSL_SIGN (xe_i) * 5e-8, 
                           x_i + (cond - 1.0) * GSL_SIGN (x_i) * 5e-8, cond * 1e-10) != 0)
            {
                printf ("\n** Expected x[%u] to be '%g', but was '%g' instead.\n",
                        (unsigned int) i, gsl_vector_get (x_expected, i), 
                        gsl_vector_get (x, i));
                printf ("   (condition number was '%g')\n", cond);
                result = -1;
            }
        }
    }
    
    return result;
}
