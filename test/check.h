
#ifndef _CHECK_H
#define _CHECK_H

#include <gsl/gsl_vector.h>

int
check_f_x (double f_expected, double f, 
           const gsl_vector *x_expected, const gsl_vector *x);

int
check_f_x_with_cond (double f_expected, double f, 
                     const gsl_vector *x_expected, const gsl_vector *x, 
                     double cond);


int
check_f (double f_expected, double f);

int
check_f_with_cond (double f_expected, double f, double cond);

int
check_x (const gsl_vector *x_expected, const gsl_vector *x);

int
check_x_with_cond (const gsl_vector *x_expected, const gsl_vector *x, double cond);


#endif // _CHECK_H
