
#include <assert.h>
#include <stdio.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include "check.h"
#include "gsl_qp.h"

static FILE *
parse_args (int argc, char **argv)
{
    FILE *infile;
    
    if (argc == 1)
        infile = stdin;
    else if (argc == 2)
        infile = fopen (argv[1], "r");
    else
    {
        printf ("Too many arguments given.\n");
        exit (-1);
    }
    
    if (!infile)
    {
        printf ("Error opening file '%s'\n", argv[1]);
        exit (-1);
    }

    return infile;
}


static gsl_matrix *
read_matrix (FILE *infile, const char *name)
{
    gsl_matrix *matrix = NULL;
    unsigned int m, n;
    
    if ((fscanf (infile, "%s", name) == 0) &&
        (fscanf (infile, " (%u,%u)\n", &m, &n) == 2) &&
        (m > 0) && (n > 0))
    {
        matrix = gsl_matrix_alloc (m, n);
            
        if (gsl_matrix_fscanf (infile, matrix) == GSL_EFAILED)
        {
            gsl_matrix_free (matrix);
            matrix = NULL;
        }
            
        fscanf (infile, "\n\n");
    }
    
    return matrix;
}


static gsl_vector *
read_vector (FILE *infile, const char *name)
{
    gsl_vector *vector = NULL;
    unsigned int n;
    
    if ((fscanf (infile, name) == 0) &&
        (fscanf (infile, " (%u)\n", &n) == 1) &&
        (n > 0))
    {
        vector = gsl_vector_alloc (n);

        if (gsl_vector_fscanf (infile, vector) == GSL_EFAILED)
        {
            gsl_vector_free (vector);
            vector = NULL;
        }
        
        fscanf (infile, "\n\n");
    }

    return vector;
}


static double
read_scalar (FILE *infile, const char *name)
{
    double scalar = 0.0 / 0.0;
    
    if ((fscanf (infile, name) == 0) &&
        (fscanf (infile, "\n%lg", &scalar) == 1))
    {
        // success
    }
    
    fscanf (infile, "\n\n");
    
    return scalar;
}


int
main (int argc, char **argv)
{
    FILE *infile = parse_args (argc, argv);
    gsl_matrix *G, *CE, *C;
    gsl_vector *a, *be, *b;
    gsl_vector *x, *x_expected;
    void *work;
    double f, f_expected;
    int failures = 0;
    
    G = read_matrix (infile, "G");
    a = read_vector (infile, "a");
    C = read_matrix (infile, "C");
    b = read_vector (infile, "b");
    CE = read_matrix (infile, "CE");
    be = read_vector (infile, "be");
    f_expected = read_scalar (infile, "f");
    x_expected = read_vector (infile, "x");
    fclose (infile);
    
    x = gsl_vector_alloc (G->size1);
    work = malloc (gsl_qp_work_size (x->size));
    gsl_qp_solve (CblasTrans, CblasTrans, G, a, C, b, CE, be, x, &f, work);
    
    failures = check_f_x (f_expected, f, x_expected, x);
    
    free (work);
    gsl_matrix_free (G);
    gsl_vector_free (a);
    if (CE) gsl_matrix_free (CE);
    if (be) gsl_vector_free (be);
    if (C) gsl_matrix_free (C);
    if (b) gsl_vector_free (b);
    gsl_vector_free (x);
    gsl_vector_free (x_expected);

    if (!failures)
    {
        printf ("Success!\n");
    }

    return failures;
}
