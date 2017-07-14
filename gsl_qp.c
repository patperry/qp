
#include <assert.h>
#include <float.h>
#include <string.h>

#include <gsl/gsl_blas.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_matrix.h>

#include "gsl_qp.h"

typedef struct _qp_t
{
    // inequality constraints
    CBLAS_TRANSPOSE_t TransC;
    const gsl_matrix *C;
    const gsl_vector *b;
    
    // equality constraints
    CBLAS_TRANSPOSE_t TransCE;
    const gsl_matrix *CE;
    const gsl_vector *be;
    
    // N^* and H
    gsl_matrix *J;
    gsl_matrix *Rt;
    double R_norm;
    
    gsl_vector *d;
    gsl_vector *z;
    gsl_vector *r;
    
    // the active set
    size_t *A; 
    size_t q;
    
    // the primal and dual solutions
    gsl_vector *x;
    gsl_vector *u;
    double f;
    
} qp_t;

static void *
alloc_work_array (void **work, size_t n, size_t size)
{
    void * result = *work;
    *work += n * size;
    
    return result;
}

static gsl_vector *
alloc_work_vector (void **work, size_t n)
{
    gsl_vector *result;
    
    *((gsl_vector_view *) *work) =
        gsl_vector_view_array (*work + sizeof (gsl_vector_view), n);
    result = &((gsl_vector_view *) *work)->vector;
    *work += sizeof (gsl_vector_view) + n * sizeof (double);
    
    return result;
}

static gsl_matrix *
alloc_work_matrix (void **work, size_t m, size_t n)
{
    gsl_matrix *result;
    
    *((gsl_matrix_view *) *work) = 
        gsl_matrix_view_array (*work + sizeof (gsl_matrix_view), m, n);
    result = &((gsl_matrix_view *) *work)->matrix;
    *work += sizeof (gsl_matrix_view) + m * n * sizeof (double);
    
    return result;
}

static void
qp_init (qp_t *qp, 
         CBLAS_UPLO_t uplo,
         CBLAS_TRANSPOSE_t TransC, CBLAS_TRANSPOSE_t TransCE,
         const gsl_matrix *G_chol,  const gsl_vector *a, 
         const gsl_matrix *C,       const gsl_vector *b, 
         const gsl_matrix *CE,      const gsl_vector *be, 
         gsl_vector *x,
         void *work)
{
    size_t n, m_e, m;
    
    n = x->size;
    m_e = be ? be->size : 0;
    m = b ? b->size : 0;

    qp->TransC  = TransC;
    qp->C       = C;
    qp->b       = b;
    
    qp->TransCE = TransCE;
    qp->CE      = CE;
    qp->be      = be;
    
    qp->J  = alloc_work_matrix (&work, n, n);
    qp->Rt = alloc_work_matrix (&work, n, n);
    qp->R_norm = 1.0;
    
    qp->d = alloc_work_vector (&work, n);
    qp->z = alloc_work_vector (&work, n);
    qp->r = alloc_work_vector (&work, n);
    
    // malloc ((m_e + m) * sizeof (size_t));
    qp->A = alloc_work_array (&work, n, sizeof (size_t));
    qp->q = 0;
    
    qp->x = x;
    qp->u = alloc_work_vector (&work, n);
        

    
    gsl_matrix_set_identity (qp->J);

    gsl_blas_dcopy (a, qp->x);
    gsl_blas_dscal (-1.0, qp->x);

    // J := (L^{T})^{-1}
    // x := -G^{-1} a
    if (uplo == CblasLower)
    {
        // G      = L L^T
        // G^{-1} = (L^{-1})^{T} L^{-1}
        gsl_blas_dtrsm (CblasLeft, CblasLower, CblasTrans, CblasNonUnit, 
                        1.0, G_chol, qp->J);
                        
        // y = L^{-1} x
        // z = (L^{-1})^{T}
        gsl_blas_dtrsv (CblasLower, CblasNoTrans, CblasNonUnit, G_chol, qp->x);
        gsl_blas_dtrsv (CblasLower, CblasTrans,   CblasNonUnit, G_chol, qp->x);
    }
    else
    {
        gsl_blas_dtrsm (CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, 
                        1.0, G_chol, qp->J);

        gsl_blas_dtrsv (CblasUpper, CblasTrans,   CblasNonUnit, G_chol, qp->x);
        gsl_blas_dtrsv (CblasUpper, CblasNoTrans, CblasNonUnit, G_chol, qp->x);
    }
    

    // f := 0.5 a^{T} x
    gsl_blas_ddot (a, qp->x, &(qp->f));
    qp->f *= 0.5;
}

size_t
gsl_qp_work_size (size_t n)
{
    return (   2 * (sizeof (gsl_matrix_view) + n * n * sizeof (double))
             + 4 * (sizeof (gsl_vector_view) +     n * sizeof (double))
             + 1 * (                               n * sizeof (size_t)) 
           );
}

/*static void
qp_teardown (qp_t *qp)
{
    gsl_matrix_free (qp->Rt);
    gsl_matrix_free (qp->J);
    gsl_vector_free (qp->d);
    gsl_vector_free (qp->z);
    gsl_vector_free (qp->r);
    gsl_vector_free (qp->u);
    free (qp->A);
}*/


static gsl_vector_view
trans_matrix_row (CBLAS_TRANSPOSE_t Trans, gsl_matrix *m, size_t i)
{
    gsl_vector_view result;
    
    if (Trans == CblasNoTrans)
        result = gsl_matrix_row (m, i);
    else
        result = gsl_matrix_column (m, i);
        
    return result;
}


static size_t
num_equality_constraints (const qp_t *qp)
{
    return (qp->be ? qp->be->size : 0);
}

static size_t
num_inequality_constraints (const qp_t *qp)
{
    return (qp->b ? qp->b->size : 0);
}

static size_t
index_in_active_set (const qp_t *qp, size_t j)
{
    size_t index = num_equality_constraints (qp);
    
    // find the index of p in the active set
    while ((index < qp->q) && (qp->A[index] != j)) 
        index++;
    
    return index;
}

static int
in_active_set (const qp_t *qp, size_t j)
{
    return (index_in_active_set (qp, j) < qp->q);
}

static int
add_constraint (qp_t *qp, size_t p)
{
    size_t j, q, n;
    double c, s;
    int success = 1;
    
    q = qp->q;
    n = qp->d->size;
    
    for (j = n - 1; j >= q + 1; j--)
    {
        gsl_vector_view column_j_minus_1;
        gsl_vector_view column_j;
        
        // compute a givens rotation to zero out d[j]
        gsl_blas_drotg (gsl_vector_ptr (qp->d, j - 1), 
                        gsl_vector_ptr (qp->d, j), 
                        &c, &s);
        
        // update J := J Q^{T}  (J^T = Q J^T)
        column_j_minus_1 = gsl_matrix_column (qp->J, j - 1);
        column_j = gsl_matrix_column (qp->J, j);
        gsl_blas_drot (&column_j_minus_1.vector,
                       &column_j.vector,
                       c, s);
    }
    
    // To update R we have to put the q components of the d vector
    // into column q of R
    gsl_vector_const_view d1 = gsl_vector_const_subvector (qp->d, 0, q + 1);
    gsl_vector_view R_column_q = gsl_matrix_row (qp->Rt, q);
    gsl_vector_view R_sub_column_q = gsl_vector_subvector (&R_column_q.vector,
                                                           0, q + 1);
    gsl_blas_dcopy (&d1.vector, &R_sub_column_q.vector);
    
    // problem degenerate
    if (fabs (gsl_vector_get (qp->d, q)) <= DBL_EPSILON * (qp->R_norm))
    {
        GSL_ERROR ("Could not add constraint.", GSL_EDOM);
        success = 0;
    }
    
    qp->R_norm = GSL_MAX_DBL (qp->R_norm, fabs (gsl_vector_get (qp->d, q)));
    
    qp->A[q] = p;
    qp->q += 1;
    
    return success;
}


static void
drop_constraint (qp_t *qp, size_t k)
{
    size_t l;
    size_t tail_len;
    size_t i;
    double c, s;
    
    l = index_in_active_set (qp, k);
    assert (l < qp->q);
        
    tail_len = qp->q - l - 1;
    
    if (tail_len > 0)
    {
        // remove the constraint from A 
        memmove (qp->A + l, qp->A + l + 1, 
                 tail_len * sizeof (size_t));

        // delete the corresponding column from R
        memmove (gsl_matrix_ptr (qp->Rt, l, 0), 
                 gsl_matrix_ptr (qp->Rt, l + 1, 0),
                 tail_len * qp->Rt->tda * sizeof (double));
    }
    
    // remove the constraint from u
    memmove (gsl_vector_ptr (qp->u, l), 
             gsl_vector_ptr (qp->u, l + 1),
             (tail_len + 1) * sizeof (double));

    // update the size of A
    qp->q -= 1;
    
    // now, we need to apply givens rotations to make R upper triangular
    for (i = l; i < qp->q; i++)
    {
        gsl_vector_view J_column_i, J_column_i_plus_1;
        
        gsl_blas_drotg (gsl_matrix_ptr (qp->Rt, i, i), 
                        gsl_matrix_ptr (qp->Rt, i, i + 1),
                        &c, &s);
        
        // R := Q R
        if (i < qp->q - 1)
        {
            gsl_matrix_view Rt_tail = gsl_matrix_submatrix (qp->Rt, i + 1, i, qp->q - i - 1, 2);
            gsl_vector_view R_tail_fst_row = gsl_matrix_column (&Rt_tail.matrix, 0);
            gsl_vector_view R_tail_snd_row = gsl_matrix_column (&Rt_tail.matrix, 1);
            gsl_blas_drot (&R_tail_fst_row.vector, &R_tail_snd_row.vector, c, s);
        }
        
        // J := J Q^T 
        J_column_i = gsl_matrix_column (qp->J, i);
        J_column_i_plus_1 = gsl_matrix_column (qp->J, i + 1);
        gsl_blas_drot (&J_column_i.vector, &J_column_i_plus_1.vector, c, s);
    }
}

// d := J^T n    
static void
compute_d (qp_t *qp, const gsl_vector *n_plus)
{
    gsl_blas_dgemv (CblasTrans, 1.0, qp->J, n_plus, 0.0, qp->d);     
}

// z := J_2 d_2
static void
update_z (qp_t *qp)
{
    size_t q = qp->q;
    size_t n = qp->J->size1;
    
    gsl_matrix_const_view J2 = gsl_matrix_const_submatrix (qp->J, 0, q, n, n - q);
    gsl_vector_const_view d2 = gsl_vector_const_subvector (qp->d, q, n - q);

    gsl_blas_dgemv (CblasNoTrans, 1.0, &J2.matrix, &d2.vector, 0.0, qp->z);             
}

// r := R^{-1} d1
static void
update_r (qp_t *qp)
{
    size_t q = qp->q;
    
    if (q > 0)
    {
        gsl_matrix_const_view R1 = gsl_matrix_const_submatrix (qp->Rt, 0, 0, q, q);
        gsl_vector_const_view d1 = gsl_vector_const_subvector (qp->d, 0, q);
        gsl_vector_view r1 = gsl_vector_subvector (qp->r, 0, q);
        
        gsl_blas_dcopy (&d1.vector, &r1.vector);
        gsl_blas_dtrsv (CblasLower, CblasTrans, CblasNonUnit, 
                        &R1.matrix, &r1.vector); 
    }
}


static void
compute_z_and_r (qp_t *qp, const gsl_vector *n_plus)
{
    compute_d (qp, n_plus);
    update_z (qp);
    update_r (qp);
}


static double
partial_step_length (const qp_t *qp, size_t *k)
{
    double t, t1 = GSL_POSINF;
    double r_j, u_j;
    size_t j;
    size_t l = 0;
    size_t q = qp->q;
    size_t m_e = num_equality_constraints (qp);
    
    for (j = m_e; j < q; j++)
    {
        r_j = gsl_vector_get (qp->r, j);
        
        if (r_j > 0.0)
        {
            u_j = gsl_vector_get (qp->u, j);
            t = u_j / r_j;
            
            if (t < t1)
            {
                t1 = t;
                l = j;
            }
        }
    }

    if (l >= m_e)
        *k = qp->A[l];

    return t1;
}

// compute the full step length t2: i.e., the minimum step in primal 
// space s.t. the contraint becomes feasible
static double
full_step_length (const qp_t *qp, double s_p, const gsl_vector *n_plus)
{
    double np_dot_z;
    double t2 = GSL_POSINF;
    
    if (gsl_blas_dnrm2 (qp->z) > DBL_EPSILON)
    {
        gsl_blas_ddot (n_plus, qp->z, &np_dot_z);
        t2 = -s_p / np_dot_z;
    }
    
    return t2;
}

// take a step in the dual space
static void
dual_step (qp_t *qp, double t)
{
    size_t q = qp->q;
    
    // u+ := u+ + t [-r 1]
    if (q > 0)
    {
        gsl_vector_view u_sub = gsl_vector_subvector (qp->u, 0, q);
        gsl_vector_const_view r_sub = gsl_vector_const_subvector (qp->r, 0, q);
        gsl_blas_daxpy (-t, &r_sub.vector, &u_sub.vector);
    }
    gsl_vector_set (qp->u, q, gsl_vector_get (qp->u, q) + t);
}

// take a step in both the primal and the dual space
static void
primal_dual_step (qp_t *qp, double t, const gsl_vector *n)
{
    size_t q = qp->q;
    double n_dot_z;
    double u_q;

    // x := x + t z
    gsl_blas_daxpy (t, qp->z, qp->x);
    
    // f := f + t n^T z (0.5 t + u^+[q])
    gsl_blas_ddot (n, qp->z, &n_dot_z);
    u_q = gsl_vector_get (qp->u, q);
    qp->f += t * n_dot_z * (0.5 * t + u_q);

    dual_step (qp, t);
}

static int
choose_violated_constraint (qp_t *qp, size_t *p, gsl_vector_view *n, double *s)
{
    int success = 0;
    double n_dot_x;
    size_t m_e = num_equality_constraints (qp);
    size_t m = num_inequality_constraints (qp);
    size_t q = qp->q;
    
    // if there are any remaining equality constraints, choose one of them
    if (q < m_e)
    {
        *p = q;
        
        *n = trans_matrix_row (qp->TransCE, (gsl_matrix *) (qp->CE), *p);
        
        gsl_blas_ddot (&n->vector, qp->x, &n_dot_x);
        *s = n_dot_x - gsl_vector_get (qp->be, *p);
        
        success = 1;
    }
    // otherwise, choose the non-negativity constraint that is most violated
    else if (q < m_e + m)
    {
        // compute by how much each constraint is violated
        double s_data[m];
        gsl_vector_view s_vec = gsl_vector_view_array (s_data, m);
                                                                   
        gsl_blas_dcopy (qp->b, &s_vec.vector);
        gsl_blas_dgemv (qp->TransC, 
                        1.0, qp->C, qp->x, -1.0, &s_vec.vector);
        
        *p = gsl_vector_min_index (&s_vec.vector);
        *s = gsl_vector_get (&s_vec.vector, *p);
        *n = trans_matrix_row (qp->TransC, (gsl_matrix *) (qp->C), *p);
        
        if ((*s < -1e-12) && // - 0.1 * DBL_EPSILON * (qp->G_cond)) && 
            !in_active_set (qp, *p))
        {
            success = 1;
        }
    }
    
    return success;
}


static int
enforce_constraints (qp_t *qp)
{
    int failed = 0;
    size_t p, k = 0;
    gsl_vector_view n;
    double s;
    double t, t1, t2;
    int full_step;
    
    // (1) Choose a violated constraint, if any
    while (choose_violated_constraint (qp, &p, &n, &s))
    {
        full_step = 0;
        
        // u^+ := [u 0]
        gsl_vector_set (qp->u, qp->q, 0.0);
        
        while (!full_step)
        {
            // (2a) Determine step direction
            compute_z_and_r (qp, &n.vector);
        
            // (2b) Compute step length
            t1 = partial_step_length (qp, &k);
            t2 = full_step_length (qp, s, &n.vector);
            t = GSL_MIN_DBL (t1, t2);
        
            // (2c.i) No step in primal or dual space
            if (gsl_isinf (t)) 
            {
                GSL_ERROR ("No step in primal or dual space.  "
                           "The QPP is infeasible.", 
                           GSL_EDOM);
                failed = 1;
                break;
            }
        
            // (2c.ii) Step in dual space
            else if (gsl_isinf (t2))
            {
                dual_step (qp, t);
                drop_constraint (qp, k);
            }
        
            // (2c.iii) Step in primal and dual space
            else
            {
                primal_dual_step (qp, t, &n.vector);
            
                if (t == t2)
                {
                    add_constraint (qp, p);
                    full_step = 1;
                }
                if (t == t1)
                {
                    drop_constraint (qp, k);
                    s *= (t2 - t) / t2;
                }
            }
        }
        
        // if we got here, then the QPP is infeasible
        if (!full_step)
            break;
    }
    
    return failed;
}


// G is destroyed
int 
gsl_qp_solve (CBLAS_TRANSPOSE_t TransC, CBLAS_TRANSPOSE_t TransCE,
              gsl_matrix *G, const gsl_vector *a, 
              const gsl_matrix *C, const gsl_vector *b, 
              const gsl_matrix *CE, const gsl_vector *be,
              gsl_vector *x, double *f, void *work)
{
    int success;
    
    // G := L L^{T}
    gsl_linalg_cholesky_decomp (G);
    success = gsl_qp_solve_with_chol (CblasLower, TransC, TransCE, 
                                      G, a, C, b, CE, be, x, f, work);
    
    return success;
}


int 
gsl_qp_solve_with_chol (CBLAS_UPLO_t uplo,
                        CBLAS_TRANSPOSE_t TransC, CBLAS_TRANSPOSE_t TransCE,
                        const gsl_matrix *G_chol, const gsl_vector *a, 
                        const gsl_matrix *C, const gsl_vector *b, 
                        const gsl_matrix *CE, const gsl_vector *be,
                        gsl_vector *x, double *f, void *work)
{
    int err;
    qp_t qp;
    
    qp_init (&qp, uplo, TransC, TransCE, G_chol, a, C, b,  CE, be, x, work);

    err = enforce_constraints (&qp);
    *f = qp.f;

    return err;
}
