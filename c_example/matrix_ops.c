#include <Python.h>
#include <numpy/arrayobject.h>

#ifdef __APPLE__
    #include <Accelerate/Accelerate.h>
    #define USE_ACCELERATE 1
#else
    #define USE_ACCELERATE 0
#endif

/*-------------------------------------------------------------------------*/
void printmatrix(int m, int n, const double * A, int lda, const char * name) {
	int i, j;
	if (m*n > 0) {
		printf("\n ");
		printf(name);
		printf(" = \n\n");
		for(i=0;i<m;i++){
			for(j=0;j<n;j++){
				if (A[i+j*lda]>0.0){printf("   %3.4e", A[i+j*lda]);}
				else if (A[i+j*lda]<0.0){printf("  %3.4e", A[i+j*lda]);}
				else {printf("   0         ");}
			}
			printf("\n");
		}
		printf("\n\n");
	}
	else {
		printf("\n ");
		printf(name);
		printf("= []");
	}
}
/*-------------------------------------------------------------------------*/



// Matrix-vector multiplication: result = matrix * vector
// matrix: m x n, vector: n x 1, result: m x 1
static PyObject* matrix_vector_multiply(PyObject* self, PyObject* args) {
    PyArrayObject *matrix, *vector, *result;
    
    // Parse input arguments
    if (!PyArg_ParseTuple(args, "OO", &matrix, &vector)) {
        return NULL;
    }
    
    // Check if inputs are numpy arrays
    if (!PyArray_Check(matrix) || !PyArray_Check(vector)) {
        PyErr_SetString(PyExc_TypeError, "Arguments must be numpy arrays");
        return NULL;
    }
    
    // Get dimensions
    int m = PyArray_DIM(matrix, 0);  // rows of matrix
    int n = PyArray_DIM(matrix, 1);  // cols of matrix
    int vec_len = PyArray_DIM(vector, 0);  // length of vector
    
    // Check dimension compatibility
    if (n != vec_len) {
        PyErr_SetString(PyExc_ValueError, "Matrix columns must equal vector length");
        return NULL;
    }
    
    // Create output array
    npy_intp dims[1] = {m};
    result = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    if (result == NULL) {
        return NULL;
    }
    
    // Get data pointers
    double *mat_data = (double*)PyArray_DATA(matrix);
    double *vec_data = (double*)PyArray_DATA(vector);
    double *res_data = (double*)PyArray_DATA(result);
    
#if USE_ACCELERATE
    // Use BLAS dgemv for optimized matrix-vector multiplication
    // dgemv: y = alpha * A * x + beta * y
    // CblasRowMajor: matrix is stored in row-major order (C-style)
    // CblasNoTrans: don't transpose the matrix
    cblas_dgemv(CblasRowMajor, CblasNoTrans, 
                m, n,           // matrix dimensions
                1.0,            // alpha = 1.0
                mat_data, n,    // matrix A and leading dimension
                vec_data, 1,    // vector x and increment
                0.0,            // beta = 0.0
                res_data, 1);   // vector y and increment
#else
    // Fallback to manual implementation
    for (int i = 0; i < m; i++) {
        res_data[i] = 0.0;
        for (int j = 0; j < n; j++) {
            res_data[i] += mat_data[i * n + j] * vec_data[j];
        }
    }
#endif
    
    return (PyObject*)result;
}

// Vector dot product
static PyObject* vector_dot_product(PyObject* self, PyObject* args) {
    PyArrayObject *vec1, *vec2;
    
    if (!PyArg_ParseTuple(args, "OO", &vec1, &vec2)) {
        return NULL;
    }
    
    if (!PyArray_Check(vec1) || !PyArray_Check(vec2)) {
        PyErr_SetString(PyExc_TypeError, "Arguments must be numpy arrays");
        return NULL;
    }
    
    int len1 = PyArray_DIM(vec1, 0);
    int len2 = PyArray_DIM(vec2, 0);
    
    if (len1 != len2) {
        PyErr_SetString(PyExc_ValueError, "Vectors must have same length");
        return NULL;
    }
    
    double *data1 = (double*)PyArray_DATA(vec1);
    double *data2 = (double*)PyArray_DATA(vec2);
    
#if USE_ACCELERATE
    // Use BLAS ddot for optimized dot product
    double result = cblas_ddot(len1, data1, 1, data2, 1);
#else
    // Fallback to manual implementation
    double result = 0.0;
    for (int i = 0; i < len1; i++) {
        result += data1[i] * data2[i];
    }
#endif
    
    return PyFloat_FromDouble(result);
}

// Additional BLAS operations using Accelerate framework
#if USE_ACCELERATE

// Matrix-matrix multiplication: C = A * B
static PyObject* matrix_matrix_multiply(PyObject* self, PyObject* args) {
    PyArrayObject *A, *B, *C;
    
    if (!PyArg_ParseTuple(args, "OO", &A, &B)) {
        return NULL;
    }
    
    if (!PyArray_Check(A) || !PyArray_Check(B)) {
        PyErr_SetString(PyExc_TypeError, "Arguments must be numpy arrays");
        return NULL;
    }
    
    int m = PyArray_DIM(A, 0);  // rows of A
    int k = PyArray_DIM(A, 1);  // cols of A
    int k2 = PyArray_DIM(B, 0); // rows of B
    int n = PyArray_DIM(B, 1);  // cols of B
    
    if (k != k2) {
        PyErr_SetString(PyExc_ValueError, "Matrix dimensions incompatible for multiplication");
        return NULL;
    }
    
    // Create output array
    npy_intp dims[2] = {m, n};
    // C = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
// Create a new array in column-major order
    C = PyArray_New(
        &PyArray_Type,        // Subtype (PyArray_Type for a standard array)
        2,                    // Number of dimensions
        dims,                 // Dimensions array
        NPY_DOUBLE,            // Data type (e.g., float)
        NULL,                 // Strides (NULL for default calculation)
        NULL,                 // Data pointer (NULL for uninitialized data)
        0,                    // Itemsize (0 for default based on type_num)
        NPY_ARRAY_F_CONTIGUOUS, // Flags: specify column-major order
        NULL                  // Base object (NULL if not a view)
    );


    if (C == NULL) {
        return NULL;
    }
    // PyArray_CLEARFLAGS((PyArrayObject*)C, NPY_ARRAY_C_CONTIGUOUS);
    // PyArray_ENABLEFLAGS((PyArrayObject*)C, NPY_ARRAY_F_CONTIGUOUS);
    
    double *A_data = (double*)PyArray_DATA(A);
    double *B_data = (double*)PyArray_DATA(B);
    double *C_data = (double*)PyArray_DATA(C);

    printmatrix(2,2,A_data,2,"A");
    printmatrix(2,2,B_data,2,"B");
    
    // Use BLAS dgemm: C = alpha * A * B + beta * C
    // cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
    //             m, n, k,        // dimensions
    //             1.0,            // alpha = 1.0
    //             A_data, k,      // matrix A and leading dimension
    //             B_data, n,      // matrix B and leading dimension
    //             0.0,            // beta = 0.0
    //             C_data, n);     // matrix C and leading dimension
    
    // Use Fortran BLAS dgemm: C = alpha * A * B + beta * C
    char transa = 'N';  // No transpose for A
    char transb = 'N';  // No transpose for B
    int m_f = m, n_f = n, k_f = k;  // Dimensions
    double alpha = 1.0;
    double beta = 0.0;
    int lda = k;  // Leading dimension of A
    int ldb = n;  // Leading dimension of B  
    int ldc = n;  // Leading dimension of C

    dgemm_(&transa, &transb, &m_f, &n_f, &k_f,
        &alpha,
        A_data, &lda,
        B_data, &ldb,
        &beta,
        C_data, &ldc);


    printmatrix(2,2,C_data,2,"C");

    return (PyObject*)C;
}

// Vector scaling: y = alpha * x
static PyObject* vector_scale(PyObject* self, PyObject* args) {
    PyArrayObject *vector, *result;
    double alpha;
    
    if (!PyArg_ParseTuple(args, "Od", &vector, &alpha)) {
        return NULL;
    }
    
    if (!PyArray_Check(vector)) {
        PyErr_SetString(PyExc_TypeError, "First argument must be numpy array");
        return NULL;
    }
    
    int n = PyArray_DIM(vector, 0);
    
    // Create output array
    npy_intp dims[1] = {n};
    result = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    if (result == NULL) {
        return NULL;
    }
    
    double *vec_data = (double*)PyArray_DATA(vector);
    double *res_data = (double*)PyArray_DATA(result);
    
    // Copy vector first
    cblas_dcopy(n, vec_data, 1, res_data, 1);
    // Scale the result
    cblas_dscal(n, alpha, res_data, 1);
    
    return (PyObject*)result;
}

// Vector addition: z = alpha * x + y
static PyObject* vector_axpy(PyObject* self, PyObject* args) {
    PyArrayObject *x, *y, *result;
    double alpha;
    
    if (!PyArg_ParseTuple(args, "dOO", &alpha, &x, &y)) {
        return NULL;
    }
    
    if (!PyArray_Check(x) || !PyArray_Check(y)) {
        PyErr_SetString(PyExc_TypeError, "Arguments must be numpy arrays");
        return NULL;
    }
    
    int nx = PyArray_DIM(x, 0);
    int ny = PyArray_DIM(y, 0);
    
    if (nx != ny) {
        PyErr_SetString(PyExc_ValueError, "Vectors must have same length");
        return NULL;
    }
    
    // Create output array
    npy_intp dims[1] = {nx};
    result = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    if (result == NULL) {
        return NULL;
    }
    
    double *x_data = (double*)PyArray_DATA(x);
    double *y_data = (double*)PyArray_DATA(y);
    double *res_data = (double*)PyArray_DATA(result);
    
    // Copy y to result first
    cblas_dcopy(nx, y_data, 1, res_data, 1);
    // Add alpha * x to result
    cblas_daxpy(nx, alpha, x_data, 1, res_data, 1);
    
    return (PyObject*)result;
}

#endif // USE_ACCELERATE

// Get information about Accelerate framework usage
static PyObject* get_accelerate_info(PyObject* self, PyObject* args) {
    PyObject *info_dict = PyDict_New();
    if (info_dict == NULL) {
        return NULL;
    }
    
#if USE_ACCELERATE
    PyDict_SetItemString(info_dict, "using_accelerate", Py_True);
    PyDict_SetItemString(info_dict, "backend", PyUnicode_FromString("Apple Accelerate Framework"));
#else
    PyDict_SetItemString(info_dict, "using_accelerate", Py_False);
    PyDict_SetItemString(info_dict, "backend", PyUnicode_FromString("Manual C implementation"));
#endif
    
    return info_dict;
}

// Method definitions
static PyMethodDef MatrixOpsMethods[] = {
    {"matrix_vector_multiply", matrix_vector_multiply, METH_VARARGS,
     "Multiply a matrix by a vector using optimized BLAS (macOS) or manual implementation"},
    {"vector_dot_product", vector_dot_product, METH_VARARGS,
     "Compute dot product of two vectors using optimized BLAS (macOS) or manual implementation"},
    {"get_accelerate_info", get_accelerate_info, METH_NOARGS,
     "Get information about whether Accelerate framework is being used"},
#if USE_ACCELERATE
    {"matrix_matrix_multiply", matrix_matrix_multiply, METH_VARARGS,
     "Multiply two matrices using BLAS dgemm (macOS only)"},
    {"vector_scale", vector_scale, METH_VARARGS,
     "Scale a vector by a scalar using BLAS dscal (macOS only)"},
    {"vector_axpy", vector_axpy, METH_VARARGS,
     "Compute alpha * x + y using BLAS daxpy (macOS only)"},
#endif
    {NULL, NULL, 0, NULL}
};

// Module definition
static struct PyModuleDef matrixopsmodule = {
    PyModuleDef_HEAD_INIT,
    "matrix_ops",
    "Matrix operations in C",
    -1,
    MatrixOpsMethods
};

// Module initialization
PyMODINIT_FUNC PyInit_matrix_ops(void) {
    import_array();  // Initialize NumPy C API
    return PyModule_Create(&matrixopsmodule);
}