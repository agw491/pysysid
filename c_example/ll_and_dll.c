#include <Python.h>
#include <numpy/arrayobject.h>

#ifdef __APPLE__
    #include <Accelerate/Accelerate.h>
    #define USE_ACCELERATE 1
#else
    #define USE_ACCELERATE 0
#endif

/*-------------------------------------------------------------------------*/
void printvector(int size, const double * vec, const char * name) {
	int i;
	if (size > 0) {
		printf("\n ");
		printf(name);
		printf(" = \n\n");
		for(i=0;i<size;i++){
			if (vec[i] >= 0.0){printf("          %3.5e\n", vec[i]);}
			else {printf("         %3.5e\n", vec[i]);}
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

/*-------------------------------------------------------------------------*/
/*
 * rchol: Robust Cholesky factorisation
 *
 * This function takes a symmetric positive semi-definite (SPS)
 * matrix A and returns an upper triangular matrix U such that
 *
 * A = U^T U
 */

/* ZERO defines the threshold below which
 * doubles are considered to be 0         */
#define ZERO 1e-15
#define EPS  1e-15

varint rchol(varint n, double * A, varint lda) {
	varint j, k, mk, itmp;
	double alpha, tol;
	varint error=0;
	varint ione =  1;
	double done   =  1.0;
    double dmone  = -1.0;
    double dzero  =  0.0;
	
	/* Perform the factorisation */
    tol = n*EPS;
    if(A[0]<= tol){
        dscal(&n,&dzero,A,&lda);
    } else {
        alpha = 1.0/sqrt(A[0]);
        dscal(&n,&alpha,A,&lda);
    }
    for(j=1;j<n;j++){
        k = n-j;
        dgemv("T", &j, &k, &dmone, &A[j*lda], &lda, &A[j*lda], &ione, &done, &A[j*lda+j], &lda);
        if(A[j*lda+j]<=tol){
            dscal(&k,&dzero,&A[j*lda+j],&lda);
        } else {
            alpha = 1.0/sqrt(A[lda*j+j]);
            dscal(&k,&alpha,&A[lda*j+j],&lda);
        }
    }
    return 0;
}

void rfbs(char *lor, char *uplo, char *tran, char *unit, int n, int m, double alp, double *A, int lda, double *B, int ldb) 
{	
    int i, j, k, mk, itmp;
	double alpha, tol;
	int error=0;
	int ione =  1;
	double done   =  1.0;
    double dmone  = -1.0;
    double dzero  =  0.0;
    
    tol = n*EPS;
    
    if(tran[0]=='N'){
        for(i=0;i<m;i++){
            if(fabs(A[(n-1)*lda+n-1])>tol){
                B[(n-1)+i*(ldb)] = B[(n-1)+i*(ldb)]/A[(n-1)*lda+n-1];
            } else {
                B[i*ldb] = 0.0;
            }
            for(j=n-2;j>=0;j--){
                if(fabs(A[j+j*lda])>tol){
                    for(k=j+1;k<n;k++){
                        B[j+i*ldb] -= A[j+k*lda]*B[k+i*ldb];
                    }
                    B[j+i*ldb] = B[j+i*ldb]/A[j+j*lda];
                }  else {
                    B[j+i*ldb] = 0.0;
                }
            }
        }
    } else {
        for(i=0;i<m;i++){
            if(fabs(A[0])>tol){
                B[i*ldb] = B[i*ldb]/A[0];
            } else {
                B[i*ldb] = 0.0;
            }
            for(j=1;j<n;j++){
                if(fabs(A[j+j*lda])>tol){
                    for(k=0;k<j;k++){
                        B[j+i*ldb] -= A[j*lda+k]*B[k+i*ldb];
                    }
                    B[j+i*ldb] = B[j+i*ldb]/A[j+j*lda];
                } else {
                    B[j+i*ldb] = 0.0;
                }
            }
        }
    }
}

void ll_and_dll(int n,
                int m,
                int p,
                int na,
                int nb,
                int N,
                int smoothing,
                double *y,
                double *u,
                double *a,
                double *b,
                double *theta,
                double *X1,
                double *P1,
                double *LL,
                double *dLL,
                double *yout,
                double *thn,
                double *xs)
{
	double *A, *B, *C, *D;
	double *Ai, *Bi, *Ci, *Di;
	double *Pp, *Pf, *Ps, *xp, *xf, *yp, *yf, *ys, *pe, *fe, *se; //*xs
	double *Ri, *K, *PE, *H, *R1, *R2, *R3, *R4, *R5, *R6, *X;
	double dtmp, *work, *Riep;
	double *Ms, *APf, *AP, *KCN, *Jt, *Jtp1, *APp1, *v, *Pt, *Pt1;
	double *mem, *Gamma, *Pi, *Pih, *Phi, *Psi, *Sig, *AA;
	
	int isB, isD, isF, isG;
	int k, i, j, t, ii, jj, kk;
	int type, strlen, error, ksem, bilin;
	int tva, tvb, tvc, tvd, tvq, tvs, tvr;
	int tta, ttb, ttc, ttd, ttq, tts, ttr;
	int ldh, itmp1, itmp2, itmp3, itmp4;
	int lwork, info, idx, ldg;
	
    // Extract Gamma and Pi from theta
    ldg   = p+n;
    Gamma = theta;
    i     = n*n*na + n*m*nb + p*n*na + p*m*nb;
    Pih   = &theta[i];

    Ci    = Gamma;
    Di    = &Gamma[ldg*n*na];
    Ai    = &Gamma[p];
    Bi    = &Gamma[p+ldg*n*na];
	
//     printmatrix(p,n*na,Ci,ldg,"C");
//     printmatrix(p,m*nb,Di,ldg,"D");
//     printmatrix(n,n*na,Ai,ldg,"A");
//     printmatrix(n,m*nb,Bi,ldg,"B");

	/* Define variables for use with dpotrf_*/
	int dpot_n, dpot_lda, itmp;
	
	int ione  = 1;
	int izero = 0;
	double done  =  1.0;
	double dzero =  0.0;
	double dmone = -1.0;
	
    // Setup for forcing smoothing and calculating the required matrices for the gradient 
    ksem      = 1;
    tva       = 1;
    tvb       = 1;
    tvc       = 1;
    tvd       = 1;
    tvq       = 0;
    tvs       = 0;
    tvr       = 0;

	/* Create some output matrices */
    /* Calculate optimal work size for QR */
	itmp  = (3*n+p)*(3*n+p);
	lwork = -1;
	dgeqrf(&itmp, &itmp, R1, &itmp, &dtmp, &dtmp, &lwork, &info);
	lwork = (int)dtmp;
    ldh = na*n + nb*m + n + p;

    i    = 0;
    i    = i + lwork+itmp;
    i    = i + ldh*ldh;
    i    = i + ldh;
    i    = i + n*(N+1);
    i    = i + n*(N);
    i    = i + p*(N);
    i    = i + p*(N);
    i    = i + p*(N);
    i    = i + p*(N);
    i    = i + 1;
    i    = i + n*n*(N+1);
    i    = i + n*n*(N);
    i    = i + p*p*(N);
    i    = i + n*p*(N);
    i    = i + (n+p)*(n+p);
    i    = i + (n+p)*(n+p);
    i    = i + (2*n)*(n);	
    i    = i + (2*n)*(n);	
    i    = i + (n+p)*(n+p);	
    i    = i + p;	
    //i    = i + n*(N+1);	
    i    = i + p*(N);	
    i    = i + p*(N);	
    i    = i + n*n*(N+1);	
    i    = i + n*n*(N);	
    i    = i + 4*n*2*n;	
    i    = i + n*n;	
    i    = i + n*n;	
    i    = i + n*n;	
    i    = i + n*n;	
    i    = i + n*n;	
    i    = i + n*n;	
    i    = i + n*n*N;	
    i    = i + n*m*N;	
    i    = i + p*n*N;	
    i    = i + p*m*N;	    
    i    = i + (n+p)*(n+p);
    i    = i + (n+p)*(n+p);

    //printf("memSize = %d\n",i);

    //Create memory and zero it
    mem  = (double *)malloc(i*sizeof(double));
    dtmp = 0.0;
    dcopy(&i,&dtmp,&izero,mem,&ione);

    i    = 0;
    work = &mem[i];
    i    = i + lwork+itmp;
    H    = &mem[i];
    i    = i + ldh*ldh;
    v    = &mem[i];
    i    = i + ldh;
    xp   = &mem[i];
    i    = i + n*(N+1);
    xf   = &mem[i];
    i    = i + n*(N);
    yp   = &mem[i];
    i    = i + p*(N);
    yf   = &mem[i];
    i    = i + p*(N);
    pe   = &mem[i]; 
    i    = i + p*(N);
    fe   = &mem[i]; 
    i    = i + p*(N);
    PE   = &mem[i]; 
    i    = i + 1;
    Pp   = &mem[i]; 
    i    = i + n*n*(N+1);
    Pf   = &mem[i]; 
    i    = i + n*n*(N);
    Ri   = &mem[i]; 
    i    = i + p*p*(N);
    K    = &mem[i]; 
    i    = i + n*p*(N);
    R1   = &mem[i]; 
    i    = i + (n+p)*(n+p);
    R2   = &mem[i]; 
    i    = i + (n+p)*(n+p);
    R3   = &mem[i]; 
    i    = i + (2*n)*(n);	
    R4   = &mem[i]; 
    i    = i + (2*n)*(n);	
    X    = &mem[i]; 
    i    = i + (n+p)*(n+p);	
	Riep = &mem[i]; 
    i    = i + p;	
	//xs   = &mem[i]; 
    //i    = i + n*(N+1);	
    ys   = &mem[i]; 
    i    = i + p*(N);	
    se   = &mem[i]; 
    i    = i + p*(N);	
    Ps   = &mem[i]; 
    i    = i + n*n*(N+1);	
    Ms   = &mem[i]; 
    i    = i + n*n*(N);	
    R5   = &mem[i]; 
    i    = i + 4*n*2*n;	
    APf  = &mem[i]; 
    i    = i + n*n;	
    AP   = &mem[i]; 
    i    = i + n*n;	
    KCN  = &mem[i]; 
    i    = i + n*n;	
    Jt   = &mem[i]; 
    i    = i + n*n;	
    Jtp1 = &mem[i]; 
    i    = i + n*n;	
    APp1 = &mem[i]; 
    i    = i + n*n;	
    A    = &mem[i]; 
    i    = i + n*n*N;	
    B    = &mem[i]; 
    i    = i + n*m*N;	
    C    = &mem[i]; 
    i    = i + p*n*N;	
    D    = &mem[i]; 
    i    = i + p*m*N;	    
    AA   = &mem[i]; 
    i    = i + (n+p)*(n+p);	    
    Pi   = &mem[i]; 
    i    = i + (n+p)*(n+p);	    

	
	/* Initialise the predicted mean and cov */
	dcopy(&n, X1, &ione, xp, &ione);
	itmp = n*n;
	dcopy(&itmp, P1, &ione, Pp, &ione);
	//if(rchol(n, Pp, n)){mexErrMsgTxt("Cholesky Factorisation of P1 failed.");};
    
	/* Handle LPV system by making A, B, C and D time varying */
	itmp1 = n*n;
	itmp2 = n*m;
	itmp3 = p*n;
	itmp4 = p*m;
	dtmp  = 0.0;
	for(t=0;t<N;t++){
        dcopy(&itmp1, &dtmp, &izero, &A[t*n*n], &ione);
		dcopy(&itmp2, &dtmp, &izero, &B[t*n*m], &ione);
		dcopy(&itmp3, &dtmp, &izero, &C[t*p*n], &ione);
		dcopy(&itmp4, &dtmp, &izero, &D[t*p*m], &ione);
        for(i=0;i<na;i++){
            for(k=0;k<n;k++){
                for(j=0;j<n;j++){
                    A[t*n*n+k*n+j] += a[t*na+i]*Ai[i*ldg*n + k*ldg + j];
                }
                for(j=0;j<p;j++){
                    C[t*p*n+k*p+j] += a[t*na+i]*Ci[i*ldg*n + k*ldg + j];
                }
            }
		}
		for(i=0;i<nb;i++){
            for(k=0;k<m;k++){
                for(j=0;j<n;j++){
                    B[t*n*m+k*n+j] += b[t*nb+i]*Bi[i*ldg*m + k*ldg + j];
                }
                for(j=0;j<p;j++){
                    D[t*p*m+k*p+j] += b[t*nb+i]*Di[i*ldg*m + k*ldg + j];
                }
            }
		}
	}
	
//     t=N-1;
//     printmatrix(p,n,&C[t*p*n],p,"C");
//     printmatrix(p,m,&D[t*p*m],p,"D");
//     printmatrix(n,n,&A[t*n*n],n,"A");
//     printmatrix(n,m,&B[t*n*m],n,"B");

	/* Initialise the current time index for the time-varying (or not) matrices */
	tta = 0;
	ttb = 0;
	ttc = 0;
	ttd = 0;
	ttq = 0;
	tts = 0;
	ttr = 0;
	
	/* Clear X and set it up */
	dtmp = 0.0;
	itmp = (p+n)*(p+n);
	dcopy(&itmp, &dtmp, &izero, X, &ione);
    k = 0;
    for(i=0;i<p+n;i++){
        for(j=0;j<=i;j++){
            Pi[j + i*(n+p)] = Pih[k];
            X[j + i*(n+p)]  = Pih[k];
            k++;
        }
    }
	
//     itmp = p+n;
//     printmatrix(itmp,itmp,X,itmp,"X");

	itmp = p+n;
	rfbs("L", "U", "N", "N", p, n, done, X, itmp, &X[p*itmp], itmp);
	
    

	/*Main Filter loop starts here */
	for(t=0;t<N;t++){		
		if ((tva==1) || (tvc==1) || (t==0)){
			dtmp = -1.0;
			itmp = p+n;
			dgemm("T", "N", &n, &n, &p, &dtmp, &X[p*itmp], &itmp, &C[ttc*p*n], &p, &done, &A[tta*n*n], &n);
		}
		if ((tvb==1) || (tvd==1) || (t==0)){
			dtmp = -1.0;
			itmp = p+n;
			dgemm("T", "N", &n, &m, &p, &dtmp, &X[p*itmp], &itmp, &D[ttd*p*m], &p, &done, &B[ttb*n*m], &n);
		}
		itmp = (n+p)*(n+p);
		dtmp = 0.0;
		dcopy(&itmp, &dtmp, &izero, R1, &ione);
		for(i=0;i<p;i++){
			for(j=0;j<=i;j++){
				R1[i*(p+n)+j] = X[i*(p+n)+j];
			}
		}
		for(i=0;i<p;i++){
			for(j=0;j<n;j++){
				R1[p+j+i*(p+n)] = C[ttc*p*n+i+j*p];
			}
		}
		itmp = p+n;
		dtrmm("L", "U", "N", "N", &n, &p, &done, &Pp[t*n*n], &n, &R1[p], &itmp);
		for(i=0;i<n;i++){
			for(j=0;j<=i;j++){
				R1[(i+p)*(p+n)+p+j] = Pp[t*n*n+j+i*n];
			}
		}        
		itmp = p+n;
		dgeqrf(&itmp, &itmp, R1, &itmp, work, &work[itmp], &lwork, &info);
        //qr(p+n,p+n,R1,p+n);
		if(info!=0){mexErrMsgTxt("QR factorization of R1 failed.");}
		
		for(i=0;i<p;i++){
			for(j=0;j<=i;j++){
				Ri[t*p*p+j+i*p] = R1[j+i*(p+n)];
			}
		}
		
		itmp = p+n;
		rfbs("L", "U", "N", "N", p, n, done, R1, itmp, &R1[p*itmp], itmp);
		for(i=0;i<p;i++){
			for(j=0;j<n;j++){
				K[t*n*p+j+i*n] = R1[(j+p)*(p+n)+i];
			}
		}
		
		for(i=0;i<n;i++){
			for(j=0;j<=i;j++){
				Pf[t*n*n+j+i*n] = R1[(i+p)*(p+n)+p+j];
			}
		}
		
		dtmp = 0.0;
		dgemv("N", &p, &n, &done, &C[ttc*p*n], &p, &xp[t*n], &ione, &dtmp, &yp[t*p], &ione);
		if(m>0){dgemv("N", &p, &m, &done, &D[ttd*p*m], &p, &u[t*m], &ione, &done, &yp[t*p], &ione);}
		dcopy(&p, &y[t*p], &ione, &pe[t*p], &ione);
		dtmp = -1.0;
		daxpy(&p, &dtmp, &yp[t*p], &ione, &pe[t*p], &ione);
		for(i=0;i<p;i++){PE[0] += pe[t*p+i]*pe[t*p+i];}
		
		dcopy(&p, &pe[t*p], &ione, Riep, &ione);
		itmp = p+n;
		rfbs("L", "U", "T", "N", p, ione, done, R1, itmp, Riep, p);
		for(i=0;i<p;i++){LL[0] += Riep[i]*Riep[i] + 2.0*log(fabs(R1[i+i*(p+n)]));}
		
		dcopy(&n, &xp[t*n], &ione, &xf[t*n], &ione);
		itmp = p+n;
		dgemv("T", &p, &n, &done, &R1[p*(p+n)], &itmp, &pe[t*p], &ione, &done, &xf[t*n], &ione);
		
		
		dtmp = 0.0;
		dgemv("N", &p, &n, &done, &C[ttc*p*n], &p, &xf[t*n], &ione, &dtmp, &yf[t*p], &ione);
		if(m>0){dgemv("N", &p, &m, &done, &D[ttd*p*m], &p, &u[t*m], &ione, &done, &yf[t*p], &ione);}
		dcopy(&p, &y[t*p], &ione, &fe[t*p], &ione);
		dtmp = -1.0;
		daxpy(&p, &dtmp, &yf[t*p], &ione, &fe[t*p], &ione);
		
		dtmp = 0.0;
		dgemv("N", &n, &n, &done, &A[tta*n*n], &n, &xf[t*n], &ione, &dtmp, &xp[(t+1)*n], &ione);
		if(m>0){dgemv("N", &n, &m, &done, &B[ttb*n*m], &n, &u[t*m], &ione, &done, &xp[(t+1)*n], &ione);}
		itmp = p+n;
		dgemv("T", &p, &n, &done, &X[p*(p+n)], &itmp, &y[t*p], &ione, &done, &xp[(t+1)*n], &ione);
		
		itmp = 2*n*n;
		dtmp = 0.0;
		dcopy(&itmp, &dtmp, &izero, R3, &ione);
		for(i=0;i<n;i++){
			for(j=0;j<n;j++){
				R3[j+2*i*n] = A[tta*n*n+i+j*n];
			}
		}
		itmp = 2*n;
		dtrmm("L", "U", "N", "N", &n, &n, &done, &Pf[t*n*n], &n, R3, &itmp);
		for(i=0;i<n;i++){
			for(j=0;j<=i;j++){
				R3[n+2*i*n+j] = X[p+j+(p+i)*(n+p)];
			}
		}
		
		itmp = 2*n;
		dgeqrf(&itmp, &n, R3, &itmp, work, &work[itmp], &lwork, &info);
        //qr(2*n,n,R3,2*n);
		if(info!=0){mexErrMsgTxt("QR factorization of R3 failed.");}
		for(i=0;i<n;i++){
			for(j=0;j<=i;j++){
				Pp[(t+1)*n*n+j+i*n] = R3[2*i*n+j];
			}
		}
		
		if(t<N-1){
			tta += tva;
			ttb += tvb;
			ttc += tvc;
			ttd += tvd;
			ttq += tvq;
			tts += tvs;
			ttr += tvr;
		}
		
	}
	
	/* Now if smoothing is NOT required then jump to the end */
	if(smoothing==0){
		goto exit_stub;
	}
		
	/* Set intial values at time t=N+1 */
	t = N-1;
	dcopy(&n, &xp[(t+1)*n], &ione, &xs[(t+1)*n], &ione);
	dcopy(&n, &xf[t*n], &ione, &xs[t*n], &ione);

	itmp = n*n;
	dcopy(&itmp, &Pp[(t+1)*n*n], &ione, &Ps[(t+1)*n*n], &ione);
	dcopy(&itmp, &Pf[t*n*n], &ione, &Ps[t*n*n], &ione);
	itmp = n*n;
	dcopy(&itmp, &A[tta*n*n], &ione, APf, &ione);
	dtrmm("R", "U", "T", "N", &n, &n, &done, &Pf[t*n*n], &n, APf, &n);
	dcopy(&itmp, APf, &ione, AP, &ione);
	dtrmm("R", "U", "N", "N", &n, &n, &done, &Pf[t*n*n], &n, AP, &n);
	dcopy(&itmp, AP, &ione, &Ms[t*n*n], &ione);
	dtmp = 0.0;
	itmp = n*n;
	dcopy(&itmp, &dtmp, &izero, KCN, &ione);
	for(i=0;i<n;i++){ KCN[i+i*n] = 1.0;}
	dtmp = -1.0;
	dgemm("N", "N", &n, &n, &p, &dtmp, &K[t*n*p], &n, &C[ttc*p*n], &p, &done, KCN, &n);
	dtmp = 0.0;
	dgemv("N", &p, &n, &done, &C[ttc*p*n], &p, &xs[t*n], &ione, &dtmp, &ys[t*p], &ione);
	if(m>0){dgemv("N", &p, &m, &done, &D[ttd*p*m], &p, &u[t*m], &ione, &done, &ys[t*p], &ione);}
	dcopy(&p, &y[t*p], &ione, &se[t*p], &ione);
	dtmp = -1.0;
	daxpy(&p, &dtmp, &ys[t*p], &ione, &se[t*p], &ione);
	
	Pt  = R1;
	Pt1 = R3;
	
	if(ksem==1){
		for(i=0;i<na;i++){
			for(j=0;j<n;j++){
				v[n*i+j] = a[t*na+i]*xs[t*n+j];
			}
		}
		idx = na*n;
		for(i=0;i<nb;i++){
			for(j=0;j<m;j++){
				v[idx+m*i+j] = b[t*nb+i]*u[t*m+j];
			}
		}
		idx += nb*m;
		for(j=0;j<p;j++){
			v[idx+j] = y[t*p+j];
		}
		idx += p;
		for(j=0;j<n;j++){
			v[idx+j] = xs[(t+1)*n+j];
		}        
		dtmp = 0.0;
		dsyrk("U", "N", &ldh, &ione, &done, v, &ldh, &dtmp, H, &ldh);
		dtmp = 0.0;
		dgemm("T", "N", &n, &n, &n, &done, &Ps[t*n*n], &n, &Ps[t*n*n], &n, &dtmp, Pt, &n);
		dgemm("T", "N", &n, &n, &n, &done, &Ps[(t+1)*n*n], &n, &Ps[(t+1)*n*n], &n, &dtmp, Pt1, &n);
		
		for (ii=0;ii<na;ii++){
			for(jj=0;jj<=ii;jj++){
				for(i=0;i<n;i++){
					for(j=0;j<n;j++){
						H[ii*ldh*n + jj*n + j+i*ldh] += a[t*na+jj]*a[t*na+ii]*Pt[j+i*n];
					}
				}
			}
		}

		idx = na*n+nb*m+p;
		for(ii=0;ii<na;ii++){
			for(i=0;i<n;i++){
				for(j=0;j<n;j++){
					H[idx*ldh + ii*n + j+i*ldh] += a[t*na+ii]*Ms[t*n*n+j*n+i];
				}
			}
		}
		for(i=0;i<n;i++){
			for(j=0;j<n;j++){
				H[idx*ldh+idx+i*ldh+j] += Pt1[j+i*n];
			}
		}
	}
	
	
	for(t=N-2;t>=0;t--){
		tta -= tva;
		ttb -= tvb;
		ttc -= tvc;
		ttd -= tvd;
		ttq -= tvq;
		tts -= tvs;
		ttr -= tvr;
		
		itmp = n*n;
		dcopy(&itmp, &A[tta*n*n], &ione, APf, &ione);
		dtrmm("R", "U", "T", "N", &n, &n, &done, &Pf[t*n*n], &n, APf, &n);
		dcopy(&itmp, APf, &ione, AP, &ione);
		dtrmm("R", "U", "N", "N", &n, &n, &done, &Pf[t*n*n], &n, AP, &n);
		dcopy(&itmp, AP, &ione, Jt, &ione);
        rfbs("L", "U", "T", "N", n, n, done, &Pp[(t+1)*n*n], n, Jt, n);
		rfbs("L", "U", "N", "N", n, n, done, &Pp[(t+1)*n*n], n, Jt, n);
		itmp = 6*n*n;
		dtmp = 0.0;
		dcopy(&itmp, &dtmp, &izero, R5, &ione);
		for(i=0;i<n;i++){
			for(j=0;j<n;j++){
				R5[j+3*i*n]         = APf[i+j*n];
				R5[2*n+3*n*(i+n)+j] = Jt[j+i*n];
			}
			for(j=0;j<=i;j++){
				R5[3*n*(i+n)+j]     = Pf[t*n*n+i*n+j];
				R5[n+3*i*n+j]       = X[p+j+(p+i)*(n+p)]; //Q[ttq*n*n+i*n+j];
			}
		}
		itmp = 3*n;
		dtrmm("L", "U", "N", "N", &n, &n, &done, &Ps[(t+1)*n*n], &n, &R5[2*n+3*n*n], &itmp);
        itmp = 3*n;
		j    = 2*n;
        dgeqrf(&itmp, &j, R5, &itmp, work, &work[itmp], &lwork, &info);
        //qr(3*n,2*n,R5,3*n);
		if(info!=0){mexErrMsgTxt("QR factorization of R5 failed.");}
		for(i=0;i<n;i++){
			for(j=0;j<=i;j++){
				Ps[t*n*n+i*n+j] = R5[n+3*n*(i+n) + j];
			}
		}
		dcopy(&n, &xs[(t+1)*n], &ione, R1, &ione);
		dtmp = -1.0;
		daxpy(&n, &dtmp, &xp[(t+1)*n], &ione, R1, &ione);
		dcopy(&n, &xf[t*n], &ione, &xs[t*n], &ione);
		dgemv("T", &n, &n, &done, Jt, &n, R1, &ione, &done, &xs[t*n], &ione);
		
		dtmp = 0.0;
		dgemv("N", &p, &n, &done, &C[ttc*p*n], &p, &xs[t*n], &ione, &dtmp, &ys[t*p], &ione);
		if(m>0){dgemv("N", &p, &m, &done, &D[ttd*p*m], &p, &u[t*m], &ione, &done, &ys[t*p], &ione);}
		dcopy(&p, &y[t*p], &ione, &se[t*p], &ione);
		dtmp = -1.0;
		daxpy(&p, &dtmp, &ys[t*p], &ione, &se[t*p], &ione);
		
		if(t==N-2){
			dtmp = 0.0;
			dgemm("N", "N", &n, &n, &n, &done, KCN, &n, AP, &n, &dtmp, &Ms[t*n*n], &n);
		} else {
			dtmp = 0.0;
			dgemm("T", "N", &n, &n, &n, &done, &Pf[(t+1)*n*n], &n, &Pf[(t+1)*n*n], &n, &dtmp, KCN, &n);
			dgemm("N", "N", &n, &n, &n, &done, Jtp1, &n, APp1, &n, &done, KCN, &n);
			dgemm("N", "N", &n, &n, &n, &done, KCN, &n, Jt, &n, &dtmp, &Ms[t*n*n], &n);
		}
		for(i=0;i<n;i++){
			for(j=0;j<n;j++){
				Jtp1[i*n+j] = Jt[j*n+i];
				APp1[i*n+j] = Ms[t*n*n+i*n+j] - AP[i*n+j];
			}
		}
		
		if(ksem==1){
			for(i=0;i<na;i++){
				for(j=0;j<n;j++){
					v[n*i+j] = a[t*na+i]*xs[t*n+j];
				}
			}
			idx = na*n;
			for(i=0;i<nb;i++){
				for(j=0;j<m;j++){
					v[idx+m*i+j] = b[t*nb+i]*u[t*m+j];
				}
			}
			idx += nb*m;
			for(j=0;j<p;j++){
				v[idx+j] = y[t*p+j];
			}            
            idx += p;
            for(j=0;j<n;j++){
				v[idx+j] = xs[(t+1)*n+j];
			}
            
			dsyrk("U", "N", &ldh, &ione, &done, v, &ldh, &done, H, &ldh);
			
			dtmp = 0.0;
			dgemm("T", "N", &n, &n, &n, &done, &Ps[t*n*n], &n, &Ps[t*n*n], &n, &dtmp, Pt, &n);
			dgemm("T", "N", &n, &n, &n, &done, &Ps[(t+1)*n*n], &n, &Ps[(t+1)*n*n], &n, &dtmp, Pt1, &n);
			
			for (ii=0;ii<na;ii++){
				for(jj=0;jj<=ii;jj++){
					for(i=0;i<n;i++){
						for(j=0;j<n;j++){
							H[ii*ldh*n + jj*n + j+i*ldh] += a[t*na+jj]*a[t*na+ii]*Pt[j+i*n];
						}
					}
				}
			}
			
			idx = na*n+nb*m+p;
			for(ii=0;ii<na;ii++){
				for(i=0;i<n;i++){
					for(j=0;j<n;j++){
						H[idx*ldh + ii*n + j+i*ldh] += a[t*na+ii]*Ms[t*n*n+j*n+i];
					}
				}
			}
			for(i=0;i<n;i++){
				for(j=0;j<n;j++){
					H[idx*ldh+idx+i*ldh+j] += Pt1[j+i*n];
				}
			}
		}
	}
	
//     printvector(n,xs,"xs(1)");
//     printmatrix(n,n,Ps,n,"Ps(1)");

    //printmatrix(ldh,ldh,H,ldh,"H");

	if(ksem==1){
		for(i=0;i<ldh;i++){
			for(j=0;j<i;j++){
				H[i+j*ldh] = H[j+i*ldh];
			}
		}
	}

//     printmatrix(ldh,ldh,H,ldh,"H");

    //Now calculate the gradient
//     %Now compute the gradient of Q
//     R1i = 1 : nx*na+nu*nb;
// 	   R2i = nx*na+nu*nb+1 : nx*na+nu*nb + nx + ny;
//     Phi = G.H(R2i,R2i);
//     Phi = Phi([nx+1:nx+ny,1:nx],[nx+1:nx+ny,1:nx]);
//     Psi = G.H(R2i,R1i);
//     Psi = Psi([nx+1:nx+ny,1:nx],:);
//     Sig = G.H(R1i,R1i);
//     GG  = [M.ss.C M.ss.D;M.ss.A M.ss.B];
//     AA  = Phi-GG*Psi'-Psi*GG'+GG*Sig*GG';          
//     G.gLL = [%-2*(P1\(P1'\(G.ss.xs(:,1)-M.ss.X1)));
//         %2*vech(diag(1./diag(P1)) - ((P1'\(BB)/P1)/P1'));
//         vec(Pi\(Pi'\(-2*Psi + 2*GG*Sig)));
//         2*vech(diag(Z.Ny./diag(Pi)) - ((Pi'\(AA)/Pi)/Pi')) ];
    idx = na*n+nb*m;
    Phi = &H[ldh*idx+idx];
    Psi = &H[idx];
    Sig = H;
    for(i=0;i<n+p;i++){
        for(j=0;j<idx;j++){
            dLL[i+(n+p)*j] = -2.0*Psi[i+ldh*j];
        }
        for(j=0;j<n+p;j++){
            AA[i+j*(n+p)] = 2.0*Phi[i+ldh*j];
        }
    }
    dtmp = 2.0;
    itmp = n+p;
	dgemm("N", "N", &itmp, &idx, &idx, &dtmp, Gamma, &ldg, Sig, &ldh, &done, dLL, &itmp);  //-2*psi + 2*GG*Sig
    //printmatrix(itmp,idx,dLL,itmp,"-2*Psi + 2*GG*Sig");

    dgemm("N", "T", &itmp, &itmp, &idx, &done, dLL, &itmp, Gamma, &ldg, &done, AA, &itmp);  //2*Phi-2*Psi*GG'+2*GG*Sig*GG'
    dtmp = -2.0;
    dgemm("N", "T", &itmp, &itmp, &idx, &dtmp, Gamma, &itmp, Psi, &ldh, &done, AA, &itmp);  //2*Phi-2*GG*Psi'-2*Psi*GG'+2*GG*Sig*GG'
    
    //printmatrix(itmp,itmp,AA,itmp,"2*Phi-2*GG*Psi'-2*Psi*GG'+2*GG*Sig*GG'");


    dtrsm("L","U","T","N",&itmp,&idx,&done,Pi,&itmp,dLL,&itmp);
    dtrsm("L","U","N","N",&itmp,&idx,&done,Pi,&itmp,dLL,&itmp); //vec(Pi\(Pi'\(-2*Psi + 2*GG*Sig)));

    //         2*vech(diag(Z.Ny./diag(Pi)) - ((Pi'\(AA)/Pi)/Pi'))
    dtrsm("L","U","T","N",&itmp,&itmp,&done,Pi,&itmp,AA,&itmp);
    dtrsm("R","U","N","N",&itmp,&itmp,&done,Pi,&itmp,AA,&itmp);
    dtrsm("R","U","T","N",&itmp,&itmp,&done,Pi,&itmp,AA,&itmp);
    k=0;
    for(i=0;i<n+p;i++){
        AA[i+i*(n+p)] = AA[i+i*(n+p)] - 2.0*N/Pi[i+i*(n+p)];
        for(j=0;j<=i;j++){
            dLL[itmp*idx+k] = -AA[j+i*(n+p)];
            k++;
        }
    }
	
    //Copy the desired predicted output
    itmp=p*N;
    dcopy(&itmp,ys,&ione,yout,&ione);

    //Now compute the EM new theta (Gamma and Pi)
    rchol(ldh,H,ldh);

    //printmatrix(ldh,ldh,H,ldh,"H");

    rfbs("L", "U", "N", "N", n*na+m*nb, n+p, 1.0, H, ldh, &H[ldh*(n*na+m*nb)], ldh);

    //printmatrix(ldh,ldh,H,ldh,"H");

    //Now copy out the right parts into thn
    for(j=0;j<n*na+m*nb;j++){
        for(i=0;i<n+p;i++){
            thn[i+(n+p)*j] = H[ldh*(n*na+m*nb) + i*ldh + j];
        }
    }
    k=0;
    for(j=0;j<n+p;j++){
        for(i=0;i<=j;i++){
            thn[(n+p)*(n*na+m*nb)+k] = H[(ldh+1)*(n*na+m*nb) + ldh*j + i]/sqrt(N);
            k++;
        }
    }

exit_stub:    
	free(mem);
}

	
	
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	double *y, *u, *P1, *X1, *a, *b, *theta, *LL, *dLL, *yout, *thn, *x;
	int n, m, p, N, na, nb, smoothing, ntheta;
	

    // n, m, p, na, nb, N, smoothing, y, u, a, b, theta, P1, X1
	n         = (int)(mxGetPr(prhs[0])[0]);
    m         = (int)(mxGetPr(prhs[1])[0]);
    p         = (int)(mxGetPr(prhs[2])[0]);
    na        = (int)(mxGetPr(prhs[3])[0]);
    nb        = (int)(mxGetPr(prhs[4])[0]);
    N         = (int)(mxGetPr(prhs[5])[0]);
    smoothing = (int)(mxGetPr(prhs[6])[0]);
    y         = mxGetPr(prhs[7]);
    u         = mxGetPr(prhs[8]);
    a         = mxGetPr(prhs[9]);
    b         = mxGetPr(prhs[10]);
    theta     = mxGetPr(prhs[11]);
    ntheta    = mxGetM(prhs[11]);
    P1        = mxGetPr(prhs[12]);
    X1        = mxGetPr(prhs[13]);

    plhs[0]   = mxCreateDoubleMatrix(1,1,mxREAL);
    LL        = mxGetPr(plhs[0]);
    plhs[1]   = mxCreateDoubleMatrix(ntheta,1,mxREAL);
    dLL       = mxGetPr(plhs[1]);
    plhs[2]   = mxCreateDoubleMatrix(p,N,mxREAL);
    yout      = mxGetPr(plhs[2]);
    plhs[3]   = mxCreateDoubleMatrix(ntheta,1,mxREAL);
    thn       = mxGetPr(plhs[3]);
    plhs[4]   = mxCreateDoubleMatrix(n,N+1,mxREAL);
    x         = mxGetPr(plhs[4]);

    ll_and_dll(n,m,p,na,nb,N,smoothing,y,u,a,b,theta,X1,P1,LL,dLL,yout,thn,x);
}