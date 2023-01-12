#include <iostream>
#include <omp.h>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <cstring>      /* strcasecmp */
#include <cstdint>
#include <assert.h>
#include <vector>       // std::vector
#include <algorithm>    // std::random_shuffle
#include <random>
#include <stdexcept>
#include <math.h>

using namespace std;

using idx_t = std::uint32_t;
using val_t = float;
using ptr_t = std::uintptr_t;

int NUM_THREAD=1;
int nthreads = 1;

/**
 * CSR structure to store search results
 */
typedef struct csr_t {
  idx_t nrows; // number of rows
  idx_t ncols; // number of cols
  idx_t * ind; // column ids
  val_t * val; // values
  ptr_t * ptr; // pointers (start of row in ind/val)

  csr_t()
  {
    nrows = ncols = 0;
    ind = nullptr;
    val = nullptr;
    ptr = nullptr;
  }

  /**
   * Reserve space for more rows or non-zeros. Structure may only grow, not shrink.
   * @param nrows Number of rows
   * @param nnz   Number of non-zeros
   */
  void reserve(const idx_t nrows, const ptr_t nnz)
  {
    if(nrows > this->nrows){
      if(ptr){
        ptr = (ptr_t*) realloc(ptr, sizeof(ptr_t) * (nrows+1));
      } else {
        ptr = (ptr_t*) malloc(sizeof(ptr_t) * (nrows+1));
        ptr[0] = 0;
      }
      if(!ptr){
        throw std::runtime_error("Could not allocate ptr array.");
      }
    }

    if(nnz > ptr[this->nrows]){
      if(ind){
        ind = (idx_t*) realloc(ind, sizeof(idx_t) * nnz);
      } else {
        ind = (idx_t*) malloc(sizeof(idx_t) * nnz);
      }
      if(!ind){
        throw std::runtime_error("Could not allocate ind array.");
      }
      if(val){
        val = (val_t*) realloc(val, sizeof(val_t) * nnz);
      } else {
        val = (val_t*) malloc(sizeof(val_t) * nnz);
      }
      if(!val){
        throw std::runtime_error("Could not allocate val array.");
      }
    }
    this->nrows = nrows;
  }

  /**
   * Reserve space for more rows or non-zeros. Structure may only grow, not shrink.
   * @param nrows Number of rows
   * @param ncols Number of columns
   * @param factor   Sparsity factor
   */
  static csr_t * random(const idx_t nrows, const idx_t ncols, const double factor)
  {
    ptr_t nnz = (ptr_t) (factor * nrows * ncols);
    if(nnz >= nrows * ncols / 2.0){
      throw std::runtime_error("Asking for too many non-zeros. Matrix is not sparse.");
    }
    auto mat = new csr_t();
    mat->reserve(nrows, nnz);
    mat->ncols = ncols;

    /* fill in ptr array; generate random row sizes */
    unsigned int seed = (unsigned long) mat;
    long double sum = 0;
    for(idx_t i=1; i <= mat->nrows; ++i){
      mat->ptr[i] = rand_r(&seed) % ncols;
      sum += mat->ptr[i];
    }
    for(idx_t i=0; i < mat->nrows; ++i){
      double percent = mat->ptr[i+1] / sum;
      mat->ptr[i+1] = mat->ptr[i] + (ptr_t)(percent * nnz);
      if(mat->ptr[i+1] > nnz){
        mat->ptr[i+1] = nnz;
      }
    }

    if(nnz - mat->ptr[mat->nrows-1] <= ncols){
      mat->ptr[mat->nrows] = nnz;
    }

    /* fill in indices and values with random numbers */
    #pragma omp parallel
    {
      int tid = omp_get_thread_num();
      unsigned int seed = (unsigned long) mat * (1+tid);
      std::vector<int> perm;
      for(idx_t i=0; i < ncols; ++i){
        perm.push_back(i);
      }
      std::random_device seeder;
      std::mt19937 engine(seeder());

      #pragma omp for
      for(idx_t i=0; i < nrows; ++i){
        std::shuffle(perm.begin(), perm.end(), engine);
        for(ptr_t j=mat->ptr[i]; j < mat->ptr[i+1]; ++j){
          mat->ind[j] = perm[j - mat->ptr[i]];
          mat->val[j] = ((double) rand_r(&seed)/rand_r(&seed));
        }
      }
    }

    return mat;
  }

  string info(const string name="") const
  {
    return (name.empty() ? "CSR" : name) + "<" + to_string(nrows) + ", " + to_string(ncols) + ", " +
      (ptr ? to_string(ptr[nrows]) : "0") + ">";
  }

  string details(const string name="", int n=-1){
    if (n<0) n=ptr[nrows];
    string str= name + ".ptr [ " ;
    for (idx_t i=0; i< nrows+1; i++)
    {
      str+="" + to_string(ptr[i]) + ", ";
    }
    str+= " ]\n";

    str+= name + ".ind [ " ;
    for (idx_t i=0; i< n; i++)
    {
      str+="" + to_string(ind[i]) + ", ";
    }
    str+= " ]\n";
    str+= name + ".val [ " ;
    for (idx_t i=0; i< n; i++)
    {
      str+="" + to_string(val[i]) + ", ";
    }
    str+= " ]\n";
    return str;
  }

  void sort_ind_and_val(){
    #pragma omp parallel
    {
      // int tid = omp_get_thread_num();
      // if(tid == 0 ) cout << "SORT NUM THREADS: " << omp_get_num_threads() << "\n" ; 
      #pragma omp for 
      for(idx_t i=0; i < nrows; ++i){
        ptr_t r_st = ptr[i];
        ptr_t r_end =  ptr[i+1];
        ptr_t ln = r_end -r_st;
        if (ln < 2 ) continue;

        vector<pair<idx_t, val_t>> row_pair_vect(ln);
  
        // Storing the respective array
        // elements in pairs.
        for (idx_t j = r_st, cnt=0; j < r_end; j++, cnt++) {
          pair<idx_t, val_t> curr;
          curr.first = ind[j];
          curr.second = val[j];
          row_pair_vect[cnt]=curr;
        }
        
        // Sorting the pair array.
        sort(begin(row_pair_vect), end(row_pair_vect));

        // adding to CSR
        for (idx_t j = r_st, cnt=0; j < r_end; j++, cnt++) {
          ind[j] = row_pair_vect[cnt].first ;
          val[j] = row_pair_vect[cnt].second ;
        }
      }
    }
  }

  val_t round(double var)
  {
    int prec=2;
    int tmp=pow(10, prec);
    double value = (int)(var * pow(10, prec)+ .5);
    return (val_t)value / tmp;
  }


  bool compare(const csr_t * mat, int debug=0){
    if (mat->nrows != this->nrows ) {
      debug && cout << "nRows not Equal ! mat-> nrows :" << mat->nrows << "this->nrows" <<  this->nrows;
      return false; 
    }

    if (mat->ncols != this->ncols ) {
      debug && cout << "nCols not Equal ! mat-> ncols :" << mat->ncols << "this->ncols" <<  this->ncols;
      return false; 
    }

    if (mat->ptr[mat->nrows] != this->ptr[this->nrows]){
      debug && cout << "NNZ not Equal ! mat-> NNZ :" << mat->ptr[mat->nrows] << "this->NNZ" <<  this->ptr[this->nrows];
      return false;
    }
    for (idx_t i=0; i< nrows+1; i++)
      if (mat->ptr[i] != this->ptr[i]){
        debug && cout << "In Ptr compare : unequal idx : " << i  << ",and mat val : " << \
        mat->ptr[i] << ", this val : " << this->ptr[i] << "\n"; 
        return false;
      } 

    if (mat->ptr[nrows] != this->ptr[nrows] ) return false; 
    for (idx_t i=0; i< ptr[nrows]; i++){
      if (mat->ind[i] != this->ind[i]){
        debug && cout << "In ind compare : unequal idx : " << i << ",and mat val : " << \
        mat->ptr[i] << ", this val : " << this->ptr[i] << "\n"; 
        return false;
      } 
      
    }

    for (idx_t i=0; i< ptr[nrows]; i++)
      if (trunc(10. * mat->val[i]) != trunc(10. * this->val[i])){
      // if ((round(mat->val[i]) != round(this->val[i]))){
        debug && cout << "In val compare : unequal idx : " << i << ",and mat val : " << \
        round(mat->ptr[i]) << ", this val : " << round(this->ptr[i]) << "\n"; 
        return false;
      } 
    return true;
  }

  ~csr_t()
  {
    if(ind){
      free(ind);
    }
    if(val){
      free(val);
    }
    if(ptr){
      free(ptr);
    }
  }
} csr_t;

/**
 * Ensure the matrix is valid
 * @param mat Matrix to test
 */
void test_matrix(csr_t * mat){
  auto nrows = mat->nrows;
  auto ncols = mat->ncols;
  assert(mat->ptr);
  auto nnz = mat->ptr[nrows];
  for(idx_t i=0; i < nrows; ++i){
    assert(mat->ptr[i] <= nnz);
  }
  for(ptr_t j=0; j < nnz; ++j){
    assert(mat->ind[j] < ncols);
  }
}


/**
 * Multiply A and B (transposed given) and write output in C.
 * Note that C has no data allocations (i.e., ptr, ind, and val pointers are null).
 * Use `csr_t::reserve` to increase C's allocations as necessary.
 * @param A  Matrix A.
 * @param B The transpose of matrix B.
 * @param C  Output matrix
 */
void sparsematmult(csr_t * A, csr_t * B, csr_t *C)
{
  int c_nnz=0;
  vector<ptr_t> res_ptr(A->nrows+1);
  vector<idx_t> res_ind(0);
  vector<val_t> res_val(0);

  for(idx_t i=0; i < A->nrows; ++i){
    idx_t a_st  = A->ptr[i];
    idx_t a_end = A->ptr[i+1];
    idx_t a_ln  = a_end - a_st ;

    if (a_ln == 0 ){
      res_ptr[i+1] = c_nnz;
      continue;
    }


    for(idx_t j=0; j < B->nrows; ++j){
      idx_t b_st  = B->ptr[j];
      idx_t b_end = B->ptr[j+1];
      idx_t b_ln  = b_end - b_st ;

      if (b_ln == 0 )
          continue;

      // double sum=0.0;
      // for (idx_t n=a_st; n < a_end; ++n){
      //   for (idx_t m=b_st; m < b_end; ++m){
      //     if (A->ind[n] < B->ind[m]){
      //       break;
      //     }
      //     if (A->ind[n] == B->ind[m]){
      //       sum+=A->val[n]*B->val[m];
      //     }
      //   }
      // }

      idx_t a_curr=a_st;
      idx_t b_curr=b_st;
      double sum=0.0;
      while (a_curr < a_end && b_curr < b_end ){
        if (A->ind[a_curr] == B->ind[b_curr]){
          sum+=A->val[a_curr]*B->val[b_curr];
          a_curr++;
          b_curr++;
        }
        else if (A->ind[a_curr] > B->ind[b_curr]){
          b_curr++;
        }
        else if (A->ind[a_curr] < B->ind[b_curr]){
          a_curr++;
        }
      }

      if (sum != 0.0){
        c_nnz++;
        res_val.push_back(sum);
        res_ind.push_back(j);
      }
    }
    res_ptr[i+1] = c_nnz;
  }

  C->reserve(A->nrows,c_nnz);
  C->ncols=B->nrows;
  for (idx_t i = 0 ; i < A->nrows+1; i++){
    C->ptr[i]=res_ptr[i];
  }
  for (int i = 0 ; i < c_nnz; i++){
    C->ind[i]=res_ind[i];
    C->val[i]=res_val[i];
  }
  test_matrix(C);
}

void para_sparsematmult(csr_t * A, csr_t * B, csr_t *C)
{
  // int c_nnz=0;
  idx_t c_rows=A->nrows;
  idx_t c_cols=B->nrows;
  vector<ptr_t> res_ptr(A->nrows+1,0);
  vector<vector<idx_t>> res_ind(A->nrows);
  vector<vector<val_t>> res_val(A->nrows);

  #pragma omp parallel 
  {
    // int tid = omp_get_thread_num();
    #pragma omp master 
    {
      NUM_THREAD=omp_get_num_threads();
    }
    #pragma omp for
    for(idx_t i=0; i < A->nrows; ++i){
      idx_t a_st  = A->ptr[i];
      idx_t a_end = A->ptr[i+1];
      idx_t a_ln  = a_end - a_st ;
      if (a_ln < 1 )
          continue;
      
      for(idx_t j=0; j < B->nrows; ++j){
        idx_t b_st  = B->ptr[j];
        idx_t b_end = B->ptr[j+1];
        idx_t b_ln  = b_end - b_st ;

        if (b_ln < 1 )
          continue;

        // val_t sum=0.0;
        // for (idx_t n=a_st; n < a_end; ++n){
        //   for (idx_t m=b_st; m < b_end; ++m){
        //     if (A->ind[n] < B->ind[m])
        //       break;
        //     if (A->ind[n] == B->ind[m])
        //       sum+=A->val[n]*B->val[m];
        //     }
        //   }
        idx_t a_curr=a_st;
        idx_t b_curr=b_st;
        double sum=0.0;
        while (a_curr < a_end && b_curr < b_end ){
          if (A->ind[a_curr] == B->ind[b_curr]){
            sum+=A->val[a_curr]*B->val[b_curr];
            a_curr++;
            b_curr++;
          }
          else if (A->ind[a_curr] > B->ind[b_curr]){
            b_curr++;
          }
          else if (A->ind[a_curr] < B->ind[b_curr]){
            a_curr++;
          }
        }

        if (sum != 0.0){
          res_val[i].push_back(sum);
          res_ind[i].push_back(j);
        }
      }
    }
  }

  res_ptr[0]=0;
  for (idx_t i = 0 ; i < c_rows; i++){
    res_ptr[i+1]=res_ptr[i] + res_ind[i].size();
  }

  C->reserve(c_rows,res_ptr[c_rows]);
  C->ncols=c_cols;
  for (idx_t i = 0 ; i < c_rows+1; i++){
    C->ptr[i]=res_ptr[i];
  }

  int cnt=0;
  for (idx_t i=0; i < c_rows ; i ++){
    for (idx_t j =0 ; j< res_ptr[i+1] - res_ptr[i] ; j++){
    C->ind[cnt]=res_ind[i][j] ;
    C->val[cnt]=res_val[i][j] ;
    cnt++;
    }
  }
  test_matrix(C);
}

int main(int argc, char *argv[])
{
  if(argc < 4){
    cerr << "Invalid options." << endl << "<program> <A_nrows> <A_ncols> <B_ncols> <fill_factor> [-t <num_threads>]" << endl;
    exit(1);
  }
  int nrows = atoi(argv[1]);
  int ncols = atoi(argv[2]);
  int ncols2 = atoi(argv[3]);
  double factor = atof(argv[4]);

  if(argc == 7 && strcasecmp(argv[5], "-t") == 0){
    nthreads = atoi(argv[6]);
    omp_set_num_threads(nthreads);
  }

  // cout << "A_nrows: " << nrows << endl;
  // cout << "A_ncols: " << ncols << endl;
  // cout << "B_ncols: " << ncols2 << endl;
  // cout << "factor: " << factor << endl;
  // cout << "nthreads: " << nthreads << endl;

  /* initialize random seed: */
  srand (time(NULL));

  auto A = csr_t::random(nrows, ncols, factor);
  auto B = csr_t::random(ncols2, ncols, factor); // Note B is already transposed.
  test_matrix(A);
  test_matrix(B);
  auto C = new csr_t(); // Note that C has no data allocations so far.
  auto D = new csr_t(); // Note that D has no data allocations so far.
  A->sort_ind_and_val();
  B->sort_ind_and_val();


  double t1,t2;
  cout << "FUNCTION NAME,ROW1,COL1,COL2,FACTOR,A_INFO,B_INFO,C_INFO,NUM THRDS DEF,NUM THRDS ALLOCATED,PASSED TEST_MATRIX,EXEC TIME\n";
  // t1 = omp_get_wtime();
  // sparsematmult(A, B, C);
  // t2 = omp_get_wtime();
  // cout << "serial sparsematmult,"<< nrows << "," << ncols<< "," << ncols2 << "," << factor << "," << (string)A->info("A") << "," << (string)B->info("B") << "," << (string)C->info("C") << "," << nthreads<< "," << NUM_THREAD<< "," << "REF"<< ",YES," << (t2-t1) << "Sec\n" ;
  
  t1 = omp_get_wtime();
  para_sparsematmult(A, B, D);
  t2 = omp_get_wtime();
  cout << "para_sparsematmult ,"<< nrows << "," << ncols<< "," << ncols2 << "," << factor << "," << (string)A->info("A") << "," << (string)B->info("B") << "," << (string)D->info("D") << "," << nthreads<< "," << NUM_THREAD<<  ",YES," << (t2-t1) << "Sec\n" ;

  delete A;
  delete B;
  delete C;

  return 0;
}
