#ifndef ENCLAVE_T_H__
#define ENCLAVE_T_H__

#include <stdint.h>
#include <wchar.h>
#include <stddef.h>
#include "sgx_edger8r.h" /* for sgx_ocall etc. */

#include "sgx_tseal.h"

#include <stdlib.h> /* for size_t */

#define SGX_CAST(type, item) ((type)(item))

#ifdef __cplusplus
extern "C" {
#endif

void secure_nn(int randseed);
void secure_train(double** data, double** gt, size_t ndata1);
void secure_test(double** datat, double** gtt, size_t ndata2);
void secure_summurize(void);
sgx_status_t seal(uint8_t* plaintext, size_t plaintext_len, sgx_sealed_data_t* sealed_data, size_t sealed_size);
sgx_status_t unseal(sgx_sealed_data_t* sealed_data, size_t sealed_size, uint8_t* plaintext, uint32_t plaintext_len);

sgx_status_t SGX_CDECL print_message(const char* str);
sgx_status_t SGX_CDECL outputresult(int* predictions, size_t ndata3);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif
