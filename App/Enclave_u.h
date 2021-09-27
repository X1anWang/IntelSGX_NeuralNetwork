#ifndef ENCLAVE_U_H__
#define ENCLAVE_U_H__

#include <stdint.h>
#include <wchar.h>
#include <stddef.h>
#include <string.h>
#include "sgx_edger8r.h" /* for sgx_satus_t etc. */

#include "sgx_tseal.h"

#include <stdlib.h> /* for size_t */

#define SGX_CAST(type, item) ((type)(item))

#ifdef __cplusplus
extern "C" {
#endif

#ifndef PRINT_MESSAGE_DEFINED__
#define PRINT_MESSAGE_DEFINED__
void SGX_UBRIDGE(SGX_NOCONVENTION, print_message, (const char* str));
#endif
#ifndef OUTPUTRESULT_DEFINED__
#define OUTPUTRESULT_DEFINED__
void SGX_UBRIDGE(SGX_NOCONVENTION, outputresult, (int* predictions, size_t ndata3));
#endif

sgx_status_t secure_nn(sgx_enclave_id_t eid, int randseed);
sgx_status_t secure_train(sgx_enclave_id_t eid, double** data, double** gt, size_t ndata1);
sgx_status_t secure_test(sgx_enclave_id_t eid, double** datat, double** gtt, size_t ndata2);
sgx_status_t secure_summurize(sgx_enclave_id_t eid);
sgx_status_t seal(sgx_enclave_id_t eid, sgx_status_t* retval, uint8_t* plaintext, size_t plaintext_len, sgx_sealed_data_t* sealed_data, size_t sealed_size);
sgx_status_t unseal(sgx_enclave_id_t eid, sgx_status_t* retval, sgx_sealed_data_t* sealed_data, size_t sealed_size, uint8_t* plaintext, uint32_t plaintext_len);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif
