#include "Enclave_u.h"
#include <errno.h>

typedef struct ms_secure_nn_t {
	int ms_randseed;
} ms_secure_nn_t;

typedef struct ms_secure_train_t {
	double** ms_data;
	double** ms_gt;
	size_t ms_ndata1;
} ms_secure_train_t;

typedef struct ms_secure_test_t {
	double** ms_datat;
	double** ms_gtt;
	size_t ms_ndata2;
} ms_secure_test_t;

typedef struct ms_seal_t {
	sgx_status_t ms_retval;
	uint8_t* ms_plaintext;
	size_t ms_plaintext_len;
	sgx_sealed_data_t* ms_sealed_data;
	size_t ms_sealed_size;
} ms_seal_t;

typedef struct ms_unseal_t {
	sgx_status_t ms_retval;
	sgx_sealed_data_t* ms_sealed_data;
	size_t ms_sealed_size;
	uint8_t* ms_plaintext;
	uint32_t ms_plaintext_len;
} ms_unseal_t;

typedef struct ms_print_message_t {
	const char* ms_str;
} ms_print_message_t;

typedef struct ms_outputresult_t {
	int* ms_predictions;
	size_t ms_ndata3;
} ms_outputresult_t;

static sgx_status_t SGX_CDECL Enclave_print_message(void* pms)
{
	ms_print_message_t* ms = SGX_CAST(ms_print_message_t*, pms);
	print_message(ms->ms_str);

	return SGX_SUCCESS;
}

static sgx_status_t SGX_CDECL Enclave_outputresult(void* pms)
{
	ms_outputresult_t* ms = SGX_CAST(ms_outputresult_t*, pms);
	outputresult(ms->ms_predictions, ms->ms_ndata3);

	return SGX_SUCCESS;
}

static const struct {
	size_t nr_ocall;
	void * table[2];
} ocall_table_Enclave = {
	2,
	{
		(void*)Enclave_print_message,
		(void*)Enclave_outputresult,
	}
};
sgx_status_t secure_nn(sgx_enclave_id_t eid, int randseed)
{
	sgx_status_t status;
	ms_secure_nn_t ms;
	ms.ms_randseed = randseed;
	status = sgx_ecall(eid, 0, &ocall_table_Enclave, &ms);
	return status;
}

sgx_status_t secure_train(sgx_enclave_id_t eid, double** data, double** gt, size_t ndata1)
{
	sgx_status_t status;
	ms_secure_train_t ms;
	ms.ms_data = data;
	ms.ms_gt = gt;
	ms.ms_ndata1 = ndata1;
	status = sgx_ecall(eid, 1, &ocall_table_Enclave, &ms);
	return status;
}

sgx_status_t secure_test(sgx_enclave_id_t eid, double** datat, double** gtt, size_t ndata2)
{
	sgx_status_t status;
	ms_secure_test_t ms;
	ms.ms_datat = datat;
	ms.ms_gtt = gtt;
	ms.ms_ndata2 = ndata2;
	status = sgx_ecall(eid, 2, &ocall_table_Enclave, &ms);
	return status;
}

sgx_status_t secure_summurize(sgx_enclave_id_t eid)
{
	sgx_status_t status;
	status = sgx_ecall(eid, 3, &ocall_table_Enclave, NULL);
	return status;
}

sgx_status_t seal(sgx_enclave_id_t eid, sgx_status_t* retval, uint8_t* plaintext, size_t plaintext_len, sgx_sealed_data_t* sealed_data, size_t sealed_size)
{
	sgx_status_t status;
	ms_seal_t ms;
	ms.ms_plaintext = plaintext;
	ms.ms_plaintext_len = plaintext_len;
	ms.ms_sealed_data = sealed_data;
	ms.ms_sealed_size = sealed_size;
	status = sgx_ecall(eid, 4, &ocall_table_Enclave, &ms);
	if (status == SGX_SUCCESS && retval) *retval = ms.ms_retval;
	return status;
}

sgx_status_t unseal(sgx_enclave_id_t eid, sgx_status_t* retval, sgx_sealed_data_t* sealed_data, size_t sealed_size, uint8_t* plaintext, uint32_t plaintext_len)
{
	sgx_status_t status;
	ms_unseal_t ms;
	ms.ms_sealed_data = sealed_data;
	ms.ms_sealed_size = sealed_size;
	ms.ms_plaintext = plaintext;
	ms.ms_plaintext_len = plaintext_len;
	status = sgx_ecall(eid, 5, &ocall_table_Enclave, &ms);
	if (status == SGX_SUCCESS && retval) *retval = ms.ms_retval;
	return status;
}

