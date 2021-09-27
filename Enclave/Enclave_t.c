#include "Enclave_t.h"

#include "sgx_trts.h" /* for sgx_ocalloc, sgx_is_outside_enclave */
#include "sgx_lfence.h" /* for sgx_lfence */

#include <errno.h>
#include <mbusafecrt.h> /* for memcpy_s etc */
#include <stdlib.h> /* for malloc/free etc */

#define CHECK_REF_POINTER(ptr, siz) do {	\
	if (!(ptr) || ! sgx_is_outside_enclave((ptr), (siz)))	\
		return SGX_ERROR_INVALID_PARAMETER;\
} while (0)

#define CHECK_UNIQUE_POINTER(ptr, siz) do {	\
	if ((ptr) && ! sgx_is_outside_enclave((ptr), (siz)))	\
		return SGX_ERROR_INVALID_PARAMETER;\
} while (0)

#define CHECK_ENCLAVE_POINTER(ptr, siz) do {	\
	if ((ptr) && ! sgx_is_within_enclave((ptr), (siz)))	\
		return SGX_ERROR_INVALID_PARAMETER;\
} while (0)


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

static sgx_status_t SGX_CDECL sgx_secure_nn(void* pms)
{
	CHECK_REF_POINTER(pms, sizeof(ms_secure_nn_t));
	//
	// fence after pointer checks
	//
	sgx_lfence();
	ms_secure_nn_t* ms = SGX_CAST(ms_secure_nn_t*, pms);
	sgx_status_t status = SGX_SUCCESS;



	secure_nn(ms->ms_randseed);


	return status;
}

static sgx_status_t SGX_CDECL sgx_secure_train(void* pms)
{
	CHECK_REF_POINTER(pms, sizeof(ms_secure_train_t));
	//
	// fence after pointer checks
	//
	sgx_lfence();
	ms_secure_train_t* ms = SGX_CAST(ms_secure_train_t*, pms);
	sgx_status_t status = SGX_SUCCESS;
	double** _tmp_data = ms->ms_data;
	size_t _tmp_ndata1 = ms->ms_ndata1;
	size_t _len_data = _tmp_ndata1 * sizeof(double*);
	double** _in_data = NULL;
	double** _tmp_gt = ms->ms_gt;
	size_t _len_gt = _tmp_ndata1 * sizeof(double*);
	double** _in_gt = NULL;

	if (sizeof(*_tmp_data) != 0 &&
		(size_t)_tmp_ndata1 > (SIZE_MAX / sizeof(*_tmp_data))) {
		return SGX_ERROR_INVALID_PARAMETER;
	}

	if (sizeof(*_tmp_gt) != 0 &&
		(size_t)_tmp_ndata1 > (SIZE_MAX / sizeof(*_tmp_gt))) {
		return SGX_ERROR_INVALID_PARAMETER;
	}

	CHECK_UNIQUE_POINTER(_tmp_data, _len_data);
	CHECK_UNIQUE_POINTER(_tmp_gt, _len_gt);

	//
	// fence after pointer checks
	//
	sgx_lfence();

	if (_tmp_data != NULL && _len_data != 0) {
		_in_data = (double**)malloc(_len_data);
		if (_in_data == NULL) {
			status = SGX_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		if (memcpy_s(_in_data, _len_data, _tmp_data, _len_data)) {
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}

	}
	if (_tmp_gt != NULL && _len_gt != 0) {
		_in_gt = (double**)malloc(_len_gt);
		if (_in_gt == NULL) {
			status = SGX_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		if (memcpy_s(_in_gt, _len_gt, _tmp_gt, _len_gt)) {
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}

	}

	secure_train(_in_data, _in_gt, _tmp_ndata1);
err:
	if (_in_data) free(_in_data);
	if (_in_gt) free(_in_gt);

	return status;
}

static sgx_status_t SGX_CDECL sgx_secure_test(void* pms)
{
	CHECK_REF_POINTER(pms, sizeof(ms_secure_test_t));
	//
	// fence after pointer checks
	//
	sgx_lfence();
	ms_secure_test_t* ms = SGX_CAST(ms_secure_test_t*, pms);
	sgx_status_t status = SGX_SUCCESS;
	double** _tmp_datat = ms->ms_datat;
	size_t _tmp_ndata2 = ms->ms_ndata2;
	size_t _len_datat = _tmp_ndata2 * sizeof(double*);
	double** _in_datat = NULL;
	double** _tmp_gtt = ms->ms_gtt;
	size_t _len_gtt = _tmp_ndata2 * sizeof(double*);
	double** _in_gtt = NULL;

	if (sizeof(*_tmp_datat) != 0 &&
		(size_t)_tmp_ndata2 > (SIZE_MAX / sizeof(*_tmp_datat))) {
		return SGX_ERROR_INVALID_PARAMETER;
	}

	if (sizeof(*_tmp_gtt) != 0 &&
		(size_t)_tmp_ndata2 > (SIZE_MAX / sizeof(*_tmp_gtt))) {
		return SGX_ERROR_INVALID_PARAMETER;
	}

	CHECK_UNIQUE_POINTER(_tmp_datat, _len_datat);
	CHECK_UNIQUE_POINTER(_tmp_gtt, _len_gtt);

	//
	// fence after pointer checks
	//
	sgx_lfence();

	if (_tmp_datat != NULL && _len_datat != 0) {
		_in_datat = (double**)malloc(_len_datat);
		if (_in_datat == NULL) {
			status = SGX_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		if (memcpy_s(_in_datat, _len_datat, _tmp_datat, _len_datat)) {
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}

	}
	if (_tmp_gtt != NULL && _len_gtt != 0) {
		_in_gtt = (double**)malloc(_len_gtt);
		if (_in_gtt == NULL) {
			status = SGX_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		if (memcpy_s(_in_gtt, _len_gtt, _tmp_gtt, _len_gtt)) {
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}

	}

	secure_test(_in_datat, _in_gtt, _tmp_ndata2);
err:
	if (_in_datat) free(_in_datat);
	if (_in_gtt) free(_in_gtt);

	return status;
}

static sgx_status_t SGX_CDECL sgx_secure_summurize(void* pms)
{
	sgx_status_t status = SGX_SUCCESS;
	if (pms != NULL) return SGX_ERROR_INVALID_PARAMETER;
	secure_summurize();
	return status;
}

static sgx_status_t SGX_CDECL sgx_seal(void* pms)
{
	CHECK_REF_POINTER(pms, sizeof(ms_seal_t));
	//
	// fence after pointer checks
	//
	sgx_lfence();
	ms_seal_t* ms = SGX_CAST(ms_seal_t*, pms);
	sgx_status_t status = SGX_SUCCESS;
	uint8_t* _tmp_plaintext = ms->ms_plaintext;
	size_t _tmp_plaintext_len = ms->ms_plaintext_len;
	size_t _len_plaintext = _tmp_plaintext_len;
	uint8_t* _in_plaintext = NULL;
	sgx_sealed_data_t* _tmp_sealed_data = ms->ms_sealed_data;
	size_t _tmp_sealed_size = ms->ms_sealed_size;
	size_t _len_sealed_data = _tmp_sealed_size;
	sgx_sealed_data_t* _in_sealed_data = NULL;

	CHECK_UNIQUE_POINTER(_tmp_plaintext, _len_plaintext);
	CHECK_UNIQUE_POINTER(_tmp_sealed_data, _len_sealed_data);

	//
	// fence after pointer checks
	//
	sgx_lfence();

	if (_tmp_plaintext != NULL && _len_plaintext != 0) {
		_in_plaintext = (uint8_t*)malloc(_len_plaintext);
		if (_in_plaintext == NULL) {
			status = SGX_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		if (memcpy_s(_in_plaintext, _len_plaintext, _tmp_plaintext, _len_plaintext)) {
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}

	}
	if (_tmp_sealed_data != NULL && _len_sealed_data != 0) {
		if ((_in_sealed_data = (sgx_sealed_data_t*)malloc(_len_sealed_data)) == NULL) {
			status = SGX_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memset((void*)_in_sealed_data, 0, _len_sealed_data);
	}

	ms->ms_retval = seal(_in_plaintext, _tmp_plaintext_len, _in_sealed_data, _tmp_sealed_size);
err:
	if (_in_plaintext) free(_in_plaintext);
	if (_in_sealed_data) {
		if (memcpy_s(_tmp_sealed_data, _len_sealed_data, _in_sealed_data, _len_sealed_data)) {
			status = SGX_ERROR_UNEXPECTED;
		}
		free(_in_sealed_data);
	}

	return status;
}

static sgx_status_t SGX_CDECL sgx_unseal(void* pms)
{
	CHECK_REF_POINTER(pms, sizeof(ms_unseal_t));
	//
	// fence after pointer checks
	//
	sgx_lfence();
	ms_unseal_t* ms = SGX_CAST(ms_unseal_t*, pms);
	sgx_status_t status = SGX_SUCCESS;
	sgx_sealed_data_t* _tmp_sealed_data = ms->ms_sealed_data;
	size_t _tmp_sealed_size = ms->ms_sealed_size;
	size_t _len_sealed_data = _tmp_sealed_size;
	sgx_sealed_data_t* _in_sealed_data = NULL;
	uint8_t* _tmp_plaintext = ms->ms_plaintext;
	uint32_t _tmp_plaintext_len = ms->ms_plaintext_len;
	size_t _len_plaintext = _tmp_plaintext_len;
	uint8_t* _in_plaintext = NULL;

	CHECK_UNIQUE_POINTER(_tmp_sealed_data, _len_sealed_data);
	CHECK_UNIQUE_POINTER(_tmp_plaintext, _len_plaintext);

	//
	// fence after pointer checks
	//
	sgx_lfence();

	if (_tmp_sealed_data != NULL && _len_sealed_data != 0) {
		_in_sealed_data = (sgx_sealed_data_t*)malloc(_len_sealed_data);
		if (_in_sealed_data == NULL) {
			status = SGX_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		if (memcpy_s(_in_sealed_data, _len_sealed_data, _tmp_sealed_data, _len_sealed_data)) {
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}

	}
	if (_tmp_plaintext != NULL && _len_plaintext != 0) {
		if ((_in_plaintext = (uint8_t*)malloc(_len_plaintext)) == NULL) {
			status = SGX_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memset((void*)_in_plaintext, 0, _len_plaintext);
	}

	ms->ms_retval = unseal(_in_sealed_data, _tmp_sealed_size, _in_plaintext, _tmp_plaintext_len);
err:
	if (_in_sealed_data) free(_in_sealed_data);
	if (_in_plaintext) {
		if (memcpy_s(_tmp_plaintext, _len_plaintext, _in_plaintext, _len_plaintext)) {
			status = SGX_ERROR_UNEXPECTED;
		}
		free(_in_plaintext);
	}

	return status;
}

SGX_EXTERNC const struct {
	size_t nr_ecall;
	struct {void* ecall_addr; uint8_t is_priv;} ecall_table[6];
} g_ecall_table = {
	6,
	{
		{(void*)(uintptr_t)sgx_secure_nn, 0},
		{(void*)(uintptr_t)sgx_secure_train, 0},
		{(void*)(uintptr_t)sgx_secure_test, 0},
		{(void*)(uintptr_t)sgx_secure_summurize, 0},
		{(void*)(uintptr_t)sgx_seal, 0},
		{(void*)(uintptr_t)sgx_unseal, 0},
	}
};

SGX_EXTERNC const struct {
	size_t nr_ocall;
	uint8_t entry_table[2][6];
} g_dyn_entry_table = {
	2,
	{
		{0, 0, 0, 0, 0, 0, },
		{0, 0, 0, 0, 0, 0, },
	}
};


sgx_status_t SGX_CDECL print_message(const char* str)
{
	sgx_status_t status = SGX_SUCCESS;
	size_t _len_str = str ? strlen(str) + 1 : 0;

	ms_print_message_t* ms = NULL;
	size_t ocalloc_size = sizeof(ms_print_message_t);
	void *__tmp = NULL;


	CHECK_ENCLAVE_POINTER(str, _len_str);

	ocalloc_size += (str != NULL) ? _len_str : 0;

	__tmp = sgx_ocalloc(ocalloc_size);
	if (__tmp == NULL) {
		sgx_ocfree();
		return SGX_ERROR_UNEXPECTED;
	}
	ms = (ms_print_message_t*)__tmp;
	__tmp = (void *)((size_t)__tmp + sizeof(ms_print_message_t));
	ocalloc_size -= sizeof(ms_print_message_t);

	if (str != NULL) {
		ms->ms_str = (const char*)__tmp;
		if (memcpy_s(__tmp, ocalloc_size, str, _len_str)) {
			sgx_ocfree();
			return SGX_ERROR_UNEXPECTED;
		}
		__tmp = (void *)((size_t)__tmp + _len_str);
		ocalloc_size -= _len_str;
	} else {
		ms->ms_str = NULL;
	}
	
	status = sgx_ocall(0, ms);

	if (status == SGX_SUCCESS) {
	}
	sgx_ocfree();
	return status;
}

sgx_status_t SGX_CDECL outputresult(int* predictions, size_t ndata3)
{
	sgx_status_t status = SGX_SUCCESS;
	size_t _len_predictions = ndata3 * sizeof(int);

	ms_outputresult_t* ms = NULL;
	size_t ocalloc_size = sizeof(ms_outputresult_t);
	void *__tmp = NULL;


	CHECK_ENCLAVE_POINTER(predictions, _len_predictions);

	ocalloc_size += (predictions != NULL) ? _len_predictions : 0;

	__tmp = sgx_ocalloc(ocalloc_size);
	if (__tmp == NULL) {
		sgx_ocfree();
		return SGX_ERROR_UNEXPECTED;
	}
	ms = (ms_outputresult_t*)__tmp;
	__tmp = (void *)((size_t)__tmp + sizeof(ms_outputresult_t));
	ocalloc_size -= sizeof(ms_outputresult_t);

	if (predictions != NULL) {
		ms->ms_predictions = (int*)__tmp;
		if (memcpy_s(__tmp, ocalloc_size, predictions, _len_predictions)) {
			sgx_ocfree();
			return SGX_ERROR_UNEXPECTED;
		}
		__tmp = (void *)((size_t)__tmp + _len_predictions);
		ocalloc_size -= _len_predictions;
	} else {
		ms->ms_predictions = NULL;
	}
	
	ms->ms_ndata3 = ndata3;
	status = sgx_ocall(1, ms);

	if (status == SGX_SUCCESS) {
	}
	sgx_ocfree();
	return status;
}

