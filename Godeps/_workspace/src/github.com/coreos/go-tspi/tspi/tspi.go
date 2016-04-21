// Copyright 2015 CoreOS, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package tspi

// #include <trousers/tss.h>
// #cgo LDFLAGS: -ltspi
import "C"
import "errors"
import "fmt"

func tspiError(tssRet C.TSS_RESULT) error {
	ret := (int)(tssRet)
	if ret == 0 {
		return nil
	}
	if (ret & 0xf000) != 0 {
		ret &= ^(0xf000)
		switch {
		case ret == C.TSS_E_FAIL:
			return errors.New("TSS_E_FAIL")
		case ret == C.TSS_E_BAD_PARAMETER:
			return errors.New("TSS_E_BAD_PARAMETER")
		case ret == C.TSS_E_INTERNAL_ERROR:
			return errors.New("TSS_E_INTERNAL_ERROR")
		case ret == C.TSS_E_OUTOFMEMORY:
			return errors.New("TSS_E_OUTOFMEMORY")
		case ret == C.TSS_E_NOTIMPL:
			return errors.New("TSS_E_NOTIMPL")
		case ret == C.TSS_E_KEY_ALREADY_REGISTERED:
			return errors.New("TSS_E_KEY_ALREADY_REGISTERED")
		case ret == C.TSS_E_TPM_UNEXPECTED:
			return errors.New("TSS_E_TPM_UNEXPECTED")
		case ret == C.TSS_E_COMM_FAILURE:
			return errors.New("TSS_E_COMM_FAILURE")
		case ret == C.TSS_E_TIMEOUT:
			return errors.New("TSS_E_TIMEOUT")
		case ret == C.TSS_E_TPM_UNSUPPORTED_FEATURE:
			return errors.New("TSS_E_TPM_UNSUPPORTED_FEATURE")
		case ret == C.TSS_E_CANCELED:
			return errors.New("TSS_E_CANCELED")
		case ret == C.TSS_E_PS_KEY_NOTFOUND:
			return errors.New("TSS_E_PS_KEY_NOTFOUND")
		case ret == C.TSS_E_PS_KEY_EXISTS:
			return errors.New("TSS_E_PS_KEY_EXISTS")
		case ret == C.TSS_E_PS_BAD_KEY_STATE:
			return errors.New("TSS_E_PS_BAD_KEY_STATE")
		case ret == C.TSS_E_INVALID_OBJECT_TYPE:
			return errors.New("TSS_E_INVALID_OBJECT_TYPE")
		case ret == C.TSS_E_NO_CONNECTION:
			return errors.New("TSS_E_NO_CONNECTION")
		case ret == C.TSS_E_CONNECTION_FAILED:
			return errors.New("TSS_E_CONNECTION_FAILED")
		case ret == C.TSS_E_CONNECTION_BROKEN:
			return errors.New("TSS_E_CONNECTION_BROKEN")
		case ret == C.TSS_E_HASH_INVALID_ALG:
			return errors.New("TSS_E_HASH_INVALID_ALG")
		case ret == C.TSS_E_HASH_INVALID_LENGTH:
			return errors.New("TSS_E_HASH_INVALID_LENGTH")
		case ret == C.TSS_E_HASH_NO_DATA:
			return errors.New("TSS_E_HASH_NO_DATA")
		case ret == C.TSS_E_INVALID_ATTRIB_FLAG:
			return errors.New("TSS_E_INVALID_ATTRIB_FLAG")
		case ret == C.TSS_E_INVALID_ATTRIB_SUBFLAG:
			return errors.New("TSS_E_INVALID_ATTRIB_SUBFLAG")
		case ret == C.TSS_E_INVALID_ATTRIB_DATA:
			return errors.New("TSS_E_INVALID_ATTRIB_DATA")
		case ret == C.TSS_E_INVALID_OBJECT_INITFLAG:
			return errors.New("TSS_E_INVALID_OBJECT_INITFLAG")
		case ret == C.TSS_E_NO_PCRS_SET:
			return errors.New("TSS_E_NO_PCRS_SET")
		case ret == C.TSS_E_KEY_NOT_LOADED:
			return errors.New("TSS_E_KEY_NOT_LOADED")
		case ret == C.TSS_E_KEY_NOT_SET:
			return errors.New("TSS_E_KEY_NOT_SET")
		case ret == C.TSS_E_VALIDATION_FAILED:
			return errors.New("TSS_E_VALIDATION_FAILED")
		case ret == C.TSS_E_TSP_AUTHREQUIRED:
			return errors.New("TSS_E_TSP_AUTHREQUIRED")
		case ret == C.TSS_E_TSP_AUTH2REQUIRED:
			return errors.New("TSS_E_TSP_AUTH2REQUIRED")
		case ret == C.TSS_E_TSP_AUTHFAIL:
			return errors.New("TSS_E_TSP_AUTHFAIL")
		case ret == C.TSS_E_TSP_AUTH2FAIL:
			return errors.New("TSS_E_TSP_AUTH2FAIL")
		case ret == C.TSS_E_KEY_NO_MIGRATION_POLICY:
			return errors.New("TSS_E_KEY_NO_MIGRATION_POLICY")
		case ret == C.TSS_E_POLICY_NO_SECRET:
			return errors.New("TSS_E_POLICY_NO_SECRET")
		case ret == C.TSS_E_INVALID_OBJ_ACCESS:
			return errors.New("TSS_E_INVALID_OBJ_ACCESS")
		case ret == C.TSS_E_INVALID_ENCSCHEME:
			return errors.New("TSS_E_INVALID_ENCSCHEME")
		case ret == C.TSS_E_INVALID_SIGSCHEME:
			return errors.New("TSS_E_INVALID_SIGSCHEME")
		case ret == C.TSS_E_ENC_INVALID_LENGTH:
			return errors.New("TSS_E_ENC_INVALID_LENGTH")
		case ret == C.TSS_E_ENC_NO_DATA:
			return errors.New("TSS_E_ENC_NO_DATA")
		case ret == C.TSS_E_ENC_INVALID_TYPE:
			return errors.New("TSS_E_ENC_INVALID_TYPE")
		case ret == C.TSS_E_INVALID_KEYUSAGE:
			return errors.New("TSS_E_INVALID_KEYUSAGE")
		case ret == C.TSS_E_VERIFICATION_FAILED:
			return errors.New("TSS_E_VERIFICATION_FAILED")
		case ret == C.TSS_E_HASH_NO_IDENTIFIER:
			return errors.New("TSS_E_HASH_NO_IDENTIFIER")
		case ret == C.TSS_E_INVALID_HANDLE:
			return errors.New("TSS_E_INVALID_HANDLE")
		case ret == C.TSS_E_SILENT_CONTEXT:
			return errors.New("TSS_E_SILENT_CONTEXT")
		case ret == C.TSS_E_EK_CHECKSUM:
			return errors.New("TSS_E_EK_CHECKSUM")
		case ret == C.TSS_E_DELEGATION_NOTSET:
			return errors.New("TSS_E_DELEGATION_NOTSET")
		case ret == C.TSS_E_DELFAMILY_NOTFOUND:
			return errors.New("TSS_E_DELFAMILY_NOTFOUND")
		case ret == C.TSS_E_DELFAMILY_ROWEXISTS:
			return errors.New("TSS_E_DELFAMILY_ROWEXISTS")
		case ret == C.TSS_E_VERSION_MISMATCH:
			return errors.New("TSS_E_VERSION_MISMATCH")
		case ret == C.TSS_E_DAA_AR_DECRYPTION_ERROR:
			return errors.New("TSS_E_DAA_AR_DECRYPTION_ERROR")
		case ret == C.TSS_E_DAA_AUTHENTICATION_ERROR:
			return errors.New("TSS_E_DAA_AUTHENTICATION_ERROR")
		case ret == C.TSS_E_DAA_CHALLENGE_RESPONSE_ERROR:
			return errors.New("TSS_E_DAA_CHALLENGE_RESPONSE_ERROR")
		case ret == C.TSS_E_DAA_CREDENTIAL_PROOF_ERROR:
			return errors.New("TSS_E_DAA_CREDENTIAL_PROOF_ERROR")
		case ret == C.TSS_E_DAA_CREDENTIAL_REQUEST_PROOF_ERROR:
			return errors.New("TSS_E_DAA_CREDENTIAL_REQUEST_PROOF_ERROR")
		case ret == C.TSS_E_DAA_ISSUER_KEY_ERROR:
			return errors.New("TSS_E_DAA_ISSUER_KEY_ERROR")
		case ret == C.TSS_E_DAA_PSEUDONYM_ERROR:
			return errors.New("TSS_E_DAA_PSEUDONYM_ERROR")
		case ret == C.TSS_E_INVALID_RESOURCE:
			return errors.New("TSS_E_INVALID_RESOURCE")
		case ret == C.TSS_E_NV_AREA_EXIST:
			return errors.New("TSS_E_NV_AREA_EXIST")
		case ret == C.TSS_E_NV_AREA_NOT_EXIST:
			return errors.New("TSS_E_NV_AREA_NOT_EXIST")
		case ret == C.TSS_E_TSP_TRANS_AUTHFAIL:
			return errors.New("TSS_E_TSP_TRANS_AUTHFAIL")
		case ret == C.TSS_E_TSP_TRANS_AUTHREQUIRED:
			return errors.New("TSS_E_TSP_TRANS_AUTHREQUIRED")
		case ret == C.TSS_E_TSP_TRANS_NOTEXCLUSIVE:
			return errors.New("TSS_E_TSP_TRANS_NOTEXCLUSIVE")
		case ret == C.TSS_E_TSP_TRANS_FAIL:
			return errors.New("TSS_E_TSP_TRANS_FAIL")
		case ret == C.TSS_E_TSP_TRANS_NO_PUBKEY:
			return errors.New("TSS_E_TSP_TRANS_NO_PUBKEY")
		case ret == C.TSS_E_NO_ACTIVE_COUNTER:
			return errors.New("TSS_E_NO_ACTIVE_COUNTER")
		}
		return fmt.Errorf("Unknown TSS error: %x", ret)
	}

	switch {
	case ret == C.TPM_E_NON_FATAL:
		return errors.New("TPM_E_NON_FATAL")
	case ret == C.TPM_E_AUTHFAIL:
		return errors.New("TPM_E_AUTHFAIL")
	case ret == C.TPM_E_BADINDEX:
		return errors.New("TPM_E_BADINDEX")
	case ret == C.TPM_E_BAD_PARAMETER:
		return errors.New("TPM_E_BAD_PARAMETER")
	case ret == C.TPM_E_AUDITFAILURE:
		return errors.New("TPM_E_AUDITFAILURE")
	case ret == C.TPM_E_CLEAR_DISABLED:
		return errors.New("TPM_E_CLEAR_DISABLED")
	case ret == C.TPM_E_DEACTIVATED:
		return errors.New("TPM_E_DEACTIVATED")
	case ret == C.TPM_E_DISABLED:
		return errors.New("TPM_E_DISABLED")
	case ret == C.TPM_E_DISABLED_CMD:
		return errors.New("TPM_E_DISABLED_CMD")
	case ret == C.TPM_E_FAIL:
		return errors.New("TPM_E_FAIL")
	case ret == C.TPM_E_BAD_ORDINAL:
		return errors.New("TPM_E_BAD_ORDINAL")
	case ret == C.TPM_E_INSTALL_DISABLED:
		return errors.New("TPM_E_INSTALL_DISABLED")
	case ret == C.TPM_E_INVALID_KEYHANDLE:
		return errors.New("TPM_E_INVALID_KEYHANDLE")
	case ret == C.TPM_E_KEYNOTFOUND:
		return errors.New("TPM_E_KEYNOTFOUND")
	case ret == C.TPM_E_INAPPROPRIATE_ENC:
		return errors.New("TPM_E_INAPPROPRIATE_ENC")
	case ret == C.TPM_E_MIGRATEFAIL:
		return errors.New("TPM_E_MIGRATEFAIL")
	case ret == C.TPM_E_INVALID_PCR_INFO:
		return errors.New("TPM_E_INVALID_PCR_INFO")
	case ret == C.TPM_E_NOSPACE:
		return errors.New("TPM_E_NOSPACE")
	case ret == C.TPM_E_NOSRK:
		return errors.New("TPM_E_NOSRK")
	case ret == C.TPM_E_NOTSEALED_BLOB:
		return errors.New("TPM_E_NOTSEALED_BLOB")
	case ret == C.TPM_E_OWNER_SET:
		return errors.New("TPM_E_OWNER_SET")
	case ret == C.TPM_E_RESOURCES:
		return errors.New("TPM_E_RESOURCES")
	case ret == C.TPM_E_SHORTRANDOM:
		return errors.New("TPM_E_SHORTRANDOM")
	case ret == C.TPM_E_SIZE:
		return errors.New("TPM_E_SIZE")
	case ret == C.TPM_E_WRONGPCRVAL:
		return errors.New("TPM_E_WRONGPCRVAL")
	case ret == C.TPM_E_BAD_PARAM_SIZE:
		return errors.New("TPM_E_BAD_PARAM_SIZE")
	case ret == C.TPM_E_SHA_THREAD:
		return errors.New("TPM_E_SHA_THREAD")
	case ret == C.TPM_E_SHA_ERROR:
		return errors.New("TPM_E_SHA_ERROR")
	case ret == C.TPM_E_FAILEDSELFTEST:
		return errors.New("TPM_E_FAILEDSELFTEST")
	case ret == C.TPM_E_AUTH2FAIL:
		return errors.New("TPM_E_AUTH2FAIL")
	case ret == C.TPM_E_BADTAG:
		return errors.New("TPM_E_BADTAG")
	case ret == C.TPM_E_IOERROR:
		return errors.New("TPM_E_IOERROR")
	case ret == C.TPM_E_ENCRYPT_ERROR:
		return errors.New("TPM_E_ENCRYPT_ERROR")
	case ret == C.TPM_E_DECRYPT_ERROR:
		return errors.New("TPM_E_DECRYPT_ERROR")
	case ret == C.TPM_E_INVALID_AUTHHANDLE:
		return errors.New("TPM_E_INVALID_AUTHHANDLE")
	case ret == C.TPM_E_NO_ENDORSEMENT:
		return errors.New("TPM_E_NO_ENDORSEMENT")
	case ret == C.TPM_E_INVALID_KEYUSAGE:
		return errors.New("TPM_E_INVALID_KEYUSAGE")
	case ret == C.TPM_E_WRONG_ENTITYTYPE:
		return errors.New("TPM_E_WRONG_ENTITYTYPE")
	case ret == C.TPM_E_INVALID_POSTINIT:
		return errors.New("TPM_E_INVALID_POSTINIT")
	case ret == C.TPM_E_INAPPROPRIATE_SIG:
		return errors.New("TPM_E_INAPPROPRIATE_SIG")
	case ret == C.TPM_E_BAD_KEY_PROPERTY:
		return errors.New("TPM_E_BAD_KEY_PROPERTY")
	case ret == C.TPM_E_BAD_MIGRATION:
		return errors.New("TPM_E_BAD_MIGRATION")
	case ret == C.TPM_E_BAD_SCHEME:
		return errors.New("TPM_E_BAD_SCHEME")
	case ret == C.TPM_E_BAD_DATASIZE:
		return errors.New("TPM_E_BAD_DATASIZE")
	case ret == C.TPM_E_BAD_MODE:
		return errors.New("TPM_E_BAD_MODE")
	case ret == C.TPM_E_BAD_PRESENCE:
		return errors.New("TPM_E_BAD_PRESENCE")
	case ret == C.TPM_E_BAD_VERSION:
		return errors.New("TPM_E_BAD_VERSION")
	case ret == C.TPM_E_NO_WRAP_TRANSPORT:
		return errors.New("TPM_E_NO_WRAP_TRANSPORT")
	case ret == C.TPM_E_AUDITFAIL_UNSUCCESSFUL:
		return errors.New("TPM_E_AUDITFAIL_UNSUCCESSFUL")
	case ret == C.TPM_E_AUDITFAIL_SUCCESSFUL:
		return errors.New("TPM_E_AUDITFAIL_SUCCESSFUL")
	case ret == C.TPM_E_NOTRESETABLE:
		return errors.New("TPM_E_NOTRESETABLE")
	case ret == C.TPM_E_NOTLOCAL:
		return errors.New("TPM_E_NOTLOCAL")
	case ret == C.TPM_E_BAD_TYPE:
		return errors.New("TPM_E_BAD_TYPE")
	case ret == C.TPM_E_INVALID_RESOURCE:
		return errors.New("TPM_E_INVALID_RESOURCE")
	case ret == C.TPM_E_NOTFIPS:
		return errors.New("TPM_E_NOTFIPS")
	case ret == C.TPM_E_INVALID_FAMILY:
		return errors.New("TPM_E_INVALID_FAMILY")
	case ret == C.TPM_E_NO_NV_PERMISSION:
		return errors.New("TPM_E_NO_NV_PERMISSION")
	case ret == C.TPM_E_REQUIRES_SIGN:
		return errors.New("TPM_E_REQUIRES_SIGN")
	case ret == C.TPM_E_KEY_NOTSUPPORTED:
		return errors.New("TPM_E_KEY_NOTSUPPORTED")
	case ret == C.TPM_E_AUTH_CONFLICT:
		return errors.New("TPM_E_AUTH_CONFLICT")
	case ret == C.TPM_E_AREA_LOCKED:
		return errors.New("TPM_E_AREA_LOCKED")
	case ret == C.TPM_E_BAD_LOCALITY:
		return errors.New("TPM_E_BAD_LOCALITY")
	case ret == C.TPM_E_READ_ONLY:
		return errors.New("TPM_E_READ_ONLY")
	case ret == C.TPM_E_PER_NOWRITE:
		return errors.New("TPM_E_PER_NOWRITE")
	case ret == C.TPM_E_FAMILYCOUNT:
		return errors.New("TPM_E_FAMILYCOUNT")
	case ret == C.TPM_E_WRITE_LOCKED:
		return errors.New("TPM_E_WRITE_LOCKED")
	case ret == C.TPM_E_BAD_ATTRIBUTES:
		return errors.New("TPM_E_BAD_ATTRIBUTES")
	case ret == C.TPM_E_INVALID_STRUCTURE:
		return errors.New("TPM_E_INVALID_STRUCTURE")
	case ret == C.TPM_E_KEY_OWNER_CONTROL:
		return errors.New("TPM_E_KEY_OWNER_CONTROL")
	case ret == C.TPM_E_BAD_COUNTER:
		return errors.New("TPM_E_BAD_COUNTER")
	case ret == C.TPM_E_NOT_FULLWRITE:
		return errors.New("TPM_E_NOT_FULLWRITE")
	case ret == C.TPM_E_CONTEXT_GAP:
		return errors.New("TPM_E_CONTEXT_GAP")
	case ret == C.TPM_E_MAXNVWRITES:
		return errors.New("TPM_E_MAXNVWRITES")
	case ret == C.TPM_E_NOOPERATOR:
		return errors.New("TPM_E_NOOPERATOR")
	case ret == C.TPM_E_RESOURCEMISSING:
		return errors.New("TPM_E_RESOURCEMISSING")
	case ret == C.TPM_E_DELEGATE_LOCK:
		return errors.New("TPM_E_DELEGATE_LOCK")
	case ret == C.TPM_E_DELEGATE_FAMILY:
		return errors.New("TPM_E_DELEGATE_FAMILY")
	case ret == C.TPM_E_DELEGATE_ADMIN:
		return errors.New("TPM_E_DELEGATE_ADMIN")
	case ret == C.TPM_E_TRANSPORT_NOTEXCLUSIVE:
		return errors.New("TPM_E_TRANSPORT_NOTEXCLUSIVE")
	case ret == C.TPM_E_OWNER_CONTROL:
		return errors.New("TPM_E_OWNER_CONTROL")
	case ret == C.TPM_E_DAA_RESOURCES:
		return errors.New("TPM_E_DAA_RESOURCES")
	case ret == C.TPM_E_DAA_INPUT_DATA0:
		return errors.New("TPM_E_DAA_INPUT_DATA0")
	case ret == C.TPM_E_DAA_INPUT_DATA1:
		return errors.New("TPM_E_DAA_INPUT_DATA1")
	case ret == C.TPM_E_DAA_ISSUER_SETTINGS:
		return errors.New("TPM_E_DAA_ISSUER_SETTINGS")
	case ret == C.TPM_E_DAA_TPM_SETTINGS:
		return errors.New("TPM_E_DAA_TPM_SETTINGS")
	case ret == C.TPM_E_DAA_STAGE:
		return errors.New("TPM_E_DAA_STAGE")
	case ret == C.TPM_E_DAA_ISSUER_VALIDITY:
		return errors.New("TPM_E_DAA_ISSUER_VALIDITY")
	case ret == C.TPM_E_DAA_WRONG_W:
		return errors.New("TPM_E_DAA_WRONG_W")
	case ret == C.TPM_E_BAD_HANDLE:
		return errors.New("TPM_E_BAD_HANDLE")
	case ret == C.TPM_E_BAD_DELEGATE:
		return errors.New("TPM_E_BAD_DELEGATE")
	case ret == C.TPM_E_BADCONTEXT:
		return errors.New("TPM_E_BADCONTEXT")
	case ret == C.TPM_E_TOOMANYCONTEXTS:
		return errors.New("TPM_E_TOOMANYCONTEXTS")
	case ret == C.TPM_E_MA_TICKET_SIGNATURE:
		return errors.New("TPM_E_MA_TICKET_SIGNATURE")
	case ret == C.TPM_E_MA_DESTINATION:
		return errors.New("TPM_E_MA_DESTINATION")
	case ret == C.TPM_E_MA_SOURCE:
		return errors.New("TPM_E_MA_SOURCE")
	case ret == C.TPM_E_MA_AUTHORITY:
		return errors.New("TPM_E_MA_AUTHORITY")
	case ret == C.TPM_E_PERMANENTEK:
		return errors.New("TPM_E_PERMANENTEK")
	case ret == C.TPM_E_BAD_SIGNATURE:
		return errors.New("TPM_E_BAD_SIGNATURE")
	case ret == C.TPM_E_NOCONTEXTSPACE:
		return errors.New("TPM_E_NOCONTEXTSPACE")
	case ret == C.TPM_E_RETRY:
		return errors.New("TPM_E_RETRY")
	case ret == C.TPM_E_NEEDS_SELFTEST:
		return errors.New("TPM_E_NEEDS_SELFTEST")
	case ret == C.TPM_E_DOING_SELFTEST:
		return errors.New("TPM_E_DOING_SELFTEST")
	case ret == C.TPM_E_DEFEND_LOCK_RUNNING:
		return errors.New("TPM_E_DEFEND_LOCK_RUNNING")
	}
	return fmt.Errorf("Unknown error: %x", ret)
}

var TSS_UUID_SRK = C.TSS_UUID{
	ulTimeLow:     0,
	usTimeMid:     0,
	usTimeHigh:    0,
	bClockSeqHigh: 0,
	bClockSeqLow:  0,
	rgbNode:       [6]C.BYTE{0, 0, 0, 0, 0, 1},
}

var TSS_UUID_SK = C.TSS_UUID{
	ulTimeLow:     0,
	usTimeMid:     0,
	usTimeHigh:    0,
	bClockSeqHigh: 0,
	bClockSeqLow:  0,
	rgbNode:       [6]C.BYTE{0, 0, 0, 0, 0, 2},
}

var TSS_UUID_RK = C.TSS_UUID{
	ulTimeLow:     0,
	usTimeMid:     0,
	usTimeHigh:    0,
	bClockSeqHigh: 0,
	bClockSeqLow:  0,
	rgbNode:       [6]C.BYTE{0, 0, 0, 0, 0, 3},
}

var TSS_UUID_CRK = C.TSS_UUID{
	ulTimeLow:     0,
	usTimeMid:     0,
	usTimeHigh:    0,
	bClockSeqHigh: 0,
	bClockSeqLow:  0,
	rgbNode:       [6]C.BYTE{0, 0, 0, 0, 0, 8},
}

var TSS_UUID_USK1 = C.TSS_UUID{
	ulTimeLow:     0,
	usTimeMid:     0,
	usTimeHigh:    0,
	bClockSeqHigh: 0,
	bClockSeqLow:  0,
	rgbNode:       [6]C.BYTE{0, 0, 0, 0, 0, 4},
}

var TSS_UUID_USK2 = C.TSS_UUID{
	ulTimeLow:     0,
	usTimeMid:     0,
	usTimeHigh:    0,
	bClockSeqHigh: 0,
	bClockSeqLow:  0,
	rgbNode:       [6]C.BYTE{0, 0, 0, 0, 0, 5},
}

var TSS_UUID_USK3 = C.TSS_UUID{
	ulTimeLow:     0,
	usTimeMid:     0,
	usTimeHigh:    0,
	bClockSeqHigh: 0,
	bClockSeqLow:  0,
	rgbNode:       [6]C.BYTE{0, 0, 0, 0, 0, 6},
}

var TSS_UUID_USK4 = C.TSS_UUID{
	ulTimeLow:     0,
	usTimeMid:     0,
	usTimeHigh:    0,
	bClockSeqHigh: 0,
	bClockSeqLow:  0,
	rgbNode:       [6]C.BYTE{0, 0, 0, 0, 0, 7},
}

var TSS_UUID_USK5 = C.TSS_UUID{
	ulTimeLow:     0,
	usTimeMid:     0,
	usTimeHigh:    0,
	bClockSeqHigh: 0,
	bClockSeqLow:  0,
	rgbNode:       [6]C.BYTE{0, 0, 0, 0, 0, 9},
}

var TSS_UUID_USK6 = C.TSS_UUID{
	ulTimeLow:     0,
	usTimeMid:     0,
	usTimeHigh:    0,
	bClockSeqHigh: 0,
	bClockSeqLow:  0,
	rgbNode:       [6]C.BYTE{0, 0, 0, 0, 0, 10},
}
