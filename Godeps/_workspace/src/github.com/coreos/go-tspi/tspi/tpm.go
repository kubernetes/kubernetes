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
import "C"
import (
	"bufio"
	"crypto/sha1"
	"encoding/binary"
	"io"
	"os"
	"unsafe"

	"github.com/coreos/go-tspi/tspiconst"
)

// TPM is a TSS TPM object
type TPM struct {
	handle  C.TSS_HTPM
	context C.TSS_HCONTEXT
}

// GetEventLog returns an array of structures representing the contents of the
// TSS event log
func (tpm *TPM) GetEventLog() ([]tspiconst.Log, error) {
	var count C.UINT32
	var events *C.TSS_PCR_EVENT
	var event C.TSS_PCR_EVENT
	var log []tspiconst.Log

	f, err := os.Open("/sys/kernel/security/tpm0/binary_bios_measurements")
	if err != nil {
		return nil, err
	}

	firmware_events := bufio.NewReader(f)

	for {
		var entry tspiconst.Log
		var datalen int32
		err := binary.Read(firmware_events, binary.LittleEndian, &entry.Pcr)
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
		err = binary.Read(firmware_events, binary.LittleEndian, &entry.Eventtype)
		if err != nil {
			return nil, err
		}
		err = binary.Read(firmware_events, binary.LittleEndian, &entry.PcrValue)
		if err != nil {
			return nil, err
		}
		err = binary.Read(firmware_events, binary.LittleEndian, &datalen)
		if err != nil {
			return nil, err
		}
		data := make([]byte, datalen)
		err = binary.Read(firmware_events, binary.LittleEndian, &data)
		if err != nil {
			return nil, err
		}
		entry.Event = data[:]
		log = append(log, entry)
	}

	err = tspiError(C.Tspi_TPM_GetEventLog(tpm.handle, &count, &events))
	if err != nil {
		return nil, err
	}

	if count == 0 {
		return log, nil
	}

	length := count * C.UINT32(unsafe.Sizeof(event))
	slice := (*[1 << 30]C.TSS_PCR_EVENT)(unsafe.Pointer(events))[:length:length]

	for i := 0; i < int(count); i++ {
		var entry tspiconst.Log
		entry.Pcr = int32(slice[i].ulPcrIndex)
		entry.Eventtype = int32(slice[i].eventType)
		copy(entry.PcrValue[:],C.GoBytes(unsafe.Pointer(slice[i].rgbPcrValue), C.int(slice[i].ulPcrValueLength)))
		entry.Event = C.GoBytes(unsafe.Pointer(slice[i].rgbEvent), C.int(slice[i].ulEventLength))
		log = append(log, entry)
	}
	C.Tspi_Context_FreeMemory(tpm.context, (*C.BYTE)(unsafe.Pointer(events)))
	return log, nil
}

// ExtendPCR extends a pcr. If event is nil, data must be pre-hashed with
// SHA1. If event is not nil, event is used to populate the TSS event
// log. If both data and event are provided, both will be used to create the
// extend hash.
func (tpm *TPM) ExtendPCR(pcr int, data []byte, eventtype int, event []byte) error {
	var outlen C.UINT32
	var pcrval *C.BYTE
	var eventstruct C.TSS_PCR_EVENT
	var err error

	shasum := sha1.Sum(data)

	if event != nil {
		var pcrdata *C.BYTE
		var pcrdatalen C.UINT32

		eventstruct.versionInfo.bMajor = 1
		eventstruct.versionInfo.bMinor = 2
		eventstruct.versionInfo.bRevMajor = 1
		eventstruct.versionInfo.bRevMinor = 0
		eventstruct.ulPcrIndex = C.UINT32(pcr)
		eventstruct.rgbPcrValue = (*C.BYTE)(&shasum[0])
		eventstruct.eventType = C.TSS_EVENTTYPE(eventtype)
		eventstruct.ulEventLength = C.UINT32(len(event))
		eventstruct.rgbEvent = (*C.BYTE)(&event[0])

		if data == nil || len(data) == 0 {
			pcrdata = nil
			pcrdatalen = C.UINT32(0)
		} else {
			pcrdata = (*C.BYTE)(&data[0])
			pcrdatalen = C.UINT32(len(data))
		}

		err = tspiError(C.Tspi_TPM_PcrExtend(tpm.handle, C.UINT32(pcr), pcrdatalen, pcrdata, &eventstruct, &outlen, &pcrval))
	} else {
		err = tspiError(C.Tspi_TPM_PcrExtend(tpm.handle, C.UINT32(pcr), C.UINT32(len(shasum)), (*C.BYTE)(&shasum[0]), nil, &outlen, &pcrval))
	}

	C.Tspi_Context_FreeMemory(tpm.context, pcrval)

	return err
}

//GetQuote takes an encrypted key blob representing the AIK, a set of PCRs
//and a challenge and returns a blob containing a hash of the PCR hashes and
//the challenge, and a validation blob signed by the AIK.
func (tpm *TPM) GetQuote(aik *Key, pcrs *PCRs, challenge []byte) ([]byte, []byte, error) {
	var validation C.TSS_VALIDATION
	challangeHash := sha1.Sum(challenge[:])

	validation.ulExternalDataLength = sha1.Size
	validation.rgbExternalData = (*C.BYTE)(&challangeHash[0])
	err := tspiError(C.Tspi_TPM_Quote(tpm.handle, aik.handle, pcrs.handle, &validation))

	if err != nil {
		return nil, nil, err
	}

	data := C.GoBytes(unsafe.Pointer(validation.rgbData), (C.int)(validation.ulDataLength))
	validationOutput := C.GoBytes(unsafe.Pointer(validation.rgbValidationData), (C.int)(validation.ulValidationDataLength))

	C.Tspi_Context_FreeMemory(tpm.context, validation.rgbData)
	C.Tspi_Context_FreeMemory(tpm.context, validation.rgbValidationData)

	return data, validationOutput, nil
}

// ActivateIdentity accepts an encrypted key blob representing the AIK and
// two blobs representing the asymmetric and symmetric challenges associated
// with the AIK. If the TPM is able to decrypt the challenges and the
// challenges correspond to the AIK, the TPM will return the original
// challenge secret.
func (tpm *TPM) ActivateIdentity(aik *Key, asymblob []byte, symblob []byte) (secret []byte, err error) {
	var creds *C.BYTE
	var credlen C.UINT32

	err = tspiError(C.Tspi_TPM_ActivateIdentity(tpm.handle, aik.handle, (C.UINT32)(len(asymblob)), (*C.BYTE)(&asymblob[0]), (C.UINT32)(len(symblob)), (*C.BYTE)(&symblob[0]), &credlen, &creds))

	if err != nil {
		return nil, err
	}

	plaintext := C.GoBytes(unsafe.Pointer(creds), (C.int)(credlen))

	C.Tspi_Context_FreeMemory(tpm.context, creds)

	return plaintext, nil
}

// GetPolicy returns the TSS policy associated with the TPM.
func (tpm *TPM) GetPolicy(poltype int) (*Policy, error) {
	var policyHandle C.TSS_HPOLICY
	err := tspiError(C.Tspi_GetPolicyObject((C.TSS_HOBJECT)(tpm.handle), (C.TSS_FLAG)(poltype), &policyHandle))
	return &Policy{handle: policyHandle, context: tpm.context}, err
}

// TakeOwnership transitions a TPM from unowned state to owned, installing the
// encrypted key blob as the SRK.
func (tpm *TPM) TakeOwnership(srk *Key) error {
	err := tspiError(C.Tspi_TPM_TakeOwnership(tpm.handle, srk.handle, 0))
	return err
}


// AssignPolicy assigns a TSS policy to the TPM.
func (tpm *TPM) AssignPolicy(policy *Policy) error {
	err := tspiError(C.Tspi_Policy_AssignToObject(policy.handle, (C.TSS_HOBJECT)(tpm.handle)))
	return err
}

// CollateIdentityRequest creates a signing request for the provided AIKq
func (tpm *TPM) CollateIdentityRequest(srk *Key, pubkey *Key, aik *Key) ([]byte, error) {
	var certLen C.UINT32
	var cCertReq *C.BYTE
	err := tspiError(C.Tspi_TPM_CollateIdentityRequest(tpm.handle, srk.handle, pubkey.handle, 0, nil, aik.handle, C.TSS_ALG_AES, &certLen, &cCertReq))
	certReq := C.GoBytes(unsafe.Pointer(cCertReq), (C.int)(certLen))
	C.Tspi_Context_FreeMemory(tpm.context, cCertReq)
	return certReq, err
}
