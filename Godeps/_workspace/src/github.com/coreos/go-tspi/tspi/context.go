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
import "unsafe"

// Context is a TSS context
type Context struct {
	context C.TSS_HCONTEXT
	tpm     TPM
}

// NewContext returns a TSS daemon context
func NewContext() (*Context, error) {
	context := new(Context)
	err := tspiError(C.Tspi_Context_Create(&context.context))
	return context, err
}

// Connect opens a connection between the context and the TSS daemon. It
// returns an error on failure.
func (context *Context) Connect() error {
	var tpmhandle C.TSS_HTPM
	err := tspiError(C.Tspi_Context_Connect(context.context, nil))
	if err != nil {
		return err
	}
	C.Tspi_Context_GetTpmObject(context.context, &tpmhandle)
	context.tpm = TPM{handle: tpmhandle, context: context.context}
	return nil
}

// Close closes the connection between the context and the TSS daemon. It
// returns an error on failure.
func (context *Context) Close() error {
	err := tspiError(C.Tspi_Context_Close(context.context))
	return err
}

// CreateNV creates a TSS object referring to a TPM NVRAM area. It returns a
// reference to the object and any error.
func (context *Context) CreateNV() (*NV, error) {
	var handle C.TSS_HNVSTORE
	err := tspiError(C.Tspi_Context_CreateObject(context.context, C.TSS_OBJECT_TYPE_NV, 0, (*C.TSS_HOBJECT)(&handle)))
	return &NV{handle: handle, context: context.context}, err
}

// CreateKey creates a TSS object referring to a TPM key. It returns a
// reference to the object and any error.
func (context *Context) CreateKey(flags int) (*Key, error) {
	var handle C.TSS_HKEY
	err := tspiError(C.Tspi_Context_CreateObject(context.context, C.TSS_OBJECT_TYPE_RSAKEY, (C.TSS_FLAG)(flags), (*C.TSS_HOBJECT)(&handle)))
	return &Key{handle: handle, context: context.context}, err
}

// LoadKeyByUUID loads the key referenced by UUID. The storetype argument
// indicates whether the key should be obtained from the system or user
// stores. It returns a reference to the key and any error.
func (context *Context) LoadKeyByUUID(storetype int, uuid C.TSS_UUID) (*Key, error) {
	var handle C.TSS_HKEY
	err := tspiError(C.Tspi_Context_LoadKeyByUUID(context.context, (C.TSS_FLAG)(storetype), uuid, &handle))
	return &Key{handle: handle, context: context.context}, err
}

// LoadKeyByBlob takes an encrypted key blob and reads it into the TPM. It
// takes a reference to the parent key and the key blob, and returns a
// reference to the key and any error.
func (context *Context) LoadKeyByBlob(parent *Key, blob []byte) (*Key, error) {
	var handle C.TSS_HKEY
	err := tspiError(C.Tspi_Context_LoadKeyByBlob(context.context, parent.handle, (C.UINT32)(len(blob)), (*C.BYTE)(unsafe.Pointer(&blob[0])), &handle))
	return &Key{handle: handle, context: context.context}, err
}

// GetTPM returns a reference to the TPM associated with this context
func (context *Context) GetTPM() *TPM {
	return &context.tpm
}

// CreatePolicy creates an object referring to a TSS policy. It returns a
// reference to the object plus any error.
func (context *Context) CreatePolicy(flags int) (*Policy, error) {
	var handle C.TSS_HPOLICY
	err := tspiError(C.Tspi_Context_CreateObject(context.context, C.TSS_OBJECT_TYPE_POLICY, (C.TSS_FLAG)(flags), (*C.TSS_HOBJECT)(&handle)))
	return &Policy{handle: handle, context: context.context}, err
}

// CreatePCRs creates an object referring to a TSS PCR composite. It returns
// a reference to the object plus any error.
func (context *Context) CreatePCRs(flags int) (*PCRs, error) {
	var handle C.TSS_HPCRS
	err := tspiError(C.Tspi_Context_CreateObject(context.context, C.TSS_OBJECT_TYPE_PCRS, (C.TSS_FLAG)(flags), (*C.TSS_HOBJECT)(&handle)))
	return &PCRs{handle: handle, context: context.context}, err
}
