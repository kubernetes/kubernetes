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

// NV is a TSS NV object
type NV struct {
	handle  C.TSS_HNVSTORE
	context C.TSS_HCONTEXT
}

// ReadValue reads length bytes from offset in the TPM NVRAM space
func (nv *NV) ReadValue(offset uint, length uint) ([]byte, error) {
	data := make([]byte, length)
	err := tspiError(C.Tspi_NV_ReadValue(nv.handle, (C.UINT32)(offset), (*C.UINT32)(unsafe.Pointer(&length)), (**C.BYTE)(unsafe.Pointer(&data))))
	return data, err
}

// SetIndex sets the TPM NVRAM index that will be referenced by ReadValue()
func (nv *NV) SetIndex(index uint) error {
	err := tspiError(C.Tspi_SetAttribUint32((C.TSS_HOBJECT)(nv.handle), C.TSS_TSPATTRIB_NV_INDEX, 0, (C.UINT32)(index)))
	return err
}

// AssignPolicy assigns a policy to the TPM NVRAM region
func (nv *NV) AssignPolicy(policy *Policy) error {
	err := tspiError(C.Tspi_Policy_AssignToObject(policy.handle, (C.TSS_HOBJECT)(nv.handle)))
	return err
}
