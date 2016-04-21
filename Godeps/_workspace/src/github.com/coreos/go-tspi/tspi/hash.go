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

// Hash is a TSS hash
type Hash struct {
	handle  C.TSS_HHASH
	context C.TSS_HCONTEXT
}

// Update updates a TSS hash with the data provided. It returns an error on
// failure.
func (hash *Hash) Update(data []byte) error {
	err := tspiError(C.Tspi_Hash_UpdateHashValue(hash.handle, (C.UINT32)(len(data)), (*C.BYTE)(&data[0])))
	return err
}

// Verify checks whether a hash matches the signature signed with the
// provided key. It returns an error on failure.
func (hash *Hash) Verify(key *Key, signature []byte) error {
	err := tspiError(C.Tspi_Hash_VerifySignature(hash.handle, key.handle, (C.UINT32)(len(signature)), (*C.BYTE)(&signature[0])))
	return err
}
