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

// Policy is a TSS policy object
type Policy struct {
	handle  C.TSS_HPOLICY
	context C.TSS_HCONTEXT
}

// SetSecret sets the secret for a policy. This policy may then be applied to
// another object.
func (policy *Policy) SetSecret(sectype int, secret []byte) error {
	err := tspiError(C.Tspi_Policy_SetSecret(policy.handle, (C.TSS_FLAG)(sectype), (C.UINT32)(len(secret)), (*C.BYTE)(&secret[0])))
	return err
}
