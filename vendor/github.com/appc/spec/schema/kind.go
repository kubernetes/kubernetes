// Copyright 2015 The appc Authors
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

package schema

import (
	"encoding/json"

	"github.com/appc/spec/schema/types"
)

type Kind struct {
	ACVersion types.SemVer `json:"acVersion"`
	ACKind    types.ACKind `json:"acKind"`
}

type kind Kind

func (k *Kind) UnmarshalJSON(data []byte) error {
	nk := kind{}
	err := json.Unmarshal(data, &nk)
	if err != nil {
		return err
	}
	*k = Kind(nk)
	return nil
}

func (k Kind) MarshalJSON() ([]byte, error) {
	return json.Marshal(kind(k))
}
