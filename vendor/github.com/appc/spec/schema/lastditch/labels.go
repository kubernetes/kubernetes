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

package lastditch

import (
	"encoding/json"
)

type Labels []Label

// a type just to avoid a recursion during unmarshalling
type labels Labels

type Label struct {
	Name  string `json:"name"`
	Value string `json:"value"`
}

func (l *Labels) UnmarshalJSON(data []byte) error {
	var jl labels
	if err := json.Unmarshal(data, &jl); err != nil {
		return err
	}
	*l = Labels(jl)
	return nil
}
