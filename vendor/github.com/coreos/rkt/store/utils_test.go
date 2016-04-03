// Copyright 2015 The rkt Authors
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

package store

import (
	"testing"

	"github.com/appc/spec/schema/types"
)

func TestLabelsToString(t *testing.T) {
	tests := []struct {
		labels types.Labels
		out    string
	}{
		{types.Labels{}, `[]`},
		{[]types.Label{{"label1", "value1"}}, `["label1":"value1"]`},
		{[]types.Label{{"label1", "value1"}, {"version", "2.1.1"}}, `["version":"2.1.1", "label1":"value1"]`},
		{[]types.Label{{"arch", "amd64"}, {"label1", "value1"}, {"os", "linux"}, {"version", "2.1.1"}}, `["version":"2.1.1", "os":"linux", "arch":"amd64", "label1":"value1"]`},
	}

	for i, tt := range tests {
		out := labelsToString(tt.labels)
		if out != tt.out {
			t.Errorf("#%d: got %v, want %v", i, out, tt.out)
		}
	}
}
