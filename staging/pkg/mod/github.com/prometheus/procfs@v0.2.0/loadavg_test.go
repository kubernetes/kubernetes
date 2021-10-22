// Copyright 2019 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package procfs

import (
	"testing"

	"github.com/google/go-cmp/cmp"
)

func TestLoadAvg(t *testing.T) {
	fs, err := NewFS(procTestFixtures)
	if err != nil {
		t.Fatalf("failed to open procfs: %v", err)
	}

	loadavg, err := fs.LoadAvg()
	if err != nil {
		t.Fatalf("failed to get loadavg: %v", err)
	}

	if diff := cmp.Diff(0.02, loadavg.Load1); diff != "" {
		t.Fatalf("unexpected LoadAvg Per a minute:\n%s", diff)
	}
	if diff := cmp.Diff(0.04, loadavg.Load5); diff != "" {
		t.Fatalf("unexpected LoadAvg Per five minutes:\n%s", diff)
	}
	if diff := cmp.Diff(0.05, loadavg.Load15); diff != "" {
		t.Fatalf("unexpected LoadAvg Per fifteen minutes:\n%s", diff)
	}
}

func Test_parseLoad(t *testing.T) {
	tests := []struct {
		name    string
		s       string
		ok      bool
		loadavg *LoadAvg
	}{
		{
			name: "empty",
			ok:   false,
		},
		{
			name: "not enough fields",
			s:    `0.00 0.03`,
			ok:   false,
		},
		{
			name: "invalid line",
			s:    `malformed line`,
			ok:   false,
		},
		{
			name:    "valid line",
			s:       `0.00 0.03 0.05 1/502 33634`,
			ok:      true,
			loadavg: &LoadAvg{Load1: 0, Load5: 0.03, Load15: 0.05},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			loadavg, err := parseLoad([]byte(tt.s))
			if err != nil {
				if tt.ok {
					t.Fatalf("failed to parse loadavg: %v", err)
				}

				t.Logf("OK error: %v", err)
				return
			}
			if !tt.ok {
				t.Fatal("expected an error, but none occurred")
			}

			if diff := cmp.Diff(tt.loadavg, loadavg); diff != "" {
				t.Errorf("unexpected loadavg(-want +got):\n%s", diff)
			}
		})
	}
}
