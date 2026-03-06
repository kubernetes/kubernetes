/*
Copyright The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package sharding

import (
	"testing"
)

func TestParse(t *testing.T) {
	tests := []struct {
		name    string
		input   string
		wantErr bool
		wantReq []ShardRangeRequirement
	}{
		{
			name:    "empty string",
			input:   "",
			wantErr: true,
		},
		{
			name:  "single requirement with uid",
			input: "shardRange(object.metadata.uid,0x0000000000000000,0x8000000000000000)",
			wantReq: []ShardRangeRequirement{
				{Key: "object.metadata.uid", Start: "0000000000000000", End: "8000000000000000"},
			},
		},
		{
			name:  "full range with 2^64 end",
			input: "shardRange(object.metadata.uid,0x0000000000000000,0x10000000000000000)",
			wantReq: []ShardRangeRequirement{
				{Key: "object.metadata.uid", Start: "0000000000000000", End: "10000000000000000"},
			},
		},
		{
			name:    "missing 0x prefix on start",
			input:   "shardRange(object.metadata.uid,0000000000000000,0x8000000000000000)",
			wantErr: true,
		},
		{
			name:    "missing 0x prefix on end",
			input:   "shardRange(object.metadata.uid,0x0000000000000000,8000000000000000)",
			wantErr: true,
		},
		{
			name:    "empty after 0x prefix",
			input:   "shardRange(object.metadata.uid,0x,0x8000000000000000)",
			wantErr: true,
		},
		{
			name:  "namespace field",
			input: "shardRange(object.metadata.namespace,0xaa,0xff)",
			wantReq: []ShardRangeRequirement{
				{Key: "object.metadata.namespace", Start: "aa", End: "ff"},
			},
		},
		{
			name:    "name field unsupported",
			input:   "shardRange(object.metadata.name,0x00,0x80)",
			wantErr: true,
		},
		{
			name:    "unsupported field",
			input:   "shardRange(object.metadata.labels,0x00,0x80)",
			wantErr: true,
		},
		{
			name:    "missing shardRange prefix",
			input:   "invalidFunc(object.metadata.uid,0x00,0x80)",
			wantErr: true,
		},
		{
			name:    "hex too long (18 chars)",
			input:   "shardRange(object.metadata.uid,0x000000000000000000,0x80)",
			wantErr: true,
		},
		{
			name:    "invalid hex char",
			input:   "shardRange(object.metadata.uid,0x0g,0x80)",
			wantErr: true,
		},
		{
			name:    "missing closing paren",
			input:   "shardRange(object.metadata.uid,0x00,0x80",
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			sel, err := Parse(tt.input)
			if tt.wantErr {
				if err == nil {
					t.Errorf("Parse(%q) expected error, got nil", tt.input)
				}
				return
			}
			if err != nil {
				t.Fatalf("Parse(%q) unexpected error: %v", tt.input, err)
			}

			reqs := sel.Requirements()
			if len(reqs) != len(tt.wantReq) {
				t.Fatalf("Parse(%q) got %d requirements, want %d", tt.input, len(reqs), len(tt.wantReq))
			}
			for i, req := range reqs {
				if req != tt.wantReq[i] {
					t.Errorf("Parse(%q) requirement[%d] = %+v, want %+v", tt.input, i, req, tt.wantReq[i])
				}
			}
		})
	}
}

func TestParseRoundTrip(t *testing.T) {
	inputs := []string{
		"shardRange(object.metadata.uid,0x0000000000000000,0x8000000000000000)",
		"shardRange(object.metadata.uid,0x8000000000000000,0x10000000000000000)",
		"shardRange(object.metadata.uid,0x0000000000000000,0x10000000000000000)",
		"shardRange(object.metadata.namespace,0xaa,0xff)",
	}

	for _, input := range inputs {
		sel, err := Parse(input)
		if err != nil {
			t.Fatalf("Parse(%q) error: %v", input, err)
		}
		output := sel.String()
		sel2, err := Parse(output)
		if err != nil {
			t.Fatalf("Parse(%q) (round-trip) error: %v", output, err)
		}
		if sel.String() != sel2.String() {
			t.Errorf("round-trip failed: %q -> %q -> %q", input, output, sel2.String())
		}
	}
}
