// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package profiler

import (
	"bytes"
	"io"
	"testing"

	"cloud.google.com/go/profiler/testdata"
	"github.com/google/pprof/profile"
)

func TestGoHeapProfile(t *testing.T) {
	oldStartCPUProfile, oldStopCPUProfile, oldWriteHeapProfile, oldSleep := startCPUProfile, stopCPUProfile, writeHeapProfile, sleep
	defer func() {
		startCPUProfile, stopCPUProfile, writeHeapProfile, sleep = oldStartCPUProfile, oldStopCPUProfile, oldWriteHeapProfile, oldSleep
	}()

	tests := []struct {
		name    string
		profile *profile.Profile
		wantErr bool
	}{
		{
			name:    "valid heap profile",
			profile: testdata.HeapProfileCollected1,
		},
		{
			name:    "profile with too few sample types",
			profile: testdata.AllocProfileUploaded,
			wantErr: true,
		},
		{
			name: "profile with incorrect sample types",
			profile: &profile.Profile{
				DurationNanos: 10e9,
				SampleType: []*profile.ValueType{
					{Type: "objects", Unit: "count"},
					{Type: "alloc_space", Unit: "bytes"},
					{Type: "inuse_objects", Unit: "count"},
					{Type: "inuse_space", Unit: "bytes"},
				},
			},
			wantErr: true,
		},
	}

	for _, tc := range tests {
		var profileBytes bytes.Buffer
		tc.profile.Write(&profileBytes)
		writeHeapProfile = func(w io.Writer) error {
			w.Write(profileBytes.Bytes())
			return nil
		}
		_, err := goHeapProfile()
		if tc.wantErr {
			if err == nil {
				t.Errorf("%s: goHeapProfile() got no error, want error", tc.name)
			}
			continue
		}
		if err != nil {
			t.Errorf("%s: goHeapProfile() got %q, want no error", tc.name, err)
		}
	}
}
