/*
Copyright 2026 The Kubernetes Authors.

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

package pidlimit

import "testing"

func TestParseRunningTaskCount(t *testing.T) {
	tests := []struct {
		name    string
		loadavg string
		want    int64
		wantErr bool
	}{
		{
			name:    "valid",
			loadavg: "1.36 3.49 4.53 2/3518 3715089\n",
			want:    3518,
		},
		{
			name:    "valid with extra whitespace",
			loadavg: "  1.36\t3.49 4.53 2/42 3715089  \n",
			want:    42,
		},
		{
			name:    "not enough fields",
			loadavg: "1.36 3.49 4.53 2/3518",
			wantErr: true,
		},
		{
			name:    "missing task separator",
			loadavg: "1.36 3.49 4.53 3518 3715089",
			wantErr: true,
		},
		{
			name:    "too many task separators",
			loadavg: "1.36 3.49 4.53 2/3518/7 3715089",
			wantErr: true,
		},
		{
			name:    "invalid task count",
			loadavg: "1.36 3.49 4.53 2/not-a-number 3715089",
			wantErr: true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got, err := parseRunningTaskCount(test.loadavg)
			if (err != nil) != test.wantErr {
				t.Fatalf("parseRunningTaskCount() error = %v, wantErr %t", err, test.wantErr)
			}
			if got != test.want {
				t.Errorf("parseRunningTaskCount() = %d, want %d", got, test.want)
			}
		})
	}
}
