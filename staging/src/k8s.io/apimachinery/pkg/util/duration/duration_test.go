/*
Copyright 2018 The Kubernetes Authors.

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

package duration

import (
	"testing"
	"time"
)

func TestHumanDuration(t *testing.T) {
	tests := []struct {
		d    time.Duration
		want string
	}{
		{d: time.Second, want: "1s"},
		{d: 70 * time.Second, want: "70s"},
		{d: 190 * time.Second, want: "3m10s"},
		{d: 70 * time.Minute, want: "70m"},
		{d: 47 * time.Hour, want: "47h"},
		{d: 49 * time.Hour, want: "2d1h"},
		{d: (8*24 + 2) * time.Hour, want: "8d"},
		{d: (367 * 24) * time.Hour, want: "367d"},
		{d: (365*2*24 + 25) * time.Hour, want: "2y1d"},
		{d: (365*8*24 + 2) * time.Hour, want: "8y"},
	}
	for _, tt := range tests {
		t.Run(tt.d.String(), func(t *testing.T) {
			if got := HumanDuration(tt.d); got != tt.want {
				t.Errorf("HumanDuration() = %v, want %v", got, tt.want)
			}
		})
	}
}
