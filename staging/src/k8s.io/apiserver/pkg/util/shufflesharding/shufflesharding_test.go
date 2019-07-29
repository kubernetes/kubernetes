/*
Copyright 2019 The Kubernetes Authors.

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

package shufflesharding

import (
	"testing"
)

func TestValidateParameters(t *testing.T) {
	tests := []struct {
		name      string
		queueSize int32
		handSize  int32
		validated bool
	}{
		{
			"queueSize is < 0",
			-100,
			8,
			false,
		},
		{
			"handSize is < 0",
			128,
			-100,
			false,
		},
		{
			"queueSize is 0",
			0,
			8,
			false,
		},
		{
			"handSize is 0",
			128,
			0,
			false,
		},
		{
			"handSize is greater than queueSize",
			128,
			129,
			false,
		},
		{
			"queueSize: 128 handSize: 6",
			128,
			6,
			true,
		},
		{
			"queueSize: 1024 handSize: 6",
			1024,
			6,
			true,
		},
		{
			"queueSize: 512 handSize: 8",
			512,
			8,
			false,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if ValidateParameters(test.queueSize, test.handSize) != test.validated {
				t.Errorf("test case %s fails", test.name)
				return
			}
		})
	}
}

func BenchmarkValidateParameters(b *testing.B) {
	for i := 0; i < b.N; i++ {
		//queueSize, handSize := uint32(rand.Intn(513)), uint32(rand.Intn(17))
		queueSize, handSize := int32(512), int32(8)
		ValidateParameters(queueSize, handSize)
	}
}
