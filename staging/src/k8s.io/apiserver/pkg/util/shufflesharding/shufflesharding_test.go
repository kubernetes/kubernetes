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
	"errors"
	"math/rand"
	"testing"
)

func TestValidateParameters(t *testing.T) {
	tests := []struct {
		name      string
		numQueues int32
		handSize  int32
		validated bool
	}{
		{
			"numQueues is < 0",
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
			"numQueues is 0",
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
			"handSize is greater than numQueues",
			128,
			129,
			false,
		},
		{
			"numQueues: 128 handSize: 6",
			128,
			6,
			true,
		},
		{
			"numQueues: 1024 handSize: 6",
			1024,
			6,
			true,
		},
		{
			"numQueues: 512 handSize: 8",
			512,
			8,
			false,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if ValidateParameters(test.numQueues, test.handSize) != test.validated {
				t.Errorf("test case %s fails", test.name)
				return
			}
		})
	}
}

func BenchmarkValidateParameters(b *testing.B) {
	queueSize, handSize := int32(512), int32(8)
	for i := 0; i < b.N; i++ {
		_ = ValidateParameters(queueSize, handSize)
	}
}

func TestDealWithValidation(t *testing.T) {
	tests := []struct {
		name      string
		numQueues int32
		handSize  int32
		pick      func(int32) error
		validated bool
	}{
		{
			"numQueues is < 0",
			-100,
			8,
			func(i int32) error {
				return nil
			},
			false,
		},
		{
			"handSize is < 0",
			128,
			-100,
			func(i int32) error {
				return nil
			},
			false,
		},
		{
			"numQueues is 0",
			0,
			8,
			func(i int32) error {
				return nil
			},
			false,
		},
		{
			"handSize is 0",
			128,
			0,
			func(i int32) error {
				return nil
			},
			false,
		},
		{
			"handSize is greater than numQueues",
			128,
			129,
			func(i int32) error {
				return nil
			},
			false,
		},
		{
			"numQueues: 128 handSize: 6",
			128,
			6,
			func(i int32) error {
				return nil
			},
			true,
		},
		{
			"numQueues: 1024 handSize: 6",
			1024,
			6,
			func(i int32) error {
				return nil
			},
			true,
		},
		{
			"numQueues: 128 handSize: 6 with bad pick",
			128,
			6,
			func(i int32) error {
				return errors.New("for test")
			},
			false,
		},
		{
			"numQueues: 512 handSize: 8",
			512,
			8,
			func(i int32) error {
				return nil
			},
			false,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if (DealWithValidation(rand.Uint64(), test.numQueues, test.handSize, test.pick) == nil) != test.validated {
				t.Errorf("test case %s fails", test.name)
				return
			}
		})
	}
}

func BenchmarkDeal(b *testing.B) {
	hashValue := rand.Uint64()
	queueSize, handSize := int32(512), int32(8)
	pick := func(int32) error {
		return nil
	}
	for i := 0; i < b.N; i++ {
		_ = Deal(hashValue, queueSize, handSize, pick)
	}
}

func TestDealToSlices(t *testing.T) {
	tests := []struct {
		name      string
		numQueues int32
		handSize  int32
		validated bool
	}{
		{
			"validation fails",
			-100,
			-100,
			false,
		},
		{
			"numQueues == handSize == 4",
			4,
			4,
			true,
		},
		{
			"numQueues == handSize == 8",
			8,
			8,
			true,
		},
		{
			"numQueues == handSize == 10",
			10,
			10,
			true,
		},
		{
			"numQueues == handSize == 12",
			12,
			12,
			true,
		},
	}
	for _, test := range tests {
		hashValue := rand.Uint64()
		t.Run(test.name, func(t *testing.T) {
			cards, err := DealToSlice(hashValue, test.numQueues, test.handSize)
			if (err == nil) != test.validated {
				t.Errorf("test case %s fails in validation check", test.name)
				return
			}

			if test.validated {
				// check cards number
				if len(cards) != int(test.handSize) {
					t.Errorf("test case %s fails in cards number", test.name)
					return
				}

				// check cards duplication
				cardMap := make(map[int32]struct{}, test.handSize)
				for _, cardIdx := range cards {
					cardMap[cardIdx] = struct{}{}
				}
				for i := int32(0); i < test.handSize; i++ {
					if _, ok := cardMap[i]; !ok {
						t.Errorf("test case %s fails in duplication check", test.name)
						return
					}
				}
			}
		})
	}
}
