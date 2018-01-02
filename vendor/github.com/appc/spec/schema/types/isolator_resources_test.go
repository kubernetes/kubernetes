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

package types

import (
	"reflect"
	"testing"

	"github.com/appc/spec/schema/types/resource"
)

func mustQuantity(s string) *resource.Quantity {
	q := resource.MustParse(s)
	return &q
}

func TestResourceMemoryIsolator(t *testing.T) {
	tests := []struct {
		inreq string
		inlim string

		wres *ResourceMemory
		werr bool
	}{
		{
			"100M",
			"200M",

			&ResourceMemory{
				ResourceBase{
					resourceValue{
						Request: mustQuantity("100M"),
						Limit:   mustQuantity("200M"),
					},
				},
			},
			false,
		},
	}
	for i, tt := range tests {
		gres, err := NewResourceMemoryIsolator(tt.inreq, tt.inlim)
		if gerr := err != nil; gerr != tt.werr {
			t.Errorf("#%d: want werr=%t, got %t (err=%v)", i, tt.werr, gerr, err)
		}
		if !reflect.DeepEqual(tt.wres, gres) {
			t.Errorf("#%d: want %s, got %s", i, tt.wres, gres)
		}
	}
}

func TestResourceCPUIsolator(t *testing.T) {
	tests := []struct {
		inreq string
		inlim string

		wres *ResourceCPU
		werr bool
	}{
		// empty is not valid
		{
			"",
			"2",

			nil,
			true,
		},
		// garbage value
		{
			"1",
			"such garbage",

			nil,
			true,
		},
		{
			"1",
			"2",

			&ResourceCPU{
				ResourceBase{
					resourceValue{
						Request: mustQuantity("1"),
						Limit:   mustQuantity("2"),
					},
				},
			},
			false,
		},
		{
			"345",
			"6",

			&ResourceCPU{
				ResourceBase{
					resourceValue{
						Request: mustQuantity("345"),
						Limit:   mustQuantity("6"),
					},
				},
			},
			false,
		},
	}
	for i, tt := range tests {
		gres, err := NewResourceCPUIsolator(tt.inreq, tt.inlim)
		if gerr := err != nil; gerr != tt.werr {
			t.Errorf("#%d: want werr=%t, got %t (err=%v)", i, tt.werr, gerr, err)
		}
		if !reflect.DeepEqual(tt.wres, gres) {
			t.Errorf("#%d: want %s, got %s", i, tt.wres, gres)
		}
	}
}
