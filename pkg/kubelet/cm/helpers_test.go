/*
Copyright 2024 The Kubernetes Authors.

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

package cm

import (
	"cmp"
	"testing"

	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/utils/ptr"
)

func TestCompareCPUQuantities(t *testing.T) {
	tests := []struct {
		name   string
		x, y   *resource.Quantity
		expect int
	}{{
		name:   "equal millivalues",
		x:      resource.NewMilliQuantity(2000, resource.DecimalSI),
		y:      resource.NewQuantity(2, resource.DecimalSI),
		expect: 0,
	}, {
		name:   "equal quantities",
		x:      resource.NewMilliQuantity(100, resource.BinarySI),
		y:      ptr.To(resource.MustParse("100m")),
		expect: 0,
	}, {
		name:   "regular less than",
		x:      resource.NewMilliQuantity(100, resource.DecimalSI),
		y:      resource.NewMilliQuantity(200, resource.BinarySI),
		expect: -1,
	}, {
		name:   "regular greater than",
		x:      resource.NewMilliQuantity(200, resource.DecimalSI),
		y:      resource.NewMilliQuantity(100, resource.BinarySI),
		expect: 1,
	}, {
		name:   "equal zero values",
		x:      resource.NewMilliQuantity(0, resource.DecimalSI),
		y:      ptr.To(resource.MustParse("0")),
		expect: 0,
	}, {
		name:   "equal nil",
		x:      nil,
		y:      nil,
		expect: 0,
	}, {
		name:   "equivalent nil and zero",
		x:      nil,
		y:      ptr.To(resource.MustParse("0")),
		expect: 0,
	}, {
		name:   "equivalent zero and nil",
		x:      &resource.Quantity{},
		y:      nil,
		expect: 0,
	}, {
		name:   "nil less than",
		x:      nil,
		y:      resource.NewMilliQuantity(1, resource.DecimalSI),
		expect: -1,
	}, {
		name:   "greater than nil",
		x:      resource.NewQuantity(1, resource.BinarySI),
		y:      nil,
		expect: 1,
	}}

	for _, test := range tests {
		assert.Equal(t, test.expect, CompareCPUQuantities(test.x, test.y, cmp.Compare[int64]), test.name)
	}
}
