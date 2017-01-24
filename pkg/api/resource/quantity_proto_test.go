/*
Copyright 2017 The Kubernetes Authors.

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

package resource

import (
	"testing"

	inf "gopkg.in/inf.v0"
)

func TestQuantityProtoMarshal(t *testing.T) {
	table := []struct {
		quantity string
		expect   Quantity
	}{
		{"0", Quantity{i: int64Amount{value: 0, scale: 0}, s: "0", Format: DecimalSI}},
		{"100m", Quantity{i: int64Amount{value: 100, scale: -3}, s: "100m", Format: DecimalSI}},
		{"50m", Quantity{i: int64Amount{value: 50, scale: -3}, s: "50m", Format: DecimalSI}},
		{"10000T", Quantity{i: int64Amount{value: 10000, scale: 12}, s: "10000T", Format: DecimalSI}},
	}
	for _, testCase := range table {
		q := MustParse(testCase.quantity)
		// Won't currently get an error as MarshalTo can't return one
		result, _ := q.Marshal()
		q.MarshalTo(result)
		if q.Cmp(testCase.expect) != 0 {
			t.Errorf("Expected: %v, Actual: %v", testCase.expect, q)
		}
	}

	nils := []struct {
		dec    *inf.Dec
		expect Quantity
	}{
		{dec(0, 0).Dec, Quantity{i: int64Amount{value: 0, scale: 0}, d: infDecAmount{dec(0, 0).Dec}, s: "0", Format: DecimalSI}},
		{dec(10, 0).Dec, Quantity{i: int64Amount{value: 0, scale: 0}, d: infDecAmount{dec(10, 0).Dec}, s: "10", Format: DecimalSI}},
		{dec(-10, 0).Dec, Quantity{i: int64Amount{value: 0, scale: 0}, d: infDecAmount{dec(-10, 0).Dec}, s: "-10", Format: DecimalSI}},
	}
	for _, nilCase := range nils {
		q := Quantity{d: infDecAmount{nilCase.dec}, Format: DecimalSI}
		// Won't currently get an error as MarshalTo can't return one
		result, _ := q.Marshal()
		q.Unmarshal(result)
		if q.Cmp(nilCase.expect) != 0 {
			t.Errorf("Expected: %v, Actual: %v", nilCase.expect, q)
		}
	}
}

func TestQuantityProtoUnmarshal(t *testing.T) {
	table := []struct {
		input  Quantity
		expect string
	}{
		{Quantity{i: int64Amount{value: 0, scale: 0}, s: "0", Format: DecimalSI}, "0"},
		{Quantity{i: int64Amount{value: 100, scale: -3}, s: "100m", Format: DecimalSI}, "100m"},
		{Quantity{i: int64Amount{value: 50, scale: -3}, s: "50m", Format: DecimalSI}, "50m"},
		{Quantity{i: int64Amount{value: 10000, scale: 12}, s: "10000T", Format: DecimalSI}, "10000T"},
	}
	for _, testCase := range table {
		var inputQ Quantity
		expectQ := MustParse(testCase.expect)
		inputByteArray, _ := testCase.input.Marshal()
		inputQ.Unmarshal(inputByteArray)
		if inputQ.Cmp(expectQ) != 0 {
			t.Errorf("Expected: %v, Actual: %v", inputQ, expectQ)
		}
	}

	nils := []struct {
		input  Quantity
		expect *inf.Dec
	}{
		{Quantity{i: int64Amount{value: 0, scale: 0}, d: infDecAmount{dec(0, 0).Dec}, s: "0", Format: DecimalSI}, dec(0, 0).Dec},
		{Quantity{i: int64Amount{value: 0, scale: 0}, d: infDecAmount{dec(10, 0).Dec}, s: "10", Format: DecimalSI}, dec(10, 0).Dec},
		{Quantity{i: int64Amount{value: 0, scale: 0}, d: infDecAmount{dec(-10, 0).Dec}, s: "-10", Format: DecimalSI}, dec(-10, 0).Dec},
	}
	for _, nilCase := range nils {
		var inputQ Quantity
		expectQ := Quantity{d: infDecAmount{nilCase.expect}, Format: DecimalSI}
		inputByteArray, _ := nilCase.input.Marshal()
		inputQ.Unmarshal(inputByteArray)
		if inputQ.Cmp(expectQ) != 0 {
			t.Errorf("Expected: %v, Actual: %v", inputQ, expectQ)
		}
	}
}
