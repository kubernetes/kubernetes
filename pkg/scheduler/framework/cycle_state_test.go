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

package framework

import (
	"testing"
)

type fakeData struct {
	data string
}

func (f *fakeData) Clone() StateData {
	copy := &fakeData{
		data: f.data,
	}
	return copy
}

func TestCycleStateClone(t *testing.T) {
	var key StateKey = "key"
	data1 := "value1"
	data2 := "value2"

	state := NewCycleState()
	originalValue := &fakeData{
		data: data1,
	}
	state.Write(key, originalValue)
	stateCopy := state.Clone()

	valueCopy, err := stateCopy.Read(key)
	if err != nil {
		t.Errorf("failed to read copied value: %v", err)
	}
	if v, ok := valueCopy.(*fakeData); ok && v.data != data1 {
		t.Errorf("clone failed, got %q, expected %q", v.data, data1)
	}

	originalValue.data = data2
	original, err := state.Read(key)
	if err != nil {
		t.Errorf("failed to read original value: %v", err)
	}
	if v, ok := original.(*fakeData); ok && v.data != data2 {
		t.Errorf("original value should change, got %q, expected %q", v.data, data2)
	}

	if v, ok := valueCopy.(*fakeData); ok && v.data != data1 {
		t.Errorf("cloned copy should not change, got %q, expected %q", v.data, data1)
	}
}

func TestCycleStateCloneNil(t *testing.T) {
	var state *CycleState
	stateCopy := state.Clone()
	if stateCopy != nil {
		t.Errorf("clone expected to be nil")
	}
}
