/*
Copyright 2016 The Kubernetes Authors.

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

package conversion

type TestObject interface {
	DeepCopyTestObject() TestObject
}

// +k8s:deepcopy-gen=true
type TestSubSubStruct struct {
	D map[string]int
	E int
}

// +k8s:deepcopy-gen=true
type TestSubStruct struct {
	A, B, C TestSubSubStruct
	X       []int
	Y       []byte
}

// +k8s:deepcopy-gen:interfaces=k8s.io/kubernetes/pkg/conversion.TestObject
// +k8s:deepcopy-gen=true
// TestStruct is used to test deepcopy
type TestStruct struct {
	Map           map[string]string
	Int           int
	String        string
	Pointer       *int
	Struct        TestSubStruct
	StructPointer *TestSubStruct
	StructSlice   []*TestSubStruct
	StructMap     map[string]*TestSubStruct
}
