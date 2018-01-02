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

// +k8s:deepcopy-gen=package

// This is a test package.
package maps

type Ttest struct {
	Byte map[string]byte
	//Int8    map[string]int8 //TODO: int8 becomes byte in SnippetWriter
	Int16   map[string]int16
	Int32   map[string]int32
	Int64   map[string]int64
	Uint8   map[string]uint8
	Uint16  map[string]uint16
	Uint32  map[string]uint32
	Uint64  map[string]uint64
	Float32 map[string]float32
	Float64 map[string]float64
	String  map[string]string
}
