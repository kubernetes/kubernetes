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

package samples

import (
	"time"

	runtime "k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/intstr"
)

// Map sample tests openAPIGen.generateMapProperty method.
type MapSample struct {
	// A sample String to String map
	StringToString map[string]string
	// A sample String to struct map
	StringToStruct map[string]MapSample

	name string
}

// Sample with simple data types.
type SimpleSample struct {
	// A simple string
	String string
	// A simple int
	Int int
	// A simple int64
	Int64 int64
	// A simple int32
	Int32 int32
	// A simple int16
	Int16 int16
	// A simple int8
	Int8 int8
	// A simple int
	Uint uint
	// A simple int64
	Uint64 uint64
	// A simple int32
	Uint32 uint32
	// A simple int16
	Uint16 uint16
	// A simple int8
	Uint8 uint8
	// A simple byte
	Byte byte
	// A simple boolean
	Bool bool
	// A simple float64
	Float64 float64
	// A simple float32
	Float32 float32
	// A simple time
	Time time.Time
	// a base64 encoded characters
	ByteArray []byte

	// a runtime object
	RObject runtime.Object

	// an int or string type
	IntOrString intstr.IntOrString
}

// Map sample tests openAPIGen.generateMapProperty method.
// +openapi=should-fail
type FailingMapSample1 struct {
	// A sample String to String map
	StringToArray map[string]map[string]string
}

// Map sample tests openAPIGen.generateMapProperty method.
// +openapi=should-fail
type FailingMapSample2 struct {
	// A sample String to String map
	StringToArray map[int]string
}

// SampleStructProperty demonstrates properties with struct type
type SampleStructProperty struct {
	// Struct is a reference to another struct in this file
	Struct SimpleSample
}

// SampleSliceProperty demonstrates properties with slice type
type SampleSliceProperty struct {
	// A simple string slice
	StringSlice []string
	// A simple struct slice
	StructSlice []SimpleSample
}

// PointerSample demonstrate pointer's properties
type PointerSample struct {
	// A string pointer
	StringPointer *string
	// A struct pointer
	StructPointer *SimpleSample
	// A slice pointer
	SlicePointer *[]string
	// A map pointer
	MapPointer *map[string]string
}
