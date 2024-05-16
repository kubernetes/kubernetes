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

// Package json implements fuzzers for json deserialization routines in
// Kubernetes. These targets are compatible with the github.com/dvyukov/go-fuzz
// fuzzing framework.
package json

import (
	"bytes"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer/json"
)

var (
	gvk                = &schema.GroupVersionKind{Version: "v1"}
	strictOpt          = json.SerializerOptions{Yaml: false, Pretty: false, Strict: true}
	strictYamlOpt      = json.SerializerOptions{Yaml: true, Pretty: false, Strict: true}
	strictPrettyOpt    = json.SerializerOptions{Yaml: false, Pretty: true, Strict: true}
	nonstrictOpt       = json.SerializerOptions{Yaml: false, Pretty: false, Strict: false}
	nonstrictYamlOpt   = json.SerializerOptions{Yaml: true, Pretty: false, Strict: false}
	nonstrictPrettyOpt = json.SerializerOptions{Yaml: false, Pretty: true, Strict: false}
	scheme             = runtime.NewScheme()
	strictSer          = json.NewSerializerWithOptions(json.DefaultMetaFactory, scheme, scheme, strictOpt)
	ysSer              = json.NewSerializerWithOptions(json.DefaultMetaFactory, scheme, scheme, strictYamlOpt)
	psSer              = json.NewSerializerWithOptions(json.DefaultMetaFactory, scheme, scheme, strictPrettyOpt)
	nonstrictSer       = json.NewSerializerWithOptions(json.DefaultMetaFactory, scheme, scheme, nonstrictOpt)
	ynsSer             = json.NewSerializerWithOptions(json.DefaultMetaFactory, scheme, scheme, nonstrictYamlOpt)
	pnsSer             = json.NewSerializerWithOptions(json.DefaultMetaFactory, scheme, scheme, nonstrictPrettyOpt)
)

// FuzzStrictDecode is a fuzz target for "k8s.io/apimachinery/pkg/runtime/serializer/json" strict decoding.
func FuzzStrictDecode(data []byte) int {
	obj0, _, err0 := strictSer.Decode(data, gvk, nil)
	obj1, _, err1 := nonstrictSer.Decode(data, gvk, nil)
	obj2, _, err2 := ysSer.Decode(data, gvk, nil)
	obj3, _, err3 := psSer.Decode(data, gvk, nil)
	if obj0 == nil {
		if obj1 != nil {
			panic("NonStrict is stricter than Strict")
		}
		if obj2 != nil {
			panic("Yaml strict different from plain strict")
		}
		if obj3 != nil {
			panic("Pretty strict different from plain strict")
		}
		if err0 == nil || err1 == nil || err2 == nil || err3 == nil {
			panic("no error")
		}
		return 0
	}

	if err0 != nil {
		panic("got object and error for strict")
	}
	if err2 != nil {
		panic("got object and error for yaml strict")
	}
	if err3 != nil {
		panic("got object and error pretty strict")
	}

	var b0 bytes.Buffer
	err4 := strictSer.Encode(obj0, &b0)
	if err4 != nil {
		panic("Can't encode decoded data")
	}
	if !bytes.Equal(b0.Bytes(), data) {
		panic("Encoded data doesn't match original")
	}

	b0.Reset()
	err5 := ysSer.Encode(obj1, &b0)
	if err5 != nil {
		panic("Can't encode yaml strict decoded data")
	}
	if !bytes.Equal(b0.Bytes(), data) {
		panic("Encoded yaml strict data doesn't match original")
	}

	b0.Reset()
	err6 := psSer.Encode(obj2, &b0)
	if err6 != nil {
		panic("Can't encode pretty strict decoded data")
	}
	if !bytes.Equal(b0.Bytes(), data) {
		panic("Encoded pretty strict data doesn't match original")
	}

	b0.Reset()
	err7 := nonstrictSer.Encode(obj3, &b0)
	if err7 != nil {
		panic("Can't encode nonstrict decoded data")
	}
	if !bytes.Equal(b0.Bytes(), data) {
		panic("Encoded nonstrict data doesn't match original")
	}
	return 1
}

// FuzzNonStrictDecode is a fuzz target for "k8s.io/apimachinery/pkg/runtime/serializer/json" non-strict decoding.
func FuzzNonStrictDecode(data []byte) int {
	obj0, _, err0 := nonstrictSer.Decode(data, gvk, nil)
	if err0 != nil {
		return 0
	}

	var b0 bytes.Buffer
	err1 := nonstrictSer.Encode(obj0, &b0)
	if err1 != nil {
		panic("Can't nonstrict encode decoded data")
	}
	_, _, err2 := nonstrictSer.Decode(b0.Bytes(), gvk, nil)
	if err2 != nil {
		panic("Can't nonstrict decode encoded data")
	}

	b0.Reset()
	err3 := ynsSer.Encode(obj0, &b0)
	if err3 != nil {
		panic("Can't yaml strict encode decoded data")
	}
	_, _, err4 := nonstrictSer.Decode(b0.Bytes(), gvk, nil)
	if err4 != nil {
		panic("Can't nonstrict decode encoded data")
	}

	b0.Reset()
	err5 := pnsSer.Encode(obj0, &b0)
	if err5 != nil {
		panic("Can't pretty strict encode decoded data")
	}
	_, _, err6 := nonstrictSer.Decode(b0.Bytes(), gvk, nil)
	if err6 != nil {
		panic("Can't nonstrict decode encoded data")
	}
	return 1
}
