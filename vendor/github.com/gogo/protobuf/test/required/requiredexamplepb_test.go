// Protocol Buffers for Go with Gadgets
//
// Copyright (c) 2013, The GoGo Authors. All rights reserved.
// http://github.com/gogo/protobuf
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

package required

import (
	"github.com/gogo/protobuf/proto"
	"github.com/gogo/protobuf/test"
	"math/rand"
	"reflect"
	"strconv"
	"testing"
	"time"
)

func TestMarshalToErrorsWhenRequiredFieldIsNotPresent(t *testing.T) {
	data := RequiredExample{}
	buf, err := proto.Marshal(&data)
	if err == nil {
		t.Fatalf("err == nil; was %v instead", err)
	}
	if err.Error() != `proto: required field "theRequiredString" not set` {
		t.Fatalf(`err.Error() != "proto: required field "theRequiredString" not set"; was "%s" instead`, err.Error())
	}
	if len(buf) != 0 {
		t.Fatalf(`len(buf) != 0; was %d instead`, len(buf))
	}
}

func TestMarshalToSucceedsWhenRequiredFieldIsPresent(t *testing.T) {
	data := RequiredExample{
		TheRequiredString: proto.String("present"),
	}
	buf, err := proto.Marshal(&data)
	if err != nil {
		t.Fatalf("err != nil; was %v instead", err)
	}
	if len(buf) == 0 {
		t.Fatalf(`len(buf) == 0; expected nonzero`)
	}
}

func TestUnmarshalErrorsWhenRequiredFieldIsNotPresent(t *testing.T) {
	missingRequiredField := []byte{0x12, 0x8, 0x6f, 0x70, 0x74, 0x69, 0x6f, 0x6e, 0x61, 0x6c}
	data := RequiredExample{}
	err := proto.Unmarshal(missingRequiredField, &data)
	if err == nil {
		t.Fatalf("err == nil; was %v instead", err)
	}
	if err.Error() != `proto: required field "theRequiredString" not set` {
		t.Fatalf(`err.Error() != "proto: required field "theRequiredString" not set"; was "%s" instead`, err.Error())
	}
}

func TestUnmarshalSucceedsWhenRequiredIsNotPresent(t *testing.T) {
	dataOut := RequiredExample{
		TheRequiredString: proto.String("present"),
	}
	encodedMessage, err := proto.Marshal(&dataOut)
	if err != nil {
		t.Fatalf("Unexpected error when marshalling dataOut: %v", err)
	}
	dataIn := RequiredExample{}
	err = proto.Unmarshal(encodedMessage, &dataIn)
	if err != nil {
		t.Fatalf("err != nil; was %v instead", err)
	}
}

func TestUnmarshalPopulatedOptionalFieldsAsRequiredSucceeds(t *testing.T) {
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	dataOut := test.NewPopulatedNidOptNative(r, true)
	encodedMessage, err := proto.Marshal(dataOut)
	if err != nil {
		t.Fatalf("Unexpected error when marshalling dataOut: %v", err)
	}
	dataIn := NidOptNative{}
	err = proto.Unmarshal(encodedMessage, &dataIn)
	if err != nil {
		t.Fatalf("err != nil; was %v instead", err)
	}
}

func TestUnmarshalPartiallyPopulatedOptionalFieldsFails(t *testing.T) {
	// Fill in all fields, then randomly remove one.
	dataOut := &test.NinOptNative{
		Field1:  proto.Float64(0),
		Field2:  proto.Float32(0),
		Field3:  proto.Int32(0),
		Field4:  proto.Int64(0),
		Field5:  proto.Uint32(0),
		Field6:  proto.Uint64(0),
		Field7:  proto.Int32(0),
		Field8:  proto.Int64(0),
		Field9:  proto.Uint32(0),
		Field10: proto.Int32(0),
		Field11: proto.Uint64(0),
		Field12: proto.Int64(0),
		Field13: proto.Bool(false),
		Field14: proto.String("0"),
		Field15: []byte("0"),
	}
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	fieldName := "Field" + strconv.Itoa(r.Intn(15)+1)
	field := reflect.ValueOf(dataOut).Elem().FieldByName(fieldName)
	fieldType := field.Type()
	field.Set(reflect.Zero(fieldType))
	encodedMessage, err := proto.Marshal(dataOut)
	if err != nil {
		t.Fatalf("Unexpected error when marshalling dataOut: %v", err)
	}
	dataIn := NidOptNative{}
	err = proto.Unmarshal(encodedMessage, &dataIn)
	if err.Error() != `proto: required field "`+fieldName+`" not set` {
		t.Fatalf(`err.Error() != "proto: required field "`+fieldName+`" not set"; was "%s" instead`, err.Error())
	}
}

func TestMarshalFailsWithoutAllFieldsSet(t *testing.T) {
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	dataOut := NewPopulatedNinOptNative(r, true)
	fieldName := "Field" + strconv.Itoa(r.Intn(15)+1)
	field := reflect.ValueOf(dataOut).Elem().FieldByName(fieldName)
	fieldType := field.Type()
	field.Set(reflect.Zero(fieldType))
	encodedMessage, err := proto.Marshal(dataOut)
	if err.Error() != `proto: required field "`+fieldName+`" not set` {
		t.Fatalf(`err.Error() != "proto: required field "`+fieldName+`" not set"; was "%s" instead`, err.Error())
	}
	if len(encodedMessage) > 0 {
		t.Fatalf("Got some bytes from marshal, expected none.")
	}
}

func TestMissingFieldsOnRepeatedNestedTypes(t *testing.T) {
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	dataOut := &NestedNinOptNative{
		NestedNinOpts: []*NinOptNative{
			NewPopulatedNinOptNative(r, true),
			NewPopulatedNinOptNative(r, true),
			NewPopulatedNinOptNative(r, true),
		},
	}
	middle := dataOut.GetNestedNinOpts()[1]
	fieldName := "Field" + strconv.Itoa(r.Intn(15)+1)
	field := reflect.ValueOf(middle).Elem().FieldByName(fieldName)
	fieldType := field.Type()
	field.Set(reflect.Zero(fieldType))
	encodedMessage, err := proto.Marshal(dataOut)
	if err.Error() != `proto: required field "`+fieldName+`" not set` {
		t.Fatalf(`err.Error() != "proto: required field "`+fieldName+`" not set"; was "%s" instead`, err.Error())
	}
	if len(encodedMessage) > 0 {
		t.Fatalf("Got some bytes from marshal, expected none.")
	}
}
