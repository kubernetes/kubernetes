// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tag_test

import (
	"reflect"
	"testing"

	"google.golang.org/protobuf/internal/encoding/tag"
	fdesc "google.golang.org/protobuf/internal/filedesc"
	"google.golang.org/protobuf/proto"
	pdesc "google.golang.org/protobuf/reflect/protodesc"
	pref "google.golang.org/protobuf/reflect/protoreflect"
)

func Test(t *testing.T) {
	fd := new(fdesc.Field)
	fd.L0.ParentFile = fdesc.SurrogateProto3
	fd.L0.FullName = "foo_field"
	fd.L1.Number = 1337
	fd.L1.Cardinality = pref.Repeated
	fd.L1.Kind = pref.BytesKind
	fd.L1.Default = fdesc.DefaultValue(pref.ValueOf([]byte("hello, \xde\xad\xbe\xef\n")), nil)

	// Marshal test.
	gotTag := tag.Marshal(fd, "")
	wantTag := `bytes,1337,rep,name=foo_field,json=fooField,proto3,def=hello, \336\255\276\357\n`
	if gotTag != wantTag {
		t.Errorf("Marshal() = `%v`, want `%v`", gotTag, wantTag)
	}

	// Unmarshal test.
	gotFD := tag.Unmarshal(wantTag, reflect.TypeOf([]byte{}), nil)
	wantFD := fd
	if !proto.Equal(pdesc.ToFieldDescriptorProto(gotFD), pdesc.ToFieldDescriptorProto(wantFD)) {
		t.Errorf("Umarshal() mismatch:\ngot  %v\nwant %v", gotFD, wantFD)
	}
}
