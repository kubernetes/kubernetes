// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Only test compatibility with the Marshal/Unmarshal functionality with
// pure protobuf reflection since there is no support for nullable fields
// in the table-driven implementation.
// +build protoreflect

package nullable

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/reflect/protoreflect"
	"google.golang.org/protobuf/testing/protocmp"
)

func init() {
	testMethods = func(t *testing.T, mt protoreflect.MessageType) {
		m1 := mt.New()
		populated := testPopulateMessage(t, m1, 2)
		b, err := proto.Marshal(m1.Interface())
		if err != nil {
			t.Errorf("proto.Marshal error: %v", err)
		}
		if populated && len(b) == 0 {
			t.Errorf("len(proto.Marshal) = 0, want >0")
		}
		m2 := mt.New()
		if err := proto.Unmarshal(b, m2.Interface()); err != nil {
			t.Errorf("proto.Unmarshal error: %v", err)
		}
		if diff := cmp.Diff(m1.Interface(), m2.Interface(), protocmp.Transform()); diff != "" {
			t.Errorf("message mismatch:\n%v", diff)
		}
		proto.Reset(m2.Interface())
		testEmptyMessage(t, m2, true)
		proto.Merge(m2.Interface(), m1.Interface())
		if diff := cmp.Diff(m1.Interface(), m2.Interface(), protocmp.Transform()); diff != "" {
			t.Errorf("message mismatch:\n%v", diff)
		}
	}
}
