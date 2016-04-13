// Copyright 2014 Google Inc. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

package appengine

import (
	"strings"
	"testing"

	"github.com/golang/protobuf/proto"
	"golang.org/x/net/context"

	"google.golang.org/appengine/internal"
	"google.golang.org/appengine/internal/aetesting"
	basepb "google.golang.org/appengine/internal/base"
)

func TestNamespaceValidity(t *testing.T) {
	testCases := []struct {
		namespace string
		ok        bool
	}{
		// data from Python's namespace_manager_test.py
		{"", true},
		{"__a.namespace.123__", true},
		{"-_A....NAMESPACE-_", true},
		{"-", true},
		{".", true},
		{".-", true},

		{"?", false},
		{"+", false},
		{"!", false},
		{" ", false},
	}
	for _, tc := range testCases {
		_, err := Namespace(context.Background(), tc.namespace)
		if err == nil && !tc.ok {
			t.Errorf("Namespace %q should be rejected, but wasn't", tc.namespace)
		} else if err != nil && tc.ok {
			t.Errorf("Namespace %q should be accepted, but wasn't", tc.namespace)
		}
	}
}

func TestNamespaceApplication(t *testing.T) {
	internal.NamespaceMods["srv"] = func(m proto.Message, namespace string) {
		sm := m.(*basepb.StringProto)
		if strings.Contains(sm.GetValue(), "-") {
			// be idempotent
			return
		}
		sm.Value = proto.String(sm.GetValue() + "-" + namespace)
	}
	ctx := aetesting.FakeSingleContext(t, "srv", "mth", func(in, out *basepb.StringProto) error {
		out.Value = proto.String(in.GetValue())
		return nil
	})
	call := func(ctx context.Context, in string) (out string, ok bool) {
		inm := &basepb.StringProto{Value: &in}
		outm := &basepb.StringProto{}
		if err := internal.Call(ctx, "srv", "mth", inm, outm); err != nil {
			t.Errorf("RPC(in=%q) failed: %v", in, err)
			return "", false
		}
		return outm.GetValue(), true
	}

	// Check without a namespace.
	got, ok := call(ctx, "foo")
	if !ok {
		t.FailNow()
	}
	if got != "foo" {
		t.Errorf("Un-namespaced RPC returned %q, want %q", got, "foo")
	}

	// Now check by applying a namespace.
	nsCtx, err := Namespace(ctx, "myns")
	if err != nil {
		t.Fatal(err)
	}
	got, ok = call(nsCtx, "bar")
	if !ok {
		t.FailNow()
	}
	if got != "bar-myns" {
		t.Errorf("Namespaced RPC returned %q, want %q", got, "bar-myns")
	}
}
