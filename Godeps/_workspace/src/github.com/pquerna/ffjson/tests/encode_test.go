// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tff

import (
	"bytes"
	"encoding/json"
	"testing"
)

func TestOmitEmpty(t *testing.T) {
	var o Optionals
	o.Sw = "something"
	o.Mr = map[string]interface{}{}
	o.Mo = map[string]interface{}{}

	got, err := json.MarshalIndent(&o, "", " ")
	if err != nil {
		t.Fatal(err)
	}
	if got := string(got); got != optionalsExpected {
		t.Errorf(" got: %s\nwant: %s\n", got, optionalsExpected)
	}
}

func TestOmitEmptyAll(t *testing.T) {
	var o OmitAll

	got, err := json.MarshalIndent(&o, "", " ")
	if err != nil {
		t.Fatal(err)
	}
	if got := string(got); got != omitAllExpected {
		t.Errorf(" got: %s\nwant: %s\n", got, omitAllExpected)
	}
}

func TestNoExported(t *testing.T) {
	var o NoExported

	got, err := json.MarshalIndent(&o, "", " ")
	if err != nil {
		t.Fatal(err)
	}
	if got := string(got); got != noExportedExpected {
		t.Errorf(" got: %s\nwant: %s\n", got, noExportedExpected)
	}
}

func TestOmitFirst(t *testing.T) {
	var o OmitFirst

	got, err := json.MarshalIndent(&o, "", " ")
	if err != nil {
		t.Fatal(err)
	}
	if got := string(got); got != omitFirstExpected {
		t.Errorf(" got: %s\nwant: %s\n", got, omitFirstExpected)
	}
}

func TestOmitLast(t *testing.T) {
	var o OmitLast

	got, err := json.MarshalIndent(&o, "", " ")
	if err != nil {
		t.Fatal(err)
	}
	if got := string(got); got != omitLastExpected {
		t.Errorf(" got: %s\nwant: %s\n", got, omitLastExpected)
	}
}

func TestStringTag(t *testing.T) {
	var s StringTag
	s.BoolStr = true
	s.IntStr = 42
	s.StrStr = "xzbit"
	got, err := json.MarshalIndent(&s, "", " ")
	if err != nil {
		t.Fatal(err)
	}
	if got := string(got); got != stringTagExpected {
		t.Fatalf(" got: %s\nwant: %s\n", got, stringTagExpected)
	}
}

func TestUnsupportedValues(t *testing.T) {
	for _, v := range unsupportedValues {
		if _, err := json.Marshal(v); err != nil {
			if _, ok := err.(*json.UnsupportedValueError); !ok {
				t.Errorf("for %v, got %T want UnsupportedValueError", v, err)
			}
		} else {
			t.Errorf("for %v, expected error", v)
		}
	}
}

func TestRefValMarshal(t *testing.T) {
	var s = struct {
		R0 Ref
		R1 *Ref
		R2 RefText
		R3 *RefText
		V0 Val
		V1 *Val
		V2 ValText
		V3 *ValText
	}{
		R0: 12,
		R1: new(Ref),
		R2: 14,
		R3: new(RefText),
		V0: 13,
		V1: new(Val),
		V2: 15,
		V3: new(ValText),
	}
	const want = `{"R0":"ref","R1":"ref","R2":"\"ref\"","R3":"\"ref\"","V0":"val","V1":"val","V2":"\"val\"","V3":"\"val\""}`
	b, err := json.Marshal(&s)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	if got := string(b); got != want {
		t.Errorf("got %q, want %q", got, want)
	}
}

func TestMarshalerEscaping(t *testing.T) {
	var c C
	want := `"\u003c\u0026\u003e"`
	b, err := json.Marshal(c)
	if err != nil {
		t.Fatalf("Marshal(c): %v", err)
	}
	if got := string(b); got != want {
		t.Errorf("Marshal(c) = %#q, want %#q", got, want)
	}

	var ct CText
	want = `"\"\u003c\u0026\u003e\""`
	b, err = json.Marshal(ct)
	if err != nil {
		t.Fatalf("Marshal(ct): %v", err)
	}
	if got := string(b); got != want {
		t.Errorf("Marshal(ct) = %#q, want %#q", got, want)
	}
}

func TestAnonymousNonstruct(t *testing.T) {
	var i IntType = 11
	a := MyStruct{i}
	const want = `{"IntType":11}`

	b, err := json.Marshal(a)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	if got := string(b); got != want {
		t.Errorf("got %q, want %q", got, want)
	}
}

// Issue 5245.
func TestEmbeddedBug(t *testing.T) {
	v := BugB{
		BugA{"A"},
		"B",
	}
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatal("Marshal:", err)
	}
	want := `{"S":"B"}`
	got := string(b)
	if got != want {
		t.Fatalf("Marshal: got %s want %s", got, want)
	}
	// Now check that the duplicate field, S, does not appear.
	x := BugX{
		A: 23,
	}
	b, err = json.Marshal(x)
	if err != nil {
		t.Fatal("Marshal:", err)
	}
	want = `{"A":23}`
	got = string(b)
	if got != want {
		t.Fatalf("Marshal: got %s want %s", got, want)
	}
}

// Test that a field with a tag dominates untagged fields.
func TestTaggedFieldDominates(t *testing.T) {
	v := BugY{
		BugA{"BugA"},
		BugD{"BugD"},
	}
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatal("Marshal:", err)
	}
	want := `{"S":"BugD"}`
	got := string(b)
	if got != want {
		t.Fatalf("Marshal: got %s want %s", got, want)
	}
}

func TestDuplicatedFieldDisappears(t *testing.T) {
	v := BugZ{
		BugA{"BugA"},
		BugC{"BugC"},
		BugY{
			BugA{"nested BugA"},
			BugD{"nested BugD"},
		},
	}
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatal("Marshal:", err)
	}
	want := `{}`
	got := string(b)
	if got != want {
		t.Fatalf("Marshal: got %s want %s", got, want)
	}
}

func TestIssue6458(t *testing.T) {
	type Foo struct {
		M json.RawMessage
	}
	x := Foo{json.RawMessage(`"foo"`)}

	b, err := json.Marshal(&x)
	if err != nil {
		t.Fatal(err)
	}
	if want := `{"M":"foo"}`; string(b) != want {
		t.Errorf("Marshal(&x) = %#q; want %#q", b, want)
	}

	b, err = json.Marshal(x)
	if err != nil {
		t.Fatal(err)
	}

	if want := `{"M":"ImZvbyI="}`; string(b) != want {
		t.Errorf("Marshal(x) = %#q; want %#q", b, want)
	}
}

func TestHTMLEscape(t *testing.T) {
	var b, want bytes.Buffer
	m := `{"M":"<html>foo &` + "\xe2\x80\xa8 \xe2\x80\xa9" + `</html>"}`
	want.Write([]byte(`{"M":"\u003chtml\u003efoo \u0026\u2028 \u2029\u003c/html\u003e"}`))
	json.HTMLEscape(&b, []byte(m))
	if !bytes.Equal(b.Bytes(), want.Bytes()) {
		t.Errorf("HTMLEscape(&b, []byte(m)) = %s; want %s", b.Bytes(), want.Bytes())
	}
}
