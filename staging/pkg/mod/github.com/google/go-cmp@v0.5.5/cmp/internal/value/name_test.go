// Copyright 2020, The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package value

import (
	"reflect"
	"strings"
	"testing"
)

type Named struct{}

var pkgPath = reflect.TypeOf(Named{}).PkgPath()

func TestTypeString(t *testing.T) {
	tests := []struct {
		in   interface{}
		want string
	}{{
		in:   bool(false),
		want: "bool",
	}, {
		in:   int(0),
		want: "int",
	}, {
		in:   float64(0),
		want: "float64",
	}, {
		in:   string(""),
		want: "string",
	}, {
		in:   Named{},
		want: "$PackagePath.Named",
	}, {
		in:   (chan Named)(nil),
		want: "chan $PackagePath.Named",
	}, {
		in:   (<-chan Named)(nil),
		want: "<-chan $PackagePath.Named",
	}, {
		in:   (chan<- Named)(nil),
		want: "chan<- $PackagePath.Named",
	}, {
		in:   (func())(nil),
		want: "func()",
	}, {
		in:   (func(Named))(nil),
		want: "func($PackagePath.Named)",
	}, {
		in:   (func() Named)(nil),
		want: "func() $PackagePath.Named",
	}, {
		in:   (func(int, Named) (int, error))(nil),
		want: "func(int, $PackagePath.Named) (int, error)",
	}, {
		in:   (func(...Named))(nil),
		want: "func(...$PackagePath.Named)",
	}, {
		in:   struct{}{},
		want: "struct{}",
	}, {
		in:   struct{ Named }{},
		want: "struct{ $PackagePath.Named }",
	}, {
		in: struct {
			Named `tag`
		}{},
		want: "struct{ $PackagePath.Named \"tag\" }",
	}, {
		in:   struct{ Named Named }{},
		want: "struct{ Named $PackagePath.Named }",
	}, {
		in: struct {
			Named Named `tag`
		}{},
		want: "struct{ Named $PackagePath.Named \"tag\" }",
	}, {
		in: struct {
			Int   int
			Named Named
		}{},
		want: "struct{ Int int; Named $PackagePath.Named }",
	}, {
		in: struct {
			_ int
			x Named
		}{},
		want: "struct{ $FieldPrefix._ int; $FieldPrefix.x $PackagePath.Named }",
	}, {
		in:   []Named(nil),
		want: "[]$PackagePath.Named",
	}, {
		in:   []*Named(nil),
		want: "[]*$PackagePath.Named",
	}, {
		in:   [10]Named{},
		want: "[10]$PackagePath.Named",
	}, {
		in:   [10]*Named{},
		want: "[10]*$PackagePath.Named",
	}, {
		in:   map[string]string(nil),
		want: "map[string]string",
	}, {
		in:   map[Named]Named(nil),
		want: "map[$PackagePath.Named]$PackagePath.Named",
	}, {
		in:   (*Named)(nil),
		want: "*$PackagePath.Named",
	}, {
		in:   (*interface{})(nil),
		want: "*interface{}",
	}, {
		in:   (*interface{ Read([]byte) (int, error) })(nil),
		want: "*interface{ Read([]uint8) (int, error) }",
	}, {
		in: (*interface {
			F1()
			F2(Named)
			F3() Named
			F4(int, Named) (int, error)
			F5(...Named)
		})(nil),
		want: "*interface{ F1(); F2($PackagePath.Named); F3() $PackagePath.Named; F4(int, $PackagePath.Named) (int, error); F5(...$PackagePath.Named) }",
	}}

	for _, tt := range tests {
		typ := reflect.TypeOf(tt.in)
		wantShort := tt.want
		wantShort = strings.Replace(wantShort, "$PackagePath", "value", -1)
		wantShort = strings.Replace(wantShort, "$FieldPrefix.", "", -1)
		if gotShort := TypeString(typ, false); gotShort != wantShort {
			t.Errorf("TypeString(%v, false) mismatch:\ngot:  %v\nwant: %v", typ, gotShort, wantShort)
		}
		wantQualified := tt.want
		wantQualified = strings.Replace(wantQualified, "$PackagePath", `"`+pkgPath+`"`, -1)
		wantQualified = strings.Replace(wantQualified, "$FieldPrefix", `"`+pkgPath+`"`, -1)
		if gotQualified := TypeString(typ, true); gotQualified != wantQualified {
			t.Errorf("TypeString(%v, true) mismatch:\ngot:  %v\nwant: %v", typ, gotQualified, wantQualified)
		}
	}
}
