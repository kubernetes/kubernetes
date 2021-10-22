// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package dwarf_test

import (
	"testing"

	. "cloud.google.com/go/cmd/go-cloud-debug-agent/internal/debug/dwarf"
	"cloud.google.com/go/cmd/go-cloud-debug-agent/internal/debug/elf"
)

var typedefTests = map[string]string{
	"t_ptr_volatile_int":                    "*volatile int",
	"t_ptr_const_char":                      "*const char",
	"t_long":                                "long int",
	"t_ushort":                              "short unsigned int",
	"t_func_int_of_float_double":            "func(float, double) int",
	"t_ptr_func_int_of_float_double":        "*func(float, double) int",
	"t_ptr_func_int_of_float_complex":       "*func(complex float) int",
	"t_ptr_func_int_of_double_complex":      "*func(complex double) int",
	"t_ptr_func_int_of_long_double_complex": "*func(complex long double) int",
	"t_func_ptr_int_of_char_schar_uchar":    "func(char, signed char, unsigned char) *int",
	"t_func_void_of_char":                   "func(char) void",
	"t_func_void_of_void":                   "func() void",
	"t_func_void_of_ptr_char_dots":          "func(*char, ...) void",
	"t_my_struct":                           "struct my_struct {vi volatile int@0; x char@4 : 1@7; y int@4 : 4@27; z [0]int@8; array [40]long long int@8; zz [0]int@328}",
	"t_my_struct1":                          "struct my_struct1 {zz [1]int@0}",
	"t_my_union":                            "union my_union {vi volatile int@0; x char@0 : 1@7; y int@0 : 4@28; array [40]long long int@0}",
	"t_my_enum":                             "enum my_enum {e1=1; e2=2; e3=-5; e4=1000000000000000}",
	"t_my_list":                             "struct list {val short int@0; next *t_my_list@8}",
	"t_my_tree":                             "struct tree {left *struct tree@0; right *struct tree@8; val long long unsigned int@16}",
}

func elfData(t *testing.T, name string) *Data {
	f, err := elf.Open(name)
	if err != nil {
		t.Fatal(err)
	}

	d, err := f.DWARF()
	if err != nil {
		t.Fatal(err)
	}
	return d
}

func TestTypedefsELF(t *testing.T) { testTypedefs(t, elfData(t, "testdata/typedef.elf"), "elf") }

func TestTypedefsELFDwarf4(t *testing.T) { testTypedefs(t, elfData(t, "testdata/typedef.elf4"), "elf") }

func testTypedefs(t *testing.T, d *Data, kind string) {
	r := d.Reader()
	seen := make(map[string]bool)
	for {
		e, err := r.Next()
		if err != nil {
			t.Fatal("r.Next:", err)
		}
		if e == nil {
			break
		}
		if e.Tag == TagTypedef {
			typ, err := d.Type(e.Offset)
			if err != nil {
				t.Fatal("d.Type:", err)
			}
			t1 := typ.(*TypedefType)
			var typstr string
			if ts, ok := t1.Type.(*StructType); ok {
				typstr = ts.Defn()
			} else {
				typstr = t1.Type.String()
			}

			if want, ok := typedefTests[t1.Name]; ok {
				if seen[t1.Name] {
					t.Errorf("multiple definitions for %s", t1.Name)
				}
				seen[t1.Name] = true
				if typstr != want {
					t.Errorf("%s:\n\thave %s\n\twant %s", t1.Name, typstr, want)
				}
			}
		}
		if e.Tag != TagCompileUnit {
			r.SkipChildren()
		}
	}

	for k := range typedefTests {
		if !seen[k] {
			t.Errorf("missing %s", k)
		}
	}
}

func TestTypeForNonTypeEntry(t *testing.T) {
	d := elfData(t, "testdata/typedef.elf")

	// The returned entry will be a Subprogram.
	ent, err := d.LookupFunction("main")
	if err != nil {
		t.Fatal("d.LookupFunction:", err)
	}

	_, err = d.Type(ent.Offset)
	if err == nil {
		t.Fatal("nil error for unreadable entry")
	}
}
