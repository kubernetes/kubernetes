// Copyright 2013 Google Inc. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package googleapi

import (
	"encoding/json"
	"reflect"
	"testing"
)

func TestTypes(t *testing.T) {
	type T struct {
		I32 Int32s
		I64 Int64s
		U32 Uint32s
		U64 Uint64s
		F64 Float64s
	}
	v := &T{
		I32: Int32s{-1, 2, 3},
		I64: Int64s{-1, 2, 1 << 33},
		U32: Uint32s{1, 2},
		U64: Uint64s{1, 2, 1 << 33},
		F64: Float64s{1.5, 3.33},
	}
	got, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	want := `{"I32":["-1","2","3"],"I64":["-1","2","8589934592"],"U32":["1","2"],"U64":["1","2","8589934592"],"F64":["1.5","3.33"]}`
	if string(got) != want {
		t.Fatalf("Marshal mismatch.\n got: %s\nwant: %s\n", got, want)
	}

	v2 := new(T)
	if err := json.Unmarshal(got, v2); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if !reflect.DeepEqual(v, v2) {
		t.Fatalf("Unmarshal didn't produce same results.\n got: %#v\nwant: %#v\n", v, v2)
	}
}
