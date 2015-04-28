// Copyright 2014 The ql Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ql

import (
	"bytes"
	"fmt"
	"math/big"
	//"reflect"
	"testing"
	"time"

	"github.com/cznic/mathutil"
)

type (
	testSchema struct {
		a  int8
		ID int64
		A  int8
		b  int
		B  int `ql:"-"`
	}

	testSchema2 struct{}

	testSchema3 struct {
		a  int8
		ID uint64
		A  int8
		b  int
		B  int `ql:"-"`
		c  bool
		C  bool `ql:"name cc"`
	}

	testSchema4 struct {
		a  int8
		ID int64 `ql:"name id"`
		A  int8
		b  int
		B  int `ql:"-"`
		c  bool
		C  bool `ql:"name cc"`
	}

	testSchema5 struct {
		I int `ql:"index x,uindex u"`
	}

	testSchema6 struct {
		A string `ql:"index x"`
	}

	testSchema7 struct {
		A int64
		B string `ql:"uindex x"`
		C bool
	}

	testSchema8 struct {
		A bool
		//B  int
		C int8
		D int16
		E int32
		F int64
		//G  uint
		H  uint8
		I  uint16
		J  uint32
		K  uint64
		L  float32
		M  float64
		N  complex64
		O  complex128
		P  []byte
		Q  big.Int
		R  big.Rat
		S  string
		T  time.Time
		U  time.Duration
		PA *bool
		//PB *int
		PC *int8
		PD *int16
		PE *int32
		PF *int64
		//PG *uint
		PH *uint8
		PI *uint16
		PJ *uint32
		PK *uint64
		PL *float32
		PM *float64
		PN *complex64
		PO *complex128
		PP *[]byte
		PQ *big.Int
		PR *big.Rat
		PS *string
		PT *time.Time
		PU *time.Duration
	}

	testSchema9 struct {
		i              int
		ID             int64  `ql:"index xID"`
		Other          string `ql:"-"`
		DepartmentName string `ql:"uindex xDepartmentName"`
	}
)

const (
	testSchemaSFFF = "begin transaction; create table if not exists testSchema (A int8); commit;"
	testSchemaSFFT = "begin transaction; create table if not exists ql_testSchema (A int8); commit;"
	testSchemaSFTF = "begin transaction; create table testSchema (A int8); commit;"
	testSchemaSFTT = "begin transaction; create table ql_testSchema (A int8); commit;"
	testSchemaSTFF = "create table if not exists testSchema (A int8)"
	testSchemaSTFT = "create table if not exists ql_testSchema (A int8)"
	testSchemaSTTF = "create table testSchema (A int8)"
	testSchemaSTTT = "create table ql_testSchema (A int8)"
	testSchema3S   = "begin transaction; create table if not exists testSchema3 (ID uint64, A int8, cc bool); commit;"
	testSchema4S   = "begin transaction; create table if not exists testSchema4 (id int64, A int8, cc bool); commit;"
	testSchema6S   = "create table testSchema6 (A string); create index x on testSchema6 (A);"
	testSchema7S   = "begin transaction; create table testSchema7 (A int64, B string, C bool); create unique index x on testSchema7 (B); commit;"
	testSchema8S   = `
		begin transaction;
			create table if not exists testSchema8 (
				A  bool,
				//B  int64,
				C  int8,
				D  int16,
				E  int32,
				F  int64,
				//G  uint64,
				H  uint8,
				I  uint16,
				J  uint32,
				K  uint64,
				L  float32,
				M  float64,
				N  complex64,
				O  complex128,
				P  blob,
				Q  bigInt,
				R  bigRat,
				S  string,
				T  time,
				U  duration,
				PA bool,
				//PB int64,
				PC int8,
				PD int16,
				PE int32,
				PF int64,
				//PG uint64,
				PH uint8,
				PI uint16,
				PJ uint32,
				PK uint64,
				PL float32,
				PM float64,
				PN complex64,
				PO complex128,
				PP blob,
				PQ bigInt,
				PR bigRat,
				PS string,
				PT time,
				PU  duration,
			);
		commit;`
	testSchema9S = `
		begin transaction;
			create table if not exists testSchema9 (DepartmentName string);
			create index if not exists xID on testSchema9 (id());
			create unique index if not exists xDepartmentName on testSchema9 (DepartmentName);
		commit;`
)

func TestSchema(t *testing.T) {
	tab := []struct {
		inst interface{}
		name string
		opts *SchemaOptions
		err  bool
		s    string
	}{
		// 0
		{inst: nil, err: true},
		{inst: interface{}(nil), err: true},
		{testSchema{}, "", nil, false, testSchemaSFFF},
		{testSchema{}, "", &SchemaOptions{}, false, testSchemaSFFF},
		{testSchema{}, "", &SchemaOptions{KeepPrefix: true}, false, testSchemaSFFT},
		// 5
		{testSchema{}, "", &SchemaOptions{NoIfNotExists: true}, false, testSchemaSFTF},
		{testSchema{}, "", &SchemaOptions{NoIfNotExists: true, KeepPrefix: true}, false, testSchemaSFTT},
		{testSchema{}, "", &SchemaOptions{NoTransaction: true}, false, testSchemaSTFF},
		{testSchema{}, "", &SchemaOptions{NoTransaction: true, KeepPrefix: true}, false, testSchemaSTFT},
		{testSchema{}, "", &SchemaOptions{NoTransaction: true, NoIfNotExists: true}, false, testSchemaSTTF},
		// 10
		{testSchema{}, "", &SchemaOptions{NoTransaction: true, NoIfNotExists: true, KeepPrefix: true}, false, testSchemaSTTT},
		{testSchema2{}, "", nil, true, ""},
		{testSchema3{}, "", nil, false, testSchema3S},
		{testSchema4{}, "", nil, false, testSchema4S},
		{testSchema5{}, "", nil, true, ""},
		// 15
		{testSchema6{}, "", &SchemaOptions{NoTransaction: true, NoIfNotExists: true}, false, testSchema6S},
		{testSchema7{}, "", &SchemaOptions{NoIfNotExists: true}, false, testSchema7S},
		{testSchema8{}, "", nil, false, testSchema8S},
		{&testSchema8{}, "", nil, false, testSchema8S},
		{&testSchema9{}, "", nil, false, testSchema9S},
	}

	for iTest, test := range tab {
		l, err := Schema(test.inst, test.name, test.opts)
		if g, e := err != nil, test.err; g != e {
			t.Fatal(iTest, g, e, err)
		}

		if err != nil {
			t.Log(iTest, err)
			continue
		}

		s, err := Compile(test.s)
		if err != nil {
			panic("internal error 070")
		}

		if g, e := l.String(), s.String(); g != e {
			t.Fatalf("%d\n----\n%s\n----\n%s", iTest, g, e)
		}
	}
}

func ExampleSchema() {
	type department struct {
		a              int    // unexported -> ignored
		ID             int64  `ql:"index xID"`
		Other          string `xml:"-" ql:"-"` // ignored by QL tag
		DepartmentName string `ql:"name Name, uindex xName" json:"foo"`
		m              bool
		HQ             int32
		z              string
	}

	schema := MustSchema((*department)(nil), "", nil)
	sel := MustCompile(`
		SELECT * FROM __Table;
		SELECT * FROM __Column;
		SELECT * FROM __Index;`,
	)
	fmt.Print(schema)

	db, err := OpenMem()
	if err != nil {
		panic(err)
	}

	if _, _, err = db.Execute(NewRWCtx(), schema); err != nil {
		panic(err)
	}

	rs, _, err := db.Execute(nil, sel)
	if err != nil {
		panic(err)
	}

	for _, rs := range rs {
		fmt.Println("----")
		if err = rs.Do(true, func(data []interface{}) (bool, error) {
			fmt.Println(data)
			return true, nil
		}); err != nil {
			panic(err)
		}
	}
	// Output:
	// BEGIN TRANSACTION;
	// 	CREATE TABLE IF NOT EXISTS department (Name string, HQ int32);
	// 	CREATE INDEX IF NOT EXISTS xID ON department (id());
	// 	CREATE UNIQUE INDEX IF NOT EXISTS xName ON department (Name);
	// COMMIT;
	// ----
	// [Name Schema]
	// [department CREATE TABLE department (Name string, HQ int32);]
	// ----
	// [TableName Ordinal Name Type]
	// [department 1 Name string]
	// [department 2 HQ int32]
	// ----
	// [TableName ColumnName Name IsUnique]
	// [department id() xID false]
	// [department Name xName true]
}

func TestMarshal(t *testing.T) {
	now := time.Now()
	dur := time.Millisecond
	schema8 := testSchema8{
		A: true,
		//B: 1,
		C: 2,
		D: 3,
		E: 4,
		F: 5,
		//G: 6,
		H: 7,
		I: 8,
		J: 9,
		K: 10,
		L: 11,
		M: 12,
		N: -1,
		O: -2,
		P: []byte("abc"),
		Q: *big.NewInt(1),
		R: *big.NewRat(3, 2),
		S: "string",
		T: now,
		U: dur,
	}
	schema8.PA = &schema8.A
	//schema8.PB = &schema8.B
	schema8.PC = &schema8.C
	schema8.PD = &schema8.D
	schema8.PE = &schema8.E
	schema8.PF = &schema8.F
	//schema8.PG = &schema8.G
	schema8.PH = &schema8.H
	schema8.PI = &schema8.I
	schema8.PJ = &schema8.J
	schema8.PK = &schema8.K
	schema8.PL = &schema8.L
	schema8.PM = &schema8.M
	schema8.PN = &schema8.N
	schema8.PO = &schema8.O
	schema8.PP = &schema8.P
	schema8.PQ = &schema8.Q
	schema8.PR = &schema8.R
	schema8.PS = &schema8.S
	schema8.PT = &schema8.T
	schema8.PU = &schema8.U

	type u int
	tab := []struct {
		inst interface{}
		err  bool
		r    []interface{}
	}{
		{42, true, nil},
		{new(u), true, nil},
		{testSchema8{}, false, []interface{}{
			false,
			//int64(0),
			int8(0),
			int16(0),
			int32(0),
			int64(0),
			//uint64(0),
			uint8(0),
			uint16(0),
			uint32(0),
			uint64(0),
			float32(0),
			float64(0),
			complex64(0),
			complex128(0),
			[]byte(nil),
			big.Int{},
			big.Rat{},
			"",
			time.Time{},
			time.Duration(0),
			nil,
			//nil,
			nil,
			nil,
			nil,
			nil,
			//nil,
			nil,
			nil,
			nil,
			nil,
			nil,
			nil,
			nil,
			nil,
			nil,
			nil,
			nil,
			nil,
			nil,
			nil,
		}},
		{&testSchema8{}, false, []interface{}{
			false,
			//int64(0),
			int8(0),
			int16(0),
			int32(0),
			int64(0),
			//uint64(0),
			uint8(0),
			uint16(0),
			uint32(0),
			uint64(0),
			float32(0),
			float64(0),
			complex64(0),
			complex128(0),
			[]byte(nil),
			big.Int{},
			big.Rat{},
			"",
			time.Time{},
			time.Duration(0),
			nil,
			//nil,
			nil,
			nil,
			nil,
			nil,
			//nil,
			nil,
			nil,
			nil,
			nil,
			nil,
			nil,
			nil,
			nil,
			nil,
			nil,
			nil,
			nil,
			nil,
			nil,
		}},
		{schema8, false, []interface{}{
			true,
			//int64(1),
			int8(2),
			int16(3),
			int32(4),
			int64(5),
			//uint64(6),
			uint8(7),
			uint16(8),
			uint32(9),
			uint64(10),
			float32(11),
			float64(12),
			complex64(-1),
			complex128(-2),
			[]byte("abc"),
			*big.NewInt(1),
			*big.NewRat(3, 2),
			"string",
			now,
			dur,
			true,
			//int64(1),
			int8(2),
			int16(3),
			int32(4),
			int64(5),
			//uint64(6),
			uint8(7),
			uint16(8),
			uint32(9),
			uint64(10),
			float32(11),
			float64(12),
			complex64(-1),
			complex128(-2),
			[]byte("abc"),
			*big.NewInt(1),
			*big.NewRat(3, 2),
			"string",
			now,
			dur,
		}},
		{&schema8, false, []interface{}{
			true,
			//int64(1),
			int8(2),
			int16(3),
			int32(4),
			int64(5),
			//uint64(6),
			uint8(7),
			uint16(8),
			uint32(9),
			uint64(10),
			float32(11),
			float64(12),
			complex64(-1),
			complex128(-2),
			[]byte("abc"),
			*big.NewInt(1),
			*big.NewRat(3, 2),
			"string",
			now,
			dur,
			true,
			//int64(1),
			int8(2),
			int16(3),
			int32(4),
			int64(5),
			//uint64(6),
			uint8(7),
			uint16(8),
			uint32(9),
			uint64(10),
			float32(11),
			float64(12),
			complex64(-1),
			complex128(-2),
			[]byte("abc"),
			*big.NewInt(1),
			*big.NewRat(3, 2),
			"string",
			now,
			dur,
		}},
	}
	for iTest, test := range tab {
		r, err := Marshal(test.inst)
		if g, e := err != nil, test.err; g != e {
			t.Fatal(iTest, g, e)
		}

		if err != nil {
			t.Log(err)
			continue
		}

		for i := 0; i < mathutil.Min(len(r), len(test.r)); i++ {
			g, e := r[i], test.r[i]
			use(e)
			switch x := g.(type) {
			case bool:
				switch y := e.(type) {
				case bool:
					if x != y {
						t.Fatal(iTest, x, y)
					}
				default:
					t.Fatalf("%d: %T <-> %T", iTest, x, y)
				}
			case int:
				switch y := e.(type) {
				case int64:
					if int64(x) != y {
						t.Fatal(iTest, x, y)
					}
				default:
					t.Fatalf("%d: %T <-> %T", iTest, x, y)
				}
			case int8:
				switch y := e.(type) {
				case int8:
					if x != y {
						t.Fatal(iTest, x, y)
					}
				default:
					t.Fatalf("%d: %T <-> %T", iTest, x, y)
				}
			case int16:
				switch y := e.(type) {
				case int16:
					if x != y {
						t.Fatal(iTest, x, y)
					}
				default:
					t.Fatalf("%d: %T <-> %T", iTest, x, y)
				}
			case int32:
				switch y := e.(type) {
				case int32:
					if x != y {
						t.Fatal(iTest, x, y)
					}
				default:
					t.Fatalf("%d: %T <-> %T", iTest, x, y)
				}
			case int64:
				switch y := e.(type) {
				case int64:
					if x != y {
						t.Fatal(iTest, x, y)
					}
				default:
					t.Fatalf("%d: %T <-> %T", iTest, x, y)
				}
			case uint:
				switch y := e.(type) {
				case uint64:
					if uint64(x) != y {
						t.Fatal(iTest, x, y)
					}
				default:
					t.Fatalf("%d: %T <-> %T", iTest, x, y)
				}
			case uint8:
				switch y := e.(type) {
				case uint8:
					if x != y {
						t.Fatal(iTest, x, y)
					}
				default:
					t.Fatalf("%d: %T <-> %T", iTest, x, y)
				}
			case uint16:
				switch y := e.(type) {
				case uint16:
					if x != y {
						t.Fatal(iTest, x, y)
					}
				default:
					t.Fatalf("%d: %T <-> %T", iTest, x, y)
				}
			case uint32:
				switch y := e.(type) {
				case uint32:
					if x != y {
						t.Fatal(iTest, x, y)
					}
				default:
					t.Fatalf("%d: %T <-> %T", iTest, x, y)
				}
			case uint64:
				switch y := e.(type) {
				case uint64:
					if x != y {
						t.Fatal(iTest, x, y)
					}
				default:
					t.Fatalf("%d: %T <-> %T", iTest, x, y)
				}
			case float32:
				switch y := e.(type) {
				case float32:
					if x != y {
						t.Fatal(iTest, x, y)
					}
				default:
					t.Fatalf("%d: %T <-> %T", iTest, x, y)
				}
			case float64:
				switch y := e.(type) {
				case float64:
					if x != y {
						t.Fatal(iTest, x, y)
					}
				default:
					t.Fatalf("%d: %T <-> %T", iTest, x, y)
				}
			case complex64:
				switch y := e.(type) {
				case complex64:
					if x != y {
						t.Fatal(iTest, x, y)
					}
				default:
					t.Fatalf("%d: %T <-> %T", iTest, x, y)
				}
			case complex128:
				switch y := e.(type) {
				case complex128:
					if x != y {
						t.Fatal(iTest, x, y)
					}
				default:
					t.Fatalf("%d: %T <-> %T", iTest, x, y)
				}
			case []byte:
				switch y := e.(type) {
				case []byte:
					if bytes.Compare(x, y) != 0 {
						t.Fatal(iTest, x, y)
					}
				default:
					t.Fatalf("%d: %T <-> %T", iTest, x, y)
				}
			case big.Int:
				switch y := e.(type) {
				case big.Int:
					if x.Cmp(&y) != 0 {
						t.Fatal(iTest, &x, &y)
					}
				default:
					t.Fatalf("%d: %T <-> %T", iTest, x, y)
				}
			case big.Rat:
				switch y := e.(type) {
				case big.Rat:
					if x.Cmp(&y) != 0 {
						t.Fatal(iTest, &x, &y)
					}
				default:
					t.Fatalf("%d: %T <-> %T", iTest, x, y)
				}
			case string:
				switch y := e.(type) {
				case string:
					if x != y {
						t.Fatal(iTest, x, y)
					}
				default:
					t.Fatalf("%d: %T <-> %T", iTest, x, y)
				}
			case time.Time:
				switch y := e.(type) {
				case time.Time:
					if !x.Equal(y) {
						t.Fatal(iTest, x, y)
					}
				default:
					t.Fatalf("%d: %T <-> %T", iTest, x, y)
				}
			case time.Duration:
				switch y := e.(type) {
				case time.Duration:
					if x != y {
						t.Fatal(iTest, x, y)
					}
				default:
					t.Fatalf("%d: %T <-> %T", iTest, x, y)
				}
			case nil:
				switch y := e.(type) {
				case nil:
					// ok
				default:
					t.Fatalf("%d: %T <-> %T", iTest, x, y)
				}
			default:
				panic(fmt.Errorf("%T", x))
			}
		}

		if g, e := len(r), len(test.r); g != e {
			t.Fatal(iTest, g, e)
		}

	}
}

func ExampleMarshal() {
	type myInt int16

	type myString string

	type item struct {
		ID   int64
		Name myString
		Qty  *myInt // pointer enables nil values
		Bar  int8
	}

	schema := MustSchema((*item)(nil), "", nil)
	ins := MustCompile(`
		BEGIN TRANSACTION;
			INSERT INTO item VALUES($1, $2, $3);
		COMMIT;`,
	)

	db, err := OpenMem()
	if err != nil {
		panic(err)
	}

	ctx := NewRWCtx()
	if _, _, err := db.Execute(ctx, schema); err != nil {
		panic(err)
	}

	if _, _, err := db.Execute(ctx, ins, MustMarshal(&item{Name: "foo", Bar: -1})...); err != nil {
		panic(err)
	}

	q := myInt(42)
	if _, _, err := db.Execute(ctx, ins, MustMarshal(&item{Name: "bar", Qty: &q})...); err != nil {
		panic(err)
	}

	rs, _, err := db.Run(nil, "SELECT * FROM item ORDER BY id();")
	if err != nil {
		panic(err)
	}

	if err = rs[0].Do(true, func(data []interface{}) (bool, error) {
		fmt.Println(data)
		return true, nil
	}); err != nil {
		panic(err)
	}
	// Output:
	// [Name Qty Bar]
	// [foo <nil> -1]
	// [bar 42 0]
}

func TestUnmarshal0(t *testing.T) {
	type t1 struct {
		I, J int64
	}

	// ---- value field
	v1 := &t1{-1, -2}
	if err := Unmarshal(v1, []interface{}{int64(42), int64(314)}); err != nil {
		t.Fatal(err)
	}

	if g, e := v1.I, int64(42); g != e {
		t.Fatal(g, e)
	}

	if g, e := v1.J, int64(314); g != e {
		t.Fatal(g, e)
	}

	type t2 struct {
		P *int64
	}

	// ---- nil into nil ptr field
	v2 := &t2{P: nil}
	if err := Unmarshal(v2, []interface{}{nil}); err != nil {
		t.Fatal(err)
	}

	if g, e := v2.P, (*int64)(nil); g != e {
		t.Fatal(g, e)
	}

	v2 = &t2{P: nil}
	if err := Unmarshal(v2, []interface{}{interface{}(nil)}); err != nil {
		t.Fatal(err)
	}

	if g, e := v2.P, (*int64)(nil); g != e {
		t.Fatal(g, e)
	}

	// ---- nil into non nil ptr field
	i := int64(42)
	v2 = &t2{P: &i}
	if err := Unmarshal(v2, []interface{}{nil}); err != nil {
		t.Fatal(err)
	}

	if g, e := v2.P, (*int64)(nil); g != e {
		t.Fatal(g, e)
	}

	if g, e := i, int64(42); g != e {
		t.Fatal(g, e)
	}

	v2 = &t2{P: &i}
	if err := Unmarshal(v2, []interface{}{interface{}(nil)}); err != nil {
		t.Fatal(err)
	}

	if g, e := v2.P, (*int64)(nil); g != e {
		t.Fatal(g, e)
	}

	if g, e := i, int64(42); g != e {
		t.Fatal(g, e)
	}

	// ---- non nil value into non nil ptr field
	i = 42
	v2 = &t2{P: &i}
	if err := Unmarshal(v2, []interface{}{int64(314)}); err != nil {
		t.Fatal(err)
	}

	if g, e := v2.P, &i; g != e {
		t.Fatal(g, e)
	}

	if g, e := i, int64(314); g != e {
		t.Fatal(g, e)
	}

	// ---- non nil value into nil ptr field
	v2 = &t2{P: nil}
	if err := Unmarshal(v2, []interface{}{int64(314)}); err != nil {
		t.Fatal(err)
	}

	if g, e := v2.P != nil, true; g != e {
		t.Fatal(g, e)
	}

	if g, e := *v2.P, int64(314); g != e {
		t.Fatal(g, e)
	}
}

func TestUnmarshal(t *testing.T) {
	type myString string

	type t1 struct {
		A bool
		B myString
	}

	type t2 struct {
		A  bool
		ID int64
		B  myString
	}

	f := func(v interface{}) int64 {
		if x, ok := v.(*t2); ok {
			return x.ID
		}

		return -1
	}

	tab := []struct {
		inst interface{}
		data []interface{}
		err  bool
	}{
		// 0
		{t1{}, []interface{}{true, "foo"}, true},      // not a ptr
		{&t1{}, []interface{}{true}, true},            // too few values
		{&t1{}, []interface{}{"foo"}, true},           // too few values
		{&t1{}, []interface{}{true, "foo", 42}, true}, // too many values
		{&t1{}, []interface{}{"foo", true, 42}, true}, // too many values
		// 5
		{&t1{}, []interface{}{true, "foo"}, false},
		{&t1{}, []interface{}{false, "bar"}, false},
		{&t1{}, []interface{}{"bar", "baz"}, true},
		{&t1{}, []interface{}{true, 42.7}, true},
		{&t2{}, []interface{}{1}, true}, // too few values
		// 10
		{&t2{}, []interface{}{1, 2, 3, 4}, true}, // too many values
		{&t2{}, []interface{}{false, int64(314), "foo"}, false},
		{&t2{}, []interface{}{true, int64(42), "foo"}, false},
		{&t2{}, []interface{}{false, "foo"}, false},
		// 15
		{&t2{}, []interface{}{true, "foo"}, false},
	}

	for iTest, test := range tab {
		inst := test.inst
		err := Unmarshal(inst, test.data)
		if g, e := err != nil, test.err; g != e {
			t.Fatal(iTest, g, e)
		}

		if err != nil {
			t.Log(iTest, err)
			continue
		}

		data, err := Marshal(inst)
		if err != nil {
			t.Fatal(iTest, err)
		}

		if g, e := len(data), len(test.data); g > e {
			t.Fatal(iTest, g, e)
		}

		j := 0
		for _, v := range data {
			v2 := test.data[j]
			j++
			if _, ok := v2.(int64); ok {
				if g, e := f(inst), v2; g != e {
					t.Fatal(iTest, g, e)
				}

				continue
			}

			if g, e := v, v2; g != e {
				t.Fatal(iTest, g, e)
			}
		}
	}
}

func ExampleUnmarshal() {
	type myString string

	type row struct {
		ID int64
		S  myString
		P  *int64
	}

	schema := MustSchema((*row)(nil), "", nil)
	ins := MustCompile(`
		BEGIN TRANSACTION;
			INSERT INTO row VALUES($1, $2);
		COMMIT;`,
	)
	sel := MustCompile(`
		SELECT id(), S, P FROM row ORDER by id();
		SELECT * FROM row ORDER by id();`,
	)

	db, err := OpenMem()
	if err != nil {
		panic(err)
	}

	ctx := NewRWCtx()
	if _, _, err = db.Execute(ctx, schema); err != nil {
		panic(err)
	}

	r := &row{S: "foo"}
	if _, _, err = db.Execute(ctx, ins, MustMarshal(r)...); err != nil {
		panic(err)
	}

	i42 := int64(42)
	r = &row{S: "bar", P: &i42}
	if _, _, err = db.Execute(ctx, ins, MustMarshal(r)...); err != nil {
		panic(err)
	}

	rs, _, err := db.Execute(nil, sel)
	if err != nil {
		panic(err)
	}

	for _, rs := range rs {
		fmt.Println("----")
		if err := rs.Do(false, func(data []interface{}) (bool, error) {
			r := &row{}
			if err := Unmarshal(r, data); err != nil {
				return false, err
			}

			fmt.Printf("ID %d, S %q, P ", r.ID, r.S)
			switch r.P == nil {
			case true:
				fmt.Println("<nil>")
			default:
				fmt.Println(*r.P)
			}
			return true, nil
		}); err != nil {
			panic(err)
		}
	}
	// Output:
	// ----
	// ID 1, S "foo", P <nil>
	// ID 2, S "bar", P 42
	// ----
	// ID 0, S "foo", P <nil>
	// ID 0, S "bar", P 42
}
