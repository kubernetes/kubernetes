// Copyright 2012 Gary Burd
//
// Licensed under the Apache License, Version 2.0 (the "License"): you may
// not use this file except in compliance with the License. You may obtain
// a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations
// under the License.

package redis_test

import (
	"fmt"
	"github.com/garyburd/redigo/redis"
	"math"
	"reflect"
	"testing"
)

var scanConversionTests = []struct {
	src  interface{}
	dest interface{}
}{
	{[]byte("-inf"), math.Inf(-1)},
	{[]byte("+inf"), math.Inf(1)},
	{[]byte("0"), float64(0)},
	{[]byte("3.14159"), float64(3.14159)},
	{[]byte("3.14"), float32(3.14)},
	{[]byte("-100"), int(-100)},
	{[]byte("101"), int(101)},
	{int64(102), int(102)},
	{[]byte("103"), uint(103)},
	{int64(104), uint(104)},
	{[]byte("105"), int8(105)},
	{int64(106), int8(106)},
	{[]byte("107"), uint8(107)},
	{int64(108), uint8(108)},
	{[]byte("0"), false},
	{int64(0), false},
	{[]byte("f"), false},
	{[]byte("1"), true},
	{int64(1), true},
	{[]byte("t"), true},
	{[]byte("hello"), "hello"},
	{[]byte("world"), []byte("world")},
	{[]interface{}{[]byte("foo")}, []interface{}{[]byte("foo")}},
	{[]interface{}{[]byte("foo")}, []string{"foo"}},
	{[]interface{}{[]byte("hello"), []byte("world")}, []string{"hello", "world"}},
	{[]interface{}{[]byte("bar")}, [][]byte{[]byte("bar")}},
	{[]interface{}{[]byte("1")}, []int{1}},
	{[]interface{}{[]byte("1"), []byte("2")}, []int{1, 2}},
	{[]interface{}{[]byte("1"), []byte("2")}, []float64{1, 2}},
	{[]interface{}{[]byte("1")}, []byte{1}},
	{[]interface{}{[]byte("1")}, []bool{true}},
}

func TestScanConversion(t *testing.T) {
	for _, tt := range scanConversionTests {
		values := []interface{}{tt.src}
		dest := reflect.New(reflect.TypeOf(tt.dest))
		values, err := redis.Scan(values, dest.Interface())
		if err != nil {
			t.Errorf("Scan(%v) returned error %v", tt, err)
			continue
		}
		if !reflect.DeepEqual(tt.dest, dest.Elem().Interface()) {
			t.Errorf("Scan(%v) returned %v, want %v", tt, dest.Elem().Interface(), tt.dest)
		}
	}
}

var scanConversionErrorTests = []struct {
	src  interface{}
	dest interface{}
}{
	{[]byte("1234"), byte(0)},
	{int64(1234), byte(0)},
	{[]byte("-1"), byte(0)},
	{int64(-1), byte(0)},
	{[]byte("junk"), false},
	{redis.Error("blah"), false},
}

func TestScanConversionError(t *testing.T) {
	for _, tt := range scanConversionErrorTests {
		values := []interface{}{tt.src}
		dest := reflect.New(reflect.TypeOf(tt.dest))
		values, err := redis.Scan(values, dest.Interface())
		if err == nil {
			t.Errorf("Scan(%v) did not return error", tt)
		}
	}
}

func ExampleScan() {
	c, err := dial()
	if err != nil {
		panic(err)
	}
	defer c.Close()

	c.Send("HMSET", "album:1", "title", "Red", "rating", 5)
	c.Send("HMSET", "album:2", "title", "Earthbound", "rating", 1)
	c.Send("HMSET", "album:3", "title", "Beat")
	c.Send("LPUSH", "albums", "1")
	c.Send("LPUSH", "albums", "2")
	c.Send("LPUSH", "albums", "3")
	values, err := redis.Values(c.Do("SORT", "albums",
		"BY", "album:*->rating",
		"GET", "album:*->title",
		"GET", "album:*->rating"))
	if err != nil {
		panic(err)
	}

	for len(values) > 0 {
		var title string
		rating := -1 // initialize to illegal value to detect nil.
		values, err = redis.Scan(values, &title, &rating)
		if err != nil {
			panic(err)
		}
		if rating == -1 {
			fmt.Println(title, "not-rated")
		} else {
			fmt.Println(title, rating)
		}
	}
	// Output:
	// Beat not-rated
	// Earthbound 1
	// Red 5
}

type s0 struct {
	X  int
	Y  int `redis:"y"`
	Bt bool
}

type s1 struct {
	X  int    `redis:"-"`
	I  int    `redis:"i"`
	U  uint   `redis:"u"`
	S  string `redis:"s"`
	P  []byte `redis:"p"`
	B  bool   `redis:"b"`
	Bt bool
	Bf bool
	s0
}

var scanStructTests = []struct {
	title string
	reply []string
	value interface{}
}{
	{"basic",
		[]string{"i", "-1234", "u", "5678", "s", "hello", "p", "world", "b", "t", "Bt", "1", "Bf", "0", "X", "123", "y", "456"},
		&s1{I: -1234, U: 5678, S: "hello", P: []byte("world"), B: true, Bt: true, Bf: false, s0: s0{X: 123, Y: 456}},
	},
}

func TestScanStruct(t *testing.T) {
	for _, tt := range scanStructTests {

		var reply []interface{}
		for _, v := range tt.reply {
			reply = append(reply, []byte(v))
		}

		value := reflect.New(reflect.ValueOf(tt.value).Type().Elem())

		if err := redis.ScanStruct(reply, value.Interface()); err != nil {
			t.Fatalf("ScanStruct(%s) returned error %v", tt.title, err)
		}

		if !reflect.DeepEqual(value.Interface(), tt.value) {
			t.Fatalf("ScanStruct(%s) returned %v, want %v", tt.title, value.Interface(), tt.value)
		}
	}
}

func TestBadScanStructArgs(t *testing.T) {
	x := []interface{}{"A", "b"}
	test := func(v interface{}) {
		if err := redis.ScanStruct(x, v); err == nil {
			t.Errorf("Expect error for ScanStruct(%T, %T)", x, v)
		}
	}

	test(nil)

	var v0 *struct{}
	test(v0)

	var v1 int
	test(&v1)

	x = x[:1]
	v2 := struct{ A string }{}
	test(&v2)
}

var scanSliceTests = []struct {
	src        []interface{}
	fieldNames []string
	ok         bool
	dest       interface{}
}{
	{
		[]interface{}{[]byte("1"), nil, []byte("-1")},
		nil,
		true,
		[]int{1, 0, -1},
	},
	{
		[]interface{}{[]byte("1"), nil, []byte("2")},
		nil,
		true,
		[]uint{1, 0, 2},
	},
	{
		[]interface{}{[]byte("-1")},
		nil,
		false,
		[]uint{1},
	},
	{
		[]interface{}{[]byte("hello"), nil, []byte("world")},
		nil,
		true,
		[][]byte{[]byte("hello"), nil, []byte("world")},
	},
	{
		[]interface{}{[]byte("hello"), nil, []byte("world")},
		nil,
		true,
		[]string{"hello", "", "world"},
	},
	{
		[]interface{}{[]byte("a1"), []byte("b1"), []byte("a2"), []byte("b2")},
		nil,
		true,
		[]struct{ A, B string }{{"a1", "b1"}, {"a2", "b2"}},
	},
	{
		[]interface{}{[]byte("a1"), []byte("b1")},
		nil,
		false,
		[]struct{ A, B, C string }{{"a1", "b1", ""}},
	},
	{
		[]interface{}{[]byte("a1"), []byte("b1"), []byte("a2"), []byte("b2")},
		nil,
		true,
		[]*struct{ A, B string }{{"a1", "b1"}, {"a2", "b2"}},
	},
	{
		[]interface{}{[]byte("a1"), []byte("b1"), []byte("a2"), []byte("b2")},
		[]string{"A", "B"},
		true,
		[]struct{ A, C, B string }{{"a1", "", "b1"}, {"a2", "", "b2"}},
	},
	{
		[]interface{}{[]byte("a1"), []byte("b1"), []byte("a2"), []byte("b2")},
		nil,
		false,
		[]struct{}{},
	},
}

func TestScanSlice(t *testing.T) {
	for _, tt := range scanSliceTests {

		typ := reflect.ValueOf(tt.dest).Type()
		dest := reflect.New(typ)

		err := redis.ScanSlice(tt.src, dest.Interface(), tt.fieldNames...)
		if tt.ok != (err == nil) {
			t.Errorf("ScanSlice(%v, []%s, %v) returned error %v", tt.src, typ, tt.fieldNames, err)
			continue
		}
		if tt.ok && !reflect.DeepEqual(dest.Elem().Interface(), tt.dest) {
			t.Errorf("ScanSlice(src, []%s) returned %#v, want %#v", typ, dest.Elem().Interface(), tt.dest)
		}
	}
}

func ExampleScanSlice() {
	c, err := dial()
	if err != nil {
		panic(err)
	}
	defer c.Close()

	c.Send("HMSET", "album:1", "title", "Red", "rating", 5)
	c.Send("HMSET", "album:2", "title", "Earthbound", "rating", 1)
	c.Send("HMSET", "album:3", "title", "Beat", "rating", 4)
	c.Send("LPUSH", "albums", "1")
	c.Send("LPUSH", "albums", "2")
	c.Send("LPUSH", "albums", "3")
	values, err := redis.Values(c.Do("SORT", "albums",
		"BY", "album:*->rating",
		"GET", "album:*->title",
		"GET", "album:*->rating"))
	if err != nil {
		panic(err)
	}

	var albums []struct {
		Title  string
		Rating int
	}
	if err := redis.ScanSlice(values, &albums); err != nil {
		panic(err)
	}
	fmt.Printf("%v\n", albums)
	// Output:
	// [{Earthbound 1} {Beat 4} {Red 5}]
}

var argsTests = []struct {
	title    string
	actual   redis.Args
	expected redis.Args
}{
	{"struct ptr",
		redis.Args{}.AddFlat(&struct {
			I  int    `redis:"i"`
			U  uint   `redis:"u"`
			S  string `redis:"s"`
			P  []byte `redis:"p"`
			Bt bool
			Bf bool
		}{
			-1234, 5678, "hello", []byte("world"), true, false,
		}),
		redis.Args{"i", int(-1234), "u", uint(5678), "s", "hello", "p", []byte("world"), "Bt", true, "Bf", false},
	},
	{"struct",
		redis.Args{}.AddFlat(struct{ I int }{123}),
		redis.Args{"I", 123},
	},
	{"slice",
		redis.Args{}.Add(1).AddFlat([]string{"a", "b", "c"}).Add(2),
		redis.Args{1, "a", "b", "c", 2},
	},
}

func TestArgs(t *testing.T) {
	for _, tt := range argsTests {
		if !reflect.DeepEqual(tt.actual, tt.expected) {
			t.Fatalf("%s is %v, want %v", tt.title, tt.actual, tt.expected)
		}
	}
}

func ExampleArgs() {
	c, err := dial()
	if err != nil {
		panic(err)
	}
	defer c.Close()

	var p1, p2 struct {
		Title  string `redis:"title"`
		Author string `redis:"author"`
		Body   string `redis:"body"`
	}

	p1.Title = "Example"
	p1.Author = "Gary"
	p1.Body = "Hello"

	if _, err := c.Do("HMSET", redis.Args{}.Add("id1").AddFlat(&p1)...); err != nil {
		panic(err)
	}

	m := map[string]string{
		"title":  "Example2",
		"author": "Steve",
		"body":   "Map",
	}

	if _, err := c.Do("HMSET", redis.Args{}.Add("id2").AddFlat(m)...); err != nil {
		panic(err)
	}

	for _, id := range []string{"id1", "id2"} {

		v, err := redis.Values(c.Do("HGETALL", id))
		if err != nil {
			panic(err)
		}

		if err := redis.ScanStruct(v, &p2); err != nil {
			panic(err)
		}

		fmt.Printf("%+v\n", p2)
	}

	// Output:
	// {Title:Example Author:Gary Body:Hello}
	// {Title:Example2 Author:Steve Body:Map}
}
