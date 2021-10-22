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

// +build linux

package peek_test

import (
	"flag"
	"fmt"
	"log"
	"os"
	"os/exec"
	"reflect"
	"regexp"
	"sync"
	"testing"

	"cloud.google.com/go/cmd/go-cloud-debug-agent/internal/debug"
	"cloud.google.com/go/cmd/go-cloud-debug-agent/internal/debug/local"
	"cloud.google.com/go/cmd/go-cloud-debug-agent/internal/debug/remote"
)

var expectedVarValues = map[string]interface{}{
	`main.Z_bool_false`: false,
	`main.Z_bool_true`:  true,
	`main.Z_complex128`: complex128(1.987654321 - 2.987654321i),
	`main.Z_complex64`:  complex64(1.54321 + 2.54321i),
	`main.Z_float32`:    float32(1.54321),
	`main.Z_float64`:    float64(1.987654321),
	`main.Z_int16`:      int16(-32321),
	`main.Z_int32`:      int32(-1987654321),
	`main.Z_int64`:      int64(-9012345678987654321),
	`main.Z_int8`:       int8(-121),
	`main.Z_uint16`:     uint16(54321),
	`main.Z_uint32`:     uint32(3217654321),
	`main.Z_uint64`:     uint64(12345678900987654321),
	`main.Z_uint8`:      uint8(231),
}

// TODO: the string forms of some types we're testing aren't stable
var expectedVars = map[string]string{
	`main.Z_array`:               `[5]int8{-121, 121, 3, 2, 1}`,
	`main.Z_array_empty`:         `[0]int8{}`,
	`main.Z_bool_false`:          `false`,
	`main.Z_bool_true`:           `true`,
	`main.Z_channel`:             `(chan int16 0xX)`,
	`main.Z_channel_2`:           `(chan int16 0xX)`,
	`main.Z_channel_buffered`:    `(chan int16 0xX [6/10])`,
	`main.Z_channel_nil`:         `(chan int16 <nil>)`,
	`main.Z_array_of_empties`:    `[2]{}{{} {}, ({} 0xX)}`,
	`main.Z_complex128`:          `(1.987654321-2.987654321i)`,
	`main.Z_complex64`:           `(1.54321+2.54321i)`,
	`main.Z_float32`:             `1.54321`,
	`main.Z_float64`:             `1.987654321`,
	`main.Z_func_int8_r_int8`:    `func(int8, *int8) void @0xX `,
	`main.Z_func_int8_r_pint8`:   `func(int8, **int8) void @0xX `,
	`main.Z_func_bar`:            `func(*main.FooStruct) void @0xX `,
	`main.Z_func_nil`:            `func(int8, *int8) void @0xX `,
	`main.Z_int`:                 `-21`,
	`main.Z_int16`:               `-32321`,
	`main.Z_int32`:               `-1987654321`,
	`main.Z_int64`:               `-9012345678987654321`,
	`main.Z_int8`:                `-121`,
	`main.Z_int_typedef`:         `88`,
	`main.Z_interface`:           `("*main.FooStruct", 0xX)`,
	`main.Z_interface_nil`:       `(<nil>, <nil>)`,
	`main.Z_interface_typed_nil`: `("*main.FooStruct", <nil>)`,
	`main.Z_map`:                 `map[-21:3.54321]`,
	`main.Z_map_2`:               `map[1024:1]`,
	`main.Z_map_3`:               `map[1024:1 512:-1]`,
	`main.Z_map_empty`:           `map[]`,
	`main.Z_map_nil`:             `map[]`,
	`main.Z_pointer`:             `0xX`,
	`main.Z_pointer_nil`:         `0x0`,
	`main.Z_slice`:               `[]uint8{115, 108, 105, 99, 101}`,
	`main.Z_slice_2`:             `[]int8{-121, 121}`,
	`main.Z_slice_nil`:           `[]uint8{}`,
	`main.Z_string`:              `"I'm a string"`,
	`main.Z_struct`:              `main.FooStruct {21, "hi"}`,
	`main.Z_uint`:                `21`,
	`main.Z_uint16`:              `54321`,
	`main.Z_uint32`:              `3217654321`,
	`main.Z_uint64`:              `12345678900987654321`,
	`main.Z_uint8`:               `231`,
	`main.Z_uintptr`:             `21`,
	`main.Z_unsafe_pointer`:      `0xX`,
	`main.Z_unsafe_pointer_nil`:  `0x0`,
}

// expectedEvaluate contains expected results of the debug.Evaluate function.
// A nil value indicates that an error is expected.
var expectedEvaluate = map[string]debug.Value{
	`x`:                                    int16(42),
	`local_array`:                          debug.Array{42, 42, 5, 8},
	`local_channel`:                        debug.Channel{42, 42, 42, 0, 0, 2, 0},
	`local_channel_buffered`:               debug.Channel{42, 42, 42, 6, 10, 2, 8},
	`local_map`:                            debug.Map{42, 42, 1},
	`local_map_2`:                          debug.Map{42, 42, 1},
	`local_map_3`:                          debug.Map{42, 42, 2},
	`local_map_empty`:                      debug.Map{42, 42, 0},
	`x + 5`:                                int16(47),
	`x - 5`:                                int16(37),
	`x / 5`:                                int16(8),
	`x % 5`:                                int16(2),
	`x & 2`:                                int16(2),
	`x | 1`:                                int16(43),
	`x ^ 3`:                                int16(41),
	`5 + x`:                                int16(47),
	`5 - x`:                                int16(-37),
	`100 / x`:                              int16(2),
	`100 % x`:                              int16(16),
	`2 & x`:                                int16(2),
	`1 | x`:                                int16(43),
	`3 ^ x`:                                int16(41),
	`12`:                                   12,
	`+42`:                                  42,
	`23i`:                                  23i,
	`34.0`:                                 34.0,
	`34.5`:                                 34.5,
	`1e5`:                                  100000.0,
	`0x42`:                                 66,
	`'c'`:                                  'c',
	`"de"`:                                 debug.String{2, `de`},
	"`ef`":                                 debug.String{2, `ef`},
	`"de" + "fg"`:                          debug.String{4, `defg`},
	`/* comment */ -5`:                     -5,
	`false`:                                false,
	`true`:                                 true,
	`!false`:                               true,
	`!true`:                                false,
	`5 + 5`:                                10,
	`true || false`:                        true,
	`false || false`:                       false,
	`true && false`:                        false,
	`true && true`:                         true,
	`!(5 > 8)`:                             true,
	`10 + 'a'`:                             'k',
	`10 + 10.5`:                            20.5,
	`10 + 10.5i`:                           10 + 10.5i,
	`'a' + 10.5`:                           107.5,
	`'a' + 10.5i`:                          97 + 10.5i,
	`10.5 + 20.5i`:                         10.5 + 20.5i,
	`10 * 20`:                              200,
	`10.0 - 20.5`:                          -10.5,
	`(6 + 8i) * 4`:                         24 + 32i,
	`(6 + 8i) * (1 + 1i)`:                  -2 + 14i,
	`(6 + 8i) * (6 - 8i)`:                  complex128(100),
	`(6 + 8i) / (3 + 4i)`:                  complex128(2),
	`local_array[2]`:                       int8(3),
	`&local_array[1]`:                      debug.Pointer{42, 42},
	`local_map[-21]`:                       float32(3.54321),
	`local_map[+21]`:                       float32(0),
	`local_map_3[1024]`:                    int8(1),
	`local_map_3[512]`:                     int8(-1),
	`local_map_empty[21]`:                  float32(0),
	`"hello"[2]`:                           uint8('l'),
	`local_array[1:3][1]`:                  int8(3),
	`local_array[0:4][2:3][0]`:             int8(3),
	`local_array[:]`:                       debug.Slice{debug.Array{42, 42, 5, 8}, 5},
	`local_array[:2]`:                      debug.Slice{debug.Array{42, 42, 2, 8}, 5},
	`local_array[2:]`:                      debug.Slice{debug.Array{42, 42, 3, 8}, 3},
	`local_array[1:3]`:                     debug.Slice{debug.Array{42, 42, 2, 8}, 4},
	`local_array[:3:4]`:                    debug.Slice{debug.Array{42, 42, 3, 8}, 4},
	`local_array[1:3:4]`:                   debug.Slice{debug.Array{42, 42, 2, 8}, 3},
	`local_array[1:][1:][1:]`:              debug.Slice{debug.Array{42, 42, 2, 8}, 2},
	`(&local_array)[:]`:                    debug.Slice{debug.Array{42, 42, 5, 8}, 5},
	`(&local_array)[:2]`:                   debug.Slice{debug.Array{42, 42, 2, 8}, 5},
	`(&local_array)[2:]`:                   debug.Slice{debug.Array{42, 42, 3, 8}, 3},
	`(&local_array)[1:3]`:                  debug.Slice{debug.Array{42, 42, 2, 8}, 4},
	`(&local_array)[:3:4]`:                 debug.Slice{debug.Array{42, 42, 3, 8}, 4},
	`(&local_array)[1:3:4]`:                debug.Slice{debug.Array{42, 42, 2, 8}, 3},
	`lookup("main.Z_array")`:               debug.Array{42, 42, 5, 8},
	`lookup("main.Z_array_empty")`:         debug.Array{42, 42, 0, 8},
	`lookup("main.Z_bool_false")`:          false,
	`lookup("main.Z_bool_true")`:           true,
	`lookup("main.Z_channel")`:             debug.Channel{42, 42, 42, 0, 0, 2, 0},
	`lookup("main.Z_channel_buffered")`:    debug.Channel{42, 42, 42, 6, 10, 2, 8},
	`lookup("main.Z_channel_nil")`:         debug.Channel{42, 0, 0, 0, 0, 2, 0},
	`lookup("main.Z_array_of_empties")`:    debug.Array{42, 42, 2, 0},
	`lookup("main.Z_complex128")`:          complex128(1.987654321 - 2.987654321i),
	`lookup("main.Z_complex64")`:           complex64(1.54321 + 2.54321i),
	`lookup("main.Z_float32")`:             float32(1.54321),
	`lookup("main.Z_float64")`:             float64(1.987654321),
	`lookup("main.Z_func_int8_r_int8")`:    debug.Func{42},
	`lookup("main.Z_func_int8_r_pint8")`:   debug.Func{42},
	`lookup("main.Z_func_bar")`:            debug.Func{42},
	`lookup("main.Z_func_nil")`:            debug.Func{0},
	`lookup("main.Z_int")`:                 -21,
	`lookup("main.Z_int16")`:               int16(-32321),
	`lookup("main.Z_int32")`:               int32(-1987654321),
	`lookup("main.Z_int64")`:               int64(-9012345678987654321),
	`lookup("main.Z_int8")`:                int8(-121),
	`lookup("main.Z_int_typedef")`:         int16(88),
	`lookup("main.Z_interface")`:           debug.Interface{},
	`lookup("main.Z_interface_nil")`:       debug.Interface{},
	`lookup("main.Z_interface_typed_nil")`: debug.Interface{},
	`lookup("main.Z_map")`:                 debug.Map{42, 42, 1},
	`lookup("main.Z_map_2")`:               debug.Map{42, 42, 1},
	`lookup("main.Z_map_3")`:               debug.Map{42, 42, 2},
	`lookup("main.Z_map_empty")`:           debug.Map{42, 42, 0},
	`lookup("main.Z_map_nil")`:             debug.Map{42, 42, 0},
	`lookup("main.Z_pointer")`:             debug.Pointer{42, 42},
	`lookup("main.Z_pointer_nil")`:         debug.Pointer{42, 0},
	`lookup("main.Z_slice")`:               debug.Slice{debug.Array{42, 42, 5, 8}, 5},
	`lookup("main.Z_slice_2")`:             debug.Slice{debug.Array{42, 42, 2, 8}, 5},
	`lookup("main.Z_slice_nil")`:           debug.Slice{debug.Array{42, 0, 0, 8}, 0},
	`lookup("main.Z_string")`:              debug.String{12, `I'm a string`},
	`lookup("main.Z_struct")`:              debug.Struct{[]debug.StructField{{"a", debug.Var{}}, {"b", debug.Var{}}}},
	`lookup("main.Z_uint")`:                uint(21),
	`lookup("main.Z_uint16")`:              uint16(54321),
	`lookup("main.Z_uint32")`:              uint32(3217654321),
	`lookup("main.Z_uint64")`:              uint64(12345678900987654321),
	`lookup("main.Z_uint8")`:               uint8(231),
	`lookup("main.Z_uintptr")`:             uint(21),
	`lookup("main.Z_unsafe_pointer")`:      debug.Pointer{0, 42},
	`lookup("main.Z_unsafe_pointer_nil")`:  debug.Pointer{0, 0},
	`lookup("main.Z_int") + lookup("main.Z_int")`:                -42,
	`lookup("main.Z_int16") < 0`:                                 true,
	`lookup("main.Z_uint32") + lookup("main.Z_uint32")`:          uint32(2140341346),
	`lookup("main.Z_bool_true") || lookup("main.Z_bool_false")`:  true,
	`lookup("main.Z_bool_true") && lookup("main.Z_bool_false")`:  false,
	`lookup("main.Z_bool_false") || lookup("main.Z_bool_false")`: false,
	`!lookup("main.Z_bool_true")`:                                false,
	`!lookup("main.Z_bool_false")`:                               true,
	`lookup("main.Z_array")[2]`:                                  int8(3),
	`lookup("main.Z_array")[1:3][1]`:                             int8(3),
	`lookup("main.Z_array")[0:4][2:3][0]`:                        int8(3),
	`lookup("main.Z_array_of_empties")[0]`:                       debug.Struct{},
	`lookup("main.Z_complex128") * 10.0`:                         complex128(19.87654321 - 29.87654321i),
	`lookup("main.Z_complex64") * 0.1`:                           complex64(0.154321 + 0.254321i),
	`lookup("main.Z_float32") * 10.0`:                            float32(15.4321),
	`lookup("main.Z_float64") * 0.1`:                             float64(0.1987654321),
	`lookup("main.Z_int") + 1`:                                   int(-20),
	`lookup("main.Z_int16") - 10`:                                int16(-32331),
	`lookup("main.Z_int32") / 10`:                                int32(-198765432),
	`lookup("main.Z_int64") / 10`:                                int64(-901234567898765432),
	`lookup("main.Z_int8") + 10`:                                 int8(-111),
	`lookup("main.Z_map")[-21]`:                                  float32(3.54321),
	`lookup("main.Z_map")[+21]`:                                  float32(0),
	`lookup("main.Z_map_empty")[21]`:                             float32(0),
	`lookup("main.Z_slice")[1]`:                                  uint8(108),
	`lookup("main.Z_slice_2")[1]`:                                int8(121),
	`lookup("main.Z_slice")[1:5][0:3][1]`:                        uint8('i'),
	`lookup("main.Z_array")[1:3:4]`:                              debug.Slice{debug.Array{42, 42, 2, 8}, 3},
	`(&lookup("main.Z_array"))[1:3:4]`:                           debug.Slice{debug.Array{42, 42, 2, 8}, 3},
	`lookup("main.Z_string") + "!"`:                              debug.String{13, `I'm a string!`},
	`lookup("main.Z_struct").a`:                                  21,
	`(&lookup("main.Z_struct")).a`:                               21,
	`lookup("main.Z_uint")/10`:                                   uint(2),
	`lookup("main.Z_uint16")/10`:                                 uint16(5432),
	`lookup("main.Z_uint32")/10`:                                 uint32(321765432),
	`lookup("main.Z_uint64")/10`:                                 uint64(1234567890098765432),
	`lookup("main.Z_uint8")/10`:                                  uint8(23),
	`lookup("main.Z_pointer").a`:                                 21,
	`(*lookup("main.Z_pointer")).a`:                              21,
	`(&*lookup("main.Z_pointer")).a`:                             21,
	`lookup("main.Z_pointer").b`:                                 debug.String{2, `hi`},
	`(*lookup("main.Z_pointer")).b`:                              debug.String{2, `hi`},
	`(&*lookup("main.Z_pointer")).b`:                             debug.String{2, `hi`},
	`lookup("main.Z_map_nil")[32]`:                               float32(0),
	`&lookup("main.Z_int16")`:                                    debug.Pointer{42, 42},
	`&lookup("main.Z_array")[1]`:                                 debug.Pointer{42, 42},
	`&lookup("main.Z_slice")[1]`:                                 debug.Pointer{42, 42},
	`*&lookup("main.Z_int16")`:                                   int16(-32321),
	`*&*&*&*&lookup("main.Z_int16")`:                             int16(-32321),
	`lookup("time.Local")`:                                       debug.Pointer{42, 42},
	`5 + false`:                                                  nil,
	``:                                                           nil,
	`x + ""`:                                                     nil,
	`x / 0`:                                                      nil,
	`0 / 0`:                                                      nil,
	`'a' / ('a'-'a')`:                                            nil,
	`0.0 / 0.0`:                                                  nil,
	`3i / 0.0`:                                                   nil,
	`x % 0`:                                                      nil,
	`0 % 0`:                                                      nil,
	`'a' % ('a'-'a')`:                                            nil,
	`local_array[-2] + 1`:                                        nil,
	`local_array[22] + 1`:                                        nil,
	`local_slice[-2] + 1`:                                        nil,
	`local_slice[22] + 1`:                                        nil,
	`local_string[-2]`:                                           nil,
	`local_string[22]`:                                           nil,
	`"hello"[-2]`:                                                nil,
	`"hello"[22]`:                                                nil,
	`local_pointer_nil.a`:                                        nil,
	`(local_struct).c`:                                           nil,
	`(&local_struct).c`:                                          nil,
	`(*local_pointer).c`:                                         nil,
	`lookup("not a real symbol")`:                                nil,
	`lookup("x")`:                                                nil,
	`lookup(x)`:                                                  nil,
	`lookup(42)`:                                                 nil,
}

func isHex(r uint8) bool {
	switch {
	case '0' <= r && r <= '9':
		return true
	case 'a' <= r && r <= 'f':
		return true
	case 'A' <= r && r <= 'F':
		return true
	default:
		return false
	}
}

// structRE is used by matches to remove 'struct ' from type names, which is not
// output by every version of the compiler.
var structRE = regexp.MustCompile("struct *")

// Check s matches the pattern in p.
// An 'X' in p greedily matches one or more hex characters in s.
func matches(p, s string) bool {
	// Remove 'struct' and following spaces from s.
	s = structRE.ReplaceAllString(s, "")
	j := 0
	for i := 0; i < len(p); i++ {
		if j == len(s) {
			return false
		}
		c := p[i]
		if c == 'X' {
			if !isHex(s[j]) {
				return false
			}
			for j < len(s) && isHex(s[j]) {
				j++
			}
			continue
		}
		if c != s[j] {
			return false
		}
		j++
	}
	return j == len(s)
}

const (
	proxySrc  = "cloud.google.com/go/cmd/go-cloud-debug-agent/internal/debug/cmd/debugproxy"
	traceeSrc = "cloud.google.com/go/cmd/go-cloud-debug-agent/internal/debug/tests/peek/testdata"
)

var (
	// Locations of the proxy and tracee executables.
	proxyBinary  = "./debugproxy.out"
	traceeBinary = "./tracee.out"
	// Onces that ensure initProxy and initTracee are called at most once.
	proxyOnce  sync.Once
	traceeOnce sync.Once
	// Flags for setting the location of the proxy and tracee, so they don't need to be built.
	proxyFlag  = flag.String("proxy", "", "Location of debugproxy.  If empty, proxy will be built.")
	traceeFlag = flag.String("target", "", "Location of target.  If empty, target will be built.")
	// Executables this test has built, which will be removed on completion of the tests.
	filesToRemove []string
)

func TestMain(m *testing.M) {
	flag.Parse()
	x := m.Run()
	for _, f := range filesToRemove {
		os.Remove(f)
	}
	os.Exit(x)
}

func run(name string, args ...string) error {
	cmd := exec.Command(name, args...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

func initProxy() {
	if *proxyFlag != "" {
		proxyBinary = *proxyFlag
		remote.DebugproxyCmd = proxyBinary
		return
	}
	if err := run("go", "build", "-o", proxyBinary, proxySrc); err != nil {
		log.Fatalf("couldn't build proxy: %v", err)
	}
	filesToRemove = append(filesToRemove, proxyBinary)
	remote.DebugproxyCmd = proxyBinary
}

func initTracee() {
	if *traceeFlag != "" {
		traceeBinary = *traceeFlag
		return
	}
	if err := run("go", "build", "-o", traceeBinary, traceeSrc); err != nil {
		log.Fatalf("couldn't build target: %v", err)
	}
	filesToRemove = append(filesToRemove, traceeBinary)
}

func TestLocalProgram(t *testing.T) {
	t.Skip("Fails looking for runtime.lessstack for some reason")
	traceeOnce.Do(initTracee)
	prog, err := local.New(traceeBinary)
	if err != nil {
		t.Fatal("local.New:", err)
	}
	testProgram(t, prog)
}

func TestRemoteProgram(t *testing.T) {
	t.Skip("Fails looking for runtime.lessstack for some reason")
	traceeOnce.Do(initTracee)
	proxyOnce.Do(initProxy)
	prog, err := remote.New("localhost", traceeBinary)
	if err != nil {
		t.Fatal("remote.New:", err)
	}
	testProgram(t, prog)
}

func testProgram(t *testing.T, prog debug.Program) {
	_, err := prog.Run("some", "arguments")
	if err != nil {
		log.Fatalf("Run: %v", err)
	}

	pcs, err := prog.BreakpointAtFunction("main.foo")
	if err != nil {
		log.Fatalf("BreakpointAtFunction: %v", err)
	}
	fmt.Printf("breakpoints set at %x\n", pcs)

	_, err = prog.Resume()
	if err != nil {
		log.Fatalf("Resume: %v", err)
	}

	gs, err := prog.Goroutines()
	if err != nil {
		t.Fatalf("Goroutines(): got error %s", err)
	}
	for _, g := range gs {
		fmt.Println(g)
		for _, f := range g.StackFrames {
			fmt.Println(f)
		}
	}

	frames, err := prog.Frames(100)
	if err != nil {
		log.Fatalf("prog.Frames error: %v", err)
	}
	fmt.Printf("%#v\n", frames)
	if len(frames) == 0 {
		t.Fatalf("no stack frames returned")
	}
	if frames[0].Function != "main.foo" {
		t.Errorf("function name: got %s expected main.foo", frames[0].Function)
	}
	if len(frames[0].Params) != 2 {
		t.Errorf("got %d parameters, expected 2", len(frames[0].Params))
	} else {
		x := frames[0].Params[0]
		y := frames[0].Params[1]
		if x.Name != "x" {
			x, y = y, x
		}
		if x.Name != "x" {
			t.Errorf("parameter name: got %s expected x", x.Name)
		}
		if y.Name != "y" {
			t.Errorf("parameter name: got %s expected y", y.Name)
		}
		if val, err := prog.Value(x.Var); err != nil {
			t.Errorf("value of x: %s", err)
		} else if val != int16(42) {
			t.Errorf("value of x: got %T(%v) expected int16(42)", val, val)
		}
		if val, err := prog.Value(y.Var); err != nil {
			t.Errorf("value of y: %s", err)
		} else if val != float32(1.5) {
			t.Errorf("value of y: got %T(%v) expected float32(1.5)", val, val)
		}
	}

	varnames, err := prog.Eval(`re:main\.Z_.*`)
	if err != nil {
		log.Fatalf("prog.Eval error: %v", err)
	}

	// Evaluate each of the variables found above, and check they match
	// expectedVars.
	seen := make(map[string]bool)
	for _, v := range varnames {
		val, err := prog.Eval("val:" + v)
		if err != nil {
			log.Fatalf("prog.Eval error for %s: %v", v, err)
		} else {
			fmt.Printf("%s = %v\n", v, val)
			if seen[v] {
				log.Fatalf("repeated variable %s\n", v)
			}
			seen[v] = true
			if len(val) != 1 {
				log.Fatalf("should be one value for %s\n", v)
			}
			expected, ok := expectedVars[v]
			if !ok {
				log.Fatalf("unexpected variable %s\n", v)
			} else {
				if !matches(expected, val[0]) {
					log.Fatalf("expected %s = %s\n", v, expected)
				}
			}
		}
	}
	for v, e := range expectedVars {
		if !seen[v] {
			log.Fatalf("didn't get %s = %s\n", v, e)
		}
	}

	// Remove the breakpoint at main.foo.
	err = prog.DeleteBreakpoints(pcs)
	if err != nil {
		log.Fatalf("DeleteBreakpoints: %v", err)
	}

	// Set a breakpoint at line 125, resume, and check we stopped there.
	pcsLine125, err := prog.BreakpointAtLine("testdata/main.go", 125)
	if err != nil {
		t.Fatal("BreakpointAtLine:", err)
	}
	status, err := prog.Resume()
	if err != nil {
		log.Fatalf("Resume: %v", err)
	}
	stoppedAt := func(pcs []uint64) bool {
		for _, pc := range pcs {
			if status.PC == pc {
				return true
			}
		}
		return false
	}
	if !stoppedAt(pcsLine125) {
		t.Errorf("stopped at %X; expected one of %X.", status.PC, pcsLine125)
	}

	for k, v := range expectedEvaluate {
		val, err := prog.Evaluate(k)
		if v == nil {
			if err == nil {
				t.Errorf("got Evaluate(%s) = %v, expected error", k, val)
			}
			continue
		}
		if err != nil {
			t.Errorf("Evaluate(%s): got error %s, expected %v", k, err, v)
			continue
		}
		typ := reflect.TypeOf(v)
		if typ != reflect.TypeOf(val) && typ != reflect.TypeOf(int(0)) && typ != reflect.TypeOf(uint(0)) {
			t.Errorf("got Evaluate(%s) = %T(%v), expected %T(%v)", k, val, val, v, v)
			continue
		}

		// For types with fields like Address, TypeID, etc., we can't know the exact
		// value, so we only test whether those fields are zero or not.
		switch v := v.(type) {
		default:
			if v != val {
				t.Errorf("got Evaluate(%s) = %T(%v), expected %T(%v)", k, val, val, v, v)
			}
		case debug.Array:
			val := val.(debug.Array)
			if v.ElementTypeID == 0 && val.ElementTypeID != 0 {
				t.Errorf("got Evaluate(%s) = %+v, expected zero ElementTypeID", k, val)
			}
			if v.ElementTypeID != 0 && val.ElementTypeID == 0 {
				t.Errorf("got Evaluate(%s) = %+v, expected non-zero ElementTypeID", k, val)
			}
			if v.Address == 0 && val.Address != 0 {
				t.Errorf("got Evaluate(%s) = %+v, expected zero Address", k, val)
			}
			if v.Address != 0 && val.Address == 0 {
				t.Errorf("got Evaluate(%s) = %+v, expected non-zero Address", k, val)
			}
		case debug.Slice:
			val := val.(debug.Slice)
			if v.ElementTypeID == 0 && val.ElementTypeID != 0 {
				t.Errorf("got Evaluate(%s) = %+v, expected zero ElementTypeID", k, val)
			}
			if v.ElementTypeID != 0 && val.ElementTypeID == 0 {
				t.Errorf("got Evaluate(%s) = %+v, expected non-zero ElementTypeID", k, val)
			}
			if v.Address == 0 && val.Address != 0 {
				t.Errorf("got Evaluate(%s) = %+v, expected zero Address", k, val)
			}
			if v.Address != 0 && val.Address == 0 {
				t.Errorf("got Evaluate(%s) = %+v, expected non-zero Address", k, val)
			}
		case debug.Map:
			val := val.(debug.Map)
			if v.TypeID == 0 && val.TypeID != 0 {
				t.Errorf("got Evaluate(%s) = %+v, expected zero TypeID", k, val)
			}
			if v.TypeID != 0 && val.TypeID == 0 {
				t.Errorf("got Evaluate(%s) = %+v, expected non-zero TypeID", k, val)
			}
			if v.Address == 0 && val.Address != 0 {
				t.Errorf("got Evaluate(%s) = %+v, expected zero Address", k, val)
			}
			if v.Address != 0 && val.Address == 0 {
				t.Errorf("got Evaluate(%s) = %+v, expected non-zero Address", k, val)
			}
		case debug.Pointer:
			val := val.(debug.Pointer)
			if v.TypeID == 0 && val.TypeID != 0 {
				t.Errorf("got Evaluate(%s) = %+v, expected zero TypeID", k, val)
			}
			if v.TypeID != 0 && val.TypeID == 0 {
				t.Errorf("got Evaluate(%s) = %+v, expected non-zero TypeID", k, val)
			}
			if v.Address == 0 && val.Address != 0 {
				t.Errorf("got Evaluate(%s) = %+v, expected zero Address", k, val)
			}
			if v.Address != 0 && val.Address == 0 {
				t.Errorf("got Evaluate(%s) = %+v, expected non-zero Address", k, val)
			}
		case debug.Channel:
			val := val.(debug.Channel)
			if v.ElementTypeID == 0 && val.ElementTypeID != 0 {
				t.Errorf("got Evaluate(%s) = %+v, expected zero ElementTypeID", k, val)
			}
			if v.ElementTypeID != 0 && val.ElementTypeID == 0 {
				t.Errorf("got Evaluate(%s) = %+v, expected non-zero ElementTypeID", k, val)
			}
			if v.Address == 0 && val.Address != 0 {
				t.Errorf("got Evaluate(%s) = %+v, expected zero Address", k, val)
			}
			if v.Address != 0 && val.Address == 0 {
				t.Errorf("got Evaluate(%s) = %+v, expected non-zero Address", k, val)
			}
			if v.Buffer == 0 && val.Buffer != 0 {
				t.Errorf("got Evaluate(%s) = %+v, expected zero Buffer", k, val)
			}
			if v.Buffer != 0 && val.Buffer == 0 {
				t.Errorf("got Evaluate(%s) = %+v, expected non-zero Buffer", k, val)
			}
		case debug.Struct:
			val := val.(debug.Struct)
			if len(v.Fields) != len(val.Fields) {
				t.Errorf("got Evaluate(%s) = %T(%v), expected %T(%v)", k, val, val, v, v)
				break
			}
			for i := range v.Fields {
				a := v.Fields[i].Name
				b := val.Fields[i].Name
				if a != b {
					t.Errorf("Evaluate(%s): field name mismatch: %s vs %s", k, a, b)
					break
				}
			}
		case debug.Func:
			val := val.(debug.Func)
			if v.Address == 0 && val.Address != 0 {
				t.Errorf("got Evaluate(%s) = %+v, expected zero Address", k, val)
			}
			if v.Address != 0 && val.Address == 0 {
				t.Errorf("got Evaluate(%s) = %+v, expected non-zero Address", k, val)
			}
		case int:
			// ints in a remote program can be returned as int32 or int64
			switch val := val.(type) {
			case int32:
				if val != int32(v) {
					t.Errorf("got Evaluate(%s) = %T(%v), expected %v", k, val, val, v)
				}
			case int64:
				if val != int64(v) {
					t.Errorf("got Evaluate(%s) = %T(%v), expected %v", k, val, val, v)
				}
			default:
				t.Errorf("got Evaluate(%s) = %T(%v), expected %T(%v)", k, val, val, v, v)
			}
		case uint:
			// uints in a remote program can be returned as uint32 or uint64
			switch val := val.(type) {
			case uint32:
				if val != uint32(v) {
					t.Errorf("got Evaluate(%s) = %T(%v), expected %v", k, val, val, v)
				}
			case uint64:
				if val != uint64(v) {
					t.Errorf("got Evaluate(%s) = %T(%v), expected %v", k, val, val, v)
				}
			default:
				t.Errorf("got Evaluate(%s) = %T(%v), expected %T(%v)", k, val, val, v, v)
			}
		}
	}

	// Evaluate a struct.
	v := `lookup("main.Z_struct")`
	val, err := prog.Evaluate(v)
	if err != nil {
		t.Fatalf("Evaluate: %s", err)
	}
	s, ok := val.(debug.Struct)
	if !ok {
		t.Fatalf("got Evaluate(%q) = %T(%v), expected debug.Struct", v, val, val)
	}
	// Check the values of its fields.
	if len(s.Fields) != 2 {
		t.Fatalf("got Evaluate(%q) = %+v, expected 2 fields", v, s)
	}
	if v0, err := prog.Value(s.Fields[0].Var); err != nil {
		t.Errorf("Value: %s", err)
	} else if v0 != int32(21) && v0 != int64(21) {
		t.Errorf("Value: got %T(%v), expected 21", v0, v0)
	}
	if v1, err := prog.Value(s.Fields[1].Var); err != nil {
		t.Errorf("Value: %s", err)
	} else if v1 != (debug.String{2, "hi"}) {
		t.Errorf("Value: got %T(%v), expected `hi`", v1, v1)
	}

	// Remove the breakpoint at line 125, set a breakpoint at main.f1 and main.f2,
	// then delete the breakpoint at main.f1.  Resume, then check we stopped at
	// main.f2.
	err = prog.DeleteBreakpoints(pcsLine125)
	if err != nil {
		log.Fatalf("DeleteBreakpoints: %v", err)
	}
	pcs1, err := prog.BreakpointAtFunction("main.f1")
	if err != nil {
		log.Fatalf("BreakpointAtFunction: %v", err)
	}
	pcs2, err := prog.BreakpointAtFunction("main.f2")
	if err != nil {
		log.Fatalf("BreakpointAtFunction: %v", err)
	}
	err = prog.DeleteBreakpoints(pcs1)
	if err != nil {
		log.Fatalf("DeleteBreakpoints: %v", err)
	}
	status, err = prog.Resume()
	if err != nil {
		log.Fatalf("Resume: %v", err)
	}
	if !stoppedAt(pcs2) {
		t.Errorf("stopped at %X; expected one of %X.", status.PC, pcs2)
	}

	// Check we get the expected results calling VarByName then Value
	// for the variables in expectedVarValues.
	for name, exp := range expectedVarValues {
		if v, err := prog.VarByName(name); err != nil {
			t.Errorf("VarByName(%s): %s", name, err)
		} else if val, err := prog.Value(v); err != nil {
			t.Errorf("value of %s: %s", name, err)
		} else if val != exp {
			t.Errorf("value of %s: got %T(%v) want %T(%v)", name, val, val, exp, exp)
		}
	}

	// Check some error cases for VarByName and Value.
	if _, err = prog.VarByName("not a real name"); err == nil {
		t.Error("VarByName for invalid name: expected error")
	}
	if _, err = prog.Value(debug.Var{}); err == nil {
		t.Error("value of invalid var: expected error")
	}
	if v, err := prog.VarByName("main.Z_int16"); err != nil {
		t.Error("VarByName(main.Z_int16) error:", err)
	} else {
		v.Address = 0
		// v now has a valid type but a bad address.
		_, err = prog.Value(v)
		if err == nil {
			t.Error("value of invalid location: expected error")
		}
	}

	// checkValue tests that we can get a Var for a variable with the given name,
	// that we can then get the value of that Var, and that calling fn for that
	// value succeeds.
	checkValue := func(name string, fn func(val debug.Value) error) {
		if v, err := prog.VarByName(name); err != nil {
			t.Errorf("VarByName(%s): %s", name, err)
		} else if val, err := prog.Value(v); err != nil {
			t.Errorf("value of %s: %s", name, err)
		} else if err := fn(val); err != nil {
			t.Errorf("value of %s: %s", name, err)
		}
	}

	checkValue("main.Z_uintptr", func(val debug.Value) error {
		if val != uint32(21) && val != uint64(21) {
			// Z_uintptr should be an unsigned integer with size equal to the debugged
			// program's address size.
			return fmt.Errorf("got %T(%v) want 21", val, val)
		}
		return nil
	})

	checkValue("main.Z_int", func(val debug.Value) error {
		if val != int32(-21) && val != int64(-21) {
			return fmt.Errorf("got %T(%v) want -21", val, val)
		}
		return nil
	})

	checkValue("main.Z_uint", func(val debug.Value) error {
		if val != uint32(21) && val != uint64(21) {
			return fmt.Errorf("got %T(%v) want 21", val, val)
		}
		return nil
	})

	checkValue("main.Z_pointer", func(val debug.Value) error {
		if _, ok := val.(debug.Pointer); !ok {
			return fmt.Errorf("got %T(%v) expected Pointer", val, val)
		}
		return nil
	})

	checkValue("main.Z_pointer_nil", func(val debug.Value) error {
		if p, ok := val.(debug.Pointer); !ok {
			return fmt.Errorf("got %T(%v) expected Pointer", val, val)
		} else if p.Address != 0 {
			return fmt.Errorf("got %T(%v) expected nil pointer", val, val)
		}
		return nil
	})

	checkValue("main.Z_array", func(val debug.Value) error {
		a, ok := val.(debug.Array)
		if !ok {
			return fmt.Errorf("got %T(%v) expected Array", val, val)
		}
		if a.Len() != 5 {
			return fmt.Errorf("got array length %d expected 5", a.Len())
		}
		expected := [5]int8{-121, 121, 3, 2, 1}
		for i := uint64(0); i < 5; i++ {
			if v, err := prog.Value(a.Element(i)); err != nil {
				return fmt.Errorf("reading element %d: %s", i, err)
			} else if v != expected[i] {
				return fmt.Errorf("element %d: got %T(%v) want %T(%d)", i, v, v, expected[i], expected[i])
			}
		}
		return nil
	})

	checkValue("main.Z_slice", func(val debug.Value) error {
		s, ok := val.(debug.Slice)
		if !ok {
			return fmt.Errorf("got %T(%v) expected Slice", val, val)
		}
		if s.Len() != 5 {
			return fmt.Errorf("got slice length %d expected 5", s.Len())
		}
		expected := []uint8{115, 108, 105, 99, 101}
		for i := uint64(0); i < 5; i++ {
			if v, err := prog.Value(s.Element(i)); err != nil {
				return fmt.Errorf("reading element %d: %s", i, err)
			} else if v != expected[i] {
				return fmt.Errorf("element %d: got %T(%v) want %T(%d)", i, v, v, expected[i], expected[i])
			}
		}
		return nil
	})

	checkValue("main.Z_map_empty", func(val debug.Value) error {
		m, ok := val.(debug.Map)
		if !ok {
			return fmt.Errorf("got %T(%v) expected Map", val, val)
		}
		if m.Length != 0 {
			return fmt.Errorf("got map length %d expected 0", m.Length)
		}
		return nil
	})

	checkValue("main.Z_map_nil", func(val debug.Value) error {
		m, ok := val.(debug.Map)
		if !ok {
			return fmt.Errorf("got %T(%v) expected Map", val, val)
		}
		if m.Length != 0 {
			return fmt.Errorf("got map length %d expected 0", m.Length)
		}
		return nil
	})

	checkValue("main.Z_map_3", func(val debug.Value) error {
		m, ok := val.(debug.Map)
		if !ok {
			return fmt.Errorf("got %T(%v) expected Map", val, val)
		}
		if m.Length != 2 {
			return fmt.Errorf("got map length %d expected 2", m.Length)
		}
		keyVar0, valVar0, err := prog.MapElement(m, 0)
		if err != nil {
			return err
		}
		keyVar1, valVar1, err := prog.MapElement(m, 1)
		if err != nil {
			return err
		}
		key0, err := prog.Value(keyVar0)
		if err != nil {
			return err
		}
		key1, err := prog.Value(keyVar1)
		if err != nil {
			return err
		}
		val0, err := prog.Value(valVar0)
		if err != nil {
			return err
		}
		val1, err := prog.Value(valVar1)
		if err != nil {
			return err
		}
		// The map should contain 1024,1 and 512,-1 in some order.
		ok1 := key0 == int16(1024) && val0 == int8(1) && key1 == int16(512) && val1 == int8(-1)
		ok2 := key1 == int16(1024) && val1 == int8(1) && key0 == int16(512) && val0 == int8(-1)
		if !ok1 && !ok2 {
			return fmt.Errorf("got values (%d,%d) and (%d,%d), expected (1024,1) and (512,-1) in some order", key0, val0, key1, val1)
		}
		_, _, err = prog.MapElement(m, 2)
		if err == nil {
			return fmt.Errorf("MapElement: reading at a bad index succeeded, expected error")
		}
		return nil
	})

	checkValue("main.Z_string", func(val debug.Value) error {
		s, ok := val.(debug.String)
		if !ok {
			return fmt.Errorf("got %T(%v) expected String", val, val)
		}
		if s.Length != 12 {
			return fmt.Errorf("got string length %d expected 12", s.Length)
		}
		expected := "I'm a string"
		if s.String != expected {
			return fmt.Errorf("got %s expected %s", s.String, expected)
		}
		return nil
	})

	checkValue("main.Z_channel", func(val debug.Value) error {
		c, ok := val.(debug.Channel)
		if !ok {
			return fmt.Errorf("got %T(%v) expected Channel", val, val)
		}
		if c.Buffer == 0 {
			return fmt.Errorf("got buffer address %d expected nonzero", c.Buffer)
		}
		if c.Length != 0 {
			return fmt.Errorf("got length %d expected 0", c.Length)
		}
		if c.Capacity != 0 {
			return fmt.Errorf("got capacity %d expected 0", c.Capacity)
		}
		return nil
	})

	checkValue("main.Z_channel_2", func(val debug.Value) error {
		c, ok := val.(debug.Channel)
		if !ok {
			return fmt.Errorf("got %T(%v) expected Channel", val, val)
		}
		if c.Buffer == 0 {
			return fmt.Errorf("got buffer address %d expected nonzero", c.Buffer)
		}
		if c.Length != 0 {
			return fmt.Errorf("got length %d expected 0", c.Length)
		}
		if c.Capacity != 0 {
			return fmt.Errorf("got capacity %d expected 0", c.Capacity)
		}
		return nil
	})

	checkValue("main.Z_channel_nil", func(val debug.Value) error {
		c, ok := val.(debug.Channel)
		if !ok {
			return fmt.Errorf("got %T(%v) expected Channel", val, val)
		}
		if c.Buffer != 0 {
			return fmt.Errorf("got buffer address %d expected 0", c.Buffer)
		}
		if c.Length != 0 {
			return fmt.Errorf("got length %d expected 0", c.Length)
		}
		if c.Capacity != 0 {
			return fmt.Errorf("got capacity %d expected 0", c.Capacity)
		}
		return nil
	})

	checkValue("main.Z_channel_buffered", func(val debug.Value) error {
		c, ok := val.(debug.Channel)
		if !ok {
			return fmt.Errorf("got %T(%v) expected Channel", val, val)
		}
		if c.Buffer == 0 {
			return fmt.Errorf("got buffer address %d expected nonzero", c.Buffer)
		}
		if c.Length != 6 {
			return fmt.Errorf("got length %d expected 6", c.Length)
		}
		if c.Capacity != 10 {
			return fmt.Errorf("got capacity %d expected 10", c.Capacity)
		}
		if c.Stride != 2 {
			return fmt.Errorf("got stride %d expected 2", c.Stride)
		}
		expected := []int16{8, 9, 10, 11, 12, 13}
		for i := uint64(0); i < 6; i++ {
			if v, err := prog.Value(c.Element(i)); err != nil {
				return fmt.Errorf("reading element %d: %s", i, err)
			} else if v != expected[i] {
				return fmt.Errorf("element %d: got %T(%v) want %T(%d)", i, v, v, expected[i], expected[i])
			}
		}
		v := c.Element(6)
		if v.Address != 0 {
			return fmt.Errorf("invalid element returned Var with address %d, expected 0", v.Address)
		}
		return nil
	})

	checkValue("main.Z_func_bar", func(val debug.Value) error {
		f, ok := val.(debug.Func)
		if !ok {
			return fmt.Errorf("got %T(%v) expected Func", val, val)
		}
		if f.Address == 0 {
			return fmt.Errorf("got func address %d expected nonzero", f.Address)
		}
		return nil
	})

	checkValue("main.Z_func_nil", func(val debug.Value) error {
		f, ok := val.(debug.Func)
		if !ok {
			return fmt.Errorf("got %T(%v) expected Func", val, val)
		}
		if f.Address != 0 {
			return fmt.Errorf("got func address %d expected zero", f.Address)
		}
		return nil
	})
}
