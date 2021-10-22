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

package main

import (
	"fmt"
	"log"
	"os"
	"time"
	"unsafe"
)

type FooInterface interface {
	Bar()
}

type FooStruct struct {
	a int
	b string
}

func (f *FooStruct) Bar() {}

type myInt int16

var (
	Z_bool_false          bool        = false
	Z_bool_true           bool        = true
	Z_int                 int         = -21
	Z_int8                int8        = -121
	Z_int16               int16       = -32321
	Z_int32               int32       = -1987654321
	Z_int64               int64       = -9012345678987654321
	Z_int_typedef         myInt       = 88
	Z_uint                uint        = 21
	Z_uint8               uint8       = 231
	Z_uint16              uint16      = 54321
	Z_uint32              uint32      = 3217654321
	Z_uint64              uint64      = 12345678900987654321
	Z_uintptr             uintptr     = 21
	Z_float32             float32     = 1.54321
	Z_float64             float64     = 1.987654321
	Z_complex64           complex64   = 1.54321 + 2.54321i
	Z_complex128          complex128  = 1.987654321 - 2.987654321i
	Z_array               [5]int8     = [5]int8{-121, 121, 3, 2, 1}
	Z_array_empty         [0]int8     = [0]int8{}
	Z_array_of_empties    [2]struct{} = [2]struct{}{{}, {}}
	Z_channel             chan int16  = make(chan int16)
	Z_channel_2           chan int16  = make(chan int16)
	Z_channel_buffered    chan int16  = make(chan int16, 10)
	Z_channel_nil         chan int16
	Z_func_bar                              = (*FooStruct).Bar
	Z_func_int8_r_int8                      = func(x int8) int8 { return x + 1 }
	Z_func_int8_r_pint8                     = func(x int8) *int8 { y := x + 1; return &y }
	Z_func_nil            func(x int8) int8 = nil
	Z_interface           FooInterface      = &Z_struct
	Z_interface_typed_nil FooInterface      = Z_pointer_nil
	Z_interface_nil       FooInterface
	Z_map                 map[int8]float32 = map[int8]float32{-21: 3.54321}
	Z_map_2               map[int16]int8   = map[int16]int8{1024: 1}
	Z_map_3               map[int16]int8   = map[int16]int8{1024: 1, 512: -1}
	Z_map_empty           map[int8]float32 = map[int8]float32{}
	Z_map_nil             map[int8]float32
	Z_pointer             *FooStruct = &Z_struct
	Z_pointer_nil         *FooStruct
	Z_slice               []byte = []byte{'s', 'l', 'i', 'c', 'e'}
	Z_slice_2             []int8 = Z_array[0:2]
	Z_slice_nil           []byte
	Z_string              string         = "I'm a string"
	Z_struct              FooStruct      = FooStruct{a: 21, b: "hi"}
	Z_unsafe_pointer      unsafe.Pointer = unsafe.Pointer(&Z_uint)
	Z_unsafe_pointer_nil  unsafe.Pointer
)

func foo(x int16, y float32) {
	var (
		local_array               [5]int8    = [5]int8{-121, 121, 3, 2, 1}
		local_bool_false          bool       = false
		local_bool_true           bool       = true
		local_channel             chan int16 = Z_channel
		local_channel_buffered    chan int16 = Z_channel_buffered
		local_channel_nil         chan int16
		local_complex128          complex128        = 1.987654321 - 2.987654321i
		local_complex64           complex64         = 1.54321 + 2.54321i
		local_float32             float32           = 1.54321
		local_float64             float64           = 1.987654321
		local_func_bar                              = (*FooStruct).Bar
		local_func_int8_r_int8                      = func(x int8) int8 { return x + 1 }
		local_func_int8_r_pint8                     = func(x int8) *int8 { y := x + 1; return &y }
		local_func_nil            func(x int8) int8 = nil
		local_int                 int               = -21
		local_int16               int16             = -32321
		local_int32               int32             = -1987654321
		local_int64               int64             = -9012345678987654321
		local_int8                int8              = -121
		local_int_typedef         myInt             = 88
		local_interface           FooInterface      = &Z_struct
		local_interface_nil       FooInterface
		local_interface_typed_nil FooInterface     = Z_pointer_nil
		local_map                 map[int8]float32 = map[int8]float32{-21: 3.54321}
		local_map_2               map[int16]int8   = map[int16]int8{1024: 1}
		local_map_3               map[int16]int8   = map[int16]int8{1024: 1, 512: -1}
		local_map_empty           map[int8]float32 = map[int8]float32{}
		local_map_nil             map[int8]float32
		local_pointer             *FooStruct = &Z_struct
		local_pointer_nil         *FooStruct
		local_slice               []byte = []byte{'s', 'l', 'i', 'c', 'e'}
		local_slice_2             []int8 = Z_array[0:2]
		local_slice_nil           []byte
		local_string              string         = "I'm a string"
		local_struct              FooStruct      = FooStruct{a: 21, b: "hi"}
		local_uint                uint           = 21
		local_uint16              uint16         = 54321
		local_uint32              uint32         = 3217654321
		local_uint64              uint64         = 12345678900987654321
		local_uint8               uint8          = 231
		local_uintptr             uintptr        = 21
		local_unsafe_pointer      unsafe.Pointer = unsafe.Pointer(&Z_uint)
		local_unsafe_pointer_nil  unsafe.Pointer
	)
	fmt.Println(Z_bool_false, Z_bool_true)
	fmt.Println(Z_int, Z_int8, Z_int16, Z_int32, Z_int64, Z_int_typedef)
	fmt.Println(Z_uint, Z_uint8, Z_uint16, Z_uint32, Z_uint64, Z_uintptr)
	fmt.Println(Z_float32, Z_float64, Z_complex64, Z_complex128)
	fmt.Println(Z_array, Z_array_empty, Z_array_of_empties)
	fmt.Println(Z_channel, Z_channel_buffered, Z_channel_nil)
	fmt.Println(Z_func_bar, Z_func_int8_r_int8, Z_func_int8_r_pint8, Z_func_nil)
	fmt.Println(Z_interface, Z_interface_nil, Z_interface_typed_nil)
	fmt.Println(Z_map, Z_map_2, Z_map_3, Z_map_empty, Z_map_nil)
	fmt.Println(Z_pointer, Z_pointer_nil)
	fmt.Println(Z_slice, Z_slice_2, Z_slice_nil)
	fmt.Println(Z_string, Z_struct)
	fmt.Println(Z_unsafe_pointer, Z_unsafe_pointer_nil)
	fmt.Println(local_bool_false, local_bool_true)
	fmt.Println(local_int, local_int8, local_int16, local_int32, local_int64, local_int_typedef)
	fmt.Println(local_uint, local_uint8, local_uint16, local_uint32, local_uint64, local_uintptr)
	fmt.Println(local_float32, local_float64, local_complex64, local_complex128, local_array)
	fmt.Println(local_channel, local_channel_buffered, local_channel_nil)
	fmt.Println(local_func_bar, local_func_int8_r_int8, local_func_int8_r_pint8, local_func_nil)
	fmt.Println(local_interface, local_interface_nil, local_interface_typed_nil)
	fmt.Println(local_map, local_map_2, local_map_3, local_map_empty, local_map_nil)
	fmt.Println(local_pointer, local_pointer_nil)
	fmt.Println(local_slice, local_slice_2, local_slice_nil)
	fmt.Println(local_string, local_struct)
	fmt.Println(local_unsafe_pointer, local_unsafe_pointer_nil)
	f1()
	f2()
}

func f1() {
	fmt.Println()
}

func f2() {
	fmt.Println()
}

func bar() {
	foo(42, 1.5)
	fmt.Print()
}

func populateChannels() {
	go func() {
		Z_channel_2 <- 8
	}()
	go func() {
		for i := int16(0); i < 14; i++ {
			Z_channel_buffered <- i
		}
	}()
	go func() {
		for i := 0; i < 8; i++ {
			<-Z_channel_buffered
		}
	}()
	time.Sleep(time.Second / 20)
}

func main() {
	args := os.Args[1:]
	expected := []string{"some", "arguments"}
	if len(args) != 2 || args[0] != expected[0] || args[1] != expected[1] {
		log.Fatalf("got command-line args %v, expected %v", args, expected)
	}
	populateChannels()
	for ; ; time.Sleep(2 * time.Second) {
		bar()
	}
	select {}
}
