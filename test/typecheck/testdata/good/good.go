/*
Copyright 2020 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package main

import "fmt"

var (
	i    int    = int(0)
	pi   *int   = new(int)
	i16  int16  = int16(0)
	pi16 *int16 = new(int16)
	i32  int32  = int32(0)
	pi32 *int32 = new(int32)
	i64  int64  = int64(0)
	pi64 *int64 = new(int64)
)

var (
	f32  float32  = float32(0.0)
	pf32 *float32 = new(float32)
	f64  float64  = float64(0.0)
	pf64 *float64 = new(float64)
)

var (
	str  string  = "a string"
	pstr *string = new(string)
)

type struc struct {
	i int
	f float64
	s string
}

var (
	stru  struc  = struc{}
	pstru *struc = &struc{}
)

var (
	sli []int          = []int{0}
	ma  map[string]int = map[string]int{"zero": 0}
)

func main() {
	fmt.Println("hello, world!")
}
