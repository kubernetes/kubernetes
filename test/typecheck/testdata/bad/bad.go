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

var i int = int16(0)
var pi *int = new(int16)
var i16 int16 = int(0)
var pi16 *int16 = new(int)
var i32 int32 = int64(0)
var pi32 *int32 = new(int64)
var i64 int64 = int32(0)
var pi64 *int64 = new(int32)

var f32 float32 = float64(0.0)
var pf32 *float32 = new(float64)
var f64 float64 = float32(0.0)
var pf64 *float64 = new(float32)

var str string = false
var pstr *string = new(bool)

type struc struct {
	i int
	f float64
	s string
}

var stru struc = &struc{}
var pstru *struc = struc{}

var sli []int = map[string]int{"zero": 0}
var ma map[string]int = []int{0}

func main() {
	fmt.Println("hello, world!")
}
