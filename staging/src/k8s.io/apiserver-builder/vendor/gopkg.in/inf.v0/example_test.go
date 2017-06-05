package inf_test

import (
	"fmt"
	"log"
)

import "gopkg.in/inf.v0"

func ExampleDec_SetString() {
	d := new(inf.Dec)
	d.SetString("012345.67890") // decimal; leading 0 ignored; trailing 0 kept
	fmt.Println(d)
	// Output: 12345.67890
}

func ExampleDec_Scan() {
	// The Scan function is rarely used directly;
	// the fmt package recognizes it as an implementation of fmt.Scanner.
	d := new(inf.Dec)
	_, err := fmt.Sscan("184467440.73709551617", d)
	if err != nil {
		log.Println("error scanning value:", err)
	} else {
		fmt.Println(d)
	}
	// Output: 184467440.73709551617
}

func ExampleDec_QuoRound_scale2RoundDown() {
	// 10 / 3 is an infinite decimal; it has no exact Dec representation
	x, y := inf.NewDec(10, 0), inf.NewDec(3, 0)
	// use 2 digits beyond the decimal point, round towards 0
	z := new(inf.Dec).QuoRound(x, y, 2, inf.RoundDown)
	fmt.Println(z)
	// Output: 3.33
}

func ExampleDec_QuoRound_scale2RoundCeil() {
	// -42 / 400 is an finite decimal with 3 digits beyond the decimal point
	x, y := inf.NewDec(-42, 0), inf.NewDec(400, 0)
	// use 2 digits beyond decimal point, round towards positive infinity
	z := new(inf.Dec).QuoRound(x, y, 2, inf.RoundCeil)
	fmt.Println(z)
	// Output: -0.10
}

func ExampleDec_QuoExact_ok() {
	// 1 / 25 is a finite decimal; it has exact Dec representation
	x, y := inf.NewDec(1, 0), inf.NewDec(25, 0)
	z := new(inf.Dec).QuoExact(x, y)
	fmt.Println(z)
	// Output: 0.04
}

func ExampleDec_QuoExact_fail() {
	// 1 / 3 is an infinite decimal; it has no exact Dec representation
	x, y := inf.NewDec(1, 0), inf.NewDec(3, 0)
	z := new(inf.Dec).QuoExact(x, y)
	fmt.Println(z)
	// Output: <nil>
}
