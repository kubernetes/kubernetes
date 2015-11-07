/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package resource

import (
	//"reflect"
	"encoding/json"
	"testing"

	fuzz "github.com/google/gofuzz"
	"github.com/spf13/pflag"
	"speter.net/go/exp/math/dec/inf"
)

var (
	testQuantityFlag = QuantityFlag("quantityFlag", "1M", "dummy flag for testing the quantity flag mechanism")
)

func dec(i int64, exponent int) *inf.Dec {
	// See the below test-- scale is the negative of an exponent.
	return inf.NewDec(i, inf.Scale(-exponent))
}

func TestDec(t *testing.T) {
	table := []struct {
		got    *inf.Dec
		expect string
	}{
		{dec(1, 0), "1"},
		{dec(1, 1), "10"},
		{dec(5, 2), "500"},
		{dec(8, 3), "8000"},
		{dec(2, 0), "2"},
		{dec(1, -1), "0.1"},
		{dec(3, -2), "0.03"},
		{dec(4, -3), "0.004"},
	}

	for _, item := range table {
		if e, a := item.expect, item.got.String(); e != a {
			t.Errorf("expected %v, got %v", e, a)
		}
	}
}

// TestQuantityParseZero ensures that when a 0 quantity is passed, its string value is 0
func TestQuantityParseZero(t *testing.T) {
	zero := MustParse("0")
	if expected, actual := "0", zero.String(); expected != actual {
		t.Errorf("Expected %v, actual %v", expected, actual)
	}
}

// Verifies that you get 0 as canonical value if internal value is 0, and not 0<suffix>
func TestQuantityCanocicalizeZero(t *testing.T) {
	val := MustParse("1000m")
	x := val.Amount
	y := dec(1, 0)
	z := val.Amount.Sub(x, y)
	zero := Quantity{z, DecimalSI}
	if expected, actual := "0", zero.String(); expected != actual {
		t.Errorf("Expected %v, actual %v", expected, actual)
	}
}

func TestQuantityCmp(t *testing.T) {
	table := []struct {
		x      string
		y      string
		expect int
	}{
		{"0", "0", 0},
		{"100m", "50m", 1},
		{"50m", "100m", -1},
		{"10000T", "100Gi", 1},
	}
	for _, testCase := range table {
		q1 := MustParse(testCase.x)
		q2 := MustParse(testCase.y)
		if result := q1.Cmp(q2); result != testCase.expect {
			t.Errorf("X: %v, Y: %v, Expected: %v, Actual: %v", testCase.x, testCase.y, testCase.expect, result)
		}
	}
}

func TestQuantityParse(t *testing.T) {
	table := []struct {
		input  string
		expect Quantity
	}{
		{"0", Quantity{dec(0, 0), DecimalSI}},
		{"0m", Quantity{dec(0, 0), DecimalSI}},
		{"0Ki", Quantity{dec(0, 0), BinarySI}},
		{"0k", Quantity{dec(0, 0), DecimalSI}},
		{"0Mi", Quantity{dec(0, 0), BinarySI}},
		{"0M", Quantity{dec(0, 0), DecimalSI}},
		{"0Gi", Quantity{dec(0, 0), BinarySI}},
		{"0G", Quantity{dec(0, 0), DecimalSI}},
		{"0Ti", Quantity{dec(0, 0), BinarySI}},
		{"0T", Quantity{dec(0, 0), DecimalSI}},

		// Binary suffixes
		{"1Ki", Quantity{dec(1024, 0), BinarySI}},
		{"8Ki", Quantity{dec(8*1024, 0), BinarySI}},
		{"7Mi", Quantity{dec(7*1024*1024, 0), BinarySI}},
		{"6Gi", Quantity{dec(6*1024*1024*1024, 0), BinarySI}},
		{"5Ti", Quantity{dec(5*1024*1024*1024*1024, 0), BinarySI}},
		{"4Pi", Quantity{dec(4*1024*1024*1024*1024*1024, 0), BinarySI}},
		{"3Ei", Quantity{dec(3*1024*1024*1024*1024*1024*1024, 0), BinarySI}},

		{"10Ti", Quantity{dec(10*1024*1024*1024*1024, 0), BinarySI}},
		{"100Ti", Quantity{dec(100*1024*1024*1024*1024, 0), BinarySI}},

		// Decimal suffixes
		{"3m", Quantity{dec(3, -3), DecimalSI}},
		{"9", Quantity{dec(9, 0), DecimalSI}},
		{"8k", Quantity{dec(8, 3), DecimalSI}},
		{"7M", Quantity{dec(7, 6), DecimalSI}},
		{"6G", Quantity{dec(6, 9), DecimalSI}},
		{"5T", Quantity{dec(5, 12), DecimalSI}},
		{"40T", Quantity{dec(4, 13), DecimalSI}},
		{"300T", Quantity{dec(3, 14), DecimalSI}},
		{"2P", Quantity{dec(2, 15), DecimalSI}},
		{"1E", Quantity{dec(1, 18), DecimalSI}},

		// Decimal exponents
		{"1E-3", Quantity{dec(1, -3), DecimalExponent}},
		{"1e3", Quantity{dec(1, 3), DecimalExponent}},
		{"1E6", Quantity{dec(1, 6), DecimalExponent}},
		{"1e9", Quantity{dec(1, 9), DecimalExponent}},
		{"1E12", Quantity{dec(1, 12), DecimalExponent}},
		{"1e15", Quantity{dec(1, 15), DecimalExponent}},
		{"1E18", Quantity{dec(1, 18), DecimalExponent}},

		// Nonstandard but still parsable
		{"1e14", Quantity{dec(1, 14), DecimalExponent}},
		{"1e13", Quantity{dec(1, 13), DecimalExponent}},
		{"1e3", Quantity{dec(1, 3), DecimalExponent}},
		{"100.035k", Quantity{dec(100035, 0), DecimalSI}},

		// Things that look like floating point
		{"0.001", Quantity{dec(1, -3), DecimalSI}},
		{"0.0005k", Quantity{dec(5, -1), DecimalSI}},
		{"0.005", Quantity{dec(5, -3), DecimalSI}},
		{"0.05", Quantity{dec(5, -2), DecimalSI}},
		{"0.5", Quantity{dec(5, -1), DecimalSI}},
		{"0.00050k", Quantity{dec(5, -1), DecimalSI}},
		{"0.00500", Quantity{dec(5, -3), DecimalSI}},
		{"0.05000", Quantity{dec(5, -2), DecimalSI}},
		{"0.50000", Quantity{dec(5, -1), DecimalSI}},
		{"0.5e0", Quantity{dec(5, -1), DecimalExponent}},
		{"0.5e-1", Quantity{dec(5, -2), DecimalExponent}},
		{"0.5e-2", Quantity{dec(5, -3), DecimalExponent}},
		{"0.5e0", Quantity{dec(5, -1), DecimalExponent}},
		{"10.035M", Quantity{dec(10035, 3), DecimalSI}},

		{"1.2e3", Quantity{dec(12, 2), DecimalExponent}},
		{"1.3E+6", Quantity{dec(13, 5), DecimalExponent}},
		{"1.40e9", Quantity{dec(14, 8), DecimalExponent}},
		{"1.53E12", Quantity{dec(153, 10), DecimalExponent}},
		{"1.6e15", Quantity{dec(16, 14), DecimalExponent}},
		{"1.7E18", Quantity{dec(17, 17), DecimalExponent}},

		{"9.01", Quantity{dec(901, -2), DecimalSI}},
		{"8.1k", Quantity{dec(81, 2), DecimalSI}},
		{"7.123456M", Quantity{dec(7123456, 0), DecimalSI}},
		{"6.987654321G", Quantity{dec(6987654321, 0), DecimalSI}},
		{"5.444T", Quantity{dec(5444, 9), DecimalSI}},
		{"40.1T", Quantity{dec(401, 11), DecimalSI}},
		{"300.2T", Quantity{dec(3002, 11), DecimalSI}},
		{"2.5P", Quantity{dec(25, 14), DecimalSI}},
		{"1.01E", Quantity{dec(101, 16), DecimalSI}},

		// Things that saturate/round
		{"3.001m", Quantity{dec(4, -3), DecimalSI}},
		{"1.1E-3", Quantity{dec(2, -3), DecimalExponent}},
		{"0.0001", Quantity{dec(1, -3), DecimalSI}},
		{"0.0005", Quantity{dec(1, -3), DecimalSI}},
		{"0.00050", Quantity{dec(1, -3), DecimalSI}},
		{"0.5e-3", Quantity{dec(1, -3), DecimalExponent}},
		{"0.9m", Quantity{dec(1, -3), DecimalSI}},
		{"0.12345", Quantity{dec(124, -3), DecimalSI}},
		{"0.12354", Quantity{dec(124, -3), DecimalSI}},
		{"9Ei", Quantity{maxAllowed, BinarySI}},
		{"9223372036854775807Ki", Quantity{maxAllowed, BinarySI}},
		{"12E", Quantity{maxAllowed, DecimalSI}},

		// We'll accept fractional binary stuff, too.
		{"100.035Ki", Quantity{dec(10243584, -2), BinarySI}},
		{"0.5Mi", Quantity{dec(.5*1024*1024, 0), BinarySI}},
		{"0.05Gi", Quantity{dec(536870912, -1), BinarySI}},
		{"0.025Ti", Quantity{dec(274877906944, -1), BinarySI}},

		// Things written by trolls
		{"0.000001Ki", Quantity{dec(2, -3), DecimalSI}}, // rounds up, changes format
		{".001", Quantity{dec(1, -3), DecimalSI}},
		{".0001k", Quantity{dec(100, -3), DecimalSI}},
		{"1.", Quantity{dec(1, 0), DecimalSI}},
		{"1.G", Quantity{dec(1, 9), DecimalSI}},
	}

	for _, item := range table {
		got, err := ParseQuantity(item.input)
		if err != nil {
			t.Errorf("%v: unexpected error: %v", item.input, err)
			continue
		}
		if e, a := item.expect.Amount, got.Amount; e.Cmp(a) != 0 {
			t.Errorf("%v: expected %v, got %v", item.input, e, a)
		}
		if e, a := item.expect.Format, got.Format; e != a {
			t.Errorf("%v: expected %#v, got %#v", item.input, e, a)
		}
	}

	// Try the negative version of everything
	desired := &inf.Dec{}
	for _, item := range table {
		got, err := ParseQuantity("-" + item.input)
		if err != nil {
			t.Errorf("-%v: unexpected error: %v", item.input, err)
			continue
		}
		desired.Neg(item.expect.Amount)
		if e, a := desired, got.Amount; e.Cmp(a) != 0 {
			t.Errorf("%v: expected %v, got %v", item.input, e, a)
		}
		if e, a := item.expect.Format, got.Format; e != a {
			t.Errorf("%v: expected %#v, got %#v", item.input, e, a)
		}
	}

	// Try everything with an explicit +
	for _, item := range table {
		got, err := ParseQuantity("+" + item.input)
		if err != nil {
			t.Errorf("-%v: unexpected error: %v", item.input, err)
			continue
		}
		if e, a := item.expect.Amount, got.Amount; e.Cmp(a) != 0 {
			t.Errorf("%v: expected %v, got %v", item.input, e, a)
		}
		if e, a := item.expect.Format, got.Format; e != a {
			t.Errorf("%v: expected %#v, got %#v", item.input, e, a)
		}
	}

	invalid := []string{
		"1.1.M",
		"1+1.0M",
		"0.1mi",
		"0.1am",
		"aoeu",
		".5i",
		"1i",
		"-3.01i",
	}
	for _, item := range invalid {
		_, err := ParseQuantity(item)
		if err == nil {
			t.Errorf("%v parsed unexpectedly", item)
		}
	}
}

func TestQuantityString(t *testing.T) {
	table := []struct {
		in     Quantity
		expect string
	}{
		{Quantity{dec(1024*1024*1024, 0), BinarySI}, "1Gi"},
		{Quantity{dec(300*1024*1024, 0), BinarySI}, "300Mi"},
		{Quantity{dec(6*1024, 0), BinarySI}, "6Ki"},
		{Quantity{dec(1001*1024*1024*1024, 0), BinarySI}, "1001Gi"},
		{Quantity{dec(1024*1024*1024*1024, 0), BinarySI}, "1Ti"},
		{Quantity{dec(5, 0), BinarySI}, "5"},
		{Quantity{dec(500, -3), BinarySI}, "500m"},
		{Quantity{dec(1, 9), DecimalSI}, "1G"},
		{Quantity{dec(1000, 6), DecimalSI}, "1G"},
		{Quantity{dec(1000000, 3), DecimalSI}, "1G"},
		{Quantity{dec(1000000000, 0), DecimalSI}, "1G"},
		{Quantity{dec(1, -3), DecimalSI}, "1m"},
		{Quantity{dec(80, -3), DecimalSI}, "80m"},
		{Quantity{dec(1080, -3), DecimalSI}, "1080m"},
		{Quantity{dec(108, -2), DecimalSI}, "1080m"},
		{Quantity{dec(10800, -4), DecimalSI}, "1080m"},
		{Quantity{dec(300, 6), DecimalSI}, "300M"},
		{Quantity{dec(1, 12), DecimalSI}, "1T"},
		{Quantity{dec(1234567, 6), DecimalSI}, "1234567M"},
		{Quantity{dec(1234567, -3), BinarySI}, "1234567m"},
		{Quantity{dec(3, 3), DecimalSI}, "3k"},
		{Quantity{dec(1025, 0), BinarySI}, "1025"},
		{Quantity{dec(0, 0), DecimalSI}, "0"},
		{Quantity{dec(0, 0), BinarySI}, "0"},
		{Quantity{dec(1, 9), DecimalExponent}, "1e9"},
		{Quantity{dec(1, -3), DecimalExponent}, "1e-3"},
		{Quantity{dec(80, -3), DecimalExponent}, "80e-3"},
		{Quantity{dec(300, 6), DecimalExponent}, "300e6"},
		{Quantity{dec(1, 12), DecimalExponent}, "1e12"},
		{Quantity{dec(1, 3), DecimalExponent}, "1e3"},
		{Quantity{dec(3, 3), DecimalExponent}, "3e3"},
		{Quantity{dec(3, 3), DecimalSI}, "3k"},
		{Quantity{dec(0, 0), DecimalExponent}, "0"},
	}
	for _, item := range table {
		got := item.in.String()
		if e, a := item.expect, got; e != a {
			t.Errorf("%#v: expected %v, got %v", item.in, e, a)
		}
	}
	desired := &inf.Dec{} // Avoid modifying the values in the table.
	for _, item := range table {
		if item.in.Amount.Cmp(decZero) == 0 {
			// Don't expect it to print "-0" ever
			continue
		}
		q := item.in
		q.Amount = desired.Neg(q.Amount)
		if e, a := "-"+item.expect, q.String(); e != a {
			t.Errorf("%#v: expected %v, got %v", item.in, e, a)
		}
	}
}

func TestQuantityParseEmit(t *testing.T) {
	table := []struct {
		in     string
		expect string
	}{
		{"1Ki", "1Ki"},
		{"1Mi", "1Mi"},
		{"1Gi", "1Gi"},
		{"1024Mi", "1Gi"},
		{"1000M", "1G"},
		{".000001Ki", "2m"},
	}

	for _, item := range table {
		q, err := ParseQuantity(item.in)
		if err != nil {
			t.Errorf("Couldn't parse %v", item.in)
			continue
		}
		if e, a := item.expect, q.String(); e != a {
			t.Errorf("%#v: expected %v, got %v", item.in, e, a)
		}
	}
	for _, item := range table {
		q, err := ParseQuantity("-" + item.in)
		if err != nil {
			t.Errorf("Couldn't parse %v", item.in)
			continue
		}
		if q.Amount.Cmp(decZero) == 0 {
			continue
		}
		if e, a := "-"+item.expect, q.String(); e != a {
			t.Errorf("%#v: expected %v, got %v", item.in, e, a)
		}
	}
}

var fuzzer = fuzz.New().Funcs(
	func(q *Quantity, c fuzz.Continue) {
		q.Amount = &inf.Dec{}
		if c.RandBool() {
			q.Format = BinarySI
			if c.RandBool() {
				q.Amount.SetScale(0)
				q.Amount.SetUnscaled(c.Int63())
				return
			}
			// Be sure to test cases like 1Mi
			q.Amount.SetScale(0)
			q.Amount.SetUnscaled(c.Int63n(1024) << uint(10*c.Intn(5)))
			return
		}
		if c.RandBool() {
			q.Format = DecimalSI
		} else {
			q.Format = DecimalExponent
		}
		if c.RandBool() {
			q.Amount.SetScale(inf.Scale(c.Intn(4)))
			q.Amount.SetUnscaled(c.Int63())
			return
		}
		// Be sure to test cases like 1M
		q.Amount.SetScale(inf.Scale(3 - c.Intn(15)))
		q.Amount.SetUnscaled(c.Int63n(1000))
	},
)

func TestJSON(t *testing.T) {
	for i := 0; i < 500; i++ {
		q := &Quantity{}
		fuzzer.Fuzz(q)
		b, err := json.Marshal(q)
		if err != nil {
			t.Errorf("error encoding %v", q)
		}
		q2 := &Quantity{}
		err = json.Unmarshal(b, q2)
		if err != nil {
			t.Errorf("%v: error decoding %v", q, string(b))
		}
		if q2.Amount.Cmp(q.Amount) != 0 {
			t.Errorf("Expected equal: %v, %v (json was '%v')", q, q2, string(b))
		}
	}
}

func TestMilliNewSet(t *testing.T) {
	table := []struct {
		value  int64
		format Format
		expect string
		exact  bool
	}{
		{1, DecimalSI, "1m", true},
		{1000, DecimalSI, "1", true},
		{1234000, DecimalSI, "1234", true},
		{1024, BinarySI, "1024m", false}, // Format changes
		{1000000, "invalidFormatDefaultsToExponent", "1e3", true},
		{1024 * 1024, BinarySI, "1048576m", false}, // Format changes
	}

	for _, item := range table {
		q := NewMilliQuantity(item.value, item.format)
		if e, a := item.expect, q.String(); e != a {
			t.Errorf("Expected %v, got %v; %#v", e, a, q)
		}
		if !item.exact {
			continue
		}
		q2, err := ParseQuantity(q.String())
		if err != nil {
			t.Errorf("Round trip failed on %v", q)
		}
		if e, a := item.value, q2.MilliValue(); e != a {
			t.Errorf("Expected %v, got %v", e, a)
		}
	}

	for _, item := range table {
		q := NewQuantity(0, item.format)
		q.SetMilli(item.value)
		if e, a := item.expect, q.String(); e != a {
			t.Errorf("Set: Expected %v, got %v; %#v", e, a, q)
		}
	}
}

func TestNewSet(t *testing.T) {
	table := []struct {
		value  int64
		format Format
		expect string
	}{
		{1, DecimalSI, "1"},
		{1000, DecimalSI, "1k"},
		{1234000, DecimalSI, "1234k"},
		{1024, BinarySI, "1Ki"},
		{1000000, "invalidFormatDefaultsToExponent", "1e6"},
		{1024 * 1024, BinarySI, "1Mi"},
	}

	for _, item := range table {
		q := NewQuantity(item.value, item.format)
		if e, a := item.expect, q.String(); e != a {
			t.Errorf("Expected %v, got %v; %#v", e, a, q)
		}
		q2, err := ParseQuantity(q.String())
		if err != nil {
			t.Errorf("Round trip failed on %v", q)
		}
		if e, a := item.value, q2.Value(); e != a {
			t.Errorf("Expected %v, got %v", e, a)
		}
	}

	for _, item := range table {
		q := NewQuantity(0, item.format)
		q.Set(item.value)
		if e, a := item.expect, q.String(); e != a {
			t.Errorf("Set: Expected %v, got %v; %#v", e, a, q)
		}
	}
}

func TestUninitializedNoCrash(t *testing.T) {
	var q Quantity

	q.Value()
	q.MilliValue()
	q.Copy()
	_ = q.String()
	q.MarshalJSON()
}

func TestCopy(t *testing.T) {
	q := NewQuantity(5, DecimalSI)
	c := q.Copy()
	c.Set(6)
	if q.Value() == 6 {
		t.Errorf("Copy didn't")
	}
}

func TestQFlagSet(t *testing.T) {
	qf := qFlag{&Quantity{}}
	qf.Set("1Ki")
	if e, a := "1Ki", qf.String(); e != a {
		t.Errorf("Unexpected result %v != %v", e, a)
	}
}

func TestQFlagIsPFlag(t *testing.T) {
	var pfv pflag.Value = qFlag{}
	if e, a := "quantity", pfv.Type(); e != a {
		t.Errorf("Unexpected result %v != %v", e, a)
	}
}
