/*
Copyright 2014 Google Inc. All rights reserved.

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
	"testing"

	"speter.net/go/exp/math/dec/inf"
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

func TestQuantityParse(t *testing.T) {
	table := []struct {
		input  string
		expect Quantity
	}{
		{"0", Quantity{dec(0, 0), DecimalSI}},
		// Binary suffixes
		{"9i", Quantity{dec(9, 0), BinarySI}},
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
		{"0.0005", Quantity{dec(5, -4), DecimalSI}},
		{"0.005", Quantity{dec(5, -3), DecimalSI}},
		{"0.05", Quantity{dec(5, -2), DecimalSI}},
		{"0.5", Quantity{dec(5, -1), DecimalSI}},
		{"0.00050", Quantity{dec(5, -4), DecimalSI}},
		{"0.00500", Quantity{dec(5, -3), DecimalSI}},
		{"0.05000", Quantity{dec(5, -2), DecimalSI}},
		{"0.50000", Quantity{dec(5, -1), DecimalSI}},
		{"0.5e-3", Quantity{dec(5, -4), DecimalExponent}},
		{"0.5e-2", Quantity{dec(5, -3), DecimalExponent}},
		{"0.5e-1", Quantity{dec(5, -2), DecimalExponent}},
		{"0.5e0", Quantity{dec(5, -1), DecimalExponent}},
		{"10.035M", Quantity{dec(10035, 3), DecimalSI}},

		{"1.1E-3", Quantity{dec(11, -4), DecimalExponent}},
		{"1.2e3", Quantity{dec(12, 2), DecimalExponent}},
		{"1.3E6", Quantity{dec(13, 5), DecimalExponent}},
		{"1.40e9", Quantity{dec(14, 8), DecimalExponent}},
		{"1.53E12", Quantity{dec(153, 10), DecimalExponent}},
		{"1.6e15", Quantity{dec(16, 14), DecimalExponent}},
		{"1.7E18", Quantity{dec(17, 17), DecimalExponent}},

		{"3.001m", Quantity{dec(3001, -6), DecimalSI}},
		{"9.01", Quantity{dec(901, -2), DecimalSI}},
		{"8.1k", Quantity{dec(81, 2), DecimalSI}},
		{"7.123456M", Quantity{dec(7123456, 0), DecimalSI}},
		{"6.987654321G", Quantity{dec(6987654321, 0), DecimalSI}},
		{"5.444T", Quantity{dec(5444, 9), DecimalSI}},
		{"40.1T", Quantity{dec(401, 11), DecimalSI}},
		{"300.2T", Quantity{dec(3002, 11), DecimalSI}},
		{"2.5P", Quantity{dec(25, 14), DecimalSI}},
		{"1.01E", Quantity{dec(101, 16), DecimalSI}},

		// Things that saturate
		//{"0.1m",
		//{"9Ei",
		//{"9223372036854775807Ki",
		//{"12E",
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

	invalid := []string{
		"1.1.M",
		"1+1.0M",
		"0.1mi",
		"0.1am",
		"0.0001i",
		"100.035Ki",
		"0.5Mi",
		"0.05Gi",
		"0.5Ti",
		"0.005i",
		"0.05i",
		"0.5i",
		"aoeu",
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
		{Quantity{dec(5, 0), BinarySI}, "5i"},
		{Quantity{dec(1, 9), DecimalSI}, "1G"},
		{Quantity{dec(1000, 6), DecimalSI}, "1G"},
		{Quantity{dec(1000000, 3), DecimalSI}, "1G"},
		{Quantity{dec(1000000000, 0), DecimalSI}, "1G"},
		{Quantity{dec(1, -3), DecimalSI}, "1m"},
		{Quantity{dec(80, -3), DecimalSI}, "80m"},
		{Quantity{dec(1080, -3), DecimalSI}, "1.080"},
		{Quantity{dec(108, -2), DecimalSI}, "1.080"},
		{Quantity{dec(10800, -4), DecimalSI}, "1.080"},
		{Quantity{dec(300, 6), DecimalSI}, "300M"},
		{Quantity{dec(1, 12), DecimalSI}, "1T"},
		{Quantity{dec(3, 3), DecimalSI}, "3k"},
		{Quantity{dec(0, 0), DecimalSI}, "0"},
		{Quantity{dec(0, 0), BinarySI}, "0i"},
		{Quantity{dec(1, 9), DecimalExponent}, "1e9"},
		{Quantity{dec(1, -3), DecimalExponent}, "1e-3"},
		{Quantity{dec(80, -3), DecimalExponent}, "80e-3"},
		{Quantity{dec(300, 6), DecimalExponent}, "300e6"},
		{Quantity{dec(1, 12), DecimalExponent}, "1e12"},
		{Quantity{dec(1, 3), DecimalExponent}, "1e3"},
		{Quantity{dec(3, 3), DecimalExponent}, "3e3"},
		{Quantity{dec(3, 3), DecimalSI}, "3k"},
		{Quantity{dec(0, 0), DecimalExponent}, "0"},

		{Quantity{dec(-1080, -3), DecimalSI}, "-1.080"},
		{Quantity{dec(-80*1024, 0), BinarySI}, "-80Ki"},
	}
	for _, item := range table {
		got := item.in.String()
		if e, a := item.expect, got; e != a {
			t.Errorf("%#v: expected %v, got %v", item.in, e, a)
		}
	}
}
