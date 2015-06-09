/**
 *  Copyright 2014 Paul Querna
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 */

package tff

import (
	"testing"
)

// Test data from https://github.com/akheron/jansson/tree/master/test/suites/valid
// jansson, Copyright (c) 2009-2014 Petri Lehtinen <petri@digip.org>
// (MIT Licensed)

func TestFloatRealCapitalENegativeExponent(t *testing.T) {
	testExpectedXValBare(t,
		0.01,
		`1E-2`,
		&Xfloat64{})
}

func TestFloatRealCapitalEPositiveExponent(t *testing.T) {
	testExpectedXValBare(t,
		100.0,
		`1E+2`,
		&Xfloat64{})
}

func TestFloatRealCapital(t *testing.T) {
	testExpectedXValBare(t,
		1e22,
		`1E22`,
		&Xfloat64{})
}

func TestFloatRealExponent(t *testing.T) {
	testExpectedXValBare(t,
		1.2299999999999999e47,
		`123e45`,
		&Xfloat64{})
}

func TestFloatRealFractionExponent(t *testing.T) {
	testExpectedXValBare(t,
		1.23456e80,
		`123.456e78`,
		&Xfloat64{})
}

func TestFloatRealNegativeExponent(t *testing.T) {
	testExpectedXValBare(t,
		0.01,
		`1e-2`,
		&Xfloat64{})
}

func TestFloatRealPositiveExponent(t *testing.T) {
	testExpectedXValBare(t,
		100.0,
		`1e2`,
		&Xfloat64{})
}

func TestFloatRealSubnormalNumber(t *testing.T) {
	testExpectedXValBare(t,
		1.8011670033376514e-308,
		`1.8011670033376514e-308`,
		&Xfloat64{})
}

func TestFloatRealUnderflow(t *testing.T) {
	testExpectedXValBare(t,
		0.0,
		`123e-10000000`,
		&Xfloat64{})
}

func TestFloatNull(t *testing.T) {
	testExpectedXValBare(t,
		0.0,
		`null`,
		&Xfloat64{})
}

func TestFloatInt(t *testing.T) {
	testExpectedXValBare(t,
		1.0,
		`1`,
		&Xfloat64{})
}
