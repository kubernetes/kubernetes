//+build prod

//
// Copyright (c) 2014 The godbc Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package godbc

import (
	"fmt"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestPrintDbcSetup(t *testing.T) {
	fmt.Println("Prod Dbc")
}

func TestRequire(t *testing.T) {
	a, b := 0, 1
	assert.NotPanics(t, func() {
		Require(a == b)
	})

	assert.NotPanics(t, func() {
		Require(a != b)
	})

	assert.NotPanics(t, func() {
		Require(a > 0)
	})
}

func TestEnsure(t *testing.T) {
	a, b := 0, 1
	assert.NotPanics(t, func() {
		Ensure(a == b)
	})

	assert.NotPanics(t, func() {
		Ensure(a != b)
	})

	assert.NotPanics(t, func() {
		Ensure(a > 0)
	})
}

func TestCheck(t *testing.T) {
	a, b := 0, 1
	assert.NotPanics(t, func() {
		Check(a == b)
	})

	assert.NotPanics(t, func() {
		Check(a != b)
	})

	assert.NotPanics(t, func() {
		Check(a > 0)
	})
}

// Date
type Date struct {
	day, month int
}

func (d *Date) Invariant() bool {
	if (1 <= d.day && d.day <= 31) &&
		(1 <= d.month && d.month <= 12) {
		return true
	}
	return false
}

func (d *Date) Set(day, month int) {
	d.day, d.month = day, month
}

func (d *Date) String() string {
	return fmt.Sprintf("Day:%d Month:%d",
		d.day, d.month)
}

func TestInvariant(t *testing.T) {
	d := &Date{0, 0}
	assert.NotPanics(t, func() {
		Invariant(d)
	})

	d.Set(1, 0)
	assert.NotPanics(t, func() {
		Invariant(d)
	})

	d.Set(0, 1)
	assert.NotPanics(t, func() {
		Invariant(d)
	})

	d.Set(1, 1)
	assert.NotPanics(t, func() {
		Invariant(d)
	})
}

// Time does not have a String() receiver
type Time struct {
	hour, min, sec int
}

func (t *Time) Invariant() bool {
	if (1 <= t.hour && t.hour <= 23) &&
		(1 <= t.min && t.min <= 59) &&
		(1 <= t.sec && t.sec <= 59) {
		return true
	}
	return false
}

func (t *Time) Set(hour, min, sec int) {
	t.hour, t.min, t.sec = hour, min, sec
}

func TestInvariantSimple(t *testing.T) {
	time := &Time{0, 0, 0}
	assert.NotPanics(t, func() {
		InvariantSimple(time)
	})

	time.Set(1, 0, 0)
	assert.NotPanics(t, func() {
		InvariantSimple(time)
	})

	time.Set(0, 1, 0)
	assert.NotPanics(t, func() {
		InvariantSimple(time)
	})

	time.Set(1, 1, 1)
	assert.NotPanics(t, func() {
		InvariantSimple(time)
	})
}
