// Copyright 2015 CoreOS, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package unit

import (
	"testing"
)

func TestAllMatch(t *testing.T) {
	tests := []struct {
		u1    []*UnitOption
		u2    []*UnitOption
		match bool
	}{
		// empty lists match
		{
			u1:    []*UnitOption{},
			u2:    []*UnitOption{},
			match: true,
		},

		// simple match of a single option
		{
			u1: []*UnitOption{
				{Section: "Unit", Name: "Description", Value: "FOO"},
			},
			u2: []*UnitOption{
				{Section: "Unit", Name: "Description", Value: "FOO"},
			},
			match: true,
		},

		// single option mismatched
		{
			u1: []*UnitOption{
				{Section: "Unit", Name: "Description", Value: "FOO"},
			},
			u2: []*UnitOption{
				{Section: "Unit", Name: "Description", Value: "BAR"},
			},
			match: false,
		},

		// multiple options match
		{
			u1: []*UnitOption{
				{Section: "Unit", Name: "Description", Value: "FOO"},
				{Section: "Unit", Name: "BindsTo", Value: "bar.service"},
				{Section: "Service", Name: "ExecStart", Value: "/bin/true"},
			},
			u2: []*UnitOption{
				{Section: "Unit", Name: "Description", Value: "FOO"},
				{Section: "Unit", Name: "BindsTo", Value: "bar.service"},
				{Section: "Service", Name: "ExecStart", Value: "/bin/true"},
			},
			match: true,
		},

		// mismatch length
		{
			u1: []*UnitOption{
				{Section: "Unit", Name: "Description", Value: "FOO"},
				{Section: "Unit", Name: "BindsTo", Value: "bar.service"},
			},
			u2: []*UnitOption{
				{Section: "Unit", Name: "Description", Value: "FOO"},
				{Section: "Unit", Name: "BindsTo", Value: "bar.service"},
				{Section: "Service", Name: "ExecStart", Value: "/bin/true"},
			},
			match: false,
		},

		// multiple options misordered
		{
			u1: []*UnitOption{
				{Section: "Unit", Name: "Description", Value: "FOO"},
				{Section: "Service", Name: "ExecStart", Value: "/bin/true"},
			},
			u2: []*UnitOption{
				{Section: "Service", Name: "ExecStart", Value: "/bin/true"},
				{Section: "Unit", Name: "Description", Value: "FOO"},
			},
			match: false,
		},

		// interleaved sections mismatch
		{
			u1: []*UnitOption{
				{Section: "Unit", Name: "Description", Value: "FOO"},
				{Section: "Unit", Name: "BindsTo", Value: "bar.service"},
				{Section: "Service", Name: "ExecStart", Value: "/bin/true"},
				{Section: "Service", Name: "ExecStop", Value: "/bin/true"},
			},
			u2: []*UnitOption{
				{Section: "Unit", Name: "Description", Value: "FOO"},
				{Section: "Service", Name: "ExecStart", Value: "/bin/true"},
				{Section: "Unit", Name: "BindsTo", Value: "bar.service"},
				{Section: "Service", Name: "ExecStop", Value: "/bin/true"},
			},
			match: false,
		},
	}

	for i, tt := range tests {
		match := AllMatch(tt.u1, tt.u2)
		if match != tt.match {
			t.Errorf("case %d: failed comparing u1 to u2 - expected match=%t, got %t", i, tt.match, match)
		}

		match = AllMatch(tt.u2, tt.u1)
		if match != tt.match {
			t.Errorf("case %d: failed comparing u2 to u1 - expected match=%t, got %t", i, tt.match, match)
		}
	}
}

func TestMatch(t *testing.T) {
	tests := []struct {
		o1    *UnitOption
		o2    *UnitOption
		match bool
	}{
		// empty options match
		{
			o1:    &UnitOption{},
			o2:    &UnitOption{},
			match: true,
		},

		// all fields match
		{
			o1: &UnitOption{
				Section: "Unit",
				Name:    "Description",
				Value:   "FOO",
			},
			o2: &UnitOption{
				Section: "Unit",
				Name:    "Description",
				Value:   "FOO",
			},
			match: true,
		},

		// Section mismatch
		{
			o1: &UnitOption{
				Section: "Unit",
				Name:    "Description",
				Value:   "FOO",
			},
			o2: &UnitOption{
				Section: "X-Other",
				Name:    "Description",
				Value:   "FOO",
			},
			match: false,
		},

		// Name mismatch
		{
			o1: &UnitOption{
				Section: "Unit",
				Name:    "Description",
				Value:   "FOO",
			},
			o2: &UnitOption{
				Section: "Unit",
				Name:    "BindsTo",
				Value:   "FOO",
			},
			match: false,
		},

		// Value mismatch
		{
			o1: &UnitOption{
				Section: "Unit",
				Name:    "Description",
				Value:   "FOO",
			},
			o2: &UnitOption{
				Section: "Unit",
				Name:    "Description",
				Value:   "BAR",
			},
			match: false,
		},
	}

	for i, tt := range tests {
		match := tt.o1.Match(tt.o2)
		if match != tt.match {
			t.Errorf("case %d: failed comparing o1 to o2 - expected match=%t, got %t", i, tt.match, match)
		}

		match = tt.o2.Match(tt.o1)
		if match != tt.match {
			t.Errorf("case %d: failed comparing o2 to o1 - expected match=%t, got %t", i, tt.match, match)
		}
	}
}
