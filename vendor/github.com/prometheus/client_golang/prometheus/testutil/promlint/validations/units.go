// Copyright 2020 The Prometheus Authors
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

package validations

import "strings"

// Units and their possible prefixes recognized by this library.  More can be
// added over time as needed.
var (
	// map a unit to the appropriate base unit.
	units = map[string]string{
		// Base units.
		"amperes": "amperes",
		"bytes":   "bytes",
		"celsius": "celsius", // Also allow Celsius because it is common in typical Prometheus use cases.
		"grams":   "grams",
		"joules":  "joules",
		"kelvin":  "kelvin", // SI base unit, used in special cases (e.g. color temperature, scientific measurements).
		"meters":  "meters", // Both American and international spelling permitted.
		"metres":  "metres",
		"seconds": "seconds",
		"volts":   "volts",

		// Non base units.
		// Time.
		"minutes": "seconds",
		"hours":   "seconds",
		"days":    "seconds",
		"weeks":   "seconds",
		// Temperature.
		"kelvins":    "kelvin",
		"fahrenheit": "celsius",
		"rankine":    "celsius",
		// Length.
		"inches": "meters",
		"yards":  "meters",
		"miles":  "meters",
		// Bytes.
		"bits": "bytes",
		// Energy.
		"calories": "joules",
		// Mass.
		"pounds": "grams",
		"ounces": "grams",
	}

	unitPrefixes = []string{
		"pico",
		"nano",
		"micro",
		"milli",
		"centi",
		"deci",
		"deca",
		"hecto",
		"kilo",
		"kibi",
		"mega",
		"mibi",
		"giga",
		"gibi",
		"tera",
		"tebi",
		"peta",
		"pebi",
	}

	// Common abbreviations that we'd like to discourage.
	unitAbbreviations = []string{
		"s",
		"ms",
		"us",
		"ns",
		"sec",
		"b",
		"kb",
		"mb",
		"gb",
		"tb",
		"pb",
		"m",
		"h",
		"d",
	}
)

// metricUnits attempts to detect known unit types used as part of a metric name,
// e.g. "foo_bytes_total" or "bar_baz_milligrams".
func metricUnits(m string) (unit, base string, ok bool) {
	ss := strings.Split(m, "_")

	for _, s := range ss {
		if base, found := units[s]; found {
			return s, base, true
		}

		for _, p := range unitPrefixes {
			if strings.HasPrefix(s, p) {
				if base, found := units[s[len(p):]]; found {
					return s, base, true
				}
			}
		}
	}

	return "", "", false
}
