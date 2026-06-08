/*
Copyright 2023 The Kubernetes Authors.

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

package knftables

import (
	"fmt"
	"strconv"
	"strings"
)

// PtrTo can be used to fill in optional field values in objects
func PtrTo[T any](val T) *T {
	return &val
}

var numericPriorities = map[string]int{
	"raw":      -300,
	"mangle":   -150,
	"dstnat":   -100,
	"filter":   0,
	"security": 50,
	"srcnat":   100,
}

var bridgeNumericPriorities = map[string]int{
	"dstnat": -300,
	"filter": -200,
	"out":    100,
	"srcnat": 300,
}

// ParsePriority tries to convert the string form of a chain priority into a number
func ParsePriority(family Family, priority string) (int, error) {
	val, err := strconv.Atoi(priority)
	if err == nil {
		return val, nil
	}

	modVal := 0
	if i := strings.IndexAny(priority, "+-"); i != -1 {
		mod := priority[i:]
		modVal, err = strconv.Atoi(mod)
		if err != nil {
			return 0, fmt.Errorf("could not parse modifier %q: %w", mod, err)
		}
		priority = priority[:i]
	}

	var found bool
	if family == BridgeFamily {
		val, found = bridgeNumericPriorities[priority]
	} else {
		val, found = numericPriorities[priority]
	}
	if !found {
		return 0, fmt.Errorf("unknown priority %q", priority)
	}

	return val + modVal, nil
}

// Concat is a helper (primarily) for constructing Rule objects. It takes a series of
// arguments and concatenates them together into a single string with spaces between the
// arguments. Strings are output as-is, string arrays are output element by element,
// numbers are output as with `fmt.Sprintf("%d")`, and all other types are output as with
// `fmt.Sprintf("%s")`. To help with set/map lookup syntax, an argument of "@" will not
// be followed by a space, so you can do, eg, `Concat("ip saddr", "@", setName)`.
func Concat(args ...interface{}) string {
	b := &strings.Builder{}
	var needSpace, wroteAt bool
	for _, arg := range args {
		switch x := arg.(type) {
		case string:
			if needSpace {
				b.WriteByte(' ')
			}
			b.WriteString(x)
			wroteAt = (x == "@")
		case []string:
			for _, s := range x {
				if needSpace {
					b.WriteByte(' ')
				}
				b.WriteString(s)
				wroteAt = (s == "@")
				needSpace = b.Len() > 0 && !wroteAt
			}
		case int, uint, int16, uint16, int32, uint32, int64, uint64:
			if needSpace {
				b.WriteByte(' ')
			}
			fmt.Fprintf(b, "%d", x)
		default:
			if needSpace {
				b.WriteByte(' ')
			}
			fmt.Fprintf(b, "%s", x)
		}

		needSpace = b.Len() > 0 && !wroteAt
	}
	return b.String()
}
