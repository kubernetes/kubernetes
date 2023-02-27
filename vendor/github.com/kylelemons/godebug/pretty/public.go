// Copyright 2013 Google Inc.  All rights reserved.
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

package pretty

import (
	"bytes"
	"fmt"
	"io"
	"net"
	"reflect"
	"time"

	"github.com/kylelemons/godebug/diff"
)

// A Config represents optional configuration parameters for formatting.
//
// Some options, notably ShortList, dramatically increase the overhead
// of pretty-printing a value.
type Config struct {
	// Verbosity options
	Compact  bool // One-line output. Overrides Diffable.
	Diffable bool // Adds extra newlines for more easily diffable output.

	// Field and value options
	IncludeUnexported   bool // Include unexported fields in output
	PrintStringers      bool // Call String on a fmt.Stringer
	PrintTextMarshalers bool // Call MarshalText on an encoding.TextMarshaler
	SkipZeroFields      bool // Skip struct fields that have a zero value.

	// Output transforms
	ShortList int // Maximum character length for short lists if nonzero.

	// Type-specific overrides
	//
	// Formatter maps a type to a function that will provide a one-line string
	// representation of the input value.  Conceptually:
	//   Formatter[reflect.TypeOf(v)](v) = "v as a string"
	//
	// Note that the first argument need not explicitly match the type, it must
	// merely be callable with it.
	//
	// When processing an input value, if its type exists as a key in Formatter:
	//   1) If the value is nil, no stringification is performed.
	//      This allows overriding of PrintStringers and PrintTextMarshalers.
	//   2) The value will be called with the input as its only argument.
	//      The function must return a string as its first return value.
	//
	// In addition to func literals, two common values for this will be:
	//   fmt.Sprint        (function) func Sprint(...interface{}) string
	//   Type.String         (method) func (Type) String() string
	//
	// Note that neither of these work if the String method is a pointer
	// method and the input will be provided as a value.  In that case,
	// use a function that calls .String on the formal value parameter.
	Formatter map[reflect.Type]interface{}

	// If TrackCycles is enabled, pretty will detect and track
	// self-referential structures. If a self-referential structure (aka a
	// "recursive" value) is detected, numbered placeholders will be emitted.
	//
	// Pointer tracking is disabled by default for performance reasons.
	TrackCycles bool
}

// Default Config objects
var (
	// DefaultFormatter is the default set of overrides for stringification.
	DefaultFormatter = map[reflect.Type]interface{}{
		reflect.TypeOf(time.Time{}):          fmt.Sprint,
		reflect.TypeOf(net.IP{}):             fmt.Sprint,
		reflect.TypeOf((*error)(nil)).Elem(): fmt.Sprint,
	}

	// CompareConfig is the default configuration used for Compare.
	CompareConfig = &Config{
		Diffable:          true,
		IncludeUnexported: true,
		Formatter:         DefaultFormatter,
	}

	// DefaultConfig is the default configuration used for all other top-level functions.
	DefaultConfig = &Config{
		Formatter: DefaultFormatter,
	}

	// CycleTracker is a convenience config for formatting and comparing recursive structures.
	CycleTracker = &Config{
		Diffable:    true,
		Formatter:   DefaultFormatter,
		TrackCycles: true,
	}
)

func (cfg *Config) fprint(buf *bytes.Buffer, vals ...interface{}) {
	ref := &reflector{
		Config: cfg,
	}
	if cfg.TrackCycles {
		ref.pointerTracker = new(pointerTracker)
	}
	for i, val := range vals {
		if i > 0 {
			buf.WriteByte('\n')
		}
		newFormatter(cfg, buf).write(ref.val2node(reflect.ValueOf(val)))
	}
}

// Print writes the DefaultConfig representation of the given values to standard output.
func Print(vals ...interface{}) {
	DefaultConfig.Print(vals...)
}

// Print writes the configured presentation of the given values to standard output.
func (cfg *Config) Print(vals ...interface{}) {
	fmt.Println(cfg.Sprint(vals...))
}

// Sprint returns a string representation of the given value according to the DefaultConfig.
func Sprint(vals ...interface{}) string {
	return DefaultConfig.Sprint(vals...)
}

// Sprint returns a string representation of the given value according to cfg.
func (cfg *Config) Sprint(vals ...interface{}) string {
	buf := new(bytes.Buffer)
	cfg.fprint(buf, vals...)
	return buf.String()
}

// Fprint writes the representation of the given value to the writer according to the DefaultConfig.
func Fprint(w io.Writer, vals ...interface{}) (n int64, err error) {
	return DefaultConfig.Fprint(w, vals...)
}

// Fprint writes the representation of the given value to the writer according to the cfg.
func (cfg *Config) Fprint(w io.Writer, vals ...interface{}) (n int64, err error) {
	buf := new(bytes.Buffer)
	cfg.fprint(buf, vals...)
	return buf.WriteTo(w)
}

// Compare returns a string containing a line-by-line unified diff of the
// values in a and b, using the CompareConfig.
//
// Each line in the output is prefixed with '+', '-', or ' ' to indicate which
// side it's from. Lines from the a side are marked with '-', lines from the
// b side are marked with '+' and lines that are the same on both sides are
// marked with ' '.
//
// The comparison is based on the intentionally-untyped output of Print, and as
// such this comparison is pretty forviving.  In particular, if the types of or
// types within in a and b are different but have the same representation,
// Compare will not indicate any differences between them.
func Compare(a, b interface{}) string {
	return CompareConfig.Compare(a, b)
}

// Compare returns a string containing a line-by-line unified diff of the
// values in got and want according to the cfg.
//
// Each line in the output is prefixed with '+', '-', or ' ' to indicate which
// side it's from. Lines from the a side are marked with '-', lines from the
// b side are marked with '+' and lines that are the same on both sides are
// marked with ' '.
//
// The comparison is based on the intentionally-untyped output of Print, and as
// such this comparison is pretty forviving.  In particular, if the types of or
// types within in a and b are different but have the same representation,
// Compare will not indicate any differences between them.
func (cfg *Config) Compare(a, b interface{}) string {
	diffCfg := *cfg
	diffCfg.Diffable = true
	return diff.Diff(cfg.Sprint(a), cfg.Sprint(b))
}
