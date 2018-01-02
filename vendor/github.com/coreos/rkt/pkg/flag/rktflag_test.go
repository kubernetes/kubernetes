// Copyright 2015 The rkt Authors
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

package flag

import (
	"flag"
	"reflect"
	"strings"
	"testing"

	"github.com/spf13/pflag"
)

// Ensure all these types implement the pflag.Value and flag.Value interface
var _ pflag.Value = (*OptionList)(nil)
var _ pflag.Value = (*PairList)(nil)
var _ flag.Value = (*OptionList)(nil)
var _ flag.Value = (*PairList)(nil)

var options = []string{"zero", "one", "two"}

func TestOptionList(t *testing.T) {
	tests := []struct {
		opts string
		ex   string
		err  bool
	}{
		{
			opts: "zero,two",
			ex:   "zero,two",
			err:  false,
		},
		{ // Duplicate test
			opts: "one,two,two",
			ex:   "",
			err:  true,
		},
		{ // Not permissible test
			opts: "one,two,three",
			ex:   "",
			err:  true,
		},
		{ // Empty string
			opts: "",
			ex:   "",
			err:  false,
		},
	}

	for i, tt := range tests {
		// test NewOptionsList
		if _, err := NewOptionList(options, tt.opts); (err != nil) != tt.err {
			t.Errorf("test %d: unexpected error in NewOptionList: %v", i, err)
		}

		// test OptionList.Set()
		ol, err := NewOptionList(options, strings.Join(options, ","))
		if err != nil {
			t.Errorf("test %d: unexpected error preparing test: %v", i, err)
		}

		if err := ol.Set(tt.opts); (err != nil) != tt.err {
			t.Errorf("test %d: could not parse options as expected: %v", i, err)
		}
		if tt.ex != "" && tt.ex != ol.String() {
			t.Errorf("test %d: resulting options not as expected: %s != %s",
				i, tt.ex, ol.String())
		}
	}
}

var bfMap = map[string]int{
	options[0]: 0,
	options[1]: 1,
	options[2]: 1 << 1,
}

func TestBitFlags(t *testing.T) {
	tests := []struct {
		opts     string
		ex       int
		parseErr bool
		logicErr bool
	}{
		{
			opts: "one,two",
			ex:   3,
		},
		{ // Duplicate test
			opts:     "zero,two,two",
			ex:       -1,
			parseErr: true,
		},
		{ // Not included test
			opts:     "zero,two,three",
			ex:       -1,
			parseErr: true,
		},
		{ // Test 10 in 11
			opts: "one,two",
			ex:   1,
		},
		{ // Test 11 not in 01
			opts:     "one",
			ex:       3,
			logicErr: true,
		},
	}

	for i, tt := range tests {
		// test NewBitFlags
		if _, err := NewBitFlags(options, tt.opts, bfMap); (err != nil) != tt.parseErr {
			t.Errorf("test %d: unexpected error in NewBitFlags: %v", i, err)
		}

		bf, err := NewBitFlags(options, strings.Join(options, ","), bfMap)
		if err != nil {
			t.Errorf("test %d: unexpected error preparing test: %v", i, err)
		}

		// test bitFlags.Set()
		if err := bf.Set(tt.opts); (err != nil) != tt.parseErr {
			t.Errorf("test %d: Could not parse options as expected: %v", i, err)
		}
		if tt.ex >= 0 && bf.HasFlag(tt.ex) == tt.logicErr {
			t.Errorf("test %d: Result was unexpected: %d != %d",
				i, tt.ex, bf.Flags)
		}
	}
}

type ss []string
type sm map[string]string

func TestPairList(t *testing.T) {
	options := map[string][]string{
		"one":   {"a", "b", "ll"},
		"two":   {},
		"three": nil,
	}

	tests := []struct {
		args     ss
		expected sm
		hasError bool
	}{
		{
			args:     ss{},
			expected: sm{},
			hasError: false,
		}, {
			args:     ss{"one=a"},
			expected: sm{"one": "a"},
			hasError: false,
		}, {
			args:     ss{"one=a", "two=bar"},
			expected: sm{"one": "a", "two": "bar"},
			hasError: false,
		}, {
			args:     ss{"one=ll,two=bar"},
			expected: sm{"one": "ll", "two": "bar"},
			hasError: false,
		}, {
			args:     ss{"one=bar,two=bar"},
			expected: sm{},
			hasError: true,
		}, {
			args:     ss{"one=ll,two=bar", "one=a"},
			expected: sm{"one": "a", "two": "bar"},
			hasError: false,
		},
	}

	for i, tt := range tests {

		pl := MustNewPairList(options, nil)

		var argErr error
		for _, arg := range tt.args {
			err := pl.Set(arg)
			if err != nil {
				argErr = err
			}
		}
		if argErr != nil {
			if !tt.hasError {
				t.Errorf("test %d: pl.Set unexpected error %v", i, argErr)
			} else {
				continue // no point in testing equality
			}
		}

		if !reflect.DeepEqual(map[string]string(tt.expected), pl.Pairs) {
			t.Errorf("test %d: pl.Pairs expected %v got %v", i, tt.expected, pl.Pairs)
		}
	}

	pl := MustNewPairList(options, nil)
	got := pl.PermissibleString()
	expected := "one=[a|b|ll] three=* two=*" // order!
	if got != expected {
		t.Errorf("pl.PermissibleString() expected %v got %v", expected, got)
	}

	pl = MustNewPairList(options, sm{"one": "a", "two": "bar"})
	got = pl.String()
	expected = "one=a two=bar"
	if got != expected {
		t.Errorf("pl.String expected %v got %v", expected, got)
	}

	ser := SerializePairs(pl.Pairs)
	pl2 := MustNewPairList(options, nil)
	pl2.Set(ser)
	if !reflect.DeepEqual(pl.Pairs, pl2.Pairs) {
		t.Errorf("serialize - unserialize did not match")
	}
}
