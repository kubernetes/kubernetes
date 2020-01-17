/*
Copyright 2018 The Kubernetes Authors.

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

package config

import (
	"bytes"
	"flag"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestInt(t *testing.T) {
	flags := flag.NewFlagSet("test", 0)
	var context struct {
		Number int `default:"5" usage:"some number"`
	}
	require.NotPanics(t, func() {
		AddOptionsToSet(flags, &context, "")
	})
	require.Equal(t, []simpleFlag{
		{
			name:     "number",
			usage:    "some number",
			defValue: "5",
		}},
		allFlags(flags))
	assert.Equal(t, 5, context.Number)
}

func TestLower(t *testing.T) {
	flags := flag.NewFlagSet("test", 0)
	var context struct {
		Ähem      string
		MixedCase string
	}
	require.NotPanics(t, func() {
		AddOptionsToSet(flags, &context, "")
	})
	require.Equal(t, []simpleFlag{
		{
			name: "mixedCase",
		},
		{
			name: "ähem",
		},
	},
		allFlags(flags))
}

func TestPrefix(t *testing.T) {
	flags := flag.NewFlagSet("test", 0)
	var context struct {
		Number int `usage:"some number"`
	}
	require.NotPanics(t, func() {
		AddOptionsToSet(flags, &context, "some.prefix")
	})
	require.Equal(t, []simpleFlag{
		{
			name:     "some.prefix.number",
			usage:    "some number",
			defValue: "0",
		}},
		allFlags(flags))
}

func TestRecursion(t *testing.T) {
	flags := flag.NewFlagSet("test", 0)
	type Nested struct {
		Number1 int `usage:"embedded number"`
	}
	var context struct {
		Nested
		A struct {
			B struct {
				C struct {
					Number2 int `usage:"some number"`
				}
			}
		}
	}
	require.NotPanics(t, func() {
		AddOptionsToSet(flags, &context, "")
	})
	require.Equal(t, []simpleFlag{
		{
			name:     "a.b.c.number2",
			usage:    "some number",
			defValue: "0",
		},
		{
			name:     "number1",
			usage:    "embedded number",
			defValue: "0",
		},
	},
		allFlags(flags))
}

func TestPanics(t *testing.T) {
	flags := flag.NewFlagSet("test", 0)
	assert.PanicsWithValue(t, `invalid default "a" for int entry prefix.number: strconv.Atoi: parsing "a": invalid syntax`, func() {
		var context struct {
			Number int `default:"a"`
		}
		AddOptionsToSet(flags, &context, "prefix")
	})

	assert.PanicsWithValue(t, `invalid default "10000000000000000000" for int entry prefix.number: strconv.Atoi: parsing "10000000000000000000": value out of range`, func() {
		var context struct {
			Number int `default:"10000000000000000000"`
		}
		AddOptionsToSet(flags, &context, "prefix")
	})

	assert.PanicsWithValue(t, `options parameter without a type - nil?!`, func() {
		AddOptionsToSet(flags, nil, "")
	})

	assert.PanicsWithValue(t, `need a pointer to a struct, got instead: *int`, func() {
		number := 0
		AddOptionsToSet(flags, &number, "")
	})

	assert.PanicsWithValue(t, `struct entry "prefix.number" not exported`, func() {
		var context struct {
			number int
		}
		AddOptionsToSet(flags, &context, "prefix")
	})

	assert.PanicsWithValue(t, `unsupported struct entry type "prefix.someNumber": config.MyInt`, func() {
		type MyInt int
		var context struct {
			SomeNumber MyInt
		}
		AddOptionsToSet(flags, &context, "prefix")
	})
}

type TypesTestCase struct {
	name      string
	copyFlags bool
}

func TestTypes(t *testing.T) {
	testcases := []TypesTestCase{
		{name: "directly"},
		{name: "CopyFlags", copyFlags: true},
	}

	for _, testcase := range testcases {
		testTypes(t, testcase)
	}
}

func testTypes(t *testing.T, testcase TypesTestCase) {
	flags := flag.NewFlagSet("test", 0)
	type Context struct {
		Bool     bool          `default:"true"`
		Duration time.Duration `default:"1ms"`
		Float64  float64       `default:"1.23456789"`
		String   string        `default:"hello world"`
		Int      int           `default:"-1" usage:"some number"`
		Int64    int64         `default:"-1234567890123456789"`
		Uint     uint          `default:"1"`
		Uint64   uint64        `default:"1234567890123456789"`
	}
	var context Context
	require.NotPanics(t, func() {
		AddOptionsToSet(flags, &context, "")
	})

	if testcase.copyFlags {
		original := bytes.Buffer{}
		flags.SetOutput(&original)
		flags.PrintDefaults()

		flags2 := flag.NewFlagSet("test", 0)
		CopyFlags(flags, flags2)
		flags = flags2

		copy := bytes.Buffer{}
		flags.SetOutput(&copy)
		flags.PrintDefaults()
		assert.Equal(t, original.String(), copy.String(), testcase.name+": help messages equal")
		assert.Contains(t, copy.String(), "some number", testcase.name+": copied help message contains defaults")
	}

	require.Equal(t, []simpleFlag{
		{
			name:     "bool",
			defValue: "true",
			isBool:   true,
		},
		{
			name:     "duration",
			defValue: "1ms",
		},
		{
			name:     "float64",
			defValue: "1.23456789",
		},
		{
			name:     "int",
			usage:    "some number",
			defValue: "-1",
		},
		{
			name:     "int64",
			defValue: "-1234567890123456789",
		},
		{
			name:     "string",
			defValue: "hello world",
		},
		{
			name:     "uint",
			defValue: "1",
		},
		{
			name:     "uint64",
			defValue: "1234567890123456789",
		},
	},
		allFlags(flags), testcase.name)
	assert.Equal(t,
		Context{true, time.Millisecond, 1.23456789, "hello world",
			-1, -1234567890123456789, 1, 1234567890123456789,
		},
		context,
		"default values must match")
	require.NoError(t, flags.Parse([]string{
		"-int", "-2",
		"-int64", "-9123456789012345678",
		"-uint", "2",
		"-uint64", "9123456789012345678",
		"-string", "pong",
		"-float64", "-1.23456789",
		"-bool=false",
		"-duration=1s",
	}), testcase.name)
	assert.Equal(t,
		Context{false, time.Second, -1.23456789, "pong",
			-2, -9123456789012345678, 2, 9123456789012345678,
		},
		context,
		testcase.name+": parsed values must match")
}

func allFlags(fs *flag.FlagSet) []simpleFlag {
	var flags []simpleFlag
	fs.VisitAll(func(f *flag.Flag) {
		s := simpleFlag{
			name:     f.Name,
			usage:    f.Usage,
			defValue: f.DefValue,
		}
		type boolFlag interface {
			flag.Value
			IsBoolFlag() bool
		}
		if fv, ok := f.Value.(boolFlag); ok && fv.IsBoolFlag() {
			s.isBool = true
		}
		flags = append(flags, s)
	})
	return flags
}

type simpleFlag struct {
	name     string
	usage    string
	defValue string
	isBool   bool
}
