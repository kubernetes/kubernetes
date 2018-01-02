// Copyright 2016 Frank Schroeder. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package properties

import (
	"bytes"
	"flag"
	"fmt"
	"math"
	"os"
	"strings"
	"testing"
	"time"

	. "github.com/magiconair/properties/_third_party/gopkg.in/check.v1"
)

func Test(t *testing.T) { TestingT(t) }

type TestSuite struct {
	prevHandler ErrorHandlerFunc
}

var (
	_       = Suite(&TestSuite{})
	verbose = flag.Bool("verbose", false, "Verbose output")
)

// --------------------------------------------------------------------

func (s *TestSuite) SetUpSuite(c *C) {
	s.prevHandler = ErrorHandler
	ErrorHandler = PanicHandler
}

// --------------------------------------------------------------------

func (s *TestSuite) TearDownSuite(c *C) {
	ErrorHandler = s.prevHandler
}

// ----------------------------------------------------------------------------

// define test cases in the form of
// {"input", "key1", "value1", "key2", "value2", ...}
var complexTests = [][]string{
	// whitespace prefix
	{" key=value", "key", "value"},     // SPACE prefix
	{"\fkey=value", "key", "value"},    // FF prefix
	{"\tkey=value", "key", "value"},    // TAB prefix
	{" \f\tkey=value", "key", "value"}, // mix prefix

	// multiple keys
	{"key1=value1\nkey2=value2\n", "key1", "value1", "key2", "value2"},
	{"key1=value1\rkey2=value2\r", "key1", "value1", "key2", "value2"},
	{"key1=value1\r\nkey2=value2\r\n", "key1", "value1", "key2", "value2"},

	// blank lines
	{"\nkey=value\n", "key", "value"},
	{"\rkey=value\r", "key", "value"},
	{"\r\nkey=value\r\n", "key", "value"},

	// escaped chars in key
	{"k\\ ey = value", "k ey", "value"},
	{"k\\:ey = value", "k:ey", "value"},
	{"k\\=ey = value", "k=ey", "value"},
	{"k\\fey = value", "k\fey", "value"},
	{"k\\ney = value", "k\ney", "value"},
	{"k\\rey = value", "k\rey", "value"},
	{"k\\tey = value", "k\tey", "value"},

	// escaped chars in value
	{"key = v\\ alue", "key", "v alue"},
	{"key = v\\:alue", "key", "v:alue"},
	{"key = v\\=alue", "key", "v=alue"},
	{"key = v\\falue", "key", "v\falue"},
	{"key = v\\nalue", "key", "v\nalue"},
	{"key = v\\ralue", "key", "v\ralue"},
	{"key = v\\talue", "key", "v\talue"},

	// silently dropped escape character
	{"k\\zey = value", "kzey", "value"},
	{"key = v\\zalue", "key", "vzalue"},

	// unicode literals
	{"key\\u2318 = value", "key⌘", "value"},
	{"k\\u2318ey = value", "k⌘ey", "value"},
	{"key = value\\u2318", "key", "value⌘"},
	{"key = valu\\u2318e", "key", "valu⌘e"},

	// multiline values
	{"key = valueA,\\\n    valueB", "key", "valueA,valueB"},   // SPACE indent
	{"key = valueA,\\\n\f\f\fvalueB", "key", "valueA,valueB"}, // FF indent
	{"key = valueA,\\\n\t\t\tvalueB", "key", "valueA,valueB"}, // TAB indent
	{"key = valueA,\\\n \f\tvalueB", "key", "valueA,valueB"},  // mix indent

	// comments
	{"# this is a comment\n! and so is this\nkey1=value1\nkey#2=value#2\n\nkey!3=value!3\n# and another one\n! and the final one", "key1", "value1", "key#2", "value#2", "key!3", "value!3"},

	// expansion tests
	{"key=value\nkey2=${key}", "key", "value", "key2", "value"},
	{"key=value\nkey2=aa${key}", "key", "value", "key2", "aavalue"},
	{"key=value\nkey2=${key}bb", "key", "value", "key2", "valuebb"},
	{"key=value\nkey2=aa${key}bb", "key", "value", "key2", "aavaluebb"},
	{"key=value\nkey2=${key}\nkey3=${key2}", "key", "value", "key2", "value", "key3", "value"},
	{"key=${USER}", "key", os.Getenv("USER")},
	{"key=${USER}\nUSER=value", "key", "value", "USER", "value"},
}

// ----------------------------------------------------------------------------

var commentTests = []struct {
	input, key, value string
	comments          []string
}{
	{"key=value", "key", "value", nil},
	{"#\nkey=value", "key", "value", []string{""}},
	{"#comment\nkey=value", "key", "value", []string{"comment"}},
	{"# comment\nkey=value", "key", "value", []string{"comment"}},
	{"#  comment\nkey=value", "key", "value", []string{"comment"}},
	{"# comment\n\nkey=value", "key", "value", []string{"comment"}},
	{"# comment1\n# comment2\nkey=value", "key", "value", []string{"comment1", "comment2"}},
	{"# comment1\n\n# comment2\n\nkey=value", "key", "value", []string{"comment1", "comment2"}},
	{"!comment\nkey=value", "key", "value", []string{"comment"}},
	{"! comment\nkey=value", "key", "value", []string{"comment"}},
	{"!  comment\nkey=value", "key", "value", []string{"comment"}},
	{"! comment\n\nkey=value", "key", "value", []string{"comment"}},
	{"! comment1\n! comment2\nkey=value", "key", "value", []string{"comment1", "comment2"}},
	{"! comment1\n\n! comment2\n\nkey=value", "key", "value", []string{"comment1", "comment2"}},
}

// ----------------------------------------------------------------------------

var errorTests = []struct {
	input, msg string
}{
	// unicode literals
	{"key\\u1 = value", "invalid unicode literal"},
	{"key\\u12 = value", "invalid unicode literal"},
	{"key\\u123 = value", "invalid unicode literal"},
	{"key\\u123g = value", "invalid unicode literal"},
	{"key\\u123", "invalid unicode literal"},

	// circular references
	{"key=${key}", "circular reference"},
	{"key1=${key2}\nkey2=${key1}", "circular reference"},

	// malformed expressions
	{"key=${ke", "malformed expression"},
	{"key=valu${ke", "malformed expression"},
}

// ----------------------------------------------------------------------------

var writeTests = []struct {
	input, output, encoding string
}{
	// ISO-8859-1 tests
	{"key = value", "key = value\n", "ISO-8859-1"},
	{"key = value \\\n   continued", "key = value continued\n", "ISO-8859-1"},
	{"key⌘ = value", "key\\u2318 = value\n", "ISO-8859-1"},
	{"ke\\ \\:y = value", "ke\\ \\:y = value\n", "ISO-8859-1"},

	// UTF-8 tests
	{"key = value", "key = value\n", "UTF-8"},
	{"key = value \\\n   continued", "key = value continued\n", "UTF-8"},
	{"key⌘ = value⌘", "key⌘ = value⌘\n", "UTF-8"},
	{"ke\\ \\:y = value", "ke\\ \\:y = value\n", "UTF-8"},
}

// ----------------------------------------------------------------------------

var writeCommentTests = []struct {
	input, output, encoding string
}{
	// ISO-8859-1 tests
	{"key = value", "key = value\n", "ISO-8859-1"},
	{"#\nkey = value", "key = value\n", "ISO-8859-1"},
	{"#\n#\n#\nkey = value", "key = value\n", "ISO-8859-1"},
	{"# comment\nkey = value", "# comment\nkey = value\n", "ISO-8859-1"},
	{"\n# comment\nkey = value", "# comment\nkey = value\n", "ISO-8859-1"},
	{"# comment\n\nkey = value", "# comment\nkey = value\n", "ISO-8859-1"},
	{"# comment1\n# comment2\nkey = value", "# comment1\n# comment2\nkey = value\n", "ISO-8859-1"},
	{"#comment1\nkey1 = value1\n#comment2\nkey2 = value2", "# comment1\nkey1 = value1\n\n# comment2\nkey2 = value2\n", "ISO-8859-1"},

	// UTF-8 tests
	{"key = value", "key = value\n", "UTF-8"},
	{"# comment⌘\nkey = value⌘", "# comment⌘\nkey = value⌘\n", "UTF-8"},
	{"\n# comment⌘\nkey = value⌘", "# comment⌘\nkey = value⌘\n", "UTF-8"},
	{"# comment⌘\n\nkey = value⌘", "# comment⌘\nkey = value⌘\n", "UTF-8"},
	{"# comment1⌘\n# comment2⌘\nkey = value⌘", "# comment1⌘\n# comment2⌘\nkey = value⌘\n", "UTF-8"},
	{"#comment1⌘\nkey1 = value1⌘\n#comment2⌘\nkey2 = value2⌘", "# comment1⌘\nkey1 = value1⌘\n\n# comment2⌘\nkey2 = value2⌘\n", "UTF-8"},
}

// ----------------------------------------------------------------------------

var boolTests = []struct {
	input, key string
	def, value bool
}{
	// valid values for TRUE
	{"key = 1", "key", false, true},
	{"key = on", "key", false, true},
	{"key = On", "key", false, true},
	{"key = ON", "key", false, true},
	{"key = true", "key", false, true},
	{"key = True", "key", false, true},
	{"key = TRUE", "key", false, true},
	{"key = yes", "key", false, true},
	{"key = Yes", "key", false, true},
	{"key = YES", "key", false, true},

	// valid values for FALSE (all other)
	{"key = 0", "key", true, false},
	{"key = off", "key", true, false},
	{"key = false", "key", true, false},
	{"key = no", "key", true, false},

	// non existent key
	{"key = true", "key2", false, false},
}

// ----------------------------------------------------------------------------

var durationTests = []struct {
	input, key string
	def, value time.Duration
}{
	// valid values
	{"key = 1", "key", 999, 1},
	{"key = 0", "key", 999, 0},
	{"key = -1", "key", 999, -1},
	{"key = 0123", "key", 999, 123},

	// invalid values
	{"key = 0xff", "key", 999, 999},
	{"key = 1.0", "key", 999, 999},
	{"key = a", "key", 999, 999},

	// non existent key
	{"key = 1", "key2", 999, 999},
}

// ----------------------------------------------------------------------------

var parsedDurationTests = []struct {
	input, key string
	def, value time.Duration
}{
	// valid values
	{"key = -1ns", "key", 999, -1 * time.Nanosecond},
	{"key = 300ms", "key", 999, 300 * time.Millisecond},
	{"key = 5s", "key", 999, 5 * time.Second},
	{"key = 3h", "key", 999, 3 * time.Hour},
	{"key = 2h45m", "key", 999, 2*time.Hour + 45*time.Minute},

	// invalid values
	{"key = 0xff", "key", 999, 999},
	{"key = 1.0", "key", 999, 999},
	{"key = a", "key", 999, 999},
	{"key = 1", "key", 999, 999},
	{"key = 0", "key", 999, 0},

	// non existent key
	{"key = 1", "key2", 999, 999},
}

// ----------------------------------------------------------------------------

var floatTests = []struct {
	input, key string
	def, value float64
}{
	// valid values
	{"key = 1.0", "key", 999, 1.0},
	{"key = 0.0", "key", 999, 0.0},
	{"key = -1.0", "key", 999, -1.0},
	{"key = 1", "key", 999, 1},
	{"key = 0", "key", 999, 0},
	{"key = -1", "key", 999, -1},
	{"key = 0123", "key", 999, 123},

	// invalid values
	{"key = 0xff", "key", 999, 999},
	{"key = a", "key", 999, 999},

	// non existent key
	{"key = 1", "key2", 999, 999},
}

// ----------------------------------------------------------------------------

var int64Tests = []struct {
	input, key string
	def, value int64
}{
	// valid values
	{"key = 1", "key", 999, 1},
	{"key = 0", "key", 999, 0},
	{"key = -1", "key", 999, -1},
	{"key = 0123", "key", 999, 123},

	// invalid values
	{"key = 0xff", "key", 999, 999},
	{"key = 1.0", "key", 999, 999},
	{"key = a", "key", 999, 999},

	// non existent key
	{"key = 1", "key2", 999, 999},
}

// ----------------------------------------------------------------------------

var uint64Tests = []struct {
	input, key string
	def, value uint64
}{
	// valid values
	{"key = 1", "key", 999, 1},
	{"key = 0", "key", 999, 0},
	{"key = 0123", "key", 999, 123},

	// invalid values
	{"key = -1", "key", 999, 999},
	{"key = 0xff", "key", 999, 999},
	{"key = 1.0", "key", 999, 999},
	{"key = a", "key", 999, 999},

	// non existent key
	{"key = 1", "key2", 999, 999},
}

// ----------------------------------------------------------------------------

var stringTests = []struct {
	input, key string
	def, value string
}{
	// valid values
	{"key = abc", "key", "def", "abc"},

	// non existent key
	{"key = abc", "key2", "def", "def"},
}

// ----------------------------------------------------------------------------

var keysTests = []struct {
	input string
	keys  []string
}{
	{"", []string{}},
	{"key = abc", []string{"key"}},
	{"key = abc\nkey2=def", []string{"key", "key2"}},
	{"key2 = abc\nkey=def", []string{"key2", "key"}},
	{"key = abc\nkey=def", []string{"key"}},
}

// ----------------------------------------------------------------------------

var filterTests = []struct {
	input   string
	pattern string
	keys    []string
	err     string
}{
	{"", "", []string{}, ""},
	{"", "abc", []string{}, ""},
	{"key=value", "", []string{"key"}, ""},
	{"key=value", "key=", []string{}, ""},
	{"key=value\nfoo=bar", "", []string{"foo", "key"}, ""},
	{"key=value\nfoo=bar", "f", []string{"foo"}, ""},
	{"key=value\nfoo=bar", "fo", []string{"foo"}, ""},
	{"key=value\nfoo=bar", "foo", []string{"foo"}, ""},
	{"key=value\nfoo=bar", "fooo", []string{}, ""},
	{"key=value\nkey2=value2\nfoo=bar", "ey", []string{"key", "key2"}, ""},
	{"key=value\nkey2=value2\nfoo=bar", "key", []string{"key", "key2"}, ""},
	{"key=value\nkey2=value2\nfoo=bar", "^key", []string{"key", "key2"}, ""},
	{"key=value\nkey2=value2\nfoo=bar", "^(key|foo)", []string{"foo", "key", "key2"}, ""},
	{"key=value\nkey2=value2\nfoo=bar", "[ abc", nil, "error parsing regexp.*"},
}

// ----------------------------------------------------------------------------

var filterPrefixTests = []struct {
	input  string
	prefix string
	keys   []string
}{
	{"", "", []string{}},
	{"", "abc", []string{}},
	{"key=value", "", []string{"key"}},
	{"key=value", "key=", []string{}},
	{"key=value\nfoo=bar", "", []string{"foo", "key"}},
	{"key=value\nfoo=bar", "f", []string{"foo"}},
	{"key=value\nfoo=bar", "fo", []string{"foo"}},
	{"key=value\nfoo=bar", "foo", []string{"foo"}},
	{"key=value\nfoo=bar", "fooo", []string{}},
	{"key=value\nkey2=value2\nfoo=bar", "key", []string{"key", "key2"}},
}

// ----------------------------------------------------------------------------

var filterStripPrefixTests = []struct {
	input  string
	prefix string
	keys   []string
}{
	{"", "", []string{}},
	{"", "abc", []string{}},
	{"key=value", "", []string{"key"}},
	{"key=value", "key=", []string{}},
	{"key=value\nfoo=bar", "", []string{"foo", "key"}},
	{"key=value\nfoo=bar", "f", []string{"foo"}},
	{"key=value\nfoo=bar", "fo", []string{"foo"}},
	{"key=value\nfoo=bar", "foo", []string{"foo"}},
	{"key=value\nfoo=bar", "fooo", []string{}},
	{"key=value\nkey2=value2\nfoo=bar", "key", []string{"key", "key2"}},
}

// ----------------------------------------------------------------------------

var setTests = []struct {
	input      string
	key, value string
	prev       string
	ok         bool
	err        string
	keys       []string
}{
	{"", "", "", "", false, "", []string{}},
	{"", "key", "value", "", false, "", []string{"key"}},
	{"key=value", "key2", "value2", "", false, "", []string{"key", "key2"}},
	{"key=value", "abc", "value3", "", false, "", []string{"key", "abc"}},
	{"key=value", "key", "value3", "value", true, "", []string{"key"}},
}

// ----------------------------------------------------------------------------

// TestBasic tests basic single key/value combinations with all possible
// whitespace, delimiter and newline permutations.
func (s *TestSuite) TestBasic(c *C) {
	testWhitespaceAndDelimiterCombinations(c, "key", "")
	testWhitespaceAndDelimiterCombinations(c, "key", "value")
	testWhitespaceAndDelimiterCombinations(c, "key", "value   ")
}

func (s *TestSuite) TestComplex(c *C) {
	for _, test := range complexTests {
		testKeyValue(c, test[0], test[1:]...)
	}
}

func (s *TestSuite) TestErrors(c *C) {
	for _, test := range errorTests {
		_, err := Load([]byte(test.input), ISO_8859_1)
		c.Assert(err, NotNil)
		c.Assert(strings.Contains(err.Error(), test.msg), Equals, true, Commentf("Expected %q got %q", test.msg, err.Error()))
	}
}

func (s *TestSuite) TestDisableExpansion(c *C) {
	input := "key=value\nkey2=${key}"
	p, err := parse(input)
	p.DisableExpansion = true
	c.Assert(err, IsNil)
	c.Assert(p.MustGet("key"), Equals, "value")
	c.Assert(p.MustGet("key2"), Equals, "${key}")

	// with expansion disabled we can introduce circular references
	p.Set("keyA", "${keyB}")
	p.Set("keyB", "${keyA}")
	c.Assert(p.MustGet("keyA"), Equals, "${keyB}")
	c.Assert(p.MustGet("keyB"), Equals, "${keyA}")
}

func (s *TestSuite) TestMustGet(c *C) {
	input := "key = value\nkey2 = ghi"
	p, err := parse(input)
	c.Assert(err, IsNil)
	c.Assert(p.MustGet("key"), Equals, "value")
	c.Assert(func() { p.MustGet("invalid") }, PanicMatches, "unknown property: invalid")
}

func (s *TestSuite) TestGetBool(c *C) {
	for _, test := range boolTests {
		p, err := parse(test.input)
		c.Assert(err, IsNil)
		c.Assert(p.Len(), Equals, 1)
		c.Assert(p.GetBool(test.key, test.def), Equals, test.value)
	}
}

func (s *TestSuite) TestMustGetBool(c *C) {
	input := "key = true\nkey2 = ghi"
	p, err := parse(input)
	c.Assert(err, IsNil)
	c.Assert(p.MustGetBool("key"), Equals, true)
	c.Assert(func() { p.MustGetBool("invalid") }, PanicMatches, "unknown property: invalid")
}

func (s *TestSuite) TestGetDuration(c *C) {
	for _, test := range durationTests {
		p, err := parse(test.input)
		c.Assert(err, IsNil)
		c.Assert(p.Len(), Equals, 1)
		c.Assert(p.GetDuration(test.key, test.def), Equals, test.value)
	}
}

func (s *TestSuite) TestMustGetDuration(c *C) {
	input := "key = 123\nkey2 = ghi"
	p, err := parse(input)
	c.Assert(err, IsNil)
	c.Assert(p.MustGetDuration("key"), Equals, time.Duration(123))
	c.Assert(func() { p.MustGetDuration("key2") }, PanicMatches, "strconv.ParseInt: parsing.*")
	c.Assert(func() { p.MustGetDuration("invalid") }, PanicMatches, "unknown property: invalid")
}

func (s *TestSuite) TestGetParsedDuration(c *C) {
	for _, test := range parsedDurationTests {
		p, err := parse(test.input)
		c.Assert(err, IsNil)
		c.Assert(p.Len(), Equals, 1)
		c.Assert(p.GetParsedDuration(test.key, test.def), Equals, test.value)
	}
}

func (s *TestSuite) TestMustGetParsedDuration(c *C) {
	input := "key = 123ms\nkey2 = ghi"
	p, err := parse(input)
	c.Assert(err, IsNil)
	c.Assert(p.MustGetParsedDuration("key"), Equals, 123*time.Millisecond)
	c.Assert(func() { p.MustGetParsedDuration("key2") }, PanicMatches, "time: invalid duration ghi")
	c.Assert(func() { p.MustGetParsedDuration("invalid") }, PanicMatches, "unknown property: invalid")
}

func (s *TestSuite) TestGetFloat64(c *C) {
	for _, test := range floatTests {
		p, err := parse(test.input)
		c.Assert(err, IsNil)
		c.Assert(p.Len(), Equals, 1)
		c.Assert(p.GetFloat64(test.key, test.def), Equals, test.value)
	}
}

func (s *TestSuite) TestMustGetFloat64(c *C) {
	input := "key = 123\nkey2 = ghi"
	p, err := parse(input)
	c.Assert(err, IsNil)
	c.Assert(p.MustGetFloat64("key"), Equals, float64(123))
	c.Assert(func() { p.MustGetFloat64("key2") }, PanicMatches, "strconv.ParseFloat: parsing.*")
	c.Assert(func() { p.MustGetFloat64("invalid") }, PanicMatches, "unknown property: invalid")
}

func (s *TestSuite) TestGetInt(c *C) {
	for _, test := range int64Tests {
		p, err := parse(test.input)
		c.Assert(err, IsNil)
		c.Assert(p.Len(), Equals, 1)
		c.Assert(p.GetInt(test.key, int(test.def)), Equals, int(test.value))
	}
}

func (s *TestSuite) TestMustGetInt(c *C) {
	input := "key = 123\nkey2 = ghi"
	p, err := parse(input)
	c.Assert(err, IsNil)
	c.Assert(p.MustGetInt("key"), Equals, int(123))
	c.Assert(func() { p.MustGetInt("key2") }, PanicMatches, "strconv.ParseInt: parsing.*")
	c.Assert(func() { p.MustGetInt("invalid") }, PanicMatches, "unknown property: invalid")
}

func (s *TestSuite) TestGetInt64(c *C) {
	for _, test := range int64Tests {
		p, err := parse(test.input)
		c.Assert(err, IsNil)
		c.Assert(p.Len(), Equals, 1)
		c.Assert(p.GetInt64(test.key, test.def), Equals, test.value)
	}
}

func (s *TestSuite) TestMustGetInt64(c *C) {
	input := "key = 123\nkey2 = ghi"
	p, err := parse(input)
	c.Assert(err, IsNil)
	c.Assert(p.MustGetInt64("key"), Equals, int64(123))
	c.Assert(func() { p.MustGetInt64("key2") }, PanicMatches, "strconv.ParseInt: parsing.*")
	c.Assert(func() { p.MustGetInt64("invalid") }, PanicMatches, "unknown property: invalid")
}

func (s *TestSuite) TestGetUint(c *C) {
	for _, test := range uint64Tests {
		p, err := parse(test.input)
		c.Assert(err, IsNil)
		c.Assert(p.Len(), Equals, 1)
		c.Assert(p.GetUint(test.key, uint(test.def)), Equals, uint(test.value))
	}
}

func (s *TestSuite) TestMustGetUint(c *C) {
	input := "key = 123\nkey2 = ghi"
	p, err := parse(input)
	c.Assert(err, IsNil)
	c.Assert(p.MustGetUint("key"), Equals, uint(123))
	c.Assert(func() { p.MustGetUint64("key2") }, PanicMatches, "strconv.ParseUint: parsing.*")
	c.Assert(func() { p.MustGetUint64("invalid") }, PanicMatches, "unknown property: invalid")
}

func (s *TestSuite) TestGetUint64(c *C) {
	for _, test := range uint64Tests {
		p, err := parse(test.input)
		c.Assert(err, IsNil)
		c.Assert(p.Len(), Equals, 1)
		c.Assert(p.GetUint64(test.key, test.def), Equals, test.value)
	}
}

func (s *TestSuite) TestMustGetUint64(c *C) {
	input := "key = 123\nkey2 = ghi"
	p, err := parse(input)
	c.Assert(err, IsNil)
	c.Assert(p.MustGetUint64("key"), Equals, uint64(123))
	c.Assert(func() { p.MustGetUint64("key2") }, PanicMatches, "strconv.ParseUint: parsing.*")
	c.Assert(func() { p.MustGetUint64("invalid") }, PanicMatches, "unknown property: invalid")
}

func (s *TestSuite) TestGetString(c *C) {
	for _, test := range stringTests {
		p, err := parse(test.input)
		c.Assert(err, IsNil)
		c.Assert(p.Len(), Equals, 1)
		c.Assert(p.GetString(test.key, test.def), Equals, test.value)
	}
}

func (s *TestSuite) TestMustGetString(c *C) {
	input := `key = value`
	p, err := parse(input)
	c.Assert(err, IsNil)
	c.Assert(p.MustGetString("key"), Equals, "value")
	c.Assert(func() { p.MustGetString("invalid") }, PanicMatches, "unknown property: invalid")
}

func (s *TestSuite) TestComment(c *C) {
	for _, test := range commentTests {
		p, err := parse(test.input)
		c.Assert(err, IsNil)
		c.Assert(p.MustGetString(test.key), Equals, test.value)
		c.Assert(p.GetComments(test.key), DeepEquals, test.comments)
		if test.comments != nil {
			c.Assert(p.GetComment(test.key), Equals, test.comments[len(test.comments)-1])
		} else {
			c.Assert(p.GetComment(test.key), Equals, "")
		}

		// test setting comments
		if len(test.comments) > 0 {
			// set single comment
			p.ClearComments()
			c.Assert(len(p.c), Equals, 0)
			p.SetComment(test.key, test.comments[0])
			c.Assert(p.GetComment(test.key), Equals, test.comments[0])

			// set multiple comments
			p.ClearComments()
			c.Assert(len(p.c), Equals, 0)
			p.SetComments(test.key, test.comments)
			c.Assert(p.GetComments(test.key), DeepEquals, test.comments)

			// clear comments for a key
			p.SetComments(test.key, nil)
			c.Assert(p.GetComment(test.key), Equals, "")
			c.Assert(p.GetComments(test.key), IsNil)
		}
	}
}

func (s *TestSuite) TestFilter(c *C) {
	for _, test := range filterTests {
		p, err := parse(test.input)
		c.Assert(err, IsNil)
		pp, err := p.Filter(test.pattern)
		if err != nil {
			c.Assert(err, ErrorMatches, test.err)
			continue
		}
		c.Assert(pp, NotNil)
		c.Assert(pp.Len(), Equals, len(test.keys))
		for _, key := range test.keys {
			v1, ok1 := p.Get(key)
			v2, ok2 := pp.Get(key)
			c.Assert(ok1, Equals, true)
			c.Assert(ok2, Equals, true)
			c.Assert(v1, Equals, v2)
		}
	}
}

func (s *TestSuite) TestFilterPrefix(c *C) {
	for _, test := range filterPrefixTests {
		p, err := parse(test.input)
		c.Assert(err, IsNil)
		pp := p.FilterPrefix(test.prefix)
		c.Assert(pp, NotNil)
		c.Assert(pp.Len(), Equals, len(test.keys))
		for _, key := range test.keys {
			v1, ok1 := p.Get(key)
			v2, ok2 := pp.Get(key)
			c.Assert(ok1, Equals, true)
			c.Assert(ok2, Equals, true)
			c.Assert(v1, Equals, v2)
		}
	}
}

func (s *TestSuite) TestKeys(c *C) {
	for _, test := range keysTests {
		p, err := parse(test.input)
		c.Assert(err, IsNil)
		c.Assert(p.Len(), Equals, len(test.keys))
		c.Assert(len(p.Keys()), Equals, len(test.keys))
		c.Assert(p.Keys(), DeepEquals, test.keys)
	}
}

func (s *TestSuite) TestSet(c *C) {
	for _, test := range setTests {
		p, err := parse(test.input)
		c.Assert(err, IsNil)
		prev, ok, err := p.Set(test.key, test.value)
		if test.err != "" {
			c.Assert(err, ErrorMatches, test.err)
			continue
		}

		c.Assert(err, IsNil)
		c.Assert(ok, Equals, test.ok)
		if ok {
			c.Assert(prev, Equals, test.prev)
		}
		c.Assert(p.Keys(), DeepEquals, test.keys)
	}
}

func (s *TestSuite) TestMustSet(c *C) {
	input := "key=${key}"
	p, err := parse(input)
	c.Assert(err, IsNil)
	c.Assert(func() { p.MustSet("key", "${key}") }, PanicMatches, "circular reference .*")
}

func (s *TestSuite) TestWrite(c *C) {
	for _, test := range writeTests {
		p, err := parse(test.input)

		buf := new(bytes.Buffer)
		var n int
		switch test.encoding {
		case "UTF-8":
			n, err = p.Write(buf, UTF8)
		case "ISO-8859-1":
			n, err = p.Write(buf, ISO_8859_1)
		}
		c.Assert(err, IsNil)
		s := string(buf.Bytes())
		c.Assert(n, Equals, len(test.output), Commentf("input=%q expected=%q obtained=%q", test.input, test.output, s))
		c.Assert(s, Equals, test.output, Commentf("input=%q expected=%q obtained=%q", test.input, test.output, s))
	}
}

func (s *TestSuite) TestWriteComment(c *C) {
	for _, test := range writeCommentTests {
		p, err := parse(test.input)

		buf := new(bytes.Buffer)
		var n int
		switch test.encoding {
		case "UTF-8":
			n, err = p.WriteComment(buf, "# ", UTF8)
		case "ISO-8859-1":
			n, err = p.WriteComment(buf, "# ", ISO_8859_1)
		}
		c.Assert(err, IsNil)
		s := string(buf.Bytes())
		c.Assert(n, Equals, len(test.output), Commentf("input=%q expected=%q obtained=%q", test.input, test.output, s))
		c.Assert(s, Equals, test.output, Commentf("input=%q expected=%q obtained=%q", test.input, test.output, s))
	}
}

func (s *TestSuite) TestCustomExpansionExpression(c *C) {
	testKeyValuePrePostfix(c, "*[", "]*", "key=value\nkey2=*[key]*", "key", "value", "key2", "value")
}

func (s *TestSuite) TestPanicOn32BitIntOverflow(c *C) {
	is32Bit = true
	var min, max int64 = math.MinInt32 - 1, math.MaxInt32 + 1
	input := fmt.Sprintf("min=%d\nmax=%d", min, max)
	p, err := parse(input)
	c.Assert(err, IsNil)
	c.Assert(p.MustGetInt64("min"), Equals, min)
	c.Assert(p.MustGetInt64("max"), Equals, max)
	c.Assert(func() { p.MustGetInt("min") }, PanicMatches, ".* out of range")
	c.Assert(func() { p.MustGetInt("max") }, PanicMatches, ".* out of range")
}

func (s *TestSuite) TestPanicOn32BitUintOverflow(c *C) {
	is32Bit = true
	var max uint64 = math.MaxUint32 + 1
	input := fmt.Sprintf("max=%d", max)
	p, err := parse(input)
	c.Assert(err, IsNil)
	c.Assert(p.MustGetUint64("max"), Equals, max)
	c.Assert(func() { p.MustGetUint("max") }, PanicMatches, ".* out of range")
}

func (s *TestSuite) TestDeleteKey(c *C) {
	input := "#comments should also be gone\nkey=to-be-deleted\nsecond=key"
	p, err := parse(input)
	c.Assert(err, IsNil)
	c.Check(len(p.m), Equals, 2)
	c.Check(len(p.c), Equals, 1)
	c.Check(len(p.k), Equals, 2)
	p.Delete("key")
	c.Check(len(p.m), Equals, 1)
	c.Check(len(p.c), Equals, 0)
	c.Check(len(p.k), Equals, 1)
}

func (s *TestSuite) TestDeleteUnknownKey(c *C) {
	input := "#comments should also be gone\nkey=to-be-deleted"
	p, err := parse(input)
	c.Assert(err, IsNil)
	c.Check(len(p.m), Equals, 1)
	c.Check(len(p.c), Equals, 1)
	c.Check(len(p.k), Equals, 1)
	p.Delete("wrong-key")
	c.Check(len(p.m), Equals, 1)
	c.Check(len(p.c), Equals, 1)
	c.Check(len(p.k), Equals, 1)
}

func (s *TestSuite) TestMerge(c *C) {
	input1 := "#comment\nkey=value\nkey2=value2"
	input2 := "#another comment\nkey=another value\nkey3=value3"
	p1, err := parse(input1)
	c.Assert(err, IsNil)
	p2, err := parse(input2)
	p1.Merge(p2)
	c.Check(len(p1.m), Equals, 3)
	c.Check(len(p1.c), Equals, 1)
	c.Check(len(p1.k), Equals, 3)
	c.Check(p1.MustGet("key"), Equals, "another value")
	c.Check(p1.GetComment("key"), Equals, "another comment")
}

// ----------------------------------------------------------------------------

// tests all combinations of delimiters, leading and/or trailing whitespace and newlines.
func testWhitespaceAndDelimiterCombinations(c *C, key, value string) {
	whitespace := []string{"", " ", "\f", "\t"}
	delimiters := []string{"", " ", "=", ":"}
	newlines := []string{"", "\r", "\n", "\r\n"}
	for _, dl := range delimiters {
		for _, ws1 := range whitespace {
			for _, ws2 := range whitespace {
				for _, nl := range newlines {
					// skip the one case where there is nothing between a key and a value
					if ws1 == "" && dl == "" && ws2 == "" && value != "" {
						continue
					}

					input := fmt.Sprintf("%s%s%s%s%s%s", key, ws1, dl, ws2, value, nl)
					testKeyValue(c, input, key, value)
				}
			}
		}
	}
}

// tests whether key/value pairs exist for a given input.
// keyvalues is expected to be an even number of strings of "key", "value", ...
func testKeyValue(c *C, input string, keyvalues ...string) {
	testKeyValuePrePostfix(c, "${", "}", input, keyvalues...)
}

// tests whether key/value pairs exist for a given input.
// keyvalues is expected to be an even number of strings of "key", "value", ...
func testKeyValuePrePostfix(c *C, prefix, postfix, input string, keyvalues ...string) {
	printf("%q\n", input)

	p, err := Load([]byte(input), ISO_8859_1)
	c.Assert(err, IsNil)
	p.Prefix = prefix
	p.Postfix = postfix
	assertKeyValues(c, input, p, keyvalues...)
}

// tests whether key/value pairs exist for a given input.
// keyvalues is expected to be an even number of strings of "key", "value", ...
func assertKeyValues(c *C, input string, p *Properties, keyvalues ...string) {
	c.Assert(p, NotNil)
	c.Assert(2*p.Len(), Equals, len(keyvalues), Commentf("Odd number of key/value pairs."))

	for i := 0; i < len(keyvalues); i += 2 {
		key, value := keyvalues[i], keyvalues[i+1]
		v, ok := p.Get(key)
		c.Assert(ok, Equals, true, Commentf("No key %q found (input=%q)", key, input))
		c.Assert(v, Equals, value, Commentf("Value %q does not match %q (input=%q)", v, value, input))
	}
}

// prints to stderr if the -verbose flag was given.
func printf(format string, args ...interface{}) {
	if *verbose {
		fmt.Fprintf(os.Stderr, format, args...)
	}
}
