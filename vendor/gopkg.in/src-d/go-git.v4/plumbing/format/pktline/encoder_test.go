package pktline_test

import (
	"bytes"
	"os"
	"strings"
	"testing"

	"gopkg.in/src-d/go-git.v4/plumbing/format/pktline"

	. "gopkg.in/check.v1"
)

func Test(t *testing.T) { TestingT(t) }

type SuiteEncoder struct{}

var _ = Suite(&SuiteEncoder{})

func (s *SuiteEncoder) TestFlush(c *C) {
	var buf bytes.Buffer
	e := pktline.NewEncoder(&buf)

	err := e.Flush()
	c.Assert(err, IsNil)

	obtained := buf.Bytes()
	c.Assert(obtained, DeepEquals, pktline.FlushPkt)
}

func (s *SuiteEncoder) TestEncode(c *C) {
	for i, test := range [...]struct {
		input    [][]byte
		expected []byte
	}{
		{
			input: [][]byte{
				[]byte("hello\n"),
			},
			expected: []byte("000ahello\n"),
		}, {
			input: [][]byte{
				[]byte("hello\n"),
				pktline.Flush,
			},
			expected: []byte("000ahello\n0000"),
		}, {
			input: [][]byte{
				[]byte("hello\n"),
				[]byte("world!\n"),
				[]byte("foo"),
			},
			expected: []byte("000ahello\n000bworld!\n0007foo"),
		}, {
			input: [][]byte{
				[]byte("hello\n"),
				pktline.Flush,
				[]byte("world!\n"),
				[]byte("foo"),
				pktline.Flush,
			},
			expected: []byte("000ahello\n0000000bworld!\n0007foo0000"),
		}, {
			input: [][]byte{
				[]byte(strings.Repeat("a", pktline.MaxPayloadSize)),
			},
			expected: []byte(
				"fff0" + strings.Repeat("a", pktline.MaxPayloadSize)),
		}, {
			input: [][]byte{
				[]byte(strings.Repeat("a", pktline.MaxPayloadSize)),
				[]byte(strings.Repeat("b", pktline.MaxPayloadSize)),
			},
			expected: []byte(
				"fff0" + strings.Repeat("a", pktline.MaxPayloadSize) +
					"fff0" + strings.Repeat("b", pktline.MaxPayloadSize)),
		},
	} {
		comment := Commentf("input %d = %v\n", i, test.input)

		var buf bytes.Buffer
		e := pktline.NewEncoder(&buf)

		err := e.Encode(test.input...)
		c.Assert(err, IsNil, comment)

		c.Assert(buf.Bytes(), DeepEquals, test.expected, comment)
	}
}

func (s *SuiteEncoder) TestEncodeErrPayloadTooLong(c *C) {
	for i, input := range [...][][]byte{
		{
			[]byte(strings.Repeat("a", pktline.MaxPayloadSize+1)),
		},
		{
			[]byte("hello world!"),
			[]byte(strings.Repeat("a", pktline.MaxPayloadSize+1)),
		},
		{
			[]byte("hello world!"),
			[]byte(strings.Repeat("a", pktline.MaxPayloadSize+1)),
			[]byte("foo"),
		},
	} {
		comment := Commentf("input %d = %v\n", i, input)

		var buf bytes.Buffer
		e := pktline.NewEncoder(&buf)

		err := e.Encode(input...)
		c.Assert(err, Equals, pktline.ErrPayloadTooLong, comment)
	}
}

func (s *SuiteEncoder) TestEncodeStrings(c *C) {
	for i, test := range [...]struct {
		input    []string
		expected []byte
	}{
		{
			input: []string{
				"hello\n",
			},
			expected: []byte("000ahello\n"),
		}, {
			input: []string{
				"hello\n",
				pktline.FlushString,
			},
			expected: []byte("000ahello\n0000"),
		}, {
			input: []string{
				"hello\n",
				"world!\n",
				"foo",
			},
			expected: []byte("000ahello\n000bworld!\n0007foo"),
		}, {
			input: []string{
				"hello\n",
				pktline.FlushString,
				"world!\n",
				"foo",
				pktline.FlushString,
			},
			expected: []byte("000ahello\n0000000bworld!\n0007foo0000"),
		}, {
			input: []string{
				strings.Repeat("a", pktline.MaxPayloadSize),
			},
			expected: []byte(
				"fff0" + strings.Repeat("a", pktline.MaxPayloadSize)),
		}, {
			input: []string{
				strings.Repeat("a", pktline.MaxPayloadSize),
				strings.Repeat("b", pktline.MaxPayloadSize),
			},
			expected: []byte(
				"fff0" + strings.Repeat("a", pktline.MaxPayloadSize) +
					"fff0" + strings.Repeat("b", pktline.MaxPayloadSize)),
		},
	} {
		comment := Commentf("input %d = %v\n", i, test.input)

		var buf bytes.Buffer
		e := pktline.NewEncoder(&buf)

		err := e.EncodeString(test.input...)
		c.Assert(err, IsNil, comment)
		c.Assert(buf.Bytes(), DeepEquals, test.expected, comment)
	}
}

func (s *SuiteEncoder) TestEncodeStringErrPayloadTooLong(c *C) {
	for i, input := range [...][]string{
		{
			strings.Repeat("a", pktline.MaxPayloadSize+1),
		},
		{
			"hello world!",
			strings.Repeat("a", pktline.MaxPayloadSize+1),
		},
		{
			"hello world!",
			strings.Repeat("a", pktline.MaxPayloadSize+1),
			"foo",
		},
	} {
		comment := Commentf("input %d = %v\n", i, input)

		var buf bytes.Buffer
		e := pktline.NewEncoder(&buf)

		err := e.EncodeString(input...)
		c.Assert(err, Equals, pktline.ErrPayloadTooLong, comment)
	}
}

func (s *SuiteEncoder) TestEncodef(c *C) {
	format := " %s %d\n"
	str := "foo"
	d := 42

	var buf bytes.Buffer
	e := pktline.NewEncoder(&buf)

	err := e.Encodef(format, str, d)
	c.Assert(err, IsNil)

	expected := []byte("000c foo 42\n")
	c.Assert(buf.Bytes(), DeepEquals, expected)
}

func ExampleEncoder() {
	// Create an encoder that writes pktlines to stdout.
	e := pktline.NewEncoder(os.Stdout)

	// Encode some data as a new pkt-line.
	_ = e.Encode([]byte("data\n")) // error checks removed for brevity

	// Encode a flush-pkt.
	_ = e.Flush()

	// Encode a couple of byte slices and a flush in one go. Each of
	// them will end up as payloads of their own pktlines.
	_ = e.Encode(
		[]byte("hello\n"),
		[]byte("world!\n"),
		pktline.Flush,
	)

	// You can also encode strings:
	_ = e.EncodeString(
		"foo\n",
		"bar\n",
		pktline.FlushString,
	)

	// You can also format and encode a payload:
	_ = e.Encodef(" %s %d\n", "foo", 42)
	// Output:
	// 0009data
	// 0000000ahello
	// 000bworld!
	// 00000008foo
	// 0008bar
	// 0000000c foo 42
}
