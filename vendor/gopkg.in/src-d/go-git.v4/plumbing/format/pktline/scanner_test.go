package pktline_test

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"strings"

	"gopkg.in/src-d/go-git.v4/plumbing/format/pktline"

	. "gopkg.in/check.v1"
)

type SuiteScanner struct{}

var _ = Suite(&SuiteScanner{})

func (s *SuiteScanner) TestInvalid(c *C) {
	for _, test := range [...]string{
		"0001", "0002", "0003", "0004",
		"0001asdfsadf", "0004foo",
		"fff1", "fff2",
		"gorka",
		"0", "003",
		"   5a", "5   a", "5   \n",
		"-001", "-000",
	} {
		r := strings.NewReader(test)
		sc := pktline.NewScanner(r)
		_ = sc.Scan()
		c.Assert(sc.Err(), ErrorMatches, pktline.ErrInvalidPktLen.Error(),
			Commentf("data = %q", test))
	}
}

func (s *SuiteScanner) TestEmptyReader(c *C) {
	r := strings.NewReader("")
	sc := pktline.NewScanner(r)
	hasPayload := sc.Scan()
	c.Assert(hasPayload, Equals, false)
	c.Assert(sc.Err(), Equals, nil)
}

func (s *SuiteScanner) TestFlush(c *C) {
	var buf bytes.Buffer
	e := pktline.NewEncoder(&buf)
	err := e.Flush()
	c.Assert(err, IsNil)

	sc := pktline.NewScanner(&buf)
	c.Assert(sc.Scan(), Equals, true)

	payload := sc.Bytes()
	c.Assert(len(payload), Equals, 0)
}

func (s *SuiteScanner) TestPktLineTooShort(c *C) {
	r := strings.NewReader("010cfoobar")

	sc := pktline.NewScanner(r)

	c.Assert(sc.Scan(), Equals, false)
	c.Assert(sc.Err(), ErrorMatches, "unexpected EOF")
}

func (s *SuiteScanner) TestScanAndPayload(c *C) {
	for _, test := range [...]string{
		"a",
		"a\n",
		strings.Repeat("a", 100),
		strings.Repeat("a", 100) + "\n",
		strings.Repeat("\x00", 100),
		strings.Repeat("\x00", 100) + "\n",
		strings.Repeat("a", pktline.MaxPayloadSize),
		strings.Repeat("a", pktline.MaxPayloadSize-1) + "\n",
	} {
		var buf bytes.Buffer
		e := pktline.NewEncoder(&buf)
		err := e.EncodeString(test)
		c.Assert(err, IsNil,
			Commentf("input len=%x, contents=%.10q\n", len(test), test))

		sc := pktline.NewScanner(&buf)
		c.Assert(sc.Scan(), Equals, true,
			Commentf("test = %.20q...", test))

		obtained := sc.Bytes()
		c.Assert(obtained, DeepEquals, []byte(test),
			Commentf("in = %.20q out = %.20q", test, string(obtained)))
	}
}

func (s *SuiteScanner) TestSkip(c *C) {
	for _, test := range [...]struct {
		input    []string
		n        int
		expected []byte
	}{
		{
			input: []string{
				"first",
				"second",
				"third"},
			n:        1,
			expected: []byte("second"),
		},
		{
			input: []string{
				"first",
				"second",
				"third"},
			n:        2,
			expected: []byte("third"),
		},
	} {
		var buf bytes.Buffer
		e := pktline.NewEncoder(&buf)
		err := e.EncodeString(test.input...)
		c.Assert(err, IsNil)

		sc := pktline.NewScanner(&buf)
		for i := 0; i < test.n; i++ {
			c.Assert(sc.Scan(), Equals, true,
				Commentf("scan error = %s", sc.Err()))
		}
		c.Assert(sc.Scan(), Equals, true,
			Commentf("scan error = %s", sc.Err()))

		obtained := sc.Bytes()
		c.Assert(obtained, DeepEquals, test.expected,
			Commentf("\nin = %.20q\nout = %.20q\nexp = %.20q",
				test.input, obtained, test.expected))
	}
}

func (s *SuiteScanner) TestEOF(c *C) {
	var buf bytes.Buffer
	e := pktline.NewEncoder(&buf)
	err := e.EncodeString("first", "second")
	c.Assert(err, IsNil)

	sc := pktline.NewScanner(&buf)
	for sc.Scan() {
	}
	c.Assert(sc.Err(), IsNil)
}

type mockReader struct{}

func (r *mockReader) Read([]byte) (int, error) { return 0, errors.New("foo") }

func (s *SuiteScanner) TestInternalReadError(c *C) {
	sc := pktline.NewScanner(&mockReader{})
	c.Assert(sc.Scan(), Equals, false)
	c.Assert(sc.Err(), ErrorMatches, "foo")
}

// A section are several non flush-pkt lines followed by a flush-pkt, which
// how the git protocol sends long messages.
func (s *SuiteScanner) TestReadSomeSections(c *C) {
	nSections := 2
	nLines := 4
	data := sectionsExample(c, nSections, nLines)
	sc := pktline.NewScanner(data)

	sectionCounter := 0
	lineCounter := 0
	for sc.Scan() {
		if len(sc.Bytes()) == 0 {
			sectionCounter++
		}
		lineCounter++
	}
	c.Assert(sc.Err(), IsNil)
	c.Assert(sectionCounter, Equals, nSections)
	c.Assert(lineCounter, Equals, (1+nLines)*nSections)
}

// returns nSection sections, each of them with nLines pkt-lines (not
// counting the flush-pkt:
//
// 0009 0.0\n
// 0009 0.1\n
// ...
// 0000
// and so on
func sectionsExample(c *C, nSections, nLines int) io.Reader {
	var buf bytes.Buffer
	e := pktline.NewEncoder(&buf)

	for section := 0; section < nSections; section++ {
		ss := []string{}
		for line := 0; line < nLines; line++ {
			line := fmt.Sprintf(" %d.%d\n", section, line)
			ss = append(ss, line)
		}
		err := e.EncodeString(ss...)
		c.Assert(err, IsNil)
		err = e.Flush()
		c.Assert(err, IsNil)
	}

	return &buf
}

func ExampleScanner() {
	// A reader is needed as input.
	input := strings.NewReader("000ahello\n" +
		"000bworld!\n" +
		"0000",
	)

	// Create the scanner...
	s := pktline.NewScanner(input)

	// and scan every pkt-line found in the input.
	for s.Scan() {
		payload := s.Bytes()
		if len(payload) == 0 { // zero sized payloads correspond to flush-pkts.
			fmt.Println("FLUSH-PKT DETECTED")
		} else { // otherwise, you will be able to access the full payload.
			fmt.Printf("PAYLOAD = %q\n", string(payload))
		}
	}

	// this will catch any error when reading from the input, if any.
	if s.Err() != nil {
		fmt.Println(s.Err())
	}

	// Output:
	// PAYLOAD = "hello\n"
	// PAYLOAD = "world!\n"
	// FLUSH-PKT DETECTED
}
