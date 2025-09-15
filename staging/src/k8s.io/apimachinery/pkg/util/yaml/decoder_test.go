/*
Copyright 2014 The Kubernetes Authors.

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

package yaml

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"math/rand"
	"reflect"
	"strings"
	"testing"
)

func TestYAMLDecoderReadBytesLength(t *testing.T) {
	d := `---
stuff: 1
	test-foo: 1
`
	testCases := []struct {
		bufLen    int
		expectLen int
		expectErr error
	}{
		{len(d), len(d), nil},
		{len(d) + 10, len(d), nil},
		{len(d) - 10, len(d) - 10, io.ErrShortBuffer},
	}

	for i, testCase := range testCases {
		r := NewDocumentDecoder(io.NopCloser(bytes.NewReader([]byte(d))))
		b := make([]byte, testCase.bufLen)
		n, err := r.Read(b)
		if err != testCase.expectErr || n != testCase.expectLen {
			t.Fatalf("%d: unexpected body: %d / %v", i, n, err)
		}
	}
}

func TestBigYAML(t *testing.T) {
	d := `
stuff: 1
`
	maxLen := 5 * 1024 * 1024
	bufferLen := 4 * 1024
	//  maxLen 5 M
	dd := strings.Repeat(d, 512*1024)
	r := NewDocumentDecoder(io.NopCloser(bytes.NewReader([]byte(dd[:maxLen-1]))))
	b := make([]byte, bufferLen)
	n, err := r.Read(b)
	if err != io.ErrShortBuffer {
		t.Fatalf("expected ErrShortBuffer: %d / %v", n, err)
	}
	b = make([]byte, maxLen)
	n, err = r.Read(b)
	if err != nil {
		t.Fatalf("expected nil: %d / %v", n, err)
	}
	r = NewDocumentDecoder(io.NopCloser(bytes.NewReader([]byte(dd))))
	b = make([]byte, maxLen)
	n, err = r.Read(b)
	if err != bufio.ErrTooLong {
		t.Fatalf("bufio.Scanner: token too long: %d / %v", n, err)
	}
}

func TestYAMLDecoderCallsAfterErrShortBufferRestOfFrame(t *testing.T) {
	d := `---
stuff: 1
	test-foo: 1`
	r := NewDocumentDecoder(io.NopCloser(bytes.NewReader([]byte(d))))
	b := make([]byte, 12)
	n, err := r.Read(b)
	if err != io.ErrShortBuffer || n != 12 {
		t.Fatalf("expected ErrShortBuffer: %d / %v", n, err)
	}
	expected := "---\nstuff: 1"
	if string(b) != expected {
		t.Fatalf("expected bytes read to be: %s  got: %s", expected, string(b))
	}
	b = make([]byte, 13)
	n, err = r.Read(b)
	if err != nil || n != 13 {
		t.Fatalf("expected nil: %d / %v", n, err)
	}
	expected = "\n\ttest-foo: 1"
	if string(b) != expected {
		t.Fatalf("expected bytes read to be: '%s'  got: '%s'", expected, string(b))
	}
	b = make([]byte, 15)
	n, err = r.Read(b)
	if err != io.EOF || n != 0 { //nolint:errorlint
		t.Fatalf("expected EOF: %d / %v", n, err)
	}
}

func TestSplitYAMLDocument(t *testing.T) {
	testCases := []struct {
		input  string
		atEOF  bool
		expect string
		adv    int
	}{
		{"foo", true, "foo", 3},
		{"fo", false, "", 0},

		{"---", true, "---", 3},
		{"---\n", true, "---\n", 4},
		{"---\n", false, "", 0},

		{"\n---\n", false, "", 5},
		{"\n---\n", true, "", 5},

		{"abc\n---\ndef", true, "abc", 8},
		{"def", true, "def", 3},
		{"", true, "", 0},
	}
	for i, testCase := range testCases {
		adv, token, err := splitYAMLDocument([]byte(testCase.input), testCase.atEOF)
		if err != nil {
			t.Errorf("%d: unexpected error: %v", i, err)
			continue
		}
		if adv != testCase.adv {
			t.Errorf("%d: advance did not match: %d %d", i, testCase.adv, adv)
		}
		if testCase.expect != string(token) {
			t.Errorf("%d: token did not match: %q %q", i, testCase.expect, string(token))
		}
	}
}

func TestGuessJSON(t *testing.T) {
	if r, _, isJSON := GuessJSONStream(bytes.NewReader([]byte(" \n{}")), 100); !isJSON {
		t.Fatalf("expected stream to be JSON")
	} else {
		b := make([]byte, 30)
		n, err := r.Read(b)
		if err != nil || n != 4 {
			t.Fatalf("unexpected body: %d / %v", n, err)
		}
		if string(b[:n]) != " \n{}" {
			t.Fatalf("unexpected body: %q", string(b[:n]))
		}
	}
}

func TestScanYAML(t *testing.T) {
	s := bufio.NewScanner(bytes.NewReader([]byte(`---
stuff: 1

---
  `)))
	s.Split(splitYAMLDocument)
	if !s.Scan() {
		t.Fatalf("should have been able to scan")
	}
	t.Logf("scan: %s", s.Text())
	if !s.Scan() {
		t.Fatalf("should have been able to scan")
	}
	t.Logf("scan: %s", s.Text())
	if s.Scan() {
		t.Fatalf("scan should have been done")
	}
	if s.Err() != nil {
		t.Fatalf("err should have been nil: %v", s.Err())
	}
}

func TestDecodeYAML(t *testing.T) {
	s := NewYAMLToJSONDecoder(bytes.NewReader([]byte(`---
stuff: 1

---   
  `)))
	obj := generic{}
	if err := s.Decode(&obj); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if fmt.Sprintf("%#v", obj) != `yaml.generic{"stuff":1}` {
		t.Errorf("unexpected object: %#v", obj)
	}
	obj = generic{}
	if err := s.Decode(&obj); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(obj) != 0 {
		t.Fatalf("unexpected object: %#v", obj)
	}
	obj = generic{}
	if err := s.Decode(&obj); err != io.EOF { //nolint:errorlint
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestDecodeYAMLSeparatorValidation(t *testing.T) {
	s := NewYAMLToJSONDecoder(bytes.NewReader([]byte(`---
stuff: 1
---    # Make sure termination happen with inline comment
stuff: 2
---
stuff: 3
--- Make sure uncommented content results YAMLSyntaxError

 `)))
	obj := generic{}
	if err := s.Decode(&obj); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if fmt.Sprintf("%#v", obj) != `yaml.generic{"stuff":1}` {
		t.Errorf("unexpected object: %#v", obj)
	}
	obj = generic{}
	if err := s.Decode(&obj); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if fmt.Sprintf("%#v", obj) != `yaml.generic{"stuff":2}` {
		t.Errorf("unexpected object: %#v", obj)
	}
	obj = generic{}
	err := s.Decode(&obj)
	if err == nil {
		t.Fatalf("expected YamlSyntaxError, got nil instead")
	}
	if _, ok := err.(YAMLSyntaxError); !ok {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestDecodeBrokenYAML(t *testing.T) {
	s := NewYAMLOrJSONDecoder(bytes.NewReader([]byte(`---
stuff: 1
		test-foo: 1

---
  `)), 100)
	obj := generic{}
	err := s.Decode(&obj)
	if err == nil {
		t.Fatal("expected error with yaml: violate, got no error")
	}
	fmt.Printf("err: %s\n", err.Error())
	if !strings.Contains(err.Error(), "yaml: line 3:") {
		t.Fatalf("expected %q to have 'yaml: line 3:' found a tab character", err.Error())
	}
}

func TestDecodeBrokenJSON(t *testing.T) {
	s := NewYAMLOrJSONDecoder(bytes.NewReader([]byte(`{
	"foo": {
		"stuff": 1
		"otherStuff": 2
	}
}
  `)), 100)
	obj := generic{}
	err := s.Decode(&obj)
	if err == nil {
		t.Fatal("expected error with json: prefix, got no error")
	}
	const msg = `json: offset 28: invalid character '"' after object key:value pair`
	if msg != err.Error() {
		t.Fatalf("expected %q, got %q", msg, err.Error())
	}
}

type generic map[string]interface{}

func TestYAMLOrJSONDecoder(t *testing.T) {
	testCases := []struct {
		input  string
		buffer int
		isJSON bool
		err    bool
		out    []generic
	}{
		{` {"1":2}{"3":4}`, 2, true, false, []generic{
			{"1": 2},
			{"3": 4},
		}},
		{" \n{}", 3, true, false, []generic{
			{},
		}},
		{" \na: b", 2, false, false, []generic{
			{"a": "b"},
		}},
		{" \n{\"a\": \"b\"}", 2, false, true, []generic{
			{"a": "b"},
		}},
		{" \n{\"a\": \"b\"}", 3, true, false, []generic{
			{"a": "b"},
		}},
		{`   {"a":"b"}`, 100, true, false, []generic{
			{"a": "b"},
		}},
		{"", 1, false, false, []generic{}},
		{"foo: bar\n---\nbaz: biz", 100, false, false, []generic{
			{"foo": "bar"},
			{"baz": "biz"},
		}},
		{"---\nfoo: bar\n--- # with Comment\nbaz: biz", 100, false, false, []generic{
			{"foo": "bar"},
			{"baz": "biz"},
		}},
		// Spaces for indent, tabs are not allowed in YAML.
		{"foo:\n  field: bar\n---\nbaz:\n  field: biz", 100, false, false, []generic{
			{"foo": map[string]any{"field": "bar"}},
			{"baz": map[string]any{"field": "biz"}},
		}},
		{"foo: bar\n---\n", 100, false, false, []generic{
			{"foo": "bar"},
		}},
		{"foo: bar\n---", 100, false, false, []generic{
			{"foo": "bar"},
		}},
		{"foo: bar\n--", 100, false, true, []generic{
			{"foo": "bar"},
		}},
		{"foo: bar\n-", 100, false, true, []generic{
			{"foo": "bar"},
		}},
		{"foo: bar\n", 100, false, false, []generic{
			{"foo": "bar"},
		}},
		// First document is JSON, second is YAML
		{"{\"foo\": \"bar\"}\n---\n{baz: biz}", 100, false, false, []generic{
			{"foo": "bar"},
			{"baz": "biz"},
		}},
		// First document is JSON, second is YAML but with smaller size.
		{"{\"foo\": \"bar\"}\n---\na: b", 100, false, false, []generic{
			{"foo": "bar"},
			{"a": "b"},
		}},
		// First document is JSON, second is YAML,but with smaller size and
		// trailing whitespace.
		{"{\"foo\": \"bar\"}    \n---\na: b", 100, false, false, []generic{
			{"foo": "bar"},
			{"a": "b"},
		}},
		// First document is JSON, second is YAML, longer than the buffer
		{"{\"foo\": \"bar\"}\n---\n{baz: biz0123456780123456780123456780123456780123456789}", 20, false, false, []generic{
			{"foo": "bar"},
			{"baz": "biz0123456780123456780123456780123456780123456789"},
		}},
		// First document is JSON, then whitespace, then YAML
		{"{\"foo\": \"bar\"}    \n---\n{baz: biz}", 100, false, false, []generic{
			{"foo": "bar"},
			{"baz": "biz"},
		}},
		// First document is YAML, second is JSON
		{"{foo: bar}\n---\n{\"baz\": \"biz\"}", 100, false, false, []generic{
			{"foo": "bar"},
			{"baz": "biz"},
		}},
		// First document is JSON, second is YAML, using spaces
		{"{\n  \"foo\": \"bar\"\n}\n---\n{\n  baz: biz\n}", 100, false, false, []generic{
			{"foo": "bar"},
			{"baz": "biz"},
		}},
		// First document is JSON, second is YAML, using tabs
		{"{\n\t\"foo\": \"bar\"\n}\n---\n{\n\tbaz: biz\n}", 100, false, false, []generic{
			{"foo": "bar"},
			{"baz": "biz"},
		}},
		// First 2 documents are JSON, third is YAML (stream is JSON)
		{"{\"foo\": \"bar\"}\n{\"baz\": \"biz\"}\n---\n{qux: zrb}", 100, true, true, nil},
	}
	for i, testCase := range testCases {
		decoder := NewYAMLOrJSONDecoder(bytes.NewReader([]byte(testCase.input)), testCase.buffer)
		objs := []generic{}

		var err error
		for {
			out := make(generic)
			err = decoder.Decode(&out)
			if err != nil {
				break
			}
			objs = append(objs, out)
		}
		if err != io.EOF { //nolint:errorlint
			switch {
			case testCase.err && err == nil:
				t.Errorf("%d: unexpected non-error", i)
				continue
			case !testCase.err && err != nil:
				t.Errorf("%d: unexpected error: %v", i, err)
				continue
			case err != nil:
				continue
			}
		}
		switch {
		case decoder.yaml != nil:
			if testCase.isJSON {
				t.Errorf("%d: expected JSON decoder, got YAML", i)
			}
		case decoder.json != nil:
			if !testCase.isJSON {
				t.Errorf("%d: expected YAML decoder, got JSON", i)
			}
		}
		if fmt.Sprintf("%#v", testCase.out) != fmt.Sprintf("%#v", objs) {
			t.Errorf("%d: objects were not equal: \n%#v\n%#v", i, testCase.out, objs)
		}
	}
}

func TestReadSingleLongLine(t *testing.T) {
	testReadLines(t, []int{128 * 1024})
}

func TestReadRandomLineLengths(t *testing.T) {
	minLength := 100
	maxLength := 96 * 1024
	maxLines := 100

	lineLengths := make([]int, maxLines)
	for i := 0; i < maxLines; i++ {
		lineLengths[i] = rand.Intn(maxLength-minLength) + minLength
	}

	testReadLines(t, lineLengths)
}

func testReadLines(t *testing.T, lineLengths []int) {
	var (
		lines       [][]byte
		inputStream []byte
	)
	for _, lineLength := range lineLengths {
		inputLine := make([]byte, lineLength+1)
		for i := 0; i < lineLength; i++ {
			char := rand.Intn('z'-'A') + 'A'
			inputLine[i] = byte(char)
		}
		inputLine[len(inputLine)-1] = '\n'
		lines = append(lines, inputLine)
	}
	for _, line := range lines {
		inputStream = append(inputStream, line...)
	}

	// init Reader
	reader := bufio.NewReader(bytes.NewReader(inputStream))
	lineReader := &LineReader{reader: reader}

	// read lines
	var readLines [][]byte
	for range lines {
		bytes, err := lineReader.Read()
		if err != nil && err != io.EOF { //nolint:errorlint
			t.Fatalf("failed to read lines: %v", err)
		}
		readLines = append(readLines, bytes)
	}

	// validate
	for i := range lines {
		if len(lines[i]) != len(readLines[i]) {
			t.Fatalf("expected line length: %d, but got %d", len(lines[i]), len(readLines[i]))
		}
		if !reflect.DeepEqual(lines[i], readLines[i]) {
			t.Fatalf("expected line: %v, but got %v", lines[i], readLines[i])
		}
	}
}

func TestTypedJSONOrYamlErrors(t *testing.T) {
	s := NewYAMLOrJSONDecoder(bytes.NewReader([]byte(`{
	"foo": {
		"stuff": 1
		"otherStuff": 2
	}
}
  `)), 100)
	obj := generic{}
	err := s.Decode(&obj)
	if err == nil {
		t.Fatal("expected error with json: prefix, got no error")
	}
	if _, ok := err.(JSONSyntaxError); !ok {
		t.Fatalf("expected %q to be of type JSONSyntaxError", err.Error())
	}

	s = NewYAMLOrJSONDecoder(bytes.NewReader([]byte(`---
stuff: 1
		test-foo: 1

---
  `)), 100)
	obj = generic{}
	err = s.Decode(&obj)
	if err == nil {
		t.Fatal("expected error with yaml: prefix, got no error")
	}
	if _, ok := err.(YAMLSyntaxError); !ok {
		t.Fatalf("expected %q to be of type YAMLSyntaxError", err.Error())
	}
}

func TestUnmarshal(t *testing.T) {
	mapWithIntegerBytes := []byte(`replicas: 1`)
	mapWithInteger := make(map[string]interface{})
	if err := Unmarshal(mapWithIntegerBytes, &mapWithInteger); err != nil {
		t.Fatalf("unexpected error unmarshaling yaml: %v", err)
	}
	if _, ok := mapWithInteger["replicas"].(int64); !ok {
		t.Fatalf(`Expected number in map to be int64 but got "%T"`, mapWithInteger["replicas"])
	}

	sliceWithIntegerBytes := []byte(`- 1`)
	var sliceWithInteger []interface{}
	if err := Unmarshal(sliceWithIntegerBytes, &sliceWithInteger); err != nil {
		t.Fatalf("unexpected error unmarshaling yaml: %v", err)
	}
	if _, ok := sliceWithInteger[0].(int64); !ok {
		t.Fatalf(`Expected number in slice to be int64 but got "%T"`, sliceWithInteger[0])
	}

	integerBytes := []byte(`1`)
	var integer interface{}
	if err := Unmarshal(integerBytes, &integer); err != nil {
		t.Fatalf("unexpected error unmarshaling yaml: %v", err)
	}
	if _, ok := integer.(int64); !ok {
		t.Fatalf(`Expected number to be int64 but got "%T"`, integer)
	}

	otherTypeBytes := []byte(`123: 2`)
	otherType := make(map[int]interface{})
	if err := Unmarshal(otherTypeBytes, &otherType); err != nil {
		t.Fatalf("unexpected error unmarshaling yaml: %v", err)
	}
	if _, ok := otherType[123].(int64); ok {
		t.Fatalf(`Expected number not to be converted to int64`)
	}
	if _, ok := otherType[123].(float64); !ok {
		t.Fatalf(`Expected number to be float64 but got "%T"`, otherType[123])
	}
}
