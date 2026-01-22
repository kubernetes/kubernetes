/*
Copyright 2024 The Kubernetes Authors.

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

// Package logparse provides a parser for the klog text format:
//
//	 I1007 13:16:55.727802 1146763 example.go:57] "Key/value encoding" logger="example" foo="bar" duration="1s" int=42 float=3.14 string="hello world" quotedString="hello \"world\"" multiLinString=<
//		hello
//		world
//	  >
//	  E1007 15:20:04.343375 1157755 example.go:41] Log using Errorf, err: fail
//
// It also supports the klog/ktesting unit test log output:
//
//	  === RUN   TestKlogr
//		    example_test.go:45: I1007 13:28:21.908998] hello world
//		    example_test.go:46: E1007 13:28:21.909034] failed err="failed: some error"
//		    example_test.go:47: I1007 13:28:21.909042] verbosity 1
//		    example_test.go:48: I1007 13:28:21.909047] main/helper: with prefix
//		    example_test.go:50: I1007 13:28:21.909076] key/value pairs int=1 float=2 pair="(1, 2)" raw={"Name":"joe","NS":"kube-system"} slice=[1,2,3,"str"] map={"a":1,"b":2} kobj="kube-system/joe" string="hello world" quotedString="hello \"world\"" multiLineString=<
//		        	hello
//		        	world
//		         >
//		    example_test.go:63: I1007 13:28:21.909085] info message level 4
//		    example_test.go:64: I1007 13:28:21.909089] info message level 5
//	  --- PASS: TestKlogr (0.00s)
//	  PASS
//	  ok  	k8s.io/klog/v2/ktesting/example	(cached)
//
// Arbitrary indention with space or tab is supported. All lines of
// a klog log entry must be indented the same way.
package logparse

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"iter"
	"regexp"
	"strings"
)

// Parse splits log output provided by the reader into individual log entries.
// The original log can be reconstructed verbatim by concatenating these
// entries. If the reader fails with an error, the last log entry will
// capture that error.
func Parse(in io.Reader) []Entry {
	var log []Entry
	for entry := range All(in) {
		log = append(log, entry)
	}
	return log
}

// All is like Parse except that it can be used in a for/range loop:
//
//	for entry := range logparse.All(reader) {
//	    // entry is the next log entry.
//	}
func All(in io.Reader) iter.Seq[Entry] {
	return func(yield func(Entry) bool) {
		parse(in, yield)
	}
}

// Entry is the base type for a log entry.
//
// Use type assertions to check for additional information. All
// of the instances behind this interface are pointers.
type Entry interface {
	// LogData returns a verbatim copy of the original log chunk,
	// including one or more line breaks. Concatenating these chunks
	// from all entries will reconstruct the parsed log.
	LogData() string
}

// ErrorEntry captures the error encountered while reading the log input.
type ErrorEntry struct {
	Err error
}

var _ Entry = &ErrorEntry{}
var _ fmt.Stringer = &ErrorEntry{}

func (e *ErrorEntry) LogData() string { return "" }
func (e *ErrorEntry) String() string  { return e.Err.Error() }

// OtherEntry captures some log line which is not part of a klog log entry.
type OtherEntry struct {
	Data string
}

var _ Entry = &OtherEntry{}
var _ fmt.Stringer = &OtherEntry{}

func (e *OtherEntry) LogData() string { return e.Data }
func (e *OtherEntry) String() string  { return e.Data }

// LogEntry captures some log entry which was recognized as coming from klog.
type KlogEntry struct {
	Data     string
	Severity Severity
}

type Severity byte

const (
	SeverityInfo    = Severity('I')
	SeverityWarning = Severity('W')
	SeverityError   = Severity('E')
	SeverityFatal   = Severity('F')
)

var _ Entry = &KlogEntry{}
var _ fmt.Stringer = &KlogEntry{}

func (e *KlogEntry) LogData() string { return e.Data }
func (e *KlogEntry) String() string  { return e.Data }

func parse(in io.Reader, yield func(Entry) bool) {
	// Read lines using arbitrary length, which can't be done with a
	// bufio.Scanner.
	reader := bufio.NewReader(in)
	line, err := reader.ReadString('\n')
	for {
		// First deliver the current line. This may need to look
		// ahead and thus returns the next line.
		var nextLine string
		var nextErr error
		if len(line) > 0 {
			var cont bool
			nextLine, cont, nextErr = parseLine(reader, line, yield)
			if !cont {
				return
			}
		} else {
			nextLine, nextErr = reader.ReadString('\n')
		}

		// Finalize parsing?
		switch {
		case err == nil:
			// Okay.
		case errors.Is(err, io.EOF):
			return
		default:
			yield(&ErrorEntry{Err: err})
			return
		}

		line = nextLine
		err = nextErr
	}
}

const (
	pid      = `(?<pid>[[:digit:]]+)`
	source   = `(?<source>[^:]+:[[:digit:]]+)`
	severity = `(?<severity>[IWEF])`
	datetime = `(?<month>[[:digit:]]{2})(?<day>[[:digit:]]{2}) (?<hour>[[:digit:]]{2}):(?<minutes>[[:digit:]]{2}):(?<seconds>[[:digit:]]{2})\.(?<microseconds>[[:digit:]]{6})`
)

var (
	klogPrefix = regexp.MustCompile(`^(?<indention>[[:blank:]]*)` +
		`(?:` + source + `: )?` + // `go test` source code
		severity +
		datetime +
		`(?: +` + pid + ` ` + source + `)?` + // klog pid + source code
		`\] `)

	indentionIndex = lookupSubmatch("indention")
	severityIndex  = lookupSubmatch("severity")
)

func lookupSubmatch(name string) int {
	names := klogPrefix.SubexpNames()
	for i, n := range names {
		if n == name {
			return i
		}
	}
	panic(fmt.Errorf("named submatch %q not found in %q", name, klogPrefix.String()))
}

// parseLine deals with one non-empty line. It returns the result of yield and
// potentially the next line and/or a read error. If it doesn't have any new
// data to process, it returns the empty string and a nil error.
func parseLine(reader *bufio.Reader, line string, yield func(Entry) bool) (string, bool, error) {
	match := klogPrefix.FindStringSubmatchIndex(line)
	if match == nil {
		cont := yield(&OtherEntry{Data: line})
		return "", cont, nil
	}

	e := &KlogEntry{
		Data:     line,
		Severity: Severity(line[match[2*severityIndex]]),
		// TODO (?): store more of the fields that have been identified
	}
	// Deal with potential line continuation of multi-line string value,
	// if necessary.
	if !strings.HasSuffix(line, "=<\n") {
		return "", yield(e), nil
	}
	indention := line[match[2*indentionIndex]:match[2*indentionIndex+1]]
	for {
		var err error
		line, err = reader.ReadString('\n')
		if !strings.HasPrefix(line, indention) ||
			!strings.HasPrefix(line[len(indention):], "\t") && !strings.HasPrefix(line[len(indention):], " >") {
			// Some other line (wrong indention or wrong continuation).
			// Yield what we have so far and the go back to processing that new line.
			cont := yield(e)
			return line, cont, err
		}
		e.Data += line
		if err != nil {
			return "", yield(e), err
		}
	}
}
