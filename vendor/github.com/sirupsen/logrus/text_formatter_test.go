package logrus

import (
	"bytes"
	"errors"
	"fmt"
	"strings"
	"testing"
	"time"
)

func TestFormatting(t *testing.T) {
	tf := &TextFormatter{DisableColors: true}

	testCases := []struct {
		value    string
		expected string
	}{
		{`foo`, "time=\"0001-01-01T00:00:00Z\" level=panic test=foo\n"},
	}

	for _, tc := range testCases {
		b, _ := tf.Format(WithField("test", tc.value))

		if string(b) != tc.expected {
			t.Errorf("formatting expected for %q (result was %q instead of %q)", tc.value, string(b), tc.expected)
		}
	}
}

func TestQuoting(t *testing.T) {
	tf := &TextFormatter{DisableColors: true}

	checkQuoting := func(q bool, value interface{}) {
		b, _ := tf.Format(WithField("test", value))
		idx := bytes.Index(b, ([]byte)("test="))
		cont := bytes.Contains(b[idx+5:], []byte("\""))
		if cont != q {
			if q {
				t.Errorf("quoting expected for: %#v", value)
			} else {
				t.Errorf("quoting not expected for: %#v", value)
			}
		}
	}

	checkQuoting(false, "")
	checkQuoting(false, "abcd")
	checkQuoting(false, "v1.0")
	checkQuoting(false, "1234567890")
	checkQuoting(false, "/foobar")
	checkQuoting(false, "foo_bar")
	checkQuoting(false, "foo@bar")
	checkQuoting(false, "foobar^")
	checkQuoting(false, "+/-_^@f.oobar")
	checkQuoting(true, "foobar$")
	checkQuoting(true, "&foobar")
	checkQuoting(true, "x y")
	checkQuoting(true, "x,y")
	checkQuoting(false, errors.New("invalid"))
	checkQuoting(true, errors.New("invalid argument"))

	// Test for quoting empty fields.
	tf.QuoteEmptyFields = true
	checkQuoting(true, "")
	checkQuoting(false, "abcd")
	checkQuoting(true, errors.New("invalid argument"))
}

func TestEscaping(t *testing.T) {
	tf := &TextFormatter{DisableColors: true}

	testCases := []struct {
		value    string
		expected string
	}{
		{`ba"r`, `ba\"r`},
		{`ba'r`, `ba'r`},
	}

	for _, tc := range testCases {
		b, _ := tf.Format(WithField("test", tc.value))
		if !bytes.Contains(b, []byte(tc.expected)) {
			t.Errorf("escaping expected for %q (result was %q instead of %q)", tc.value, string(b), tc.expected)
		}
	}
}

func TestEscaping_Interface(t *testing.T) {
	tf := &TextFormatter{DisableColors: true}

	ts := time.Now()

	testCases := []struct {
		value    interface{}
		expected string
	}{
		{ts, fmt.Sprintf("\"%s\"", ts.String())},
		{errors.New("error: something went wrong"), "\"error: something went wrong\""},
	}

	for _, tc := range testCases {
		b, _ := tf.Format(WithField("test", tc.value))
		if !bytes.Contains(b, []byte(tc.expected)) {
			t.Errorf("escaping expected for %q (result was %q instead of %q)", tc.value, string(b), tc.expected)
		}
	}
}

func TestTimestampFormat(t *testing.T) {
	checkTimeStr := func(format string) {
		customFormatter := &TextFormatter{DisableColors: true, TimestampFormat: format}
		customStr, _ := customFormatter.Format(WithField("test", "test"))
		timeStart := bytes.Index(customStr, ([]byte)("time="))
		timeEnd := bytes.Index(customStr, ([]byte)("level="))
		timeStr := customStr[timeStart+5+len("\"") : timeEnd-1-len("\"")]
		if format == "" {
			format = time.RFC3339
		}
		_, e := time.Parse(format, (string)(timeStr))
		if e != nil {
			t.Errorf("time string \"%s\" did not match provided time format \"%s\": %s", timeStr, format, e)
		}
	}

	checkTimeStr("2006-01-02T15:04:05.000000000Z07:00")
	checkTimeStr("Mon Jan _2 15:04:05 2006")
	checkTimeStr("")
}

func TestDisableTimestampWithColoredOutput(t *testing.T) {
	tf := &TextFormatter{DisableTimestamp: true, ForceColors: true}

	b, _ := tf.Format(WithField("test", "test"))
	if strings.Contains(string(b), "[0000]") {
		t.Error("timestamp not expected when DisableTimestamp is true")
	}
}

// TODO add tests for sorting etc., this requires a parser for the text
// formatter output.
