package klogr

import (
	"bytes"
	"encoding/json"
	"errors"
	"flag"
	"strings"
	"testing"

	"k8s.io/klog/v2"

	"github.com/go-logr/logr"
)

const (
	formatDefault = "Default"
	formatNew     = "New"
)

func testOutput(t *testing.T, format string) {
	new := func() logr.Logger {
		switch format {
		case formatNew:
			return New()
		case formatDefault:
			return NewWithOptions()
		default:
			return NewWithOptions(WithFormat(Format(format)))
		}
	}
	tests := map[string]struct {
		klogr              logr.Logger
		text               string
		keysAndValues      []interface{}
		err                error
		expectedOutput     string
		expectedKlogOutput string
	}{
		"should log with values passed to keysAndValues": {
			klogr:         new().V(0),
			text:          "test",
			keysAndValues: []interface{}{"akey", "avalue"},
			expectedOutput: ` "msg"="test"  "akey"="avalue"
`,
			expectedKlogOutput: `"test" akey="avalue"
`,
		},
		"should log with name and values passed to keysAndValues": {
			klogr:         new().V(0).WithName("me"),
			text:          "test",
			keysAndValues: []interface{}{"akey", "avalue"},
			expectedOutput: `me "msg"="test"  "akey"="avalue"
`,
			expectedKlogOutput: `"me: test" akey="avalue"
`,
		},
		"should log with multiple names and values passed to keysAndValues": {
			klogr:         new().V(0).WithName("hello").WithName("world"),
			text:          "test",
			keysAndValues: []interface{}{"akey", "avalue"},
			expectedOutput: `hello/world "msg"="test"  "akey"="avalue"
`,
			expectedKlogOutput: `"hello/world: test" akey="avalue"
`,
		},
		"should not print duplicate keys with the same value": {
			klogr:         new().V(0),
			text:          "test",
			keysAndValues: []interface{}{"akey", "avalue", "akey", "avalue"},
			expectedOutput: ` "msg"="test"  "akey"="avalue"
`,
			expectedKlogOutput: `"test" akey="avalue"
`,
		},
		"should only print the last duplicate key when the values are passed to Info": {
			klogr:         new().V(0),
			text:          "test",
			keysAndValues: []interface{}{"akey", "avalue", "akey", "avalue2"},
			expectedOutput: ` "msg"="test"  "akey"="avalue2"
`,
			expectedKlogOutput: `"test" akey="avalue2"
`,
		},
		"should only print the duplicate key that is passed to Info if one was passed to the logger": {
			klogr:         new().WithValues("akey", "avalue"),
			text:          "test",
			keysAndValues: []interface{}{"akey", "avalue"},
			expectedOutput: ` "msg"="test"  "akey"="avalue"
`,
			expectedKlogOutput: `"test" akey="avalue"
`,
		},
		"should sort within logger and parameter key/value pairs in the default format and dump the logger pairs first": {
			klogr:         new().WithValues("akey9", "avalue9", "akey8", "avalue8", "akey1", "avalue1"),
			text:          "test",
			keysAndValues: []interface{}{"akey5", "avalue5", "akey4", "avalue4"},
			expectedOutput: ` "msg"="test" "akey1"="avalue1" "akey8"="avalue8" "akey9"="avalue9" "akey4"="avalue4" "akey5"="avalue5"
`,
			expectedKlogOutput: `"test" akey9="avalue9" akey8="avalue8" akey1="avalue1" akey5="avalue5" akey4="avalue4"
`,
		},
		"should only print the key passed to Info when one is already set on the logger": {
			klogr:         new().WithValues("akey", "avalue"),
			text:          "test",
			keysAndValues: []interface{}{"akey", "avalue2"},
			expectedOutput: ` "msg"="test"  "akey"="avalue2"
`,
			expectedKlogOutput: `"test" akey="avalue2"
`,
		},
		"should correctly handle odd-numbers of KVs": {
			text:          "test",
			keysAndValues: []interface{}{"akey", "avalue", "akey2"},
			expectedOutput: ` "msg"="test"  "akey"="avalue" "akey2"=null
`,
			expectedKlogOutput: `"test" akey="avalue" akey2=<nil>
`,
		},
		"should correctly html characters": {
			text:          "test",
			keysAndValues: []interface{}{"akey", "<&>"},
			expectedOutput: ` "msg"="test"  "akey"="<&>"
`,
			expectedKlogOutput: `"test" akey="<&>"
`,
		},
		"should correctly handle odd-numbers of KVs in both log values and Info args": {
			klogr:         new().WithValues("basekey1", "basevar1", "basekey2"),
			text:          "test",
			keysAndValues: []interface{}{"akey", "avalue", "akey2"},
			expectedOutput: ` "msg"="test" "basekey1"="basevar1" "basekey2"=null "akey"="avalue" "akey2"=null
`,
			expectedKlogOutput: `"test" basekey1="basevar1" basekey2=<nil> akey="avalue" akey2=<nil>
`,
		},
		"should correctly print regular error types": {
			klogr:         new().V(0),
			text:          "test",
			keysAndValues: []interface{}{"err", errors.New("whoops")},
			expectedOutput: ` "msg"="test"  "err"="whoops"
`,
			expectedKlogOutput: `"test" err="whoops"
`,
		},
		"should use MarshalJSON in the default format if an error type implements it": {
			klogr:         new().V(0),
			text:          "test",
			keysAndValues: []interface{}{"err", &customErrorJSON{"whoops"}},
			expectedOutput: ` "msg"="test"  "err"="WHOOPS"
`,
			expectedKlogOutput: `"test" err="whoops"
`,
		},
		"should correctly print regular error types when using logr.Error": {
			klogr: new().V(0),
			text:  "test",
			err:   errors.New("whoops"),
			// The message is printed to three different log files (info, warning, error), so we see it three times in our output buffer.
			expectedOutput: ` "msg"="test" "error"="whoops"  
 "msg"="test" "error"="whoops"  
 "msg"="test" "error"="whoops"  
`,
			expectedKlogOutput: `"test" err="whoops"
"test" err="whoops"
"test" err="whoops"
`,
		},
	}
	for n, test := range tests {
		t.Run(n, func(t *testing.T) {
			klogr := test.klogr
			if klogr == nil {
				klogr = new()
			}

			// hijack the klog output
			tmpWriteBuffer := bytes.NewBuffer(nil)
			klog.SetOutput(tmpWriteBuffer)

			if test.err != nil {
				klogr.Error(test.err, test.text, test.keysAndValues...)
			} else {
				klogr.Info(test.text, test.keysAndValues...)
			}

			// call Flush to ensure the text isn't still buffered
			klog.Flush()

			actual := tmpWriteBuffer.String()
			expectedOutput := test.expectedOutput
			if format == string(FormatKlog) || format == formatDefault {
				expectedOutput = test.expectedKlogOutput
			}
			if actual != expectedOutput {
				t.Errorf("expected %q did not match actual %q", expectedOutput, actual)
			}
		})
	}
}

func TestOutput(t *testing.T) {
	klog.InitFlags(nil)
	flag.CommandLine.Set("v", "10")
	flag.CommandLine.Set("skip_headers", "true")
	flag.CommandLine.Set("logtostderr", "false")
	flag.CommandLine.Set("alsologtostderr", "false")
	flag.CommandLine.Set("stderrthreshold", "10")
	flag.Parse()

	formats := []string{
		formatNew,
		formatDefault,
		string(FormatSerialize),
		string(FormatKlog),
	}
	for _, format := range formats {
		t.Run(format, func(t *testing.T) {
			testOutput(t, format)
		})
	}
}

type customErrorJSON struct {
	s string
}

func (e *customErrorJSON) Error() string {
	return e.s
}

func (e *customErrorJSON) MarshalJSON() ([]byte, error) {
	return json.Marshal(strings.ToUpper(e.s))
}
