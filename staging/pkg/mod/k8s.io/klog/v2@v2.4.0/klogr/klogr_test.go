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

func TestInfo(t *testing.T) {
	klog.InitFlags(nil)
	flag.CommandLine.Set("v", "10")
	flag.CommandLine.Set("skip_headers", "true")
	flag.CommandLine.Set("logtostderr", "false")
	flag.CommandLine.Set("alsologtostderr", "false")
	flag.Parse()

	tests := map[string]struct {
		klogr          logr.Logger
		text           string
		keysAndValues  []interface{}
		expectedOutput string
	}{
		"should log with values passed to keysAndValues": {
			klogr:         New().V(0),
			text:          "test",
			keysAndValues: []interface{}{"akey", "avalue"},
			expectedOutput: ` "msg"="test"  "akey"="avalue"
`,
		},
		"should not print duplicate keys with the same value": {
			klogr:         New().V(0),
			text:          "test",
			keysAndValues: []interface{}{"akey", "avalue", "akey", "avalue"},
			expectedOutput: ` "msg"="test"  "akey"="avalue"
`,
		},
		"should only print the last duplicate key when the values are passed to Info": {
			klogr:         New().V(0),
			text:          "test",
			keysAndValues: []interface{}{"akey", "avalue", "akey", "avalue2"},
			expectedOutput: ` "msg"="test"  "akey"="avalue2"
`,
		},
		"should only print the duplicate key that is passed to Info if one was passed to the logger": {
			klogr:         New().WithValues("akey", "avalue"),
			text:          "test",
			keysAndValues: []interface{}{"akey", "avalue"},
			expectedOutput: ` "msg"="test"  "akey"="avalue"
`,
		},
		"should only print the key passed to Info when one is already set on the logger": {
			klogr:         New().WithValues("akey", "avalue"),
			text:          "test",
			keysAndValues: []interface{}{"akey", "avalue2"},
			expectedOutput: ` "msg"="test"  "akey"="avalue2"
`,
		},
		"should correctly handle odd-numbers of KVs": {
			text:          "test",
			keysAndValues: []interface{}{"akey", "avalue", "akey2"},
			expectedOutput: ` "msg"="test"  "akey"="avalue" "akey2"=null
`,
		},
		"should correctly html characters": {
			text:          "test",
			keysAndValues: []interface{}{"akey", "<&>"},
			expectedOutput: ` "msg"="test"  "akey"="<&>"
`,
		},
		"should correctly handle odd-numbers of KVs in both log values and Info args": {
			klogr:         New().WithValues("basekey1", "basevar1", "basekey2"),
			text:          "test",
			keysAndValues: []interface{}{"akey", "avalue", "akey2"},
			expectedOutput: ` "msg"="test" "basekey1"="basevar1" "basekey2"=null "akey"="avalue" "akey2"=null
`,
		},
		"should correctly print regular error types": {
			klogr:         New().V(0),
			text:          "test",
			keysAndValues: []interface{}{"err", errors.New("whoops")},
			expectedOutput: ` "msg"="test"  "err"="whoops"
`,
		},
		"should use MarshalJSON if an error type implements it": {
			klogr:         New().V(0),
			text:          "test",
			keysAndValues: []interface{}{"err", &customErrorJSON{"whoops"}},
			expectedOutput: ` "msg"="test"  "err"="WHOOPS"
`,
		},
	}
	for n, test := range tests {
		t.Run(n, func(t *testing.T) {
			klogr := test.klogr
			if klogr == nil {
				klogr = New()
			}

			// hijack the klog output
			tmpWriteBuffer := bytes.NewBuffer(nil)
			klog.SetOutput(tmpWriteBuffer)

			klogr.Info(test.text, test.keysAndValues...)
			// call Flush to ensure the text isn't still buffered
			klog.Flush()

			actual := tmpWriteBuffer.String()
			if actual != test.expectedOutput {
				t.Errorf("expected %q did not match actual %q", test.expectedOutput, actual)
			}
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
