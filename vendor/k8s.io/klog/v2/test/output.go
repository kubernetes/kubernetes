/*
Copyright 2021 The Kubernetes Authors.

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

// Package test contains a reusable unit test for logging output and behavior.
package test

import (
	"bytes"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/go-logr/logr"

	"k8s.io/klog/v2"
	"k8s.io/klog/v2/textlogger"
)

// InitKlog must be called in a test to configure klog for testing.
// The previous klog configuration will be restored automatically
// after the test.
//
// The returned flag set has the klog flags registered. It can
// be used to make further changes to the klog configuration.
func InitKlog(tb testing.TB) *flag.FlagSet {
	state := klog.CaptureState()
	tb.Cleanup(state.Restore)

	expectNoError := func(err error) {
		if err != nil {
			tb.Fatalf("unexpected error: %v", err)
		}
	}

	// klog gets configured so that it writes to a single output file that
	// will be set during tests with SetOutput.
	var fs flag.FlagSet
	klog.InitFlags(&fs)
	expectNoError(fs.Set("v", "10"))
	expectNoError(fs.Set("log_file", "/dev/null"))
	expectNoError(fs.Set("logtostderr", "false"))
	expectNoError(fs.Set("alsologtostderr", "false"))
	expectNoError(fs.Set("stderrthreshold", "10"))

	return &fs
}

// OutputConfig contains optional settings for Output.
type OutputConfig struct {
	// NewLogger is called to create a new logger. If nil, output via klog
	// is tested. Support for -vmodule is optional.  ClearLogger is called
	// after each test, therefore it is okay to user SetLogger without
	// undoing that in the callback.
	NewLogger func(out io.Writer, v int, vmodule string) logr.Logger

	// AsBackend enables testing through klog and the logger set there with
	// SetLogger.
	AsBackend bool

	// ExpectedOutputMapping replaces the builtin expected output for test
	// cases with something else. If nil or a certain case is not present,
	// the original text is used.
	//
	// The expected output uses <LINE> as a placeholder for the line of the
	// log call. The source code is always the output.go file itself. When
	// testing a logger directly, <WITH-VALUES-LINE> is used for the first
	// WithValues call, <WITH-VALUES-LINE-2> for a second and
	// <WITH-VALUES-LINE-3> for a third.
	ExpectedOutputMapping map[string]string

	// SupportsVModule indicates that the logger supports the vmodule
	// parameter. Ignored when logging through klog.
	SupportsVModule bool
}

type testcase struct {
	withHelper bool // use wrappers that get skipped during stack unwinding
	withNames  []string
	// For a first WithValues call: logger1 := logger.WithValues()
	withValues []interface{}
	// For another WithValues call: logger2 := logger1.WithValues()
	moreValues []interface{}
	// For another WithValues call on the same logger as before: logger3 := logger1.WithValues()
	evenMoreValues []interface{}
	v              int
	vmodule        string
	text           string
	values         []interface{}
	err            error
	expectedOutput string
}

var tests = map[string]testcase{
	"log with values": {
		text:   "test",
		values: []interface{}{"akey", "avalue"},
		expectedOutput: `I output.go:<LINE>] "test" akey="avalue"
`,
	},
	"call depth": {
		text:       "helper",
		withHelper: true,
		values:     []interface{}{"akey", "avalue"},
		expectedOutput: `I output.go:<LINE>] "helper" akey="avalue"
`,
	},
	"verbosity enabled": {
		text: "you see me",
		v:    9,
		expectedOutput: `I output.go:<LINE>] "you see me"
`,
	},
	"verbosity disabled": {
		text: "you don't see me",
		v:    11,
	},
	"vmodule": {
		text:    "v=11: you see me because of -vmodule output=11",
		v:       11,
		vmodule: "output=11",
		expectedOutput: `I output.go:<LINE>] "v=11: you see me because of -vmodule output=11"
`,
	},
	"other vmodule": {
		text:    "v=11: you still don't see me because of -vmodule output_helper=11",
		v:       11,
		vmodule: "output_helper=11",
	},
	"vmodule with helper": {
		text:       "v=11: you see me because of -vmodule output=11",
		withHelper: true,
		v:          11,
		vmodule:    "output=11",
		expectedOutput: `I output.go:<LINE>] "v=11: you see me because of -vmodule output=11"
`,
	},
	"other vmodule with helper": {
		text:       "v=11: you still don't see me because of -vmodule output_helper=11",
		withHelper: true,
		v:          11,
		vmodule:    "output_helper=11",
	},
	"log with name and values": {
		withNames: []string{"me"},
		text:      "test",
		values:    []interface{}{"akey", "avalue"},
		expectedOutput: `I output.go:<LINE>] "test" logger="me" akey="avalue"
`,
	},
	"log with multiple names and values": {
		withNames: []string{"hello", "world"},
		text:      "test",
		values:    []interface{}{"akey", "avalue"},
		expectedOutput: `I output.go:<LINE>] "test" logger="hello.world" akey="avalue"
`,
	},
	"override single value": {
		withValues: []interface{}{"akey", "avalue"},
		text:       "test",
		values:     []interface{}{"akey", "avalue2"},
		expectedOutput: `I output.go:<LINE>] "test" akey="avalue2"
`,
	},
	"override WithValues": {
		withValues: []interface{}{"duration", time.Hour, "X", "y"},
		text:       "test",
		values:     []interface{}{"duration", time.Minute, "A", "b"},
		expectedOutput: `I output.go:<LINE>] "test" X="y" duration="1m0s" A="b"
`,
	},
	"odd WithValues": {
		withValues: []interface{}{"keyWithoutValue"},
		moreValues: []interface{}{"anotherKeyWithoutValue"},
		text:       "odd WithValues",
		expectedOutput: `I output.go:<LINE>] "odd WithValues" keyWithoutValue="(MISSING)"
I output.go:<LINE>] "odd WithValues" keyWithoutValue="(MISSING)" anotherKeyWithoutValue="(MISSING)"
I output.go:<LINE>] "odd WithValues" keyWithoutValue="(MISSING)"
`,
	},
	"multiple WithValues": {
		withValues:     []interface{}{"firstKey", 1},
		moreValues:     []interface{}{"secondKey", 2},
		evenMoreValues: []interface{}{"secondKey", 3},
		text:           "test",
		expectedOutput: `I output.go:<LINE>] "test" firstKey=1
I output.go:<LINE>] "test" firstKey=1 secondKey=2
I output.go:<LINE>] "test" firstKey=1
I output.go:<LINE>] "test" firstKey=1 secondKey=3
`,
	},
	"empty WithValues": {
		withValues: []interface{}{},
		text:       "test",
		expectedOutput: `I output.go:<LINE>] "test"
`,
	},
	"print duplicate keys in arguments": {
		text:   "test",
		values: []interface{}{"akey", "avalue", "akey", "avalue2"},
		expectedOutput: `I output.go:<LINE>] "test" akey="avalue" akey="avalue2"
`,
	},
	"preserve order of key/value pairs": {
		withValues: []interface{}{"akey9", "avalue9", "akey8", "avalue8", "akey1", "avalue1"},
		text:       "test",
		values:     []interface{}{"akey5", "avalue5", "akey4", "avalue4"},
		expectedOutput: `I output.go:<LINE>] "test" akey9="avalue9" akey8="avalue8" akey1="avalue1" akey5="avalue5" akey4="avalue4"
`,
	},
	"handle odd-numbers of KVs": {
		text:   "odd arguments",
		values: []interface{}{"akey", "avalue", "akey2"},
		expectedOutput: `I output.go:<LINE>] "odd arguments" akey="avalue" akey2="(MISSING)"
`,
	},
	"html characters": {
		text:   "test",
		values: []interface{}{"akey", "<&>"},
		expectedOutput: `I output.go:<LINE>] "test" akey="<&>"
`,
	},
	"quotation": {
		text:   `"quoted"`,
		values: []interface{}{"key", `"quoted value"`},
		expectedOutput: `I output.go:<LINE>] "\"quoted\"" key="\"quoted value\""
`,
	},
	"handle odd-numbers of KVs in both log values and Info args": {
		withValues: []interface{}{"basekey1", "basevar1", "basekey2"},
		text:       "both odd",
		values:     []interface{}{"akey", "avalue", "akey2"},
		expectedOutput: `I output.go:<LINE>] "both odd" basekey1="basevar1" basekey2="(MISSING)" akey="avalue" akey2="(MISSING)"
`,
	},
	"KObj": {
		text:   "test",
		values: []interface{}{"pod", klog.KObj(&kmeta{Name: "pod-1", Namespace: "kube-system"})},
		expectedOutput: `I output.go:<LINE>] "test" pod="kube-system/pod-1"
`,
	},
	"KObjs": {
		text: "KObjs",
		values: []interface{}{"pods",
			klog.KObjs([]interface{}{
				&kmeta{Name: "pod-1", Namespace: "kube-system"},
				&kmeta{Name: "pod-2", Namespace: "kube-system"},
			})},
		expectedOutput: `I output.go:<LINE>] "KObjs" pods=[{"name":"pod-1","namespace":"kube-system"},{"name":"pod-2","namespace":"kube-system"}]
`,
	},
	"KObjSlice okay": {
		text: "KObjSlice",
		values: []interface{}{"pods",
			klog.KObjSlice([]interface{}{
				&kmeta{Name: "pod-1", Namespace: "kube-system"},
				&kmeta{Name: "pod-2", Namespace: "kube-system"},
			})},
		expectedOutput: `I output.go:<LINE>] "KObjSlice" pods=["kube-system/pod-1","kube-system/pod-2"]
`,
	},
	"KObjSlice nil arg": {
		text: "test",
		values: []interface{}{"pods",
			klog.KObjSlice(nil)},
		expectedOutput: `I output.go:<LINE>] "test" pods=null
`,
	},
	"KObjSlice int arg": {
		text: "test",
		values: []interface{}{"pods",
			klog.KObjSlice(1)},
		expectedOutput: `I output.go:<LINE>] "test" pods="<KObjSlice needs a slice, got type int>"
`,
	},
	"KObjSlice nil entry": {
		text: "test",
		values: []interface{}{"pods",
			klog.KObjSlice([]interface{}{
				&kmeta{Name: "pod-1", Namespace: "kube-system"},
				nil,
			})},
		expectedOutput: `I output.go:<LINE>] "test" pods=["kube-system/pod-1",null]
`,
	},
	"KObjSlice ints": {
		text: "test",
		values: []interface{}{"ints",
			klog.KObjSlice([]int{1, 2, 3})},
		expectedOutput: `I output.go:<LINE>] "test" ints=["<KObjSlice needs a slice of values implementing KMetadata, got type int>"]
`,
	},
	"regular error types as value": {
		text:   "test",
		values: []interface{}{"err", errors.New("whoops")},
		expectedOutput: `I output.go:<LINE>] "test" err="whoops"
`,
	},
	"ignore MarshalJSON": {
		text:   "test",
		values: []interface{}{"err", &customErrorJSON{"whoops"}},
		expectedOutput: `I output.go:<LINE>] "test" err="whoops"
`,
	},
	"regular error types when using logr.Error": {
		text: "test",
		err:  errors.New("whoops"),
		expectedOutput: `E output.go:<LINE>] "test" err="whoops"
`,
	},
	"Error() for nil": {
		text: "error nil",
		err:  (*customErrorJSON)(nil),
		expectedOutput: `E output.go:<LINE>] "error nil" err="<panic: runtime error: invalid memory address or nil pointer dereference>"
`,
	},
	"String() for nil": {
		text:   "stringer nil",
		values: []interface{}{"stringer", (*stringer)(nil)},
		expectedOutput: `I output.go:<LINE>] "stringer nil" stringer="<panic: runtime error: invalid memory address or nil pointer dereference>"
`,
	},
	"MarshalLog() for nil": {
		text:   "marshaler nil",
		values: []interface{}{"obj", (*klog.ObjectRef)(nil)},
		expectedOutput: `I output.go:<LINE>] "marshaler nil" obj="<panic: value method k8s.io/klog/v2.ObjectRef.WriteText called using nil *ObjectRef pointer>"
`,
	},
	"Error() that panics": {
		text: "error panic",
		err:  faultyError{},
		expectedOutput: `E output.go:<LINE>] "error panic" err="<panic: fake Error panic>"
`,
	},
	"String() that panics": {
		text:   "stringer panic",
		values: []interface{}{"stringer", faultyStringer{}},
		expectedOutput: `I output.go:<LINE>] "stringer panic" stringer="<panic: fake String panic>"
`,
	},
	"MarshalLog() that panics": {
		text:   "marshaler panic",
		values: []interface{}{"obj", faultyMarshaler{}},
		expectedOutput: `I output.go:<LINE>] "marshaler panic" obj="<panic: fake MarshalLog panic>"
`,
	},
	"MarshalLog() that returns itself": {
		text:   "marshaler recursion",
		values: []interface{}{"obj", recursiveMarshaler{}},
		expectedOutput: `I output.go:<LINE>] "marshaler recursion" obj={}
`,
	},
	"handle integer keys": {
		withValues: []interface{}{1, "value", 2, "value2"},
		text:       "integer keys",
		values:     []interface{}{"akey", "avalue", "akey2"},
		expectedOutput: `I output.go:<LINE>] "integer keys" %!s(int=1)="value" %!s(int=2)="value2" akey="avalue" akey2="(MISSING)"
`,
	},
	"struct keys": {
		withValues: []interface{}{struct{ name string }{"name"}, "value", "test", "other value"},
		text:       "struct keys",
		values:     []interface{}{"key", "val"},
		expectedOutput: `I output.go:<LINE>] "struct keys" {name}="value" test="other value" key="val"
`,
	},
	"map keys": {
		withValues: []interface{}{},
		text:       "map keys",
		values:     []interface{}{map[string]bool{"test": true}, "test"},
		expectedOutput: `I output.go:<LINE>] "map keys" map[test:%!s(bool=true)]="test"
`,
	},
	"map values": {
		text:   "maps",
		values: []interface{}{"s", map[string]string{"hello": "world"}, "i", map[int]int{1: 2, 3: 4}},
		expectedOutput: `I output.go:<LINE>] "maps" s={"hello":"world"} i={"1":2,"3":4}
`,
	},
	"slice values": {
		text:   "slices",
		values: []interface{}{"s", []string{"hello", "world"}, "i", []int{1, 2, 3}},
		expectedOutput: `I output.go:<LINE>] "slices" s=["hello","world"] i=[1,2,3]
`,
	},
	"struct values": {
		text:   "structs",
		values: []interface{}{"s", struct{ Name, Kind, hidden string }{Name: "worker", Kind: "pod", hidden: "ignore"}},
		expectedOutput: `I output.go:<LINE>] "structs" s={"Name":"worker","Kind":"pod"}
`,
	},
	"klog.Format": {
		text:   "klog.Format",
		values: []interface{}{"s", klog.Format(struct{ Name, Kind, hidden string }{Name: "worker", Kind: "pod", hidden: "ignore"})},
		expectedOutput: `I output.go:<LINE>] "klog.Format" s=<
	{
	  "Name": "worker",
	  "Kind": "pod"
	}
 >
`,
	},
	"cyclic list": {
		text:   "cycle",
		values: []interface{}{"list", newCyclicList()},
		expectedOutput: `I output.go:<LINE>] "cycle" list="<internal error: json: unsupported value: encountered a cycle via *test.myList>"
`,
	},
}

func printWithLogger(logger logr.Logger, test testcase) {
	for _, name := range test.withNames {
		logger = logger.WithName(name)
	}
	// When we have multiple WithValues calls, we test
	// first with the initial set of additional values, then
	// the combination, then again the original logger.
	// It must not have been modified. This produces
	// three log entries.
	logger = logger.WithValues(test.withValues...) // <WITH-VALUES>
	loggers := []logr.Logger{logger}
	if test.moreValues != nil {
		loggers = append(loggers, logger.WithValues(test.moreValues...), logger) // <WITH-VALUES-2>
	}
	if test.evenMoreValues != nil {
		loggers = append(loggers, logger.WithValues(test.evenMoreValues...)) // <WITH-VALUES-3>
	}
	for _, logger := range loggers {
		if test.withHelper {
			loggerHelper(logger.V(test.v), test.text, test.values) // <LINE>
		} else if test.err != nil {
			logger.Error(test.err, test.text, test.values...) // <LINE>
		} else {
			logger.V(test.v).Info(test.text, test.values...) // <LINE>
		}
	}
}

var _, _, printWithLoggerLine, _ = runtime.Caller(0) // anchor for finding the line numbers above

func initPrintWithKlog(tb testing.TB, test testcase) {
	if test.withHelper && test.vmodule != "" {
		tb.Skip("klog does not support -vmodule properly when using helper functions")
	}

	state := klog.CaptureState()
	tb.Cleanup(state.Restore)

	var fs flag.FlagSet
	klog.InitFlags(&fs)
	if err := fs.Set("v", "10"); err != nil {
		tb.Fatalf("unexpected error: %v", err)
	}
	if err := fs.Set("vmodule", test.vmodule); err != nil {
		tb.Fatalf("unexpected error: %v", err)
	}
}

func printWithKlog(test testcase) {
	kv := []interface{}{}
	haveKeyInValues := func(key interface{}) bool {
		for i := 0; i < len(test.values); i += 2 {
			if key == test.values[i] {
				return true
			}
		}
		return false
	}
	appendKV := func(withValues ...interface{}) {
		if len(withValues)%2 != 0 {
			withValues = append(withValues, "(MISSING)")
		}
		for i := 0; i < len(withValues); i += 2 {
			if !haveKeyInValues(withValues[i]) {
				kv = append(kv, withValues[i], withValues[i+1])
			}
		}
	}
	// Here we need to emulate the handling of WithValues above.
	if len(test.withNames) > 0 {
		appendKV("logger", strings.Join(test.withNames, "."))
	}
	appendKV(test.withValues...)
	kvs := [][]interface{}{copySlice(kv)}
	if test.moreValues != nil {
		appendKV(test.moreValues...)
		kvs = append(kvs, copySlice(kv), copySlice(kvs[0]))
	}
	if test.evenMoreValues != nil {
		kv = copySlice(kvs[0])
		appendKV(test.evenMoreValues...)
		kvs = append(kvs, copySlice(kv))
	}
	for _, kv := range kvs {
		if len(test.values) > 0 {
			kv = append(kv, test.values...)
		}
		text := test.text
		if test.withHelper {
			klogHelper(klog.Level(test.v), text, kv)
		} else if test.err != nil {
			klog.ErrorS(test.err, text, kv...)
		} else {
			klog.V(klog.Level(test.v)).InfoS(text, kv...)
		}
	}
}

var _, _, printWithKlogLine, _ = runtime.Caller(0) // anchor for finding the line numbers above

// Output covers various special cases of emitting log output.
// It can be used for arbitrary logr.Logger implementations.
//
// The expected output is what klog would print. When testing loggers
// that emit different output, a mapping from klog output to the
// corresponding logger output must be provided, otherwise the
// test will compare against the expected klog output.
//
// Loggers will be tested with direct calls to Info or
// as backend for klog.
func Output(t *testing.T, config OutputConfig) {
	for n, test := range tests {
		t.Run(n, func(t *testing.T) {
			initPrintWithKlog(t, test)

			testOutput := func(t *testing.T, expectedLine int, print func(buffer *bytes.Buffer)) {
				var tmpWriteBuffer bytes.Buffer
				klog.SetOutput(&tmpWriteBuffer)
				print(&tmpWriteBuffer)
				klog.Flush()

				actual := tmpWriteBuffer.String()
				// Strip varying header.
				re := `(?m)^(.).... ..:..:......... ....... output.go`
				actual = regexp.MustCompile(re).ReplaceAllString(actual, `${1} output.go`)

				// Inject expected line. This matches the if checks above, which are
				// the same for both printWithKlog and printWithLogger.
				callLine := expectedLine
				if test.withHelper {
					callLine -= 8
				} else if test.err != nil {
					callLine -= 6
				} else {
					callLine -= 4
				}
				expected := test.expectedOutput
				if repl, ok := config.ExpectedOutputMapping[expected]; ok {
					expected = repl
				}
				expectedWithPlaceholder := expected
				expected = strings.ReplaceAll(expected, "<LINE>", fmt.Sprintf("%d", callLine))
				expected = strings.ReplaceAll(expected, "<WITH-VALUES>", fmt.Sprintf("%d", expectedLine-18))
				expected = strings.ReplaceAll(expected, "<WITH-VALUES-2>", fmt.Sprintf("%d", expectedLine-15))
				expected = strings.ReplaceAll(expected, "<WITH-VALUES-3>", fmt.Sprintf("%d", expectedLine-12))
				if actual != expected {
					if expectedWithPlaceholder == test.expectedOutput {
						t.Errorf("Output mismatch. Expected:\n%s\nActual:\n%s\n", expectedWithPlaceholder, actual)
					} else {
						t.Errorf("Output mismatch. klog:\n%s\nExpected:\n%s\nActual:\n%s\n", test.expectedOutput, expectedWithPlaceholder, actual)
					}
				}
			}

			if config.NewLogger == nil {
				// Test klog.
				testOutput(t, printWithKlogLine-1, func(_ *bytes.Buffer) {
					printWithKlog(test)
				})
				return
			}

			if config.AsBackend {
				testOutput(t, printWithKlogLine-1, func(buffer *bytes.Buffer) {
					setLogger(config.NewLogger(buffer, 10, test.vmodule))
					printWithKlog(test)
				})
				return
			}

			if test.vmodule != "" && !config.SupportsVModule {
				t.Skip("vmodule not supported")
			}

			testOutput(t, printWithLoggerLine-1, func(buffer *bytes.Buffer) {
				printWithLogger(config.NewLogger(buffer, 10, test.vmodule), test)
			})
		})
	}

	if config.NewLogger == nil || config.AsBackend {
		configStruct := klog.Format(myConfig{typeMeta: typeMeta{Kind: "config"}, RealField: 42})
		configStructOutput := `I output.go:<LINE>] "Format" config=<
	{
	  "Kind": "config",
	  "RealField": 42
	}
 >
`

		// Test all klog output functions.
		//
		// Each test case must be defined with the same number of
		// lines, then the source code location of the call itself
		// can be computed below.
		tests := []struct {
			name    string
			logFunc func()
			output  string
		}{
			{
				name:    "Info",
				logFunc: func() { klog.Info("hello", "world") },
				output:  "I output.go:<LINE>] helloworld\n", // This looks odd, but simply is how klog works.
			},
			{
				name:    "InfoDepth",
				logFunc: func() { klog.InfoDepth(0, "hello", "world") },
				output:  "I output.go:<LINE>] helloworld\n",
			},
			{
				name:    "Infoln",
				logFunc: func() { klog.Infoln("hello", "world") },
				output:  "I output.go:<LINE>] hello world\n",
			},
			{
				name:    "InfolnDepth",
				logFunc: func() { klog.InfolnDepth(0, "hello", "world") },
				output:  "I output.go:<LINE>] hello world\n",
			},
			{
				name:    "Infof",
				logFunc: func() { klog.Infof("hello %s", "world") },
				output:  "I output.go:<LINE>] hello world\n",
			},
			{
				name:    "InfofDepth",
				logFunc: func() { klog.InfofDepth(0, "hello %s", "world") },
				output:  "I output.go:<LINE>] hello world\n",
			},
			{
				name:    "InfoS",
				logFunc: func() { klog.InfoS("hello", "what", "world") },
				output:  "I output.go:<LINE>] \"hello\" what=\"world\"\n",
			},
			{
				name:    "InfoSDepth",
				logFunc: func() { klog.InfoSDepth(0, "hello", "what", "world") },
				output:  "I output.go:<LINE>] \"hello\" what=\"world\"\n",
			},
			{
				name:    "Warning",
				logFunc: func() { klog.Warning("hello", "world") },
				output:  "W output.go:<LINE>] helloworld\n",
			},
			{
				name:    "WarningDepth",
				logFunc: func() { klog.WarningDepth(0, "hello", "world") },
				output:  "W output.go:<LINE>] helloworld\n",
			},
			{
				name:    "Warningln",
				logFunc: func() { klog.Warningln("hello", "world") },
				output:  "W output.go:<LINE>] hello world\n",
			},
			{
				name:    "WarninglnDepth",
				logFunc: func() { klog.WarninglnDepth(0, "hello", "world") },
				output:  "W output.go:<LINE>] hello world\n",
			},
			{
				name:    "Warningf",
				logFunc: func() { klog.Warningf("hello %s", "world") },
				output:  "W output.go:<LINE>] hello world\n",
			},
			{
				name:    "WarningfDepth",
				logFunc: func() { klog.WarningfDepth(0, "hello %s", "world") },
				output:  "W output.go:<LINE>] hello world\n",
			},
			{
				name:    "Error",
				logFunc: func() { klog.Error("hello", "world") },
				output:  "E output.go:<LINE>] helloworld\n",
			},
			{
				name:    "ErrorDepth",
				logFunc: func() { klog.ErrorDepth(0, "hello", "world") },
				output:  "E output.go:<LINE>] helloworld\n",
			},
			{
				name:    "Errorln",
				logFunc: func() { klog.Errorln("hello", "world") },
				output:  "E output.go:<LINE>] hello world\n",
			},
			{
				name:    "ErrorlnDepth",
				logFunc: func() { klog.ErrorlnDepth(0, "hello", "world") },
				output:  "E output.go:<LINE>] hello world\n",
			},
			{
				name:    "Errorf",
				logFunc: func() { klog.Errorf("hello %s", "world") },
				output:  "E output.go:<LINE>] hello world\n",
			},
			{
				name:    "ErrorfDepth",
				logFunc: func() { klog.ErrorfDepth(0, "hello %s", "world") },
				output:  "E output.go:<LINE>] hello world\n",
			},
			{
				name:    "ErrorS",
				logFunc: func() { klog.ErrorS(errors.New("hello"), "world") },
				output:  "E output.go:<LINE>] \"world\" err=\"hello\"\n",
			},
			{
				name:    "ErrorSDepth",
				logFunc: func() { klog.ErrorSDepth(0, errors.New("hello"), "world") },
				output:  "E output.go:<LINE>] \"world\" err=\"hello\"\n",
			},
			{
				name:    "V().Info",
				logFunc: func() { klog.V(1).Info("hello", "one", "world") },
				output:  "I output.go:<LINE>] hellooneworld\n",
			},
			{
				name:    "V().InfoDepth",
				logFunc: func() { klog.V(1).InfoDepth(0, "hello", "one", "world") },
				output:  "I output.go:<LINE>] hellooneworld\n",
			},
			{
				name:    "V().Infoln",
				logFunc: func() { klog.V(1).Infoln("hello", "one", "world") },
				output:  "I output.go:<LINE>] hello one world\n",
			},
			{
				name:    "V().InfolnDepth",
				logFunc: func() { klog.V(1).InfolnDepth(0, "hello", "one", "world") },
				output:  "I output.go:<LINE>] hello one world\n",
			},
			{
				name:    "V().Infof",
				logFunc: func() { klog.V(1).Infof("hello %s %s", "one", "world") },
				output:  "I output.go:<LINE>] hello one world\n",
			},
			{
				name:    "V().InfofDepth",
				logFunc: func() { klog.V(1).InfofDepth(0, "hello %s %s", "one", "world") },
				output:  "I output.go:<LINE>] hello one world\n",
			},
			{
				name:    "V().InfoS",
				logFunc: func() { klog.V(1).InfoS("hello", "what", "one world") },
				output:  "I output.go:<LINE>] \"hello\" what=\"one world\"\n",
			},
			{
				name:    "V().InfoSDepth",
				logFunc: func() { klog.V(1).InfoSDepth(0, "hello", "what", "one world") },
				output:  "I output.go:<LINE>] \"hello\" what=\"one world\"\n",
			},
			{
				name:    "V().ErrorS",
				logFunc: func() { klog.V(1).ErrorS(errors.New("hello"), "one world") },
				output:  "E output.go:<LINE>] \"one world\" err=\"hello\"\n",
			},
			{
				name:    "Format InfoS",
				logFunc: func() { klog.InfoS("Format", "config", configStruct) },
				output:  configStructOutput,
			},
		}
		_, _, line, _ := runtime.Caller(0)

		for i, test := range tests {
			t.Run(test.name, func(t *testing.T) {
				var buffer bytes.Buffer
				haveWriteKlogBuffer := false
				if config.NewLogger == nil {
					klog.SetOutput(&buffer)
				} else {
					haveWriteKlogBuffer = setLogger(config.NewLogger(&buffer, 10, ""))
					defer klog.ClearLogger()
				}
				test.logFunc()
				klog.Flush()

				actual := buffer.String()
				// Strip varying header.
				re := `(?m)^(.).... ..:..:......... ....... output.go`
				actual = regexp.MustCompile(re).ReplaceAllString(actual, `${1} output.go`)

				// Inject expected line. This matches the if checks above, which are
				// the same for both printWithKlog and printWithLogger.
				callLine := line + 1 - (len(tests)-i)*5
				expected := test.output

				// When klog does string formating for
				// non-structured calls, it passes the entire
				// result, including a trailing newline, to
				// Logger.Info.
				if config.NewLogger != nil &&
					!haveWriteKlogBuffer &&
					!strings.HasSuffix(test.name, "S") &&
					!strings.HasSuffix(test.name, "SDepth") {
					// klog: I output.go:<LINE>] hello world
					// with logger: I output.go:<LINE>] "hello world\n"
					index := strings.Index(expected, "] ")
					if index == -1 {
						t.Fatalf("did not find ] separator: %s", expected)
					}
					expected = expected[0:index+2] + strconv.Quote(expected[index+2:]) + "\n"

					// Warnings become info messages.
					if strings.HasPrefix(expected, "W") {
						expected = "I" + expected[1:]
					}
				}

				if repl, ok := config.ExpectedOutputMapping[expected]; ok {
					expected = repl
				}
				expectedWithPlaceholder := expected
				expected = strings.ReplaceAll(expected, "<LINE>", fmt.Sprintf("%d", callLine))
				if actual != expected {
					if expectedWithPlaceholder == test.output {
						t.Errorf("Output mismatch. Expected:\n%s\nActual:\n%s\n", expectedWithPlaceholder, actual)
					} else {
						t.Errorf("Output mismatch. klog:\n%s\nExpected:\n%s\nActual:\n%s\n", test.output, expectedWithPlaceholder, actual)
					}
				}
			})
		}
	}
}

// Benchmark covers various special cases of emitting log output.
// It can be used for arbitrary logr.Logger implementations.
//
// Loggers will be tested with direct calls to Info or
// as backend for klog.
func Benchmark(b *testing.B, config OutputConfig) {
	for n, test := range tests {
		b.Run(n, func(b *testing.B) {
			state := klog.CaptureState()
			defer state.Restore()
			klog.SetOutput(io.Discard)
			initPrintWithKlog(b, test)
			b.ResetTimer()

			if config.NewLogger == nil {
				// Test klog.
				for i := 0; i < b.N; i++ {
					printWithKlog(test)
				}
				return
			}

			if config.AsBackend {
				setLogger(config.NewLogger(io.Discard, 10, ""))
				for i := 0; i < b.N; i++ {
					printWithKlog(test)
				}
				return
			}

			if test.vmodule != "" && !config.SupportsVModule {
				b.Skip("vmodule not supported")
			}

			logger := config.NewLogger(io.Discard, 10, test.vmodule)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				printWithLogger(logger, test)
			}
		})
	}
}

func setLogger(logger logr.Logger) bool {
	haveWriteKlogBuffer := false
	var opts []klog.LoggerOption
	if writer, ok := logger.GetSink().(textlogger.KlogBufferWriter); ok {
		opts = append(opts, klog.WriteKlogBuffer(writer.WriteKlogBuffer))
		haveWriteKlogBuffer = true
	}
	klog.SetLoggerWithOptions(logger, opts...)
	return haveWriteKlogBuffer
}

func copySlice(in []interface{}) []interface{} {
	return append([]interface{}{}, in...)
}

type kmeta struct {
	Name, Namespace string
}

func (k kmeta) GetName() string {
	return k.Name
}

func (k kmeta) GetNamespace() string {
	return k.Namespace
}

var _ klog.KMetadata = kmeta{}

type customErrorJSON struct {
	s string
}

var _ error = &customErrorJSON{}
var _ json.Marshaler = &customErrorJSON{}

func (e *customErrorJSON) Error() string {
	return e.s
}

func (e *customErrorJSON) MarshalJSON() ([]byte, error) {
	return json.Marshal(strings.ToUpper(e.s))
}

type stringer struct {
	s string
}

// String crashes when called for nil.
func (s *stringer) String() string {
	return s.s
}

var _ fmt.Stringer = &stringer{}

type faultyStringer struct{}

// String always panics.
func (f faultyStringer) String() string {
	panic("fake String panic")
}

var _ fmt.Stringer = faultyStringer{}

type faultyMarshaler struct{}

// MarshalLog always panics.
func (f faultyMarshaler) MarshalLog() interface{} {
	panic("fake MarshalLog panic")
}

var _ logr.Marshaler = faultyMarshaler{}

type recursiveMarshaler struct{}

// MarshalLog returns itself, which could cause the logger to recurse infinitely.
func (r recursiveMarshaler) MarshalLog() interface{} {
	return r
}

var _ logr.Marshaler = recursiveMarshaler{}

type faultyError struct{}

// Error always panics.
func (f faultyError) Error() string {
	panic("fake Error panic")
}

var _ error = faultyError{}

// typeMeta implements fmt.Stringer and logr.Marshaler. config below
// inherits those (incomplete!) implementations.
type typeMeta struct {
	Kind string
}

func (t typeMeta) String() string {
	return "kind is " + t.Kind
}

func (t typeMeta) MarshalLog() interface{} {
	return t.Kind
}

type myConfig struct {
	typeMeta

	RealField int
}

var _ logr.Marshaler = myConfig{}
var _ fmt.Stringer = myConfig{}

// This is a linked list. It can contain a cycle, which cannot be expressed in JSON.
type myList struct {
	Value int
	Next  *myList
}

func newCyclicList() *myList {
	a := &myList{Value: 1}
	b := &myList{Value: 2, Next: a}
	a.Next = b
	return a
}
