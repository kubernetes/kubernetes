/*
Copyright 2019 The Kubernetes Authors.

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

package trace

import (
	"bytes"
	"context"
	"flag"
	"os"
	"strings"
	"testing"
	"time"

	"k8s.io/klog/v2"
)

func init() {
	klog.InitFlags(flag.CommandLine)
	flag.CommandLine.Lookup("logtostderr").Value.Set("false")
}

func TestStep(t *testing.T) {
	tests := []struct {
		name          string
		inputString   string
		expectedTrace *Trace
	}{
		{
			name:        "When string is empty",
			inputString: "",
			expectedTrace: &Trace{
				traceItems: []traceItem{
					traceStep{stepTime: time.Now(), msg: ""},
				},
			},
		},
		{
			name:        "When string is not empty",
			inputString: "test2",
			expectedTrace: &Trace{
				traceItems: []traceItem{
					traceStep{stepTime: time.Now(), msg: "test2"},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			sampleTrace := &Trace{}
			sampleTrace.Step(tt.inputString)
			if sampleTrace.traceItems[0].(traceStep).msg != tt.expectedTrace.traceItems[0].(traceStep).msg {
				t.Errorf("Expected %v \n Got %v \n", tt.expectedTrace, sampleTrace)
			}
		})
	}
}

func TestNestedTrace(t *testing.T) {
	tests := []struct {
		name          string
		inputString   string
		expectedTrace *Trace
	}{
		{
			name:        "Empty string",
			inputString: "",
			expectedTrace: &Trace{
				traceItems: []traceItem{
					&Trace{startTime: time.Now(), name: ""},
				},
			},
		},
		{
			name:        "Non-empty string",
			inputString: "Inner trace",
			expectedTrace: &Trace{
				traceItems: []traceItem{
					&Trace{
						startTime: time.Now(),
						name:      "Inner trace",
						traceItems: []traceItem{
							&Trace{name: "Inner trace"},
						},
					},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			sampleTrace := &Trace{}
			innerSampleTrace := sampleTrace.Nest(tt.inputString)
			innerSampleTrace.Nest(tt.inputString)
			if sampleTrace.traceItems[0].(*Trace).name != tt.expectedTrace.traceItems[0].(*Trace).name {
				t.Errorf("Expected %v \n Got %v \n", tt.expectedTrace, sampleTrace)
			}
		})
	}
}

func TestTotalTime(t *testing.T) {
	test := struct {
		name       string
		inputTrace *Trace
	}{
		name: "Test with current system time",
		inputTrace: &Trace{
			startTime: time.Now(),
		},
	}

	t.Run(test.name, func(t *testing.T) {
		got := test.inputTrace.TotalTime()
		if got == 0 {
			t.Errorf("Expected total time 0, got %d \n", got)
		}
	})
}

func TestLog(t *testing.T) {
	tests := []struct {
		name             string
		msg              string
		fields           []Field
		expectedMessages []string
		sampleTrace      *Trace
	}{
		{
			name: "Check the log dump with 3 msg",
			expectedMessages: []string{
				"msg1", "msg2", "msg3",
			},
			sampleTrace: &Trace{
				name: "Sample Trace",
				traceItems: []traceItem{
					&traceStep{stepTime: time.Now(), msg: "msg1"},
					&traceStep{stepTime: time.Now(), msg: "msg2"},
					&traceStep{stepTime: time.Now(), msg: "msg3"},
				},
			},
		},
		{
			name: "Check formatting",
			expectedMessages: []string{
				"URL:/api,count:3", `"msg1" str:text,int:2,bool:false`, `"msg2" x:1`,
			},
			sampleTrace: &Trace{
				name:   "Sample Trace",
				fields: []Field{{"URL", "/api"}, {"count", 3}},
				traceItems: []traceItem{
					&traceStep{stepTime: time.Now(), msg: "msg1", fields: []Field{{"str", "text"}, {"int", 2}, {"bool",
						false}}},
					&traceStep{stepTime: time.Now(), msg: "msg2", fields: []Field{{"x", "1"}}},
				},
			},
		},
		{
			name: "Check fixture formatted",
			expectedMessages: []string{
				"URL:/api,count:3", `"msg1" str:text,int:2,bool:false`, `"msg2" x:1`,
			},
			sampleTrace: fieldsTraceFixture(),
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var buf bytes.Buffer
			klog.SetOutput(&buf)
			test.sampleTrace.Log()
			for _, msg := range test.expectedMessages {
				if !strings.Contains(buf.String(), msg) {
					t.Errorf("\nMsg %q not found in log: \n%v\n", msg, buf.String())
				}
			}
		})
	}
}

func TestNestedTraceLog(t *testing.T) {
	currentTime := time.Now()
	tests := []struct {
		name             string
		msg              string
		fields           []Field
		expectedMessages []string
		sampleTrace      *Trace
	}{
		{
			name: "Check the log dump with 3 nestedTraces",
			expectedMessages: []string{
				"Sample Trace", "msg1", "msg2", "msg3",
			},
			sampleTrace: &Trace{
				name: "Sample Trace",
				traceItems: []traceItem{
					&Trace{startTime: currentTime, endTime: &currentTime, name: "msg1"},
					&Trace{startTime: currentTime, endTime: &currentTime, name: "msg2"},
					&Trace{startTime: currentTime, endTime: &currentTime, name: "msg3"},
				},
			},
		},
		{
			name: "Check the log dump with 3 nestedTraces and steps",
			expectedMessages: []string{
				"Sample Trace", "msg1", "msg2", "msg3", "step1", "step2", "step3",
			},
			sampleTrace: &Trace{
				name: "Sample Trace",
				traceItems: []traceItem{
					&Trace{startTime: currentTime, endTime: &currentTime, name: "msg1"},
					&Trace{startTime: currentTime, endTime: &currentTime, name: "msg2"},
					&Trace{startTime: currentTime, endTime: &currentTime, name: "msg3"},
					&traceStep{stepTime: currentTime, msg: "step1"},
					&traceStep{stepTime: currentTime, msg: "step2"},
					&traceStep{stepTime: currentTime, msg: "step3"},
				},
			},
		},
		{
			name: "Check the log dump with nestedTrace with fields",
			expectedMessages: []string{
				"Sample Trace", `"msg1" str:text,int:2,bool:false`,
			},
			sampleTrace: &Trace{
				name: "Sample Trace",
				traceItems: []traceItem{
					&Trace{
						startTime: currentTime,
						endTime:   &currentTime,
						name:      "msg1", fields: []Field{
							{"str", "text"},
							{"int", 2},
							{"bool", false}}},
				},
			},
		},
		{
			name: "Check the log dump with doubly nestedTrace",
			expectedMessages: []string{
				"Sample Trace", "msg1", "nested1",
			},
			sampleTrace: &Trace{
				name: "Sample Trace",
				traceItems: []traceItem{
					&Trace{
						startTime:  currentTime,
						endTime:    &currentTime,
						name:       "msg1",
						traceItems: []traceItem{&Trace{name: "nested1", startTime: currentTime, endTime: &currentTime}},
					},
				},
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var buf bytes.Buffer
			klog.SetOutput(&buf)
			test.sampleTrace.Log()
			for _, msg := range test.expectedMessages {
				if !strings.Contains(buf.String(), msg) {
					t.Errorf("\nMsg %q not found in log: \n%v\n", msg, buf.String())
				}
			}
		})
	}
}

func fieldsTraceFixture() *Trace {
	trace := New("Sample Trace", Field{"URL", "/api"}, Field{"count", 3})
	trace.Step("msg1", Field{"str", "text"}, Field{"int", 2}, Field{"bool", false})
	trace.Step("msg2", Field{"x", "1"})
	return trace
}

func TestLogIfLong(t *testing.T) {
	currentTime := time.Now()
	type mutate struct {
		delay time.Duration
		msg   string
	}

	tests := []*struct {
		name             string
		expectedMessages []string
		sampleTrace      *Trace
		threshold        time.Duration
		mutateInfo       []mutate // mutateInfo contains the information to mutate step's time to simulate multiple tests without waiting.

	}{
		{
			name: "When threshold is 500 and msg 2 has highest share",
			expectedMessages: []string{
				"msg2",
			},
			mutateInfo: []mutate{
				{10, "msg1"},
				{1000, "msg2"},
				{0, "msg3"},
			},
			threshold: 500,
		},
		{
			name: "When threshold is 10 and msg 3 has highest share",
			expectedMessages: []string{
				"msg3",
			},
			mutateInfo: []mutate{
				{0, "msg1"},
				{0, "msg2"},
				{50, "msg3"},
			},
			threshold: 10,
		},
		{
			name: "When threshold is 0 and all msg have same share",
			expectedMessages: []string{
				"msg1", "msg2", "msg3",
			},
			mutateInfo: []mutate{
				{0, "msg1"},
				{0, "msg2"},
				{0, "msg3"},
			},
			threshold: 0,
		},
		{
			name:             "When threshold is 20 and all msg 1 has highest share",
			expectedMessages: []string{},
			mutateInfo: []mutate{
				{10, "msg1"},
				{0, "msg2"},
				{0, "msg3"},
			},
			threshold: 20,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var buf bytes.Buffer
			klog.SetOutput(&buf)

			tt.sampleTrace = New("Test trace")
			for _, mod := range tt.mutateInfo {
				tt.sampleTrace.traceItems = append(tt.sampleTrace.traceItems,
					&traceStep{stepTime: currentTime.Add(mod.delay), msg: mod.msg})
			}

			tt.sampleTrace.LogIfLong(tt.threshold)

			for _, msg := range tt.expectedMessages {
				if msg != "" && !strings.Contains(buf.String(), msg) {
					t.Errorf("Msg %q expected in trace log: \n%v\n", msg, buf.String())
				}
			}
		})
	}
}

func TestLogNestedTrace(t *testing.T) {
	twoHundred := 200 * time.Millisecond
	five := 5 * time.Millisecond
	currentTime := time.Now()

	tests := []struct {
		name          string
		expectedMsgs  []string
		unexpectedMsg []string
		trace         *Trace
		verbosity     klog.Level
	}{
		{
			name:          "Log nested trace when it surpasses threshold",
			expectedMsgs:  []string{"inner1"},
			unexpectedMsg: []string{"msg"},
			trace: &Trace{
				name:      "msg",
				startTime: currentTime.Add(10),
				traceItems: []traceItem{
					&Trace{
						name:      "inner1",
						threshold: &five,
						startTime: currentTime.Add(-10 * time.Millisecond),
						endTime:   &currentTime,
					},
				},
			},
		},
		{
			name:          "Log inner nested trace when it surpasses threshold",
			expectedMsgs:  []string{"inner2"},
			unexpectedMsg: []string{"msg", "inner1"},
			trace: &Trace{
				name:      "msg",
				startTime: currentTime.Add(10),
				traceItems: []traceItem{
					&Trace{
						name:      "inner1",
						threshold: &five,
						startTime: currentTime.Add(-1 * time.Millisecond),
						endTime:   &currentTime,
						traceItems: []traceItem{
							&Trace{
								name:      "inner2",
								threshold: &five,
								startTime: currentTime.Add(-10 * time.Millisecond),
								endTime:   &currentTime,
							},
						},
					},
				},
			},
		},
		{
			name:          "Log inner nested trace when it surpasses threshold and surrounding trace not ended",
			expectedMsgs:  []string{"inner2"},
			unexpectedMsg: []string{"msg", "inner1"},
			trace: &Trace{
				name:      "msg",
				startTime: currentTime.Add(10),
				traceItems: []traceItem{
					&Trace{
						name:      "inner1",
						threshold: &five,
						traceItems: []traceItem{
							&Trace{
								name:      "inner2",
								threshold: &five,
								startTime: currentTime.Add(-10 * time.Millisecond),
								endTime:   &currentTime,
							},
						},
					},
				},
			},
		},
		{
			name:          "Do not log nested trace that does not surpass threshold",
			expectedMsgs:  []string{"msg", "inner2"},
			unexpectedMsg: []string{"inner1"},
			trace: &Trace{
				name:      "msg",
				startTime: currentTime.Add(-300 * time.Millisecond),
				traceItems: []traceItem{
					&Trace{
						name:      "inner1",
						threshold: &five,
						startTime: currentTime.Add(-1 * time.Millisecond),
						endTime:   &currentTime,
						traceItems: []traceItem{
							&Trace{
								name:      "inner2",
								threshold: &five,
								startTime: currentTime.Add(-10 * time.Millisecond),
								endTime:   &currentTime,
							},
						},
					},
				},
			},
		},
		{
			name:          "Log all nested traces that surpass threshold",
			expectedMsgs:  []string{"inner1", "inner2"},
			unexpectedMsg: []string{"msg"},
			trace: &Trace{
				name:      "msg",
				startTime: currentTime.Add(10),
				traceItems: []traceItem{
					&Trace{
						name:      "inner1",
						threshold: &five,
						startTime: currentTime.Add(-10 * time.Millisecond),
						endTime:   &currentTime,
						traceItems: []traceItem{
							&Trace{
								name:      "inner2",
								threshold: &five,
								startTime: currentTime.Add(-10 * time.Millisecond),
								endTime:   &currentTime,
							},
						},
					},
				},
			},
		},
		{
			name:          "Log all nested traces that surpass threshold and their children if klog verbosity is >= 4",
			expectedMsgs:  []string{"inner1", "inner2"},
			unexpectedMsg: []string{"msg"},
			verbosity:     4,
			trace: &Trace{
				name:      "msg",
				startTime: currentTime.Add(10),
				traceItems: []traceItem{
					&Trace{
						name:      "inner1",
						threshold: &five,
						startTime: currentTime.Add(-10 * time.Millisecond),
						endTime:   &currentTime,
						traceItems: []traceItem{
							&Trace{
								name:      "inner2",
								threshold: &twoHundred,
								startTime: currentTime.Add(-10 * time.Millisecond),
								endTime:   &currentTime,
							},
						},
					},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var buf bytes.Buffer
			klog.SetOutput(&buf)

			if tt.verbosity > 0 {
				orig := klogV
				klogV = func(l klog.Level) bool {
					return l <= tt.verbosity
				}
				defer func() {
					klogV = orig
				}()
			}

			tt.trace.LogIfLong(twoHundred)

			for _, msg := range tt.expectedMsgs {
				if msg != "" && !strings.Contains(buf.String(), msg) {
					t.Errorf("Msg %q expected in trace log: \n%v\n", msg, buf.String())
				}
			}

			for _, msg := range tt.unexpectedMsg {
				if msg != "" && strings.Contains(buf.String(), msg) {
					t.Errorf("Msg %q not expected in trace log: \n%v\n", msg, buf.String())
				}
			}
		})

	}
}

func TestStepThreshold(t *testing.T) {

	thousandMs := 1000 * time.Millisecond
	sixHundred := 600 * time.Millisecond
	hundredMs := 100 * time.Millisecond
	twoThousandMs := 1200 * time.Millisecond

	tests := []struct {
		name              string
		inputTrace        *Trace
		expectedThreshold time.Duration
	}{
		{
			name: "Trace with  nested traces",
			inputTrace: &Trace{
				threshold: &thousandMs,
				traceItems: []traceItem{
					traceStep{msg: "trace 1"},
					traceStep{msg: "trace 2"},
					&Trace{threshold: &sixHundred},
					&Trace{name: "msg 1"},
				},
			},
			expectedThreshold: 100 * time.Millisecond,
		},
		{
			name: "Trace with  nested traces",
			inputTrace: &Trace{
				threshold: &thousandMs,
				traceItems: []traceItem{
					traceStep{msg: "trace 1"},
					traceStep{msg: "trace 2"},
					&Trace{threshold: &sixHundred},
					&Trace{name: "msg 1", threshold: &hundredMs},
				},
			},
			expectedThreshold: 100 * time.Millisecond,
		},
		{
			name: "Trace with nested traces with a large threshold",
			inputTrace: &Trace{
				threshold: &thousandMs,
				traceItems: []traceItem{
					&Trace{threshold: &twoThousandMs},
				},
			},
			expectedThreshold: 125 * time.Millisecond,
		},
	}

	for _, tt := range tests {
		actualThreshold := *tt.inputTrace.calculateStepThreshold()
		if actualThreshold != tt.expectedThreshold {
			t.Errorf("Expecting %v threshold but got %v", tt.expectedThreshold, actualThreshold)
		}
	}
}

func TestContext(t *testing.T) {
	ctx := context.Background()

	trace1 := FromContext(ctx).Nest("op1")
	ctx = ContextWithTrace(ctx, trace1)
	defer trace1.Log()
	func(ctx context.Context) {
		trace2 := FromContext(ctx).Nest("op2")
		defer trace2.Log()
	}(ctx)
	if len(trace1.traceItems) != 1 {
		t.Fatalf("expected len(trace1.traceItems) == 1, but got %d", len(trace1.traceItems))
	}
	nested, ok := trace1.traceItems[0].(*Trace)
	if !ok {
		t.Fatal("expected trace1.traceItems[0] to be a nested trace")
	}
	if nested.name != "op2" {
		t.Errorf("expected trace named op2 to be nested in op1, but got %s", nested.name)
	}
}

func ExampleTrace_Step() {
	t := New("frobber")

	time.Sleep(5 * time.Millisecond)
	t.Step("reticulated splines") // took 5ms

	time.Sleep(10 * time.Millisecond)
	t.Step("sequenced particles") // took 10ms

	klog.SetOutput(os.Stdout) // change output from stderr to stdout
	t.Log()
}
