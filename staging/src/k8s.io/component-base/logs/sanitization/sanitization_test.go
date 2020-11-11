/*
Copyright 2020 The Kubernetes Authors.

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

package sanitization

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

type withDatapolTag struct {
	Key string `json:"key" datapolicy:"password"`
}

func datapolItem() interface{} {
	return withDatapolTag{Key: "hunter2"}
}

func TestFilter(t *testing.T) {
	filter := &SanitizingFilter{}
	testcases := []struct {
		input  []interface{}
		output []interface{}
	}{{
		input:  []interface{}{},
		output: []interface{}{},
	}, {
		input: []interface{}{
			"nothing special", "really",
		},
		output: []interface{}{
			"nothing special", "really",
		},
	}, {
		input: []interface{}{
			datapolItem(),
		},
		output: []interface{}{
			"Log message has been redacted. Log argument #0 contains: [password]",
		},
	}, {
		input: []interface{}{
			"nothing special", datapolItem(),
		},
		output: []interface{}{
			"Log message has been redacted. Log argument #1 contains: [password]",
		},
	}}
	for _, tc := range testcases {
		output := filter.Filter(tc.input)
		if !assert.ElementsMatch(t, tc.output, output) {
			t.Errorf("Unexpected filter output for %v, want: %v, got %v", tc.input, tc.output, output)
		}
	}
}

func TestFilterF(t *testing.T) {
	filter := &SanitizingFilter{}
	testcases := []struct {
		inputFmt  string
		input     []interface{}
		outputFmt string
		output    []interface{}
	}{{
		inputFmt:  "",
		input:     []interface{}{},
		outputFmt: "",
		output:    []interface{}{},
	}, {
		inputFmt: "%s: %s",
		input: []interface{}{
			"nothing special", "really",
		},
		outputFmt: "%s: %s",
		output: []interface{}{
			"nothing special", "really",
		},
	}, {
		inputFmt: "%v",
		input: []interface{}{
			datapolItem(),
		},
		outputFmt: "Log message has been redacted. Log argument #%d contains: %v",
		output: []interface{}{
			0, []string{"password"},
		},
	}, {
		inputFmt: "%v",
		input: []interface{}{
			"nothing special", datapolItem(),
		},
		outputFmt: "Log message has been redacted. Log argument #%d contains: %v",
		output: []interface{}{
			1, []string{"password"},
		},
	}}
	for _, tc := range testcases {
		outputFmt, output := filter.FilterF(tc.inputFmt, tc.input)
		correctFmt := outputFmt == tc.outputFmt
		correctArgs := assert.ElementsMatch(t, tc.output, output)
		if !correctFmt || !correctArgs {
			t.Errorf("Error while executing testcase %s, %v", tc.inputFmt, tc.input)
		}
		if !correctFmt {
			t.Errorf("Unexpected output format string want %v, got %v", tc.outputFmt, outputFmt)
		}
		if !correctArgs {
			t.Errorf("Unexpected filter output arguments want: %v, got %v", tc.output, output)
		}
	}
}

func TestFilterS(t *testing.T) {
	filter := &SanitizingFilter{}
	testcases := []struct {
		inputMsg  string
		input     []interface{}
		outputMsg string
		output    []interface{}
	}{{
		inputMsg:  "",
		input:     []interface{}{},
		outputMsg: "",
		output:    []interface{}{},
	}, {
		inputMsg: "Message",
		input: []interface{}{
			"nothing special", "really",
		},
		outputMsg: "Message",
		output: []interface{}{
			"nothing special", "really",
		},
	}, {
		inputMsg: "%v",
		input: []interface{}{
			datapolItem(), "value1", "key2", "value2",
		},
		outputMsg: "Log message has been redacted.",
		output: []interface{}{
			"key_index", 0, "types", []string{"password"},
		},
	}, {
		inputMsg: "%v",
		input: []interface{}{
			"key1", "value1", datapolItem(), "value2",
		},
		outputMsg: "Log message has been redacted.",
		output: []interface{}{
			"key_index", 2, "types", []string{"password"},
		},
	}, {
		inputMsg: "%v",
		input: []interface{}{
			"key1", datapolItem(), "key2", "value2",
		},
		outputMsg: "Log message has been redacted.",
		output: []interface{}{
			"key", "key1", "types", []string{"password"},
		},
	}, {
		inputMsg: "%v",
		input: []interface{}{
			"key1", "value1", "key2", datapolItem(),
		},
		outputMsg: "Log message has been redacted.",
		output: []interface{}{
			"key", "key2", "types", []string{"password"},
		},
	}}
	for _, tc := range testcases {
		outputMsg, output := filter.FilterS(tc.inputMsg, tc.input)
		correctMsg := outputMsg == tc.outputMsg
		correctArgs := assert.ElementsMatch(t, tc.output, output)
		if !correctMsg || !correctArgs {
			t.Errorf("Error while executing testcase %s, %v", tc.inputMsg, tc.input)
		}
		if !correctMsg {
			t.Errorf("Unexpected output format string want %v, got %v", tc.outputMsg, outputMsg)
		}
		if !correctArgs {
			t.Errorf("Unexpected filter output arguments want: %v, got %v", tc.output, output)
		}
	}
}
