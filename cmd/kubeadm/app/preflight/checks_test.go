/*
Copyright 2016 The Kubernetes Authors.

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

package preflight

import (
	"bytes"
	"fmt"
	"testing"
)

type preflightCheckTest struct {
	msg string
}

func (pfct preflightCheckTest) Check() (warning, errors []error) {
	if pfct.msg == "warning" {
		return []error{fmt.Errorf("warning")}, nil
	}
	if pfct.msg != "" {
		return nil, []error{fmt.Errorf("fake error")}
	}
	return
}

func TestRunChecks(t *testing.T) {
	var tokenTest = []struct {
		p        []PreFlightCheck
		expected bool
		output   string
	}{
		{[]PreFlightCheck{}, true, ""},
		{[]PreFlightCheck{preflightCheckTest{"warning"}}, true, "WARNING: warning\n"}, // should just print warning
		{[]PreFlightCheck{preflightCheckTest{"error"}}, false, ""},
		{[]PreFlightCheck{preflightCheckTest{"test"}}, false, ""},
	}
	for _, rt := range tokenTest {
		buf := new(bytes.Buffer)
		actual := runChecks(rt.p, buf)
		if (actual == nil) != rt.expected {
			t.Errorf(
				"failed runChecks:\n\texpected: %t\n\t  actual: %t",
				rt.expected,
				(actual == nil),
			)
		}
		if buf.String() != rt.output {
			t.Errorf(
				"failed runChecks:\n\texpected: %s\n\t  actual: %s",
				rt.output,
				buf.String(),
			)
		}
	}
}
