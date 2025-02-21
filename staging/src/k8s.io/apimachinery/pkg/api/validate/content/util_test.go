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

package content

import (
	"bytes"
	"regexp"
	"strconv"
	"testing"
)

// fmtMsgs formats a list of strings for for nicer output.  It will use
// multiple lines if the length of the input is greater than 1.
func fmtMsgs(msgs []string) string {
	if len(msgs) == 0 {
		return "<no errors>"
	}
	if len(msgs) == 1 {
		return strconv.Quote(msgs[0])
	}
	buf := bytes.Buffer{}
	for _, m := range msgs {
		buf.WriteString("\n")
		buf.WriteString(strconv.Quote(m))
	}
	return buf.String()
}

// mkMsgs is a helper for nicer test setup.
func mkMsgs(msgs ...string) []string {
	return msgs
}

// testVerify checks that the result matches the expected results.
func testVerify[T any](t *testing.T, casenum int, input T, expect []string, result []string) {
	t.Helper()

	if len(result)+len(expect) == 0 {
		return
	}
	if len(result) != 0 && len(expect) == 0 {
		t.Errorf("case %d(%v): unexpected failure: %v", casenum, input, fmtMsgs(result))
		return
	}
	if len(result) == 0 && len(expect) != 0 {
		t.Errorf("case %d(%v): unexpected success: expected %v", casenum, input, fmtMsgs(expect))
		return
	}
	if len(result) != len(expect) {
		t.Errorf("case %d(%v): wrong errors\nexpected: %v\n     got: %v", casenum, input, fmtMsgs(expect), fmtMsgs(result))
		return
	}
	for i := range expect {
		want := expect[i]
		got := result[i]

		if re := regexp.MustCompile(want); !re.MatchString(got) {
			t.Errorf("case %d(%v): wrong errors\nexpected: %v\n     got: %v", casenum, input, fmtMsgs(expect), fmtMsgs(result))
			return
		}
	}
}
