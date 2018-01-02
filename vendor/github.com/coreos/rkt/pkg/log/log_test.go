// Copyright 2016 The rkt Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package log

import (
	"bytes"
	"errors"
	"fmt"
	"testing"

	"github.com/hashicorp/errwrap"
)

const (
	er1Msg          = "this is the first message"
	er2Msg          = "this is the second message"
	accompanyingMsg = "the accompanying message"
	prefix          = "prefix"
)

var outputTests = []struct {
	debug  bool
	expect string
}{
	{
		false,
		fmt.Sprintf("%s: %s: %s\n", prefix, accompanyingMsg, er2Msg),
	},
	{
		true,
		fmt.Sprintf("%s: %s\n  └─%s\n    └─%s\n", prefix, accompanyingMsg, er1Msg, er2Msg),
	},
}

func genNestedError() error {
	e1 := errors.New(er1Msg)
	e2 := errors.New(er2Msg)
	err := errwrap.Wrap(e1, e2)

	return err
}

func TestLogOutput(t *testing.T) {
	for _, tt := range outputTests {
		var logBuf bytes.Buffer
		l := New(&logBuf, prefix, tt.debug)

		l.PrintE(accompanyingMsg, genNestedError())
		errOut := logBuf.String()
		if errOut != tt.expect {
			t.Errorf("Log output not as expected: %s != %s", errOut, tt.expect)
		}
	}
}

func TestLogFormatting(t *testing.T) {
	var logBuf bytes.Buffer
	l := New(&logBuf, "prefix", false)

	l.Errorf("format args: %s %d", "string", 1)

	expected := "prefix: format args: string 1\n"

	if logBuf.String() != expected {
		t.Errorf("expected %q, got %q", expected, logBuf.String())
	}
}
