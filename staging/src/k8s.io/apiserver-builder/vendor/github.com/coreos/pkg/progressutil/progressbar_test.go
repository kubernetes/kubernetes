// Copyright 2016 CoreOS Inc
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

package progressutil

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"strings"
	"testing"
)

func TestNoBarsAdded(t *testing.T) {
	pbp := &ProgressBarPrinter{}
	allDone, err := pbp.Print(ioutil.Discard)
	if allDone {
		t.Errorf("shouldn't have gotten all done when no bars have been added")
	}
	if err != ErrorNoBarsAdded {
		t.Errorf("was expecting ErrorNoBarsAdded, got this instead: %v", err)
	}
}

func TestProgressOutOfBounds(t *testing.T) {
	pbp := &ProgressBarPrinter{}
	pb := pbp.AddProgressBar()

	for _, testcase := range []struct {
		progress    float64
		expectedErr error
	}{
		{-0.1, ErrorProgressOutOfBounds},
		{0, nil},
		{0.5, nil},
		{1, nil},
		{1.1, ErrorProgressOutOfBounds},
	} {
		err := pb.SetCurrentProgress(testcase.progress)
		if err != testcase.expectedErr {
			t.Errorf("got unexpected error. expected=%v actual=%v", testcase.expectedErr, err)
		}
		if err == nil {
			currProgress := pb.GetCurrentProgress()
			if currProgress != testcase.progress {
				t.Errorf("no error was returned, but the progress wasn't updated. should be: %d, actual: %d", testcase.progress, currProgress)
			}
		}
	}
}

func TestDrawOne(t *testing.T) {
	pbp := ProgressBarPrinter{}
	pbp.printToTTYAlways = true
	pb := pbp.AddProgressBar()

	pbp.Print(ioutil.Discard)

	for _, testcase := range []struct {
		beforeText   string
		progress     float64
		afterText    string
		shouldBeDone bool
	}{
		{"before", 0, "after", false},
		{"before2", 0.1, "after2", false},
		{"before3", 0.5, "after3", false},
		{"before4", 1, "after4", true},
	} {
		buf := &bytes.Buffer{}
		pb.SetPrintBefore(testcase.beforeText)
		pb.SetPrintAfter(testcase.afterText)
		err := pb.SetCurrentProgress(testcase.progress)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		done, err := pbp.Print(buf)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if done != testcase.shouldBeDone {
			t.Errorf("unexpected done, expected=%b actual=%b", testcase.shouldBeDone, done)
		}

		output := buf.String()

		progressBarSize := 80 - len(fmt.Sprintf("%s [] %s", testcase.beforeText, testcase.afterText))
		currentProgress := int(testcase.progress * float64(progressBarSize))

		bar := fmt.Sprintf("[%s%s]",
			strings.Repeat("=", currentProgress),
			strings.Repeat(" ", progressBarSize-currentProgress))

		expectedOutput := fmt.Sprintf("\033[1A%s %s %s\n", testcase.beforeText, bar, testcase.afterText)

		if output != expectedOutput {
			t.Errorf("unexpected output:\nexpected:\n\n%sactual:\n\n%s", expectedOutput, output)
		}
	}
}
