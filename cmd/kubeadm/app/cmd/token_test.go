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

package cmd

import (
	"bytes"
	"regexp"
	"testing"
)

const (
	TokenExpectedRegex = "^\\S{6}\\:\\S{16}\n$"
)

func TestRunGenerateToken(t *testing.T) {
	var buf bytes.Buffer

	err := RunGenerateToken(&buf)
	if err != nil {
		t.Errorf("RunGenerateToken returned an error: %v", err)
	}

	output := buf.String()

	matched, err := regexp.MatchString(TokenExpectedRegex, output)
	if err != nil {
		t.Fatalf("encountered an error while trying to match RunGenerateToken's output: %v", err)
	}
	if !matched {
		t.Errorf("RunGenerateToken's output did not match expected regex; wanted: [%s], got: [%s]", TokenExpectedRegex, output)
	}
}
