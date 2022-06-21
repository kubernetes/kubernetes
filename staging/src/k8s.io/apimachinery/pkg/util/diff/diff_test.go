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

package diff

import (
	"testing"
)

func TestStringDiff(t *testing.T) {
	diff := StringDiff("aaabb", "aaacc")
	expect := "aaa\n\nA: bb\n\nB: cc\n\n"
	if diff != expect {
		t.Errorf("diff returned %v", diff)
	}
}

func TestLegacyDiff(t *testing.T) {
	equalResult := legacyDiff(true, true)
	diffResult := legacyDiff(true, "string")

	if equalResult != "" {
		t.Errorf("two param type are same, result should be empty, but now the result is: %v", equalResult)
	}

	expectDiffResult := "  interface{}(\n- \tbool(true),\n+ \tstring(\"string\"),\n  )\n"
	if diffResult != expectDiffResult {
		t.Errorf("two param type are not same, expect result is: %v, but now the result is: %v", expectDiffResult, diffResult)
	}
}

func TestObjectGoPrintSideBySide(t *testing.T) {
	result := ObjectGoPrintSideBySide("a", "ab")
	expectResult := "(string) (len=1) \"a\" (string) (len=2) \"ab\"\n                     \n"

	if expectResult != result {
		t.Errorf("expectResult is: %v, but now the result is: %v", expectResult, result)
	}
}
