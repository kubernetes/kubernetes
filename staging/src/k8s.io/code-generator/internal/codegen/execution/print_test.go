/*
Copyright 2023 The Kubernetes Authors.

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

package execution_test

import (
	"bytes"
	"k8s.io/code-generator/internal/codegen/execution"
	"testing"
)

func TestPrint(t *testing.T) {
	t.Parallel()
	var out bytes.Buffer
	v := execution.New()
	v.Out = &out
	v.Print("foo")
	v.Printf("bar%s", "\n")
	v.Println("fizz")

	if out.String() != "foobar\nfizz\n" {
		t.Errorf("unexpected output: %s", out.String())
	}
}
