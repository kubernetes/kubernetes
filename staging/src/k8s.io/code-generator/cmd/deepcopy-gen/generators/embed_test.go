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

package generators_test

import (
	"bytes"
	"errors"
	"testing"

	"github.com/spf13/pflag"
	"k8s.io/code-generator/cmd/deepcopy-gen/generators"
)

func TestGenerateDeepCopy(t *testing.T) {
	t.Parallel()
	var out bytes.Buffer
	fs := pflag.NewFlagSet("test", pflag.ContinueOnError)
	fs.SetOutput(&out)
	args := []string{"--help"}

	err := generators.GenerateDeepCopy(fs, args)
	if !errors.Is(err, pflag.ErrHelp) {
		t.Errorf("unexpected error: %v", err)
	}
	if out.Len() == 0 {
		t.Errorf("expected output, got none")
	}
}
