/*
Copyright 2014 The Kubernetes Authors.

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

package options

import (
	"testing"

	"github.com/spf13/pflag"
)

func TestAddFlagsFlag(t *testing.T) {
	// TODO: This only tests the enable-swagger-ui flag for now.
	// Expand the test to include other flags as well.
	f := pflag.NewFlagSet("addflagstest", pflag.ContinueOnError)
	s := NewServerRunOptions()
	s.AddFlags(f)
	if s.Features.EnableSwaggerUI {
		t.Errorf("Expected s.EnableSwaggerUI to be false by default")
	}

	args := []string{
		"--enable-swagger-ui=true",
		"--request-timeout=2m",
	}
	f.Parse(args)
	if !s.Features.EnableSwaggerUI {
		t.Errorf("Expected s.EnableSwaggerUI to be true")
	}
}
