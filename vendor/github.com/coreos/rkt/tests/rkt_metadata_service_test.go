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

// +build host coreos src kvm

package main

import (
	"fmt"
	"os"
	"testing"

	"github.com/coreos/rkt/tests/testutils"
)

// Launch an app that polls the Annotations metadata
// service - (see https://github.com/appc/spec/blob/master/spec/ace.md#app-container-metadata-service)

// The app's source is at tests/inspect/inspect.go

func TestFetchAppAnnotation(t *testing.T) {
	ctx := testutils.NewRktRunCtx()
	defer ctx.Cleanup()

	if err := ctx.LaunchMDS(); err != nil {
		t.Fatalf("Failed to launch metadata service: %v", err)
	}

	patches := []string{"--exec=/inspect --print-app-annotation=coreos.com/rkt/stage1/run"}
	fileName := patchTestACI("rkt-get-annotation.aci", patches...)
	defer os.Remove(fileName)

	cmd := fmt.Sprintf("%s run --insecure-options=image --mds-register=true %s", ctx.Cmd(), fileName)
	runRktAndCheckOutput(t, cmd, "Annotation coreos.com/rkt/stage1/run=/ex/run", false)
}
