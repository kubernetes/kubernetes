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

// +build !fly

package main

import (
	"testing"

	"github.com/coreos/rkt/common"
	"github.com/coreos/rkt/tests/testutils"
)

func TestExport(t *testing.T) {
	ctx := testutils.NewRktRunCtx()
	defer ctx.Cleanup()
	overlay := (common.SupportsOverlay() == nil)
	userns := (common.SupportsUserNS() && checkUserNS() == nil && !TestedFlavor.Kvm)

	for name, testCase := range exportTestCases {
		if testCase.NeedsOverlay && !overlay {
			t.Logf("TestExport/%v needs overlay, skipping", name)
			continue
		}
		if testCase.NeedsUserNS && !userns {
			t.Logf("TestExport/%v needs userns, skipping", name)
			continue
		}
		t.Logf("TestExport/%v", name)
		testCase.Execute(t, ctx)
	}
}
