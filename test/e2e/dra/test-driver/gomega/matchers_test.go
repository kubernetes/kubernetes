/*
Copyright The Kubernetes Authors.

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

package app

import (
	"errors"
	"testing"

	"github.com/onsi/gomega/types"

	testdriver "k8s.io/kubernetes/test/e2e/dra/test-driver/app"
)

const unprepare = "/k8s.io.kubelet.pkg.apis.dra.v1.DRAPlugin/NodeUnprepareResources"

func TestNodeUnprepareResourcesMatchers(t *testing.T) {
	inProgress := testdriver.GRPCCall{FullMethod: unprepare}
	succeeded := testdriver.GRPCCall{FullMethod: unprepare, Response: struct{}{}}
	failed := testdriver.GRPCCall{FullMethod: unprepare, Err: errors.New("boom")}
	other := testdriver.GRPCCall{FullMethod: "/pluginregistration.Registration/GetInfo", Response: struct{}{}}

	for name, tc := range map[string]struct {
		calls   []testdriver.GRPCCall
		matcher types.GomegaMatcher
		want    bool
	}{
		"in-progress: none":            {nil, NodeUnprepareResourcesInProgress, false},
		"in-progress: only completed":  {[]testdriver.GRPCCall{succeeded, failed, other}, NodeUnprepareResourcesInProgress, false},
		"in-progress: blocked call":    {[]testdriver.GRPCCall{other, inProgress}, NodeUnprepareResourcesInProgress, true},
		"after: none":                  {nil, NodeUnprepareResourcesSucceededAfter(0), false},
		"after: old success only":      {[]testdriver.GRPCCall{succeeded, other}, NodeUnprepareResourcesSucceededAfter(1), false},
		"after: new success":           {[]testdriver.GRPCCall{succeeded, other, succeeded}, NodeUnprepareResourcesSucceededAfter(1), true},
		"after: first new call":        {[]testdriver.GRPCCall{succeeded, succeeded}, NodeUnprepareResourcesSucceededAfter(1), true},
		"after: new call still failed": {[]testdriver.GRPCCall{succeeded, failed}, NodeUnprepareResourcesSucceededAfter(1), false},
		"after: new call in progress":  {[]testdriver.GRPCCall{succeeded, inProgress}, NodeUnprepareResourcesSucceededAfter(1), false},
	} {
		t.Run(name, func(t *testing.T) {
			ok, err := tc.matcher.Match(tc.calls)
			if err != nil {
				t.Fatalf("Match returned an error: %v", err)
			}
			if ok != tc.want {
				t.Errorf("got %v, want %v", ok, tc.want)
			}
		})
	}
}
