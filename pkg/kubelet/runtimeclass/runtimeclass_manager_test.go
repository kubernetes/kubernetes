/*
Copyright 2018 The Kubernetes Authors.

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

package runtimeclass_test

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"

	"k8s.io/kubernetes/pkg/kubelet/runtimeclass"
	rctest "k8s.io/kubernetes/pkg/kubelet/runtimeclass/testing"
	"k8s.io/utils/pointer"
)

func TestLookupRuntimeHandler(t *testing.T) {
	tests := []struct {
		rcn         *string
		expected    string
		expectError bool
	}{
		{rcn: pointer.String(""), expected: ""},
		{rcn: pointer.String(rctest.EmptyRuntimeClass), expected: ""},
		{rcn: pointer.String(rctest.SandboxRuntimeClass), expected: "kata-containers"},
		{rcn: pointer.String("phantom"), expectError: true},
	}

	manager := runtimeclass.NewManager(rctest.NewPopulatedClient())
	defer rctest.StartManagerSync(manager)()

	for _, test := range tests {
		tname := "nil"
		if test.rcn != nil {
			tname = *test.rcn
		}
		t.Run(fmt.Sprintf("%q->%q(err:%v)", tname, test.expected, test.expectError), func(t *testing.T) {
			handler, err := manager.LookupRuntimeHandler(test.rcn)
			if test.expectError {
				assert.Error(t, err, "handler=%q", handler)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, test.expected, handler)
			}
		})
	}
}
