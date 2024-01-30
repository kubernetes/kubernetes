/*
Copyright 2024 The Kubernetes Authors.

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

package testing

import (
	"fmt"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"k8s.io/client-go/features"
)

func TestDriveInitDefaultFeatureGates(t *testing.T) {
	featureGates := features.FeatureGates()
	assertFunctionPanicsWithMessage(t, func() { featureGates.Enabled("FakeFeatureGate") }, "features.FeatureGates().Enabled", fmt.Sprintf("feature %q is not registered in FeatureGate", "FakeFeatureGate"))

	fakeFeatureGates := &alwaysEnabledFakeGates{}
	require.True(t, fakeFeatureGates.Enabled("FakeFeatureGate"))

	features.ReplaceFeatureGates(fakeFeatureGates)
	featureGates = features.FeatureGates()

	assertFeatureGatesType(t, featureGates)
	require.True(t, featureGates.Enabled("FakeFeatureGate"))
}

type alwaysEnabledFakeGates struct{}

func (f *alwaysEnabledFakeGates) Enabled(features.Feature) bool {
	return true
}

func assertFeatureGatesType(t *testing.T, fg features.Gates) {
	_, ok := fg.(*alwaysEnabledFakeGates)
	if !ok {
		t.Fatalf("passed features.FeatureGates() is NOT of type *alwaysEnabledFakeGates, it is of type = %T", fg)
	}
}

func assertFunctionPanicsWithMessage(t *testing.T, f func(), fName, errMessage string) {
	didPanic, panicMessage := didFunctionPanic(f)
	if !didPanic {
		t.Fatalf("function %q did not panicked", fName)
	}

	panicError, ok := panicMessage.(error)
	if !ok || !strings.Contains(panicError.Error(), errMessage) {
		t.Fatalf("func %q should panic with error message:\t%#v\n\tPanic value:\t%#v\n", fName, errMessage, panicMessage)
	}
}

func didFunctionPanic(f func()) (didPanic bool, panicMessage interface{}) {
	didPanic = true

	defer func() {
		panicMessage = recover()
	}()

	f()
	didPanic = false

	return
}
