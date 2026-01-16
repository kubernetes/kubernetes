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

package testing

import (
	"slices"
	"testing"

	"k8s.io/component-helpers/nodedeclaredfeatures"
	"k8s.io/component-helpers/nodedeclaredfeatures/types"
)

// SetFrameworkDuringTest overrides the global default nodedeclaredfeatures.Framework for the
// duration of the test.
func SetFrameworkDuringTest(tb testing.TB, framework nodedeclaredfeatures.Framework) {
	tb.Helper()
	originalDefaultFramework := nodedeclaredfeatures.DefaultFramework
	nodedeclaredfeatures.DefaultFramework = &framework
	tb.Cleanup(func() {
		tb.Helper()
		nodedeclaredfeatures.DefaultFramework = originalDefaultFramework
	})
}

// NewMockFramework generates a map of mock features with the given names, and creates a Framework
// with those features registered.
func NewMockFramework(tb testing.TB, features ...string) (nodedeclaredfeatures.Framework, map[string]*MockFeature) {
	slices.Sort(features)
	var allFeatures []types.Feature
	mockFeatures := map[string]*MockFeature{}

	for _, name := range features {
		m := NewMockFeature(tb)
		m.EXPECT().Name().Return(name).Maybe()

		allFeatures = append(allFeatures, m)
		mockFeatures[name] = m
	}

	return *nodedeclaredfeatures.New(allFeatures), mockFeatures
}
