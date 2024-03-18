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
	"sync"
	"testing"

	clientfeatures "k8s.io/client-go/features"
)

var (
	lock                           sync.Mutex
	currentlyOverridingTestName    string
	originalGatesForOverridingTest clientfeatures.Gates

	unsafeSkipCheckingKnownFeature bool
)

// SetFeatureDuringTest sets the specified feature to the specified value for the duration of the test.
//
// Example use:
//
//	clientfeaturestesting.SetFeatureDuringTest(t, clientfeatures.WatchListClient, false)
func SetFeatureDuringTest(tb testing.TB, feature clientfeatures.Feature, featureValue bool) {
	tb.Helper()
	lock.Lock()
	defer lock.Unlock()

	currentFeatureGates := clientfeatures.FeatureGates()
	if !unsafeSkipCheckingKnownFeature {
		assertKnownFeature(tb, feature)
	}

	overrideCleanup, err := overrideFeatureGatesLocked(tb, currentFeatureGates)
	if err != nil {
		tb.Fatal(err)
	}
	clientfeatures.ReplaceFeatureGates(featureOverride{feature: feature, featureValue: featureValue, originalGates: currentFeatureGates})

	tb.Cleanup(func() {
		tb.Helper()
		overrideCleanup()
	})
}

func overrideFeatureGatesLocked(tb testing.TB, originalGates clientfeatures.Gates) (func(), error) {
	tb.Helper()

	if currentlyOverridingTestName != "" && !sameTestOrSubtest(tb, currentlyOverridingTestName) {
		return nil, fmt.Errorf("client-go feature gates are currently overridden by %q test and cannot be also modified by %q", currentlyOverridingTestName, tb.Name())
	}

	currentlyOverridingTestName = tb.Name()
	originalGatesForOverridingTest = originalGates

	return func() {
		tb.Helper()
		lock.Lock()
		defer lock.Unlock()
		currentlyOverridingTestName = ""
		clientfeatures.ReplaceFeatureGates(originalGatesForOverridingTest)
	}, nil
}

func unsafeSkipCheckingKnownFeaturesForDurationOfTest(t *testing.T) {
	t.Helper()
	lock.Lock()
	defer lock.Unlock()
	unsafeSkipCheckingKnownFeature = true
	t.Cleanup(func() {
		unsafeSkipCheckingKnownFeature = false
	})
}

// copied from component-base/featuregate/testing
func sameTestOrSubtest(tb testing.TB, testName string) bool {
	return tb.Name() == testName || strings.HasPrefix(tb.Name(), testName+"/")
}

func assertKnownFeature(t testing.TB, feature clientfeatures.Feature) {
	t.Helper()
	registry := &featuresRegistry{features: map[clientfeatures.Feature]clientfeatures.FeatureSpec{}}
	clientfeatures.AddFeaturesToExistingFeatureGates(registry)
	if _, ok := registry.features[feature]; !ok {
		t.Fatalf("Unknown feature %s", feature)
	}
}

type featuresRegistry struct {
	features map[clientfeatures.Feature]clientfeatures.FeatureSpec
}

func (r *featuresRegistry) Add(features map[clientfeatures.Feature]clientfeatures.FeatureSpec) error {
	r.features = features
	return nil
}

type featureOverride struct {
	feature      clientfeatures.Feature
	featureValue bool

	originalGates clientfeatures.Gates
}

func (s featureOverride) Enabled(f clientfeatures.Feature) bool {
	if f == s.feature {
		return s.featureValue
	}
	return s.originalGates.Enabled(f)
}
