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
	overriddenFeaturesLock sync.Mutex
	overriddenFeatures     map[clientfeatures.Feature]string
)

func init() {
	overriddenFeatures = map[clientfeatures.Feature]string{}
}

type featureGatesSetter interface {
	clientfeatures.Gates

	Set(clientfeatures.Feature, bool) error
}

// SetFeatureDuringTest sets the specified feature to the specified value for the duration of the test.
//
// Example use:
//
//	clientfeaturestesting.SetFeatureDuringTest(t, clientfeatures.WatchListClient, true)
func SetFeatureDuringTest(tb testing.TB, feature clientfeatures.Feature, featureValue bool) {
	if err := setFeatureDuringTestInternal(tb, feature, featureValue); err != nil {
		tb.Fatal(err)
	}
}

func setFeatureDuringTestInternal(tb testing.TB, feature clientfeatures.Feature, featureValue bool) error {
	overriddenFeaturesLock.Lock()
	defer overriddenFeaturesLock.Unlock()

	currentFeatureGates := clientfeatures.FeatureGates()
	featureGates, ok := currentFeatureGates.(featureGatesSetter)
	if !ok {
		panic(fmt.Errorf("clientfeatures.FeatureGates(): %T does not implement featureGatesSetter interface", currentFeatureGates))
	}

	originalFeatureValue := featureGates.Enabled(feature)
	if overridingTestName, ok := overriddenFeatures[feature]; ok {
		if !sameTestOrSubtest(tb, overridingTestName) {
			return fmt.Errorf("client-go feature %q is currently overridden by %q test and cannot be also modified by %q", feature, overridingTestName, tb.Name())
		}
	}

	if err := featureGates.Set(feature, featureValue); err != nil {
		return err
	}
	overriddenFeatures[feature] = tb.Name()

	tb.Cleanup(func() {
		overriddenFeaturesLock.Lock()
		defer overriddenFeaturesLock.Unlock()
		delete(overriddenFeatures, feature)
		// if default is not set
		if err := featureGates.Set(feature, originalFeatureValue); err != nil {
			tb.Errorf("failed restoring client-go feature: %v to its original value: %v, err: %v", feature, originalFeatureValue, err)
		}
	})
	return nil
}

// copied from component-base/featuregate/testing
func sameTestOrSubtest(tb testing.TB, testName string) bool {
	return tb.Name() == testName || strings.HasPrefix(tb.Name(), testName+"/")
}
