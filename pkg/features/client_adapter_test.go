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

package features

import (
	"testing"

	clientfeatures "k8s.io/client-go/features"
	"k8s.io/component-base/featuregate"
)

func TestClientAdapterEnabled(t *testing.T) {
	fg := featuregate.NewFeatureGate()
	if err := fg.Add(map[featuregate.Feature]featuregate.FeatureSpec{
		"Foo": {Default: true},
	}); err != nil {
		t.Fatal(err)
	}

	a := &clientAdapter{fg}
	if !a.Enabled("Foo") {
		t.Error("expected Enabled(\"Foo\") to return true")
	}
	var r interface{}
	func() {
		defer func() {
			r = recover()
		}()
		a.Enabled("Bar")
	}()
	if r == nil {
		t.Error("expected Enabled(\"Bar\") to panic due to unknown feature name")
	}
}

func TestClientAdapterAdd(t *testing.T) {
	fg := featuregate.NewFeatureGate()
	a := &clientAdapter{fg}
	defaults := fg.GetAll()
	if err := a.Add(map[clientfeatures.Feature]clientfeatures.FeatureSpec{
		"FeatureAlpha":      {PreRelease: clientfeatures.Alpha, Default: true},
		"FeatureBeta":       {PreRelease: clientfeatures.Beta, Default: false},
		"FeatureGA":         {PreRelease: clientfeatures.GA, Default: true, LockToDefault: true},
		"FeatureDeprecated": {PreRelease: clientfeatures.Deprecated, Default: false, LockToDefault: true},
	}); err != nil {
		t.Fatal(err)
	}
	all := fg.GetAll()
	allexpected := map[featuregate.Feature]featuregate.FeatureSpec{
		"FeatureAlpha":      {PreRelease: featuregate.Alpha, Default: true},
		"FeatureBeta":       {PreRelease: featuregate.Beta, Default: false},
		"FeatureGA":         {PreRelease: featuregate.GA, Default: true, LockToDefault: true},
		"FeatureDeprecated": {PreRelease: featuregate.Deprecated, Default: false, LockToDefault: true},
	}
	for name, spec := range defaults {
		allexpected[name] = spec
	}
	if len(all) != len(allexpected) {
		t.Errorf("expected %d registered features, got %d", len(allexpected), len(all))
	}
	for name, expected := range allexpected {
		actual, ok := all[name]
		if !ok {
			t.Errorf("expected feature %q not found", name)
			continue
		}

		if actual != expected {
			t.Errorf("expected feature %q spec %#v, got spec %#v", name, expected, actual)
		}
	}

	var r interface{}
	func() {
		defer func() {
			r = recover()
		}()
		_ = a.Add(map[clientfeatures.Feature]clientfeatures.FeatureSpec{
			"FeatureAlpha": {PreRelease: "foobar"},
		})
	}()
	if r == nil {
		t.Error("expected panic when adding feature with unknown prerelease")
	}
}
