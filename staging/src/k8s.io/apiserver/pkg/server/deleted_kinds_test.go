/*
Copyright 2020 The Kubernetes Authors.

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

package server

import (
	"reflect"
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/dump"
	"k8s.io/apimachinery/pkg/util/sets"
	apimachineryversion "k8s.io/apimachinery/pkg/util/version"
	"k8s.io/apiserver/pkg/registry/rest"
)

func Test_newResourceExpirationEvaluator(t *testing.T) {
	tests := []struct {
		name           string
		currentVersion string
		expected       resourceExpirationEvaluator
		expectedErr    string
	}{
		{
			name:           "beta",
			currentVersion: "v1.20.0-beta.0.62+a5d22854a2ac21",
			expected:       resourceExpirationEvaluator{currentVersion: apimachineryversion.MajorMinor(1, 20)},
		},
		{
			name:           "alpha",
			currentVersion: "v1.20.0-alpha.0.62+a5d22854a2ac21",
			expected:       resourceExpirationEvaluator{currentVersion: apimachineryversion.MajorMinor(1, 20), isAlpha: true},
		},
		{
			name:           "maintenance",
			currentVersion: "v1.20.1",
			expected:       resourceExpirationEvaluator{currentVersion: apimachineryversion.MajorMinor(1, 20)},
		},
		{
			name:           "no v prefix",
			currentVersion: "1.20.1",
			expected:       resourceExpirationEvaluator{currentVersion: apimachineryversion.MajorMinor(1, 20)},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			actual, actualErr := NewResourceExpirationEvaluator(apimachineryversion.MustParse(tt.currentVersion))

			checkErr(t, actualErr, tt.expectedErr)
			if actualErr != nil {
				return
			}

			actual.(*resourceExpirationEvaluator).strictRemovedHandlingInAlpha = false
			if !reflect.DeepEqual(tt.expected, *actual.(*resourceExpirationEvaluator)) {
				t.Fatal(actual)
			}
		})
	}
}

func storageRemovedIn(major, minor int) *removedInStorage {
	return &removedInStorage{major: major, minor: minor}
}

func storageNeverRemoved() *removedInStorage {
	return &removedInStorage{neverRemoved: true}
}

type removedInStorage struct {
	major, minor int
	neverRemoved bool
}

func (r *removedInStorage) New() runtime.Object {
	if r.neverRemoved {
		return &defaultObj{}
	}
	return &removedInObj{major: r.major, minor: r.minor}
}

func (r *removedInStorage) Destroy() {
}

type defaultObj struct {
}

func (r *defaultObj) GetObjectKind() schema.ObjectKind {
	panic("don't do this")
}
func (r *defaultObj) DeepCopyObject() runtime.Object {
	panic("don't do this either")
}

type removedInObj struct {
	major, minor int
}

func (r *removedInObj) GetObjectKind() schema.ObjectKind {
	panic("don't do this")
}
func (r *removedInObj) DeepCopyObject() runtime.Object {
	panic("don't do this either")
}
func (r *removedInObj) APILifecycleRemoved() (major, minor int) {
	return r.major, r.minor
}

func storageIntroducedIn(major, minor int) *introducedInStorage {
	return &introducedInStorage{major: major, minor: minor}
}

type introducedInStorage struct {
	major, minor int
}

func (r *introducedInStorage) New() runtime.Object {
	if r.major == 0 && r.minor == 0 {
		return &defaultObj{}
	}
	return &IntroducedInObj{major: r.major, minor: r.minor}
}

func (r *introducedInStorage) Destroy() {
}

type IntroducedInObj struct {
	major, minor int
}

func (r *IntroducedInObj) GetObjectKind() schema.ObjectKind {
	panic("don't do this")
}
func (r *IntroducedInObj) DeepCopyObject() runtime.Object {
	panic("don't do this either")
}
func (r *IntroducedInObj) APILifecycleIntroduced() (major, minor int) {
	return r.major, r.minor
}

func Test_resourceExpirationEvaluator_shouldServe(t *testing.T) {
	tests := []struct {
		name                        string
		resourceExpirationEvaluator resourceExpirationEvaluator
		restStorage                 rest.Storage
		expected                    bool
	}{
		{
			name: "removed-in-curr",
			resourceExpirationEvaluator: resourceExpirationEvaluator{
				currentVersion: apimachineryversion.MajorMinor(1, 20),
			},
			restStorage: storageRemovedIn(1, 20),
			expected:    false,
		},
		{
			name: "removed-in-curr-but-deferred",
			resourceExpirationEvaluator: resourceExpirationEvaluator{
				currentVersion:                 apimachineryversion.MajorMinor(1, 20),
				serveRemovedAPIsOneMoreRelease: true,
			},
			restStorage: storageRemovedIn(1, 20),
			expected:    true,
		},
		{
			name: "removed-in-curr-but-alpha",
			resourceExpirationEvaluator: resourceExpirationEvaluator{
				currentVersion: apimachineryversion.MajorMinor(1, 20),
				isAlpha:        true,
			},
			restStorage: storageRemovedIn(1, 20),
			expected:    true,
		},
		{
			name: "removed-in-curr-but-alpha-but-strict",
			resourceExpirationEvaluator: resourceExpirationEvaluator{
				currentVersion:               apimachineryversion.MajorMinor(1, 20),
				isAlpha:                      true,
				strictRemovedHandlingInAlpha: true,
			},
			restStorage: storageRemovedIn(1, 20),
			expected:    false,
		},
		{
			name: "removed-in-prev-deferral-does-not-help",
			resourceExpirationEvaluator: resourceExpirationEvaluator{
				currentVersion:                 apimachineryversion.MajorMinor(1, 21),
				serveRemovedAPIsOneMoreRelease: true,
			},
			restStorage: storageRemovedIn(1, 20),
			expected:    false,
		},
		{
			name: "removed-in-prev-major",
			resourceExpirationEvaluator: resourceExpirationEvaluator{
				currentVersion:                 apimachineryversion.MajorMinor(2, 20),
				serveRemovedAPIsOneMoreRelease: true,
			},
			restStorage: storageRemovedIn(1, 20),
			expected:    false,
		},
		{
			name: "removed-in-future",
			resourceExpirationEvaluator: resourceExpirationEvaluator{
				currentVersion: apimachineryversion.MajorMinor(1, 20),
			},
			restStorage: storageRemovedIn(1, 21),
			expected:    true,
		},
		{
			name: "never-removed",
			resourceExpirationEvaluator: resourceExpirationEvaluator{
				currentVersion: apimachineryversion.MajorMinor(1, 20),
			},
			restStorage: storageNeverRemoved(),
			expected:    true,
		},
		{
			name: "introduced-in-curr",
			resourceExpirationEvaluator: resourceExpirationEvaluator{
				currentVersion: apimachineryversion.MajorMinor(1, 20),
			},
			restStorage: storageIntroducedIn(1, 20),
			expected:    true,
		},
		{
			name: "introduced-in-prev-major",
			resourceExpirationEvaluator: resourceExpirationEvaluator{
				currentVersion: apimachineryversion.MajorMinor(1, 20),
			},
			restStorage: storageIntroducedIn(1, 19),
			expected:    true,
		},
		{
			name: "introduced-in-future",
			resourceExpirationEvaluator: resourceExpirationEvaluator{
				currentVersion: apimachineryversion.MajorMinor(1, 20),
			},
			restStorage: storageIntroducedIn(1, 21),
			expected:    false,
		},
		{
			name: "missing-introduced",
			resourceExpirationEvaluator: resourceExpirationEvaluator{
				currentVersion: apimachineryversion.MajorMinor(1, 20),
			},
			restStorage: storageIntroducedIn(0, 0),
			expected:    true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gv := schema.GroupVersion{Group: "mygroup", Version: "myversion"}
			convertor := &dummyConvertor{}
			if actual := tt.resourceExpirationEvaluator.shouldServe(gv, convertor, tt.restStorage); actual != tt.expected {
				t.Errorf("shouldServe() = %v, want %v", actual, tt.expected)
			}
			if !reflect.DeepEqual(convertor.called, gv) {
				t.Errorf("expected converter to be called with %#v, got %#v", gv, convertor.called)
			}
		})
	}
}

type dummyConvertor struct {
	called runtime.GroupVersioner
}

func (d *dummyConvertor) ConvertToVersion(in runtime.Object, gv runtime.GroupVersioner) (runtime.Object, error) {
	d.called = gv
	return in, nil
}

func checkErr(t *testing.T, actual error, expected string) {
	t.Helper()
	switch {
	case len(expected) == 0 && actual == nil:
	case len(expected) == 0 && actual != nil:
		t.Fatal(actual)
	case len(expected) != 0 && actual == nil:
		t.Fatalf("missing %q, <nil>", expected)
	case len(expected) != 0 && actual != nil && !strings.Contains(actual.Error(), expected):
		t.Fatalf("missing %q, %v", expected, actual)
	}
}

func Test_removeDeletedKinds(t *testing.T) {
	tests := []struct {
		name                         string
		resourceExpirationEvaluator  resourceExpirationEvaluator
		versionedResourcesStorageMap map[string]map[string]rest.Storage
		expectedStorage              map[string]map[string]rest.Storage
	}{
		{
			name: "remove-one-of-two",
			resourceExpirationEvaluator: resourceExpirationEvaluator{
				currentVersion: apimachineryversion.MajorMinor(1, 20),
			},
			versionedResourcesStorageMap: map[string]map[string]rest.Storage{
				"v1": {
					"twenty":    storageRemovedIn(1, 20),
					"twentyone": storageRemovedIn(1, 21),
				},
			},
			expectedStorage: map[string]map[string]rest.Storage{
				"v1": {
					"twentyone": storageRemovedIn(1, 21),
				},
			},
		},
		{
			name: "remove-nested-not-expired",
			resourceExpirationEvaluator: resourceExpirationEvaluator{
				currentVersion: apimachineryversion.MajorMinor(1, 20),
			},
			versionedResourcesStorageMap: map[string]map[string]rest.Storage{
				"v1": {
					"twenty":       storageRemovedIn(1, 20),
					"twenty/scale": storageRemovedIn(1, 21),
					"twentyone":    storageRemovedIn(1, 21),
				},
			},
			expectedStorage: map[string]map[string]rest.Storage{
				"v1": {
					"twentyone": storageRemovedIn(1, 21),
				},
			},
		},
		{
			name: "remove-all-of-version",
			resourceExpirationEvaluator: resourceExpirationEvaluator{
				currentVersion: apimachineryversion.MajorMinor(1, 20),
			},
			versionedResourcesStorageMap: map[string]map[string]rest.Storage{
				"v1": {
					"twenty": storageRemovedIn(1, 20),
				},
				"v2": {
					"twenty":    storageRemovedIn(1, 20),
					"twentyone": storageRemovedIn(1, 21),
				},
			},
			expectedStorage: map[string]map[string]rest.Storage{
				"v2": {
					"twentyone": storageRemovedIn(1, 21),
				},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			convertor := &dummyConvertor{}
			tt.resourceExpirationEvaluator.RemoveDeletedKinds("group.name", convertor, tt.versionedResourcesStorageMap)
			if !reflect.DeepEqual(tt.expectedStorage, tt.versionedResourcesStorageMap) {
				t.Fatal(dump.Pretty(tt.versionedResourcesStorageMap))
			}
		})
	}
}

func Test_shouldRemoveResource(t *testing.T) {
	tests := []struct {
		name              string
		resourcesToRemove sets.String
		resourceName      string
		want              bool
	}{
		{
			name:              "prefix-matches",
			resourcesToRemove: sets.NewString("foo"),
			resourceName:      "foo/scale",
			want:              true,
		},
		{
			name:              "exact-matches",
			resourcesToRemove: sets.NewString("foo"),
			resourceName:      "foo",
			want:              true,
		},
		{
			name:              "no-match",
			resourcesToRemove: sets.NewString("foo"),
			resourceName:      "bar",
			want:              false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if actual := shouldRemoveResourceAndSubresources(tt.resourcesToRemove, tt.resourceName); actual != tt.want {
				t.Errorf("shouldRemoveResourceAndSubresources() = %v, want %v", actual, tt.want)
			}
		})
	}
}
