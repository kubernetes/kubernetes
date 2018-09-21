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

package meta

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"k8s.io/apimachinery/pkg/runtime/schema"
)

func TestAmbiguousResourceError(t *testing.T) {
	gvr := schema.GroupVersionResource{Group: "scheduling.io", Version: "v1alpha", Resource: "priority"}

	tcs := []struct {
		name   string
		mrs    []schema.GroupVersionResource
		mks    []schema.GroupVersionKind
		result string
	}{
		{
			name:   "both of MatchingResources and MatchingKinds are empty",
			mrs:    []schema.GroupVersionResource{},
			mks:    []schema.GroupVersionKind{},
			result: "{scheduling.io v1alpha priority} matches multiple resources or kinds",
		},
		{
			name:   "MatchingResources ist empty but MatchingKinds is not empty",
			mrs:    []schema.GroupVersionResource{},
			mks:    []schema.GroupVersionKind{{Group: "scheduling.io", Version: "v1alpha", Kind: "priorityClass"}},
			result: "{scheduling.io v1alpha priority} matches multiple kinds [scheduling.io/v1alpha, Kind=priorityClass]",
		},
		{
			name:   "MatchingResources is not empty but MatchingKinds is empty",
			mrs:    []schema.GroupVersionResource{{Group: "scheduling.io", Version: "v1alpha", Resource: "priority"}},
			mks:    []schema.GroupVersionKind{},
			result: "{scheduling.io v1alpha priority} matches multiple resources [{scheduling.io v1alpha priority}]",
		},
		{
			name:   "both of MatchingResources and MatchingKinds are not empty",
			mrs:    []schema.GroupVersionResource{{Group: "scheduling.io", Version: "v1alpha", Resource: "priority"}},
			mks:    []schema.GroupVersionKind{{Group: "scheduling.io", Version: "v1alpha", Kind: "priorityClass"}},
			result: "{scheduling.io v1alpha priority} matches multiple resources [{scheduling.io v1alpha priority}] and kinds [scheduling.io/v1alpha, Kind=priorityClass]",
		},
	}

	for _, tc := range tcs {
		instance := &AmbiguousResourceError{PartialResource: gvr, MatchingResources: tc.mrs, MatchingKinds: tc.mks}
		realResult := instance.Error()
		assert.EqualValues(t, tc.result, realResult, "Test failed when %s", tc.name)
	}
}

func TestAmbiguousKindError(t *testing.T) {
	gk := schema.GroupVersionKind{Group: "scheduling.io", Version: "v1alpha", Kind: "priorityClass"}

	tcs := []struct {
		name   string
		mrs    []schema.GroupVersionResource
		mks    []schema.GroupVersionKind
		result string
	}{
		{
			name:   "both of MatchingResources and MatchingKinds are empty",
			mrs:    []schema.GroupVersionResource{},
			mks:    []schema.GroupVersionKind{},
			result: "scheduling.io/v1alpha, Kind=priorityClass matches multiple resources or kinds",
		},
		{
			name:   "MatchingResources ist empty but MatchingKinds is not empty",
			mrs:    []schema.GroupVersionResource{},
			mks:    []schema.GroupVersionKind{{Group: "scheduling.io", Version: "v1alpha", Kind: "priorityClass"}},
			result: "scheduling.io/v1alpha, Kind=priorityClass matches multiple kinds [scheduling.io/v1alpha, Kind=priorityClass]",
		},
		{
			name:   "MatchingResources is not empty but MatchingKinds is empty",
			mrs:    []schema.GroupVersionResource{{Group: "scheduling.io", Version: "v1alpha", Resource: "priority"}},
			mks:    []schema.GroupVersionKind{},
			result: "scheduling.io/v1alpha, Kind=priorityClass matches multiple resources [{scheduling.io v1alpha priority}]",
		},
		{
			name:   "both of MatchingResources and MatchingKinds are not empty",
			mrs:    []schema.GroupVersionResource{{Group: "scheduling.io", Version: "v1alpha", Resource: "priority"}},
			mks:    []schema.GroupVersionKind{{Group: "scheduling.io", Version: "v1alpha", Kind: "priorityClass"}},
			result: "scheduling.io/v1alpha, Kind=priorityClass matches multiple resources [{scheduling.io v1alpha priority}] and kinds [scheduling.io/v1alpha, Kind=priorityClass]",
		},
	}

	for _, tc := range tcs {
		instance := &AmbiguousKindError{PartialKind: gk, MatchingResources: tc.mrs, MatchingKinds: tc.mks}
		realResult := instance.Error()
		assert.EqualValues(t, tc.result, realResult, "Test failed when %s", tc.name)
	}
}

func TestIsAmbiguousError(t *testing.T) {
	tcs := []struct {
		name   string
		input  error
		result bool
	}{
		{name: "error is nil", input: nil, result: false},
		{name: "error is AmbiguousResourceError", input: &AmbiguousResourceError{}, result: true},
		{name: "error is AmbiguousKindError", input: &AmbiguousKindError{}, result: true},
		{name: "error is NoResourceMatchError", input: &NoResourceMatchError{}, result: false},
	}

	for _, tc := range tcs {
		realResult := IsAmbiguousError(tc.input)
		assert.EqualValues(t, tc.result, realResult, "Test failed when %s", tc.name)
	}
}

func TestNoResourceMatchError(t *testing.T) {
	tcs := []struct {
		name   string
		pr     schema.GroupVersionResource
		result string
	}{
		{
			name:   "PartialResource with nothing",
			pr:     schema.GroupVersionResource{},
			result: "no matches for {  }",
		}, {
			name:   "PartialResource without group exist",
			pr:     schema.GroupVersionResource{Version: "v1", Resource: "pods"},
			result: "no matches for { v1 pods}",
		},
		{
			name:   "PartialResource with group exist",
			pr:     schema.GroupVersionResource{Group: "scheduling.io", Version: "v1alpha", Resource: "PriorityClass"},
			result: "no matches for {scheduling.io v1alpha PriorityClass}",
		},
	}

	for _, tc := range tcs {
		instance := &NoResourceMatchError{PartialResource: tc.pr}
		realResult := instance.Error()
		assert.EqualValues(t, tc.result, realResult, "Test failed when %s", tc.name)
	}
}

func TestNoKindMatchError(t *testing.T) {
	tcs := []struct {
		name     string
		gk       schema.GroupKind
		versions []string
		result   string
	}{
		{
			name:     "GroupKind without group and SearchedVersions not exist",
			gk:       schema.GroupKind{Kind: "priorityClass"},
			versions: []string{},
			result:   "no matches for kind \"priorityClass\" in group \"\"",
		},
		{
			name:     "GroupKind without group and length of SearchedVersions is 1",
			gk:       schema.GroupKind{Kind: "priorityClass"},
			versions: []string{"v1alpha"},
			result:   "no matches for kind \"priorityClass\" in version \"v1alpha\"",
		},
		{
			name:     "GroupKind without group and length of SearchedVersions is 2",
			gk:       schema.GroupKind{Kind: "priorityClass"},
			versions: []string{"v1", "v1alpha"},
			result:   "no matches for kind \"priorityClass\" in versions [\"v1\" \"v1alpha\"]",
		},
		{
			name:     "GroupKind with group and SearchedVersions not exist",
			gk:       schema.GroupKind{Group: "scheduling.io", Kind: "priorityClass"},
			versions: []string{},
			result:   "no matches for kind \"priorityClass\" in group \"scheduling.io\"",
		},
		{
			name:     "GroupKind with group and length of SearchedVersions is 1",
			gk:       schema.GroupKind{Group: "scheduling.io", Kind: "priorityClass"},
			versions: []string{"v1alpha"},
			result:   "no matches for kind \"priorityClass\" in version \"scheduling.io/v1alpha\"",
		},
		{
			name:     "GroupKind with group and length of SearchedVersions is 2",
			gk:       schema.GroupKind{Group: "scheduling.io", Kind: "priorityClass"},
			versions: []string{"v1", "v1alpha"},
			result:   "no matches for kind \"priorityClass\" in versions [\"scheduling.io/v1\" \"scheduling.io/v1alpha\"]",
		},
	}

	for _, tc := range tcs {
		// initialize instance
		instance := &NoKindMatchError{GroupKind: tc.gk, SearchedVersions: tc.versions}
		realResult := instance.Error()
		assert.EqualValues(t, tc.result, realResult, "Test failed when %s", tc.name)
	}
}

func TestIsNoMatchError(t *testing.T) {
	tcs := []struct {
		name   string
		input  error
		result bool
	}{
		{name: "error is nil", input: nil, result: false},
		{name: "error is NoResourceMatchError", input: &NoResourceMatchError{}, result: true},
		{name: "error is NoKindMatchError", input: &NoKindMatchError{}, result: true},
		{name: "error is AmbiguousKindError", input: &AmbiguousKindError{}, result: false},
	}

	for _, tc := range tcs {
		realResult := IsNoMatchError(tc.input)
		assert.EqualValues(t, tc.result, realResult, "Test failed when %s", tc.name)
	}
}
