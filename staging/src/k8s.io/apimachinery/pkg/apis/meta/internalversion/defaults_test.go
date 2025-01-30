/*
Copyright 2023 The Kubernetes Authors.

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

package internalversion

import (
	"testing"

	"github.com/google/go-cmp/cmp"

	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
)

func TestSetListOptionsDefaults(t *testing.T) {
	boolPtrFn := func(b bool) *bool {
		return &b
	}

	scenarios := []struct {
		name                    string
		watchListFeatureEnabled bool
		targetObj               ListOptions
		expectedObj             ListOptions
	}{
		{
			name:                    "no-op, RV doesn't match",
			watchListFeatureEnabled: true,
			targetObj:               ListOptions{LabelSelector: labels.Everything(), FieldSelector: fields.Everything(), Watch: true, ResourceVersion: "1"},
			expectedObj:             ListOptions{LabelSelector: labels.Everything(), FieldSelector: fields.Everything(), Watch: true, ResourceVersion: "1"},
		},
		{
			name:                    "no-op, SendInitialEvents set",
			watchListFeatureEnabled: true,
			targetObj:               ListOptions{LabelSelector: labels.Everything(), FieldSelector: fields.Everything(), Watch: true, SendInitialEvents: boolPtrFn(true)},
			expectedObj:             ListOptions{LabelSelector: labels.Everything(), FieldSelector: fields.Everything(), Watch: true, SendInitialEvents: boolPtrFn(true)},
		},
		{
			name:                    "no-op, ResourceVersionMatch set",
			watchListFeatureEnabled: true,
			targetObj:               ListOptions{LabelSelector: labels.Everything(), FieldSelector: fields.Everything(), Watch: true, ResourceVersionMatch: "m"},
			expectedObj:             ListOptions{LabelSelector: labels.Everything(), FieldSelector: fields.Everything(), Watch: true, ResourceVersionMatch: "m"},
		},
		{
			name:                    "no-op, Watch=false",
			watchListFeatureEnabled: true,
			targetObj:               ListOptions{LabelSelector: labels.Everything(), FieldSelector: fields.Everything()},
			expectedObj:             ListOptions{LabelSelector: labels.Everything(), FieldSelector: fields.Everything()},
		},
		{
			name:                    "defaults applied, match on empty RV",
			watchListFeatureEnabled: true,
			targetObj:               ListOptions{LabelSelector: labels.Everything(), FieldSelector: fields.Everything(), Watch: true},
			expectedObj:             ListOptions{LabelSelector: labels.Everything(), FieldSelector: fields.Everything(), Watch: true, SendInitialEvents: boolPtrFn(true), ResourceVersionMatch: metav1.ResourceVersionMatchNotOlderThan},
		},
		{
			name:                    "defaults applied, match on RV=0",
			watchListFeatureEnabled: true,
			targetObj:               ListOptions{LabelSelector: labels.Everything(), FieldSelector: fields.Everything(), Watch: true, ResourceVersion: "0"},
			expectedObj:             ListOptions{LabelSelector: labels.Everything(), FieldSelector: fields.Everything(), Watch: true, ResourceVersion: "0", SendInitialEvents: boolPtrFn(true), ResourceVersionMatch: metav1.ResourceVersionMatchNotOlderThan},
		},
		{
			name:        "no-op, match on empty RV but watch-list fg is off",
			targetObj:   ListOptions{LabelSelector: labels.Everything(), FieldSelector: fields.Everything(), Watch: true},
			expectedObj: ListOptions{LabelSelector: labels.Everything(), FieldSelector: fields.Everything(), Watch: true},
		},
		{
			name:                    "no-op, match on empty RV but SendInitialEvents is on",
			watchListFeatureEnabled: true,
			targetObj:               ListOptions{LabelSelector: labels.Everything(), FieldSelector: fields.Everything(), Watch: true, SendInitialEvents: boolPtrFn(true)},
			expectedObj:             ListOptions{LabelSelector: labels.Everything(), FieldSelector: fields.Everything(), Watch: true, SendInitialEvents: boolPtrFn(true)},
		},
		{
			name:                    "no-op, match on empty RV but SendInitialEvents is off",
			watchListFeatureEnabled: true,
			targetObj:               ListOptions{LabelSelector: labels.Everything(), FieldSelector: fields.Everything(), Watch: true, SendInitialEvents: boolPtrFn(false)},
			expectedObj:             ListOptions{LabelSelector: labels.Everything(), FieldSelector: fields.Everything(), Watch: true, SendInitialEvents: boolPtrFn(false)},
		},
		{
			name:                    "no-op, match on empty RV but ResourceVersionMatch set",
			watchListFeatureEnabled: true,
			targetObj:               ListOptions{LabelSelector: labels.Everything(), FieldSelector: fields.Everything(), Watch: true, ResourceVersionMatch: "m"},
			expectedObj:             ListOptions{LabelSelector: labels.Everything(), FieldSelector: fields.Everything(), Watch: true, ResourceVersionMatch: "m"},
		},
	}

	for _, scenario := range scenarios {
		t.Run(scenario.name, func(t *testing.T) {
			SetListOptionsDefaults(&scenario.targetObj, scenario.watchListFeatureEnabled)
			if !apiequality.Semantic.DeepEqual(&scenario.expectedObj, &scenario.targetObj) {
				t.Errorf("expected and defaulted objects are different:\n%s", cmp.Diff(&scenario.expectedObj, &scenario.targetObj))
			}
		})
	}
}
