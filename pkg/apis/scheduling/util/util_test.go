/*
Copyright 2019 The Kubernetes Authors.

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

package util

import (
	"fmt"
	"reflect"
	"testing"

	"k8s.io/apimachinery/pkg/util/diff"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/scheduling"
	"k8s.io/kubernetes/pkg/features"
)

func TestDropNonPreemptingPriority(t *testing.T) {
	pcWithoutNonPreemptingPriority := func() *scheduling.PriorityClass {
		return &scheduling.PriorityClass{}
	}
	pcWithNonPreemptingPriority := func() *scheduling.PriorityClass {
		preemptionPolicy := core.PreemptNever
		return &scheduling.PriorityClass{
			PreemptionPolicy: &preemptionPolicy,
		}
	}

	pcInfo := []struct {
		description              string
		hasNonPreemptingPriority bool
		pc                       func() *scheduling.PriorityClass
	}{
		{
			description:              "PriorityClass Without NonPreemptingPriority",
			hasNonPreemptingPriority: false,
			pc:                       pcWithoutNonPreemptingPriority,
		},
		{
			description:              "PriorityClass With NonPreemptingPriority",
			hasNonPreemptingPriority: true,
			pc:                       pcWithNonPreemptingPriority,
		},
		{
			description:              "is nil",
			hasNonPreemptingPriority: false,
			pc:                       func() *scheduling.PriorityClass { return nil },
		},
	}

	for _, enabled := range []bool{true, false} {
		for _, oldPriorityClassInfo := range pcInfo {
			for _, newPriorityClassInfo := range pcInfo {
				oldPriorityClassHasNonPreemptingPriority, oldPriorityClass := oldPriorityClassInfo.hasNonPreemptingPriority, oldPriorityClassInfo.pc()
				newPriorityClassHasNonPreemptingPriority, newPriorityClass := newPriorityClassInfo.hasNonPreemptingPriority, newPriorityClassInfo.pc()
				if newPriorityClass == nil {
					continue
				}

				t.Run(fmt.Sprintf("feature enabled=%v, old PriorityClass %v, new PriorityClass %v", enabled, oldPriorityClassInfo.description, newPriorityClassInfo.description), func(t *testing.T) {
					defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.NonPreemptingPriority, enabled)()

					DropDisabledFields(newPriorityClass, oldPriorityClass)

					// old PriorityClass should never be changed
					if !reflect.DeepEqual(oldPriorityClass, oldPriorityClassInfo.pc()) {
						t.Errorf("old PriorityClass changed: %v", diff.ObjectReflectDiff(oldPriorityClass, oldPriorityClassInfo.pc()))
					}

					switch {
					case enabled || oldPriorityClassHasNonPreemptingPriority:
						// new PriorityClass should not be changed if the feature is enabled, or if the old PriorityClass had NonPreemptingPriority
						if !reflect.DeepEqual(newPriorityClass, newPriorityClassInfo.pc()) {
							t.Errorf("new PriorityClass changed: %v", diff.ObjectReflectDiff(newPriorityClass, newPriorityClassInfo.pc()))
						}
					case newPriorityClassHasNonPreemptingPriority:
						// new PriorityClass should be changed
						if reflect.DeepEqual(newPriorityClass, newPriorityClassInfo.pc()) {
							t.Errorf("new PriorityClass was not changed")
						}
						// new PriorityClass should not have NonPreemptingPriority
						if !reflect.DeepEqual(newPriorityClass, pcWithoutNonPreemptingPriority()) {
							t.Errorf("new PriorityClass had PriorityClassNonPreemptingPriority: %v", diff.ObjectReflectDiff(newPriorityClass, pcWithoutNonPreemptingPriority()))
						}
					default:
						// new PriorityClass should not need to be changed
						if !reflect.DeepEqual(newPriorityClass, newPriorityClassInfo.pc()) {
							t.Errorf("new PriorityClass changed: %v", diff.ObjectReflectDiff(newPriorityClass, newPriorityClassInfo.pc()))
						}
					}
				})
			}
		}
	}
}
