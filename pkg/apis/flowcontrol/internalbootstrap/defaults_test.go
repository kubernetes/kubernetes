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

package internalbootstrap

import (
	"testing"

	fcv1a1 "k8s.io/api/flowcontrol/v1alpha1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apiserver/pkg/apis/flowcontrol/bootstrap"
)

func TestMandatoryAlreadyDefaulted(t *testing.T) {
	scheme := NewAPFScheme()
	for _, obj := range bootstrap.MandatoryFlowSchemas {
		obj2 := obj.DeepCopyObject().(*fcv1a1.FlowSchema)
		scheme.Default(obj2)
		if apiequality.Semantic.DeepEqual(obj, obj2) {
			t.Logf("Defaulting makes no change to %#+v", *obj)
		} else {
			t.Errorf("Defaulting changed %#+v to %#+v", *obj, *obj2)
		}
	}
	for _, obj := range bootstrap.MandatoryPriorityLevelConfigurations {
		obj2 := obj.DeepCopyObject().(*fcv1a1.PriorityLevelConfiguration)
		scheme.Default(obj2)
		if apiequality.Semantic.DeepEqual(obj, obj2) {
			t.Logf("Defaulting makes no change to %#+v", *obj)
		} else {
			t.Errorf("Defaulting changed %#+v to %#+v", *obj, *obj2)
		}
	}
}
