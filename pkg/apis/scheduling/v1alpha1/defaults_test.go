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

package v1alpha1_test

import (
	"reflect"
	"testing"

	apiv1 "k8s.io/api/core/v1"
	"k8s.io/api/scheduling/v1alpha1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/pkg/api/legacyscheme"

	// ensure types are installed
	_ "k8s.io/kubernetes/pkg/apis/scheduling/install"
)

func roundTrip(t *testing.T, obj runtime.Object) runtime.Object {
	codec := legacyscheme.Codecs.LegacyCodec(v1alpha1.SchemeGroupVersion)
	data, err := runtime.Encode(codec, obj)
	if err != nil {
		t.Errorf("%v\n %#v", err, obj)
		return nil
	}
	obj2, err := runtime.Decode(codec, data)
	if err != nil {
		t.Errorf("%v\nData: %s\nSource: %#v", err, string(data), obj)
		return nil
	}
	obj3 := reflect.New(reflect.TypeOf(obj).Elem()).Interface().(runtime.Object)
	err = legacyscheme.Scheme.Convert(obj2, obj3, nil)
	if err != nil {
		t.Errorf("%v\nSource: %#v", err, obj2)
		return nil
	}
	return obj3
}

func TestSetDefaultPreempting(t *testing.T) {
	priorityClass := &v1alpha1.PriorityClass{}

	output := roundTrip(t, runtime.Object(priorityClass)).(*v1alpha1.PriorityClass)
	if output.PreemptionPolicy == nil || *output.PreemptionPolicy != apiv1.PreemptLowerPriority {
		t.Errorf("Expected PriorityClass.Preempting value: %+v\ngot: %+v\n", apiv1.PreemptLowerPriority, output.PreemptionPolicy)
	}
}
