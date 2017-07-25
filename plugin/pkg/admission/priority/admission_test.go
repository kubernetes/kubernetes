/*
Copyright 2017 The Kubernetes Authors.

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

package admission

import (
	"fmt"
	"testing"

	"github.com/golang/glog"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/admission"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/scheduling"
	informers "k8s.io/kubernetes/pkg/client/informers/informers_generated/internalversion"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/features"
)

func addPriorityClasses(ctrl *priorityPlugin, priorityClasses []*scheduling.PriorityClass) {
	informerFactory := informers.NewSharedInformerFactory(nil, controller.NoResyncPeriodFunc())
	ctrl.SetInternalKubeInformerFactory(informerFactory)
	// First add the existing classes to the cache.
	for _, c := range priorityClasses {
		informerFactory.Scheduling().InternalVersion().PriorityClasses().Informer().GetStore().Add(c)
	}
}

var defaultClass1 = &scheduling.PriorityClass{
	TypeMeta: metav1.TypeMeta{
		Kind: "PriorityClass",
	},
	ObjectMeta: metav1.ObjectMeta{
		Name: "default1",
	},
	Value:         1000,
	GlobalDefault: true,
}

var defaultClass2 = &scheduling.PriorityClass{
	TypeMeta: metav1.TypeMeta{
		Kind: "PriorityClass",
	},
	ObjectMeta: metav1.ObjectMeta{
		Name: "default2",
	},
	Value:         2000,
	GlobalDefault: true,
}

var nondefaultClass1 = &scheduling.PriorityClass{
	TypeMeta: metav1.TypeMeta{
		Kind: "PriorityClass",
	},
	ObjectMeta: metav1.ObjectMeta{
		Name: "nondefault1",
	},
	Value:       2000,
	Description: "Just a test priority class",
}

func TestPriorityClassAdmission(t *testing.T) {
	var tooHighPriorityClass = &scheduling.PriorityClass{
		TypeMeta: metav1.TypeMeta{
			Kind: "PriorityClass",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: "toohighclass",
		},
		Value:       HighestUserDefinablePriority + 1,
		Description: "Just a test priority class",
	}

	var systemClass = &scheduling.PriorityClass{
		TypeMeta: metav1.TypeMeta{
			Kind: "PriorityClass",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: "system-cluster-critical",
		},
		Value:       HighestUserDefinablePriority + 1,
		Description: "Name conflicts with system priority class names",
	}

	tests := []struct {
		name            string
		existingClasses []*scheduling.PriorityClass
		newClass        *scheduling.PriorityClass
		expectError     bool
	}{
		{
			"one default class",
			[]*scheduling.PriorityClass{},
			defaultClass1,
			false,
		},
		{
			"more than one default classes",
			[]*scheduling.PriorityClass{defaultClass1},
			defaultClass2,
			true,
		},
		{
			"too high PriorityClass value",
			[]*scheduling.PriorityClass{},
			tooHighPriorityClass,
			true,
		},
		{
			"system name conflict",
			[]*scheduling.PriorityClass{},
			systemClass,
			true,
		},
	}

	for _, test := range tests {
		glog.V(4).Infof("starting test %q", test.name)

		ctrl := NewPlugin().(*priorityPlugin)
		// Add existing priority classes.
		addPriorityClasses(ctrl, test.existingClasses)
		// Now add the new class.
		attrs := admission.NewAttributesRecord(
			test.newClass,
			nil,
			api.Kind("PriorityClass").WithVersion("version"),
			"",
			"",
			api.Resource("priorityclasses").WithVersion("version"),
			"",
			admission.Create,
			nil,
		)
		err := ctrl.Admit(attrs)
		glog.Infof("Got %v", err)
		if err != nil && !test.expectError {
			t.Errorf("Test %q: unexpected error received: %v", test.name, err)
		}
		if err == nil && test.expectError {
			t.Errorf("Test %q: expected error and no error recevied", test.name)
		}
	}
}

// TestDefaultPriority tests that default priority is resolved correctly.
func TestDefaultPriority(t *testing.T) {
	pcResource := api.Resource("priorityclasses").WithVersion("version")
	pcKind := api.Kind("PriorityClass").WithVersion("version")
	updatedDefaultClass1 := *defaultClass1
	updatedDefaultClass1.GlobalDefault = false

	tests := []struct {
		name                  string
		classesBefore         []*scheduling.PriorityClass
		classesAfter          []*scheduling.PriorityClass
		attributes            admission.Attributes
		expectedDefaultBefore int32
		expectedDefaultAfter  int32
	}{
		{
			name:                  "simple resolution with a default class",
			classesBefore:         []*scheduling.PriorityClass{defaultClass1},
			classesAfter:          []*scheduling.PriorityClass{defaultClass1},
			attributes:            nil,
			expectedDefaultBefore: defaultClass1.Value,
			expectedDefaultAfter:  defaultClass1.Value,
		},
		{
			name:                  "add a default class",
			classesBefore:         []*scheduling.PriorityClass{nondefaultClass1},
			classesAfter:          []*scheduling.PriorityClass{nondefaultClass1, defaultClass1},
			attributes:            admission.NewAttributesRecord(defaultClass1, nil, pcKind, "", defaultClass1.Name, pcResource, "", admission.Create, nil),
			expectedDefaultBefore: scheduling.DefaultPriorityWhenNoDefaultClassExists,
			expectedDefaultAfter:  defaultClass1.Value,
		},
		{
			name:                  "delete default priority class",
			classesBefore:         []*scheduling.PriorityClass{defaultClass1},
			classesAfter:          []*scheduling.PriorityClass{},
			attributes:            admission.NewAttributesRecord(nil, nil, pcKind, "", defaultClass1.Name, pcResource, "", admission.Delete, nil),
			expectedDefaultBefore: defaultClass1.Value,
			expectedDefaultAfter:  scheduling.DefaultPriorityWhenNoDefaultClassExists,
		},
		{
			name:                  "update default class and remove its global default",
			classesBefore:         []*scheduling.PriorityClass{defaultClass1},
			classesAfter:          []*scheduling.PriorityClass{&updatedDefaultClass1},
			attributes:            admission.NewAttributesRecord(&updatedDefaultClass1, defaultClass1, pcKind, "", defaultClass1.Name, pcResource, "", admission.Update, nil),
			expectedDefaultBefore: defaultClass1.Value,
			expectedDefaultAfter:  scheduling.DefaultPriorityWhenNoDefaultClassExists,
		},
	}

	for _, test := range tests {
		glog.V(4).Infof("starting test %q", test.name)
		ctrl := NewPlugin().(*priorityPlugin)
		addPriorityClasses(ctrl, test.classesBefore)
		defaultPriority, err := ctrl.getDefaultPriority()
		if err != nil {
			t.Errorf("Test %q: unexpected error while getting default priority: %v", test.name, err)
		}
		if err == nil && defaultPriority != test.expectedDefaultBefore {
			t.Errorf("Test %q: expected default priority %d, but got %d", test.name, test.expectedDefaultBefore, defaultPriority)
		}
		if test.attributes != nil {
			err := ctrl.Admit(test.attributes)
			if err != nil {
				t.Errorf("Test %q: unexpected error received: %v", test.name, err)
			}
		}
		addPriorityClasses(ctrl, test.classesAfter)
		defaultPriority, err = ctrl.getDefaultPriority()
		if err != nil {
			t.Errorf("Test %q: unexpected error while getting default priority: %v", test.name, err)
		}
		if err == nil && defaultPriority != test.expectedDefaultAfter {
			t.Errorf("Test %q: expected default priority %d, but got %d", test.name, test.expectedDefaultAfter, defaultPriority)
		}
	}
}

var intPriority = int32(1000)

func TestPodAdmission(t *testing.T) {
	containerName := "container"

	pods := []*api.Pod{
		// pod[0]: Pod with a proper priority class.
		{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "pod-w-priorityclass",
				Namespace: "namespace",
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Name: containerName,
					},
				},
				PriorityClassName: "default1",
			},
		},
		// pod[1]: Pod with no priority class
		{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "pod-wo-priorityclass",
				Namespace: "namespace",
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Name: containerName,
					},
				},
			},
		},
		// pod[2]: Pod with non-existing priority class
		{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "pod-w-non-existing-priorityclass",
				Namespace: "namespace",
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Name: containerName,
					},
				},
				PriorityClassName: "non-existing",
			},
		},
		// pod[3]: Pod with integer value of priority
		{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "pod-w-integer-priority",
				Namespace: "namespace",
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Name: containerName,
					},
				},
				PriorityClassName: "default1",
				Priority:          &intPriority,
			},
		},
		// pod[4]: Pod with a system priority class name
		{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "pod-w-system-priority",
				Namespace: "namespace",
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Name: containerName,
					},
				},
				PriorityClassName: "system-cluster-critical",
			},
		},
	}
	// Enable PodPriority feature gate.
	utilfeature.DefaultFeatureGate.Set(fmt.Sprintf("%s=true", features.PodPriority))
	tests := []struct {
		name            string
		existingClasses []*scheduling.PriorityClass
		// Admission controller changes pod spec. So, we take an api.Pod instead of
		// *api.Pod to avoid interfering with other tests.
		pod              api.Pod
		expectedPriority int32
		expectError      bool
	}{
		{
			"Pod with priority class",
			[]*scheduling.PriorityClass{defaultClass1, nondefaultClass1},
			*pods[0],
			1000,
			false,
		},

		{
			"Pod without priority class",
			[]*scheduling.PriorityClass{defaultClass1},
			*pods[1],
			1000,
			false,
		},
		{
			"pod without priority class and no existing priority class",
			[]*scheduling.PriorityClass{},
			*pods[1],
			scheduling.DefaultPriorityWhenNoDefaultClassExists,
			false,
		},
		{
			"pod without priority class and no default class",
			[]*scheduling.PriorityClass{nondefaultClass1},
			*pods[1],
			scheduling.DefaultPriorityWhenNoDefaultClassExists,
			false,
		},
		{
			"pod with a system priority class",
			[]*scheduling.PriorityClass{},
			*pods[4],
			SystemCriticalPriority,
			false,
		},
		{
			"Pod with non-existing priority class",
			[]*scheduling.PriorityClass{defaultClass1, nondefaultClass1},
			*pods[2],
			0,
			true,
		},
		{
			"pod with integer priority",
			[]*scheduling.PriorityClass{},
			*pods[3],
			0,
			true,
		},
	}

	for _, test := range tests {
		glog.V(4).Infof("starting test %q", test.name)

		ctrl := NewPlugin().(*priorityPlugin)
		// Add existing priority classes.
		addPriorityClasses(ctrl, test.existingClasses)

		// Create pod.
		attrs := admission.NewAttributesRecord(
			&test.pod,
			nil,
			api.Kind("Pod").WithVersion("version"),
			test.pod.ObjectMeta.Namespace,
			"",
			api.Resource("pods").WithVersion("version"),
			"",
			admission.Create,
			nil,
		)
		err := ctrl.Admit(attrs)
		glog.Infof("Got %v", err)
		if !test.expectError {
			if err != nil {
				t.Errorf("Test %q: unexpected error received: %v", test.name, err)
			}
			if *test.pod.Spec.Priority != test.expectedPriority {
				t.Errorf("Test %q: expected priority is %d, but got %d.", test.name, test.expectedPriority, *test.pod.Spec.Priority)
			}
		}
		if err == nil && test.expectError {
			t.Errorf("Test %q: expected error and no error recevied", test.name)
		}
	}
}
