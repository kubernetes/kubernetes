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

package priority

import (
	"testing"

	"k8s.io/klog"

	schedulingv1beta1 "k8s.io/api/scheduling/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/authentication/user"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	utilfeaturetesting "k8s.io/apiserver/pkg/util/feature/testing"
	"k8s.io/client-go/informers"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/scheduling"
	"k8s.io/kubernetes/pkg/apis/scheduling/v1beta1"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/features"
)

func addPriorityClasses(ctrl *priorityPlugin, priorityClasses []*scheduling.PriorityClass) error {
	informerFactory := informers.NewSharedInformerFactory(nil, controller.NoResyncPeriodFunc())
	ctrl.SetExternalKubeInformerFactory(informerFactory)
	// First add the existing classes to the cache.
	for _, c := range priorityClasses {
		s := &schedulingv1beta1.PriorityClass{}
		if err := v1beta1.Convert_scheduling_PriorityClass_To_v1beta1_PriorityClass(c, s, nil); err != nil {
			return err
		}
		informerFactory.Scheduling().V1beta1().PriorityClasses().Informer().GetStore().Add(s)
	}
	return nil
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

var systemClusterCritical = &scheduling.PriorityClass{
	TypeMeta: metav1.TypeMeta{
		Kind: "PriorityClass",
	},
	ObjectMeta: metav1.ObjectMeta{
		Name: scheduling.SystemClusterCritical,
	},
	Value:         scheduling.SystemCriticalPriority,
	GlobalDefault: true,
}

func TestPriorityClassAdmission(t *testing.T) {
	var systemClass = &scheduling.PriorityClass{
		TypeMeta: metav1.TypeMeta{
			Kind: "PriorityClass",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: scheduling.SystemPriorityClassPrefix + "test",
		},
		Value:       scheduling.HighestUserDefinablePriority + 1,
		Description: "Name has system critical prefix",
	}

	tests := []struct {
		name            string
		existingClasses []*scheduling.PriorityClass
		newClass        *scheduling.PriorityClass
		userInfo        user.Info
		expectError     bool
	}{
		{
			"one default class",
			[]*scheduling.PriorityClass{},
			defaultClass1,
			nil,
			false,
		},
		{
			"more than one default classes",
			[]*scheduling.PriorityClass{defaultClass1},
			defaultClass2,
			nil,
			true,
		},
		{
			"system name and value are allowed by admission controller",
			[]*scheduling.PriorityClass{},
			systemClass,
			&user.DefaultInfo{
				Name: user.APIServerUser,
			},
			false,
		},
	}

	for _, test := range tests {
		klog.V(4).Infof("starting test %q", test.name)

		ctrl := newPlugin()
		// Add existing priority classes.
		if err := addPriorityClasses(ctrl, test.existingClasses); err != nil {
			t.Errorf("Test %q: unable to add object to informer: %v", test.name, err)
		}
		// Now add the new class.
		attrs := admission.NewAttributesRecord(
			test.newClass,
			nil,
			scheduling.Kind("PriorityClass").WithVersion("version"),
			"",
			"",
			scheduling.Resource("priorityclasses").WithVersion("version"),
			"",
			admission.Create,
			false,
			test.userInfo,
		)
		err := ctrl.Validate(attrs)
		klog.Infof("Got %v", err)
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
	pcResource := scheduling.Resource("priorityclasses").WithVersion("version")
	pcKind := scheduling.Kind("PriorityClass").WithVersion("version")
	updatedDefaultClass1 := *defaultClass1
	updatedDefaultClass1.GlobalDefault = false

	tests := []struct {
		name                      string
		classesBefore             []*scheduling.PriorityClass
		classesAfter              []*scheduling.PriorityClass
		attributes                admission.Attributes
		expectedDefaultBefore     int32
		expectedDefaultNameBefore string
		expectedDefaultAfter      int32
		expectedDefaultNameAfter  string
	}{
		{
			name:                      "simple resolution with a default class",
			classesBefore:             []*scheduling.PriorityClass{defaultClass1},
			classesAfter:              []*scheduling.PriorityClass{defaultClass1},
			attributes:                nil,
			expectedDefaultBefore:     defaultClass1.Value,
			expectedDefaultNameBefore: defaultClass1.Name,
			expectedDefaultAfter:      defaultClass1.Value,
			expectedDefaultNameAfter:  defaultClass1.Name,
		},
		{
			name:                      "add a default class",
			classesBefore:             []*scheduling.PriorityClass{nondefaultClass1},
			classesAfter:              []*scheduling.PriorityClass{nondefaultClass1, defaultClass1},
			attributes:                admission.NewAttributesRecord(defaultClass1, nil, pcKind, "", defaultClass1.Name, pcResource, "", admission.Create, false, nil),
			expectedDefaultBefore:     scheduling.DefaultPriorityWhenNoDefaultClassExists,
			expectedDefaultNameBefore: "",
			expectedDefaultAfter:      defaultClass1.Value,
			expectedDefaultNameAfter:  defaultClass1.Name,
		},
		{
			name:                      "multiple default classes resolves to the minimum value among them",
			classesBefore:             []*scheduling.PriorityClass{defaultClass1, defaultClass2},
			classesAfter:              []*scheduling.PriorityClass{defaultClass2},
			attributes:                admission.NewAttributesRecord(nil, nil, pcKind, "", defaultClass1.Name, pcResource, "", admission.Delete, false, nil),
			expectedDefaultBefore:     defaultClass1.Value,
			expectedDefaultNameBefore: defaultClass1.Name,
			expectedDefaultAfter:      defaultClass2.Value,
			expectedDefaultNameAfter:  defaultClass2.Name,
		},
		{
			name:                      "delete default priority class",
			classesBefore:             []*scheduling.PriorityClass{defaultClass1},
			classesAfter:              []*scheduling.PriorityClass{},
			attributes:                admission.NewAttributesRecord(nil, nil, pcKind, "", defaultClass1.Name, pcResource, "", admission.Delete, false, nil),
			expectedDefaultBefore:     defaultClass1.Value,
			expectedDefaultNameBefore: defaultClass1.Name,
			expectedDefaultAfter:      scheduling.DefaultPriorityWhenNoDefaultClassExists,
			expectedDefaultNameAfter:  "",
		},
		{
			name:                      "update default class and remove its global default",
			classesBefore:             []*scheduling.PriorityClass{defaultClass1},
			classesAfter:              []*scheduling.PriorityClass{&updatedDefaultClass1},
			attributes:                admission.NewAttributesRecord(&updatedDefaultClass1, defaultClass1, pcKind, "", defaultClass1.Name, pcResource, "", admission.Update, false, nil),
			expectedDefaultBefore:     defaultClass1.Value,
			expectedDefaultNameBefore: defaultClass1.Name,
			expectedDefaultAfter:      scheduling.DefaultPriorityWhenNoDefaultClassExists,
			expectedDefaultNameAfter:  "",
		},
	}

	for _, test := range tests {
		klog.V(4).Infof("starting test %q", test.name)
		ctrl := newPlugin()
		if err := addPriorityClasses(ctrl, test.classesBefore); err != nil {
			t.Errorf("Test %q: unable to add object to informer: %v", test.name, err)
		}
		pcName, defaultPriority, err := ctrl.getDefaultPriority()
		if err != nil {
			t.Errorf("Test %q: unexpected error while getting default priority: %v", test.name, err)
		}
		if err == nil &&
			(defaultPriority != test.expectedDefaultBefore || pcName != test.expectedDefaultNameBefore) {
			t.Errorf("Test %q: expected default priority %s(%d), but got %s(%d)",
				test.name, test.expectedDefaultNameBefore, test.expectedDefaultBefore, pcName, defaultPriority)
		}
		if test.attributes != nil {
			err := ctrl.Validate(test.attributes)
			if err != nil {
				t.Errorf("Test %q: unexpected error received: %v", test.name, err)
			}
		}
		if err := addPriorityClasses(ctrl, test.classesAfter); err != nil {
			t.Errorf("Test %q: unable to add object to informer: %v", test.name, err)
		}
		pcName, defaultPriority, err = ctrl.getDefaultPriority()
		if err != nil {
			t.Errorf("Test %q: unexpected error while getting default priority: %v", test.name, err)
		}
		if err == nil &&
			(defaultPriority != test.expectedDefaultAfter || pcName != test.expectedDefaultNameAfter) {
			t.Errorf("Test %q: expected default priority %s(%d), but got %s(%d)",
				test.name, test.expectedDefaultNameAfter, test.expectedDefaultAfter, pcName, defaultPriority)
		}
	}
}

var zeroPriority = int32(0)
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
				Namespace: metav1.NamespaceSystem,
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Name: containerName,
					},
				},
				PriorityClassName: scheduling.SystemClusterCritical,
			},
		},
		// pod[5]: mirror Pod with a system priority class name
		{
			ObjectMeta: metav1.ObjectMeta{
				Name:        "mirror-pod-w-system-priority",
				Namespace:   metav1.NamespaceSystem,
				Annotations: map[string]string{api.MirrorPodAnnotationKey: ""},
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
		// pod[6]: mirror Pod with integer value of priority
		{
			ObjectMeta: metav1.ObjectMeta{
				Name:        "mirror-pod-w-integer-priority",
				Namespace:   "namespace",
				Annotations: map[string]string{api.MirrorPodAnnotationKey: ""},
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
		// pod[7]: Pod with a critical priority annotation. This needs to be automatically assigned
		// system-cluster-critical
		{
			ObjectMeta: metav1.ObjectMeta{
				Name:        "pod-w-system-priority",
				Namespace:   "kube-system",
				Annotations: map[string]string{"scheduler.alpha.kubernetes.io/critical-pod": ""},
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Name: containerName,
					},
				},
			},
		},
		// pod[8]: Pod with a system priority class name in non-system namespace
		{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "pod-w-system-priority-in-nonsystem-namespace",
				Namespace: "non-system-namespace",
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Name: containerName,
					},
				},
				PriorityClassName: scheduling.SystemClusterCritical,
			},
		},
		// pod[9]: Pod with a priority value that matches the resolved priority
		{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "pod-w-zero-priority-in-nonsystem-namespace",
				Namespace: "non-system-namespace",
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Name: containerName,
					},
				},
				Priority: &zeroPriority,
			},
		},
		// pod[10]: Pod with a priority value that matches the resolved default priority
		{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "pod-w-priority-matching-default-priority",
				Namespace: "non-system-namespace",
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Name: containerName,
					},
				},
				Priority: &defaultClass2.Value,
			},
		},
		// pod[11]: Pod with a priority value that matches the resolved priority
		{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "pod-w-priority-matching-resolved-default-priority",
				Namespace: metav1.NamespaceSystem,
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Name: containerName,
					},
				},
				PriorityClassName: systemClusterCritical.Name,
				Priority:          &systemClusterCritical.Value,
			},
		},
	}
	// Enable PodPriority feature gate.
	defer utilfeaturetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodPriority, true)()
	// Enable ExperimentalCriticalPodAnnotation feature gate.
	defer utilfeaturetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ExperimentalCriticalPodAnnotation, true)()
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
			[]*scheduling.PriorityClass{systemClusterCritical},
			*pods[4],
			scheduling.SystemCriticalPriority,
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
		{
			"mirror pod with system priority class",
			[]*scheduling.PriorityClass{systemClusterCritical},
			*pods[5],
			scheduling.SystemCriticalPriority,
			false,
		},
		{
			"mirror pod with integer priority",
			[]*scheduling.PriorityClass{},
			*pods[6],
			0,
			true,
		},
		{
			"pod with critical pod annotation",
			[]*scheduling.PriorityClass{systemClusterCritical},
			*pods[7],
			scheduling.SystemCriticalPriority,
			false,
		},
		{
			"pod with system critical priority in non-system namespace",
			[]*scheduling.PriorityClass{systemClusterCritical},
			*pods[8],
			scheduling.SystemCriticalPriority,
			true,
		},
		{
			"pod with priority that matches computed priority",
			[]*scheduling.PriorityClass{nondefaultClass1},
			*pods[9],
			0,
			false,
		},
		{
			"pod with priority that matches default priority",
			[]*scheduling.PriorityClass{defaultClass2},
			*pods[10],
			defaultClass2.Value,
			false,
		},
		{
			"pod with priority that matches resolved priority",
			[]*scheduling.PriorityClass{systemClusterCritical},
			*pods[11],
			systemClusterCritical.Value,
			false,
		},
	}

	for _, test := range tests {
		klog.V(4).Infof("starting test %q", test.name)

		ctrl := newPlugin()
		// Add existing priority classes.
		if err := addPriorityClasses(ctrl, test.existingClasses); err != nil {
			t.Errorf("Test %q: unable to add object to informer: %v", test.name, err)
		}

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
			false,
			nil,
		)
		err := ctrl.Admit(attrs)
		klog.Infof("Got %v", err)
		if !test.expectError {
			if err != nil {
				t.Errorf("Test %q: unexpected error received: %v", test.name, err)
			} else if *test.pod.Spec.Priority != test.expectedPriority {
				t.Errorf("Test %q: expected priority is %d, but got %d.", test.name, test.expectedPriority, *test.pod.Spec.Priority)
			}
		}
		if err == nil && test.expectError {
			t.Errorf("Test %q: expected error and no error recevied", test.name)
		}
	}
}
