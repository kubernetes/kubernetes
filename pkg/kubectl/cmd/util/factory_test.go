/*
Copyright 2014 The Kubernetes Authors.

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
	"bytes"
	"fmt"
	"sort"
	"strings"
	"testing"
	"time"

	"k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/util/flag"
	manualfake "k8s.io/client-go/rest/fake"
	testcore "k8s.io/client-go/testing"
	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/fake"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/kubectl"
	"k8s.io/kubernetes/pkg/kubectl/resource"
)

func TestNewFactoryDefaultFlagBindings(t *testing.T) {
	factory := NewFactory(nil)

	if !factory.FlagSet().HasFlags() {
		t.Errorf("Expected flags, but didn't get any")
	}
}

func TestNewFactoryNoFlagBindings(t *testing.T) {
	clientConfig := clientcmd.NewDefaultClientConfig(*clientcmdapi.NewConfig(), &clientcmd.ConfigOverrides{})
	factory := NewFactory(clientConfig)

	if factory.FlagSet().HasFlags() {
		t.Errorf("Expected zero flags, but got %v", factory.FlagSet())
	}
}

func TestPortsForObject(t *testing.T) {
	f := NewFactory(nil)

	pod := &api.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "baz", Namespace: "test", ResourceVersion: "12"},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Ports: []api.ContainerPort{
						{
							ContainerPort: 101,
						},
					},
				},
			},
		},
	}

	expected := sets.NewString("101")
	ports, err := f.PortsForObject(pod)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	got := sets.NewString(ports...)

	if !expected.Equal(got) {
		t.Fatalf("Ports mismatch! Expected %v, got %v", expected, got)
	}
}

func TestProtocolsForObject(t *testing.T) {
	f := NewFactory(nil)

	pod := &api.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "baz", Namespace: "test", ResourceVersion: "12"},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Ports: []api.ContainerPort{
						{
							ContainerPort: 101,
							Protocol:      api.ProtocolTCP,
						},
						{
							ContainerPort: 102,
							Protocol:      api.ProtocolUDP,
						},
					},
				},
			},
		},
	}

	expected := sets.NewString("101/TCP", "102/UDP")
	protocolsMap, err := f.ProtocolsForObject(pod)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	protocolsString := kubectl.MakeProtocols(protocolsMap)
	protocolsStrings := strings.Split(protocolsString, ",")
	got := sets.NewString(protocolsStrings...)

	if !expected.Equal(got) {
		t.Fatalf("Protocols mismatch! Expected %v, got %v", expected, got)
	}
}

func TestLabelsForObject(t *testing.T) {
	f := NewFactory(nil)

	tests := []struct {
		name     string
		object   runtime.Object
		expected string
		err      error
	}{
		{
			name: "successful re-use of labels",
			object: &api.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "baz", Namespace: "test", Labels: map[string]string{"svc": "test"}},
				TypeMeta:   metav1.TypeMeta{Kind: "Service", APIVersion: "v1"},
			},
			expected: "svc=test",
			err:      nil,
		},
		{
			name: "empty labels",
			object: &api.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "test", Labels: map[string]string{}},
				TypeMeta:   metav1.TypeMeta{Kind: "Service", APIVersion: "v1"},
			},
			expected: "",
			err:      nil,
		},
		{
			name: "nil labels",
			object: &api.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "zen", Namespace: "test", Labels: nil},
				TypeMeta:   metav1.TypeMeta{Kind: "Service", APIVersion: "v1"},
			},
			expected: "",
			err:      nil,
		},
	}

	for _, test := range tests {
		gotLabels, err := f.LabelsForObject(test.object)
		if err != test.err {
			t.Fatalf("%s: Error mismatch: Expected %v, got %v", test.name, test.err, err)
		}
		got := kubectl.MakeLabels(gotLabels)
		if test.expected != got {
			t.Fatalf("%s: Labels mismatch! Expected %s, got %s", test.name, test.expected, got)
		}

	}
}

func TestCanBeExposed(t *testing.T) {
	factory := NewFactory(nil)
	tests := []struct {
		kind      schema.GroupKind
		expectErr bool
	}{
		{
			kind:      api.Kind("ReplicationController"),
			expectErr: false,
		},
		{
			kind:      api.Kind("Node"),
			expectErr: true,
		},
	}

	for _, test := range tests {
		err := factory.CanBeExposed(test.kind)
		if test.expectErr && err == nil {
			t.Error("unexpected non-error")
		}
		if !test.expectErr && err != nil {
			t.Errorf("unexpected error: %v", err)
		}
	}
}

func TestFlagUnderscoreRenaming(t *testing.T) {
	factory := NewFactory(nil)

	factory.FlagSet().SetNormalizeFunc(flag.WordSepNormalizeFunc)
	factory.FlagSet().Bool("valid_flag", false, "bool value")

	// In case of failure of this test check this PR: spf13/pflag#23
	if factory.FlagSet().Lookup("valid_flag").Name != "valid-flag" {
		t.Fatalf("Expected flag name to be valid-flag, got %s", factory.FlagSet().Lookup("valid_flag").Name)
	}
}

func newPodList(count, isUnready, isUnhealthy int, labels map[string]string) *api.PodList {
	pods := []api.Pod{}
	for i := 0; i < count; i++ {
		newPod := api.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:              fmt.Sprintf("pod-%d", i+1),
				Namespace:         metav1.NamespaceDefault,
				CreationTimestamp: metav1.Date(2016, time.April, 1, 1, 0, i, 0, time.UTC),
				Labels:            labels,
			},
			Status: api.PodStatus{
				Conditions: []api.PodCondition{
					{
						Status: api.ConditionTrue,
						Type:   api.PodReady,
					},
				},
			},
		}
		pods = append(pods, newPod)
	}
	if isUnready > -1 && isUnready < count {
		pods[isUnready].Status.Conditions[0].Status = api.ConditionFalse
	}
	if isUnhealthy > -1 && isUnhealthy < count {
		pods[isUnhealthy].Status.ContainerStatuses = []api.ContainerStatus{{RestartCount: 5}}
	}
	return &api.PodList{
		Items: pods,
	}
}

func TestGetFirstPod(t *testing.T) {
	labelSet := map[string]string{"test": "selector"}
	tests := []struct {
		name string

		podList  *api.PodList
		watching []watch.Event
		sortBy   func([]*v1.Pod) sort.Interface

		expected    *api.Pod
		expectedNum int
		expectedErr bool
	}{
		{
			name:    "kubectl logs - two ready pods",
			podList: newPodList(2, -1, -1, labelSet),
			sortBy:  func(pods []*v1.Pod) sort.Interface { return controller.ByLogging(pods) },
			expected: &api.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "pod-1",
					Namespace:         metav1.NamespaceDefault,
					CreationTimestamp: metav1.Date(2016, time.April, 1, 1, 0, 0, 0, time.UTC),
					Labels:            map[string]string{"test": "selector"},
				},
				Status: api.PodStatus{
					Conditions: []api.PodCondition{
						{
							Status: api.ConditionTrue,
							Type:   api.PodReady,
						},
					},
				},
			},
			expectedNum: 2,
		},
		{
			name:    "kubectl logs - one unhealthy, one healthy",
			podList: newPodList(2, -1, 1, labelSet),
			sortBy:  func(pods []*v1.Pod) sort.Interface { return controller.ByLogging(pods) },
			expected: &api.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "pod-2",
					Namespace:         metav1.NamespaceDefault,
					CreationTimestamp: metav1.Date(2016, time.April, 1, 1, 0, 1, 0, time.UTC),
					Labels:            map[string]string{"test": "selector"},
				},
				Status: api.PodStatus{
					Conditions: []api.PodCondition{
						{
							Status: api.ConditionTrue,
							Type:   api.PodReady,
						},
					},
					ContainerStatuses: []api.ContainerStatus{{RestartCount: 5}},
				},
			},
			expectedNum: 2,
		},
		{
			name:    "kubectl attach - two ready pods",
			podList: newPodList(2, -1, -1, labelSet),
			sortBy:  func(pods []*v1.Pod) sort.Interface { return sort.Reverse(controller.ActivePods(pods)) },
			expected: &api.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "pod-1",
					Namespace:         metav1.NamespaceDefault,
					CreationTimestamp: metav1.Date(2016, time.April, 1, 1, 0, 0, 0, time.UTC),
					Labels:            map[string]string{"test": "selector"},
				},
				Status: api.PodStatus{
					Conditions: []api.PodCondition{
						{
							Status: api.ConditionTrue,
							Type:   api.PodReady,
						},
					},
				},
			},
			expectedNum: 2,
		},
		{
			name:    "kubectl attach - wait for ready pod",
			podList: newPodList(1, 1, -1, labelSet),
			watching: []watch.Event{
				{
					Type: watch.Modified,
					Object: &api.Pod{
						ObjectMeta: metav1.ObjectMeta{
							Name:              "pod-1",
							Namespace:         metav1.NamespaceDefault,
							CreationTimestamp: metav1.Date(2016, time.April, 1, 1, 0, 0, 0, time.UTC),
							Labels:            map[string]string{"test": "selector"},
						},
						Status: api.PodStatus{
							Conditions: []api.PodCondition{
								{
									Status: api.ConditionTrue,
									Type:   api.PodReady,
								},
							},
						},
					},
				},
			},
			sortBy: func(pods []*v1.Pod) sort.Interface { return sort.Reverse(controller.ActivePods(pods)) },
			expected: &api.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "pod-1",
					Namespace:         metav1.NamespaceDefault,
					CreationTimestamp: metav1.Date(2016, time.April, 1, 1, 0, 0, 0, time.UTC),
					Labels:            map[string]string{"test": "selector"},
				},
				Status: api.PodStatus{
					Conditions: []api.PodCondition{
						{
							Status: api.ConditionTrue,
							Type:   api.PodReady,
						},
					},
				},
			},
			expectedNum: 1,
		},
	}

	for i := range tests {
		test := tests[i]
		fake := fake.NewSimpleClientset(test.podList)
		if len(test.watching) > 0 {
			watcher := watch.NewFake()
			for _, event := range test.watching {
				switch event.Type {
				case watch.Added:
					go watcher.Add(event.Object)
				case watch.Modified:
					go watcher.Modify(event.Object)
				}
			}
			fake.PrependWatchReactor("pods", testcore.DefaultWatchReactor(watcher, nil))
		}
		selector := labels.Set(labelSet).AsSelector()

		pod, numPods, err := GetFirstPod(fake.Core(), metav1.NamespaceDefault, selector.String(), 1*time.Minute, test.sortBy)
		pod.Spec.SecurityContext = nil
		if !test.expectedErr && err != nil {
			t.Errorf("%s: unexpected error: %v", test.name, err)
			continue
		}
		if test.expectedErr && err == nil {
			t.Errorf("%s: expected an error", test.name)
			continue
		}
		if test.expectedNum != numPods {
			t.Errorf("%s: expected %d pods, got %d", test.name, test.expectedNum, numPods)
			continue
		}
		if !apiequality.Semantic.DeepEqual(test.expected, pod) {
			t.Errorf("%s:\nexpected pod:\n%#v\ngot:\n%#v\n\n", test.name, test.expected, pod)
		}
	}
}

func TestPrintObjectSpecificMessage(t *testing.T) {
	f := NewFactory(nil)
	tests := []struct {
		obj          runtime.Object
		expectOutput bool
	}{
		{
			obj:          &api.Service{},
			expectOutput: false,
		},
		{
			obj:          &api.Pod{},
			expectOutput: false,
		},
		{
			obj:          &api.Service{Spec: api.ServiceSpec{Type: api.ServiceTypeLoadBalancer}},
			expectOutput: false,
		},
		{
			obj:          &api.Service{Spec: api.ServiceSpec{Type: api.ServiceTypeNodePort}},
			expectOutput: true,
		},
	}
	for _, test := range tests {
		buff := &bytes.Buffer{}
		f.PrintObjectSpecificMessage(test.obj, buff)
		if test.expectOutput && buff.Len() == 0 {
			t.Errorf("Expected output, saw none for %v", test.obj)
		}
		if !test.expectOutput && buff.Len() > 0 {
			t.Errorf("Expected no output, saw %s for %v", buff.String(), test.obj)
		}
	}
}

func TestMakePortsString(t *testing.T) {
	tests := []struct {
		ports          []api.ServicePort
		useNodePort    bool
		expectedOutput string
	}{
		{ports: nil, expectedOutput: ""},
		{ports: []api.ServicePort{}, expectedOutput: ""},
		{ports: []api.ServicePort{
			{
				Port:     80,
				Protocol: "TCP",
			},
		},
			expectedOutput: "tcp:80",
		},
		{ports: []api.ServicePort{
			{
				Port:     80,
				Protocol: "TCP",
			},
			{
				Port:     8080,
				Protocol: "UDP",
			},
			{
				Port:     9000,
				Protocol: "TCP",
			},
		},
			expectedOutput: "tcp:80,udp:8080,tcp:9000",
		},
		{ports: []api.ServicePort{
			{
				Port:     80,
				NodePort: 9090,
				Protocol: "TCP",
			},
			{
				Port:     8080,
				NodePort: 80,
				Protocol: "UDP",
			},
		},
			useNodePort:    true,
			expectedOutput: "tcp:9090,udp:80",
		},
	}
	for _, test := range tests {
		output := makePortsString(test.ports, test.useNodePort)
		if output != test.expectedOutput {
			t.Errorf("expected: %s, saw: %s.", test.expectedOutput, output)
		}
	}
}

func fakeClient() resource.ClientMapper {
	return resource.ClientMapperFunc(func(*meta.RESTMapping) (resource.RESTClient, error) {
		return &manualfake.RESTClient{}, nil
	})
}

func TestDiscoveryReplaceAliases(t *testing.T) {
	tests := []struct {
		name     string
		arg      string
		expected string
	}{
		{
			name:     "no-replacement",
			arg:      "service",
			expected: "service",
		},
		{
			name:     "all-replacement",
			arg:      "all",
			expected: "pods,replicationcontrollers,services,statefulsets.apps,horizontalpodautoscalers.autoscaling,jobs.batch,cronjobs.batch,daemonsets.extensions,deployments.extensions,replicasets.extensions",
		},
		{
			name:     "alias-in-comma-separated-arg",
			arg:      "all,secrets",
			expected: "pods,replicationcontrollers,services,statefulsets.apps,horizontalpodautoscalers.autoscaling,jobs.batch,cronjobs.batch,daemonsets.extensions,deployments.extensions,replicasets.extensions,secrets",
		},
	}

	ds := &fakeDiscoveryClient{}
	mapper, err := NewShortcutExpander(testapi.Default.RESTMapper(), ds)
	if err != nil {
		t.Fatalf("Unable to create shortcut expander, err = %s", err.Error())
	}
	b := resource.NewBuilder(mapper, resource.LegacyCategoryExpander, legacyscheme.Scheme, fakeClient(), testapi.Default.Codec())

	for _, test := range tests {
		replaced := b.ReplaceAliases(test.arg)
		if replaced != test.expected {
			t.Errorf("%s: unexpected argument: expected %s, got %s", test.name, test.expected, replaced)
		}
	}
}
