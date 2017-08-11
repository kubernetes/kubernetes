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

package pod

import (
	"net/url"
	"reflect"
	"testing"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/kubernetes/pkg/api"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/kubernetes/pkg/kubelet/client"

	// install all api groups for testing
	_ "k8s.io/kubernetes/pkg/api/testapi"
)

func TestMatchPod(t *testing.T) {
	testCases := []struct {
		in            *api.Pod
		fieldSelector fields.Selector
		expectMatch   bool
	}{
		{
			in: &api.Pod{
				Spec: api.PodSpec{NodeName: "nodeA"},
			},
			fieldSelector: fields.ParseSelectorOrDie("spec.nodeName=nodeA"),
			expectMatch:   true,
		},
		{
			in: &api.Pod{
				Spec: api.PodSpec{NodeName: "nodeB"},
			},
			fieldSelector: fields.ParseSelectorOrDie("spec.nodeName=nodeA"),
			expectMatch:   false,
		},
		{
			in: &api.Pod{
				Spec: api.PodSpec{RestartPolicy: api.RestartPolicyAlways},
			},
			fieldSelector: fields.ParseSelectorOrDie("spec.restartPolicy=Always"),
			expectMatch:   true,
		},
		{
			in: &api.Pod{
				Spec: api.PodSpec{RestartPolicy: api.RestartPolicyAlways},
			},
			fieldSelector: fields.ParseSelectorOrDie("spec.restartPolicy=Never"),
			expectMatch:   false,
		},
		{
			in: &api.Pod{
				Status: api.PodStatus{Phase: api.PodRunning},
			},
			fieldSelector: fields.ParseSelectorOrDie("status.phase=Running"),
			expectMatch:   true,
		},
		{
			in: &api.Pod{
				Status: api.PodStatus{Phase: api.PodRunning},
			},
			fieldSelector: fields.ParseSelectorOrDie("status.phase=Pending"),
			expectMatch:   false,
		},
		{
			in: &api.Pod{
				Status: api.PodStatus{PodIP: "1.2.3.4"},
			},
			fieldSelector: fields.ParseSelectorOrDie("status.podIP=1.2.3.4"),
			expectMatch:   true,
		},
		{
			in: &api.Pod{
				Status: api.PodStatus{PodIP: "1.2.3.4"},
			},
			fieldSelector: fields.ParseSelectorOrDie("status.podIP=4.3.2.1"),
			expectMatch:   false,
		},
	}
	for _, testCase := range testCases {
		m := MatchPod(labels.Everything(), testCase.fieldSelector)
		result, err := m.Matches(testCase.in)
		if err != nil {
			t.Errorf("Unexpected error %v", err)
		}
		if result != testCase.expectMatch {
			t.Errorf("Result %v, Expected %v, Selector: %v, Pod: %v", result, testCase.expectMatch, testCase.fieldSelector.String(), testCase.in)
		}
	}
}

func getResourceList(cpu, memory string) api.ResourceList {
	res := api.ResourceList{}
	if cpu != "" {
		res[api.ResourceCPU] = resource.MustParse(cpu)
	}
	if memory != "" {
		res[api.ResourceMemory] = resource.MustParse(memory)
	}
	return res
}

func addResource(rName, value string, rl api.ResourceList) api.ResourceList {
	rl[api.ResourceName(rName)] = resource.MustParse(value)
	return rl
}

func getResourceRequirements(requests, limits api.ResourceList) api.ResourceRequirements {
	res := api.ResourceRequirements{}
	res.Requests = requests
	res.Limits = limits
	return res
}

func newContainer(name string, requests api.ResourceList, limits api.ResourceList) api.Container {
	return api.Container{
		Name:      name,
		Resources: getResourceRequirements(requests, limits),
	}
}

func newPod(name string, containers []api.Container) *api.Pod {
	return &api.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: api.PodSpec{
			Containers: containers,
		},
	}
}

func TestGetPodQOS(t *testing.T) {
	testCases := []struct {
		pod      *api.Pod
		expected api.PodQOSClass
	}{
		{
			pod: newPod("guaranteed", []api.Container{
				newContainer("guaranteed", getResourceList("100m", "100Mi"), getResourceList("100m", "100Mi")),
			}),
			expected: api.PodQOSGuaranteed,
		},
		{
			pod: newPod("best-effort", []api.Container{
				newContainer("best-effort", getResourceList("", ""), getResourceList("", "")),
			}),
			expected: api.PodQOSBestEffort,
		},
		{
			pod: newPod("burstable", []api.Container{
				newContainer("burstable", getResourceList("100m", "100Mi"), getResourceList("", "")),
			}),
			expected: api.PodQOSBurstable,
		},
	}
	for id, testCase := range testCases {
		Strategy.PrepareForCreate(genericapirequest.NewContext(), testCase.pod)
		actual := testCase.pod.Status.QOSClass
		if actual != testCase.expected {
			t.Errorf("[%d]: invalid qos pod %s, expected: %s, actual: %s", id, testCase.pod.Name, testCase.expected, actual)
		}
	}
}

func TestCheckGracefulDelete(t *testing.T) {
	defaultGracePeriod := int64(30)
	tcs := []struct {
		in          *api.Pod
		gracePeriod int64
	}{
		{
			in: &api.Pod{
				Spec:   api.PodSpec{NodeName: "something"},
				Status: api.PodStatus{Phase: api.PodPending},
			},

			gracePeriod: defaultGracePeriod,
		},
		{
			in: &api.Pod{
				Spec:   api.PodSpec{NodeName: "something"},
				Status: api.PodStatus{Phase: api.PodFailed},
			},
			gracePeriod: 0,
		},
		{
			in: &api.Pod{
				Spec:   api.PodSpec{},
				Status: api.PodStatus{Phase: api.PodPending},
			},
			gracePeriod: 0,
		},
		{
			in: &api.Pod{
				Spec:   api.PodSpec{},
				Status: api.PodStatus{Phase: api.PodSucceeded},
			},
			gracePeriod: 0,
		},
		{
			in: &api.Pod{
				Spec:   api.PodSpec{},
				Status: api.PodStatus{},
			},
			gracePeriod: 0,
		},
	}
	for _, tc := range tcs {
		out := &metav1.DeleteOptions{GracePeriodSeconds: &defaultGracePeriod}
		Strategy.CheckGracefulDelete(genericapirequest.NewContext(), tc.in, out)
		if out.GracePeriodSeconds == nil {
			t.Errorf("out grace period was nil but supposed to be %v", tc.gracePeriod)
		}
		if *(out.GracePeriodSeconds) != tc.gracePeriod {
			t.Errorf("out grace period was %v but was expected to be %v", *out, tc.gracePeriod)
		}
	}
}

type mockPodGetter struct {
	pod *api.Pod
}

func (g mockPodGetter) Get(genericapirequest.Context, string, *metav1.GetOptions) (runtime.Object, error) {
	return g.pod, nil
}

func TestCheckLogLocation(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	tcs := []struct {
		in          *api.Pod
		opts        *api.PodLogOptions
		expectedErr error
	}{
		{
			in: &api.Pod{
				Spec:   api.PodSpec{},
				Status: api.PodStatus{},
			},
			opts:        &api.PodLogOptions{},
			expectedErr: errors.NewBadRequest("a container name must be specified for pod test"),
		},
		{
			in: &api.Pod{
				Spec: api.PodSpec{
					Containers: []api.Container{
						{Name: "mycontainer"},
					},
				},
				Status: api.PodStatus{},
			},
			opts:        &api.PodLogOptions{},
			expectedErr: nil,
		},
		{
			in: &api.Pod{
				Spec: api.PodSpec{
					Containers: []api.Container{
						{Name: "container1"},
						{Name: "container2"},
					},
				},
				Status: api.PodStatus{},
			},
			opts:        &api.PodLogOptions{},
			expectedErr: errors.NewBadRequest("a container name must be specified for pod test, choose one of: [container1 container2]"),
		},
		{
			in: &api.Pod{
				Spec: api.PodSpec{
					Containers: []api.Container{
						{Name: "container1"},
						{Name: "container2"},
					},
					InitContainers: []api.Container{
						{Name: "initcontainer1"},
					},
				},
				Status: api.PodStatus{},
			},
			opts:        &api.PodLogOptions{},
			expectedErr: errors.NewBadRequest("a container name must be specified for pod test, choose one of: [container1 container2] or one of the init containers: [initcontainer1]"),
		},
		{
			in: &api.Pod{
				Spec: api.PodSpec{
					Containers: []api.Container{
						{Name: "container1"},
						{Name: "container2"},
					},
				},
				Status: api.PodStatus{},
			},
			opts: &api.PodLogOptions{
				Container: "unknown",
			},
			expectedErr: errors.NewBadRequest("container unknown is not valid for pod test"),
		},
		{
			in: &api.Pod{
				Spec: api.PodSpec{
					Containers: []api.Container{
						{Name: "container1"},
						{Name: "container2"},
					},
				},
				Status: api.PodStatus{},
			},
			opts: &api.PodLogOptions{
				Container: "container2",
			},
			expectedErr: nil,
		},
	}
	for _, tc := range tcs {
		getter := &mockPodGetter{tc.in}
		_, _, err := LogLocation(getter, nil, ctx, "test", tc.opts)
		if !reflect.DeepEqual(err, tc.expectedErr) {
			t.Errorf("expected %v, got %v", tc.expectedErr, err)
		}
	}
}

func TestSelectableFieldLabelConversions(t *testing.T) {
	apitesting.TestSelectableFieldLabelConversionsOfKind(t,
		api.Registry.GroupOrDie(api.GroupName).GroupVersion.String(),
		"Pod",
		PodToSelectableFields(&api.Pod{}),
		nil,
	)
}

type mockConnectionInfoGetter struct {
	info *client.ConnectionInfo
}

func (g mockConnectionInfoGetter) GetConnectionInfo(nodeName types.NodeName) (*client.ConnectionInfo, error) {
	return g.info, nil
}

func TestPortForwardLocation(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	tcs := []struct {
		in          *api.Pod
		info        *client.ConnectionInfo
		opts        *api.PodPortForwardOptions
		expectedErr error
		expectedURL *url.URL
	}{
		{
			in: &api.Pod{
				Spec: api.PodSpec{},
			},
			opts:        &api.PodPortForwardOptions{},
			expectedErr: errors.NewBadRequest("pod test does not have a host assigned"),
		},
		{
			in: &api.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: "ns",
					Name:      "pod1",
				},
				Spec: api.PodSpec{
					NodeName: "node1",
				},
			},
			info:        &client.ConnectionInfo{},
			opts:        &api.PodPortForwardOptions{},
			expectedURL: &url.URL{Host: ":", Path: "/portForward/ns/pod1"},
		},
		{
			in: &api.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: "ns",
					Name:      "pod1",
				},
				Spec: api.PodSpec{
					NodeName: "node1",
				},
			},
			info:        &client.ConnectionInfo{},
			opts:        &api.PodPortForwardOptions{Ports: []int32{80}},
			expectedURL: &url.URL{Host: ":", Path: "/portForward/ns/pod1", RawQuery: "port=80"},
		},
	}
	for _, tc := range tcs {
		getter := &mockPodGetter{tc.in}
		connectionGetter := &mockConnectionInfoGetter{tc.info}
		loc, _, err := PortForwardLocation(getter, connectionGetter, ctx, "test", tc.opts)
		if !reflect.DeepEqual(err, tc.expectedErr) {
			t.Errorf("expected %v, got %v", tc.expectedErr, err)
		}
		if !reflect.DeepEqual(loc, tc.expectedURL) {
			t.Errorf("expected %v, got %v", tc.expectedURL, loc)
		}
	}
}
