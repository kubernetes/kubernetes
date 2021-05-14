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
	"context"
	"fmt"
	"net/http"
	"net/url"
	"reflect"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/tools/cache"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/kubernetes/pkg/apis/core"
	api "k8s.io/kubernetes/pkg/apis/core"
	_ "k8s.io/kubernetes/pkg/apis/core/install"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/client"
	utilpointer "k8s.io/utils/pointer"
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
				Spec: api.PodSpec{SchedulerName: "scheduler1"},
			},
			fieldSelector: fields.ParseSelectorOrDie("spec.schedulerName=scheduler1"),
			expectMatch:   true,
		},
		{
			in: &api.Pod{
				Spec: api.PodSpec{SchedulerName: "scheduler1"},
			},
			fieldSelector: fields.ParseSelectorOrDie("spec.schedulerName=scheduler2"),
			expectMatch:   false,
		},
		{
			in: &api.Pod{
				Spec: api.PodSpec{ServiceAccountName: "serviceAccount1"},
			},
			fieldSelector: fields.ParseSelectorOrDie("spec.serviceAccountName=serviceAccount1"),
			expectMatch:   true,
		},
		{
			in: &api.Pod{
				Spec: api.PodSpec{SchedulerName: "serviceAccount1"},
			},
			fieldSelector: fields.ParseSelectorOrDie("spec.serviceAccountName=serviceAccount2"),
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
				Status: api.PodStatus{
					PodIPs: []api.PodIP{
						{IP: "1.2.3.4"},
					},
				},
			},
			fieldSelector: fields.ParseSelectorOrDie("status.podIP=1.2.3.4"),
			expectMatch:   true,
		},
		{
			in: &api.Pod{
				Status: api.PodStatus{
					PodIPs: []api.PodIP{
						{IP: "1.2.3.4"},
					},
				},
			},
			fieldSelector: fields.ParseSelectorOrDie("status.podIP=4.3.2.1"),
			expectMatch:   false,
		},
		{
			in: &api.Pod{
				Status: api.PodStatus{NominatedNodeName: "node1"},
			},
			fieldSelector: fields.ParseSelectorOrDie("status.nominatedNodeName=node1"),
			expectMatch:   true,
		},
		{
			in: &api.Pod{
				Status: api.PodStatus{NominatedNodeName: "node1"},
			},
			fieldSelector: fields.ParseSelectorOrDie("status.nominatedNodeName=node2"),
			expectMatch:   false,
		},
		{
			in: &api.Pod{
				Status: api.PodStatus{
					PodIPs: []api.PodIP{
						{IP: "2001:db8::"},
					},
				},
			},
			fieldSelector: fields.ParseSelectorOrDie("status.podIP=2001:db8::"),
			expectMatch:   true,
		},
		{
			in: &api.Pod{
				Status: api.PodStatus{
					PodIPs: []api.PodIP{
						{IP: "2001:db8::"},
					},
				},
			},
			fieldSelector: fields.ParseSelectorOrDie("status.podIP=2001:db7::"),
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
		name              string
		pod               *api.Pod
		deleteGracePeriod *int64
		gracePeriod       int64
	}{
		{
			name: "in pending phase with has node name",
			pod: &api.Pod{
				Spec:   api.PodSpec{NodeName: "something"},
				Status: api.PodStatus{Phase: api.PodPending},
			},
			deleteGracePeriod: &defaultGracePeriod,
			gracePeriod:       defaultGracePeriod,
		},
		{
			name: "in failed phase with has node name",
			pod: &api.Pod{
				Spec:   api.PodSpec{NodeName: "something"},
				Status: api.PodStatus{Phase: api.PodFailed},
			},
			deleteGracePeriod: &defaultGracePeriod,
			gracePeriod:       0,
		},
		{
			name: "in failed phase",
			pod: &api.Pod{
				Spec:   api.PodSpec{},
				Status: api.PodStatus{Phase: api.PodPending},
			},
			deleteGracePeriod: &defaultGracePeriod,
			gracePeriod:       0,
		},
		{
			name: "in succeeded phase",
			pod: &api.Pod{
				Spec:   api.PodSpec{},
				Status: api.PodStatus{Phase: api.PodSucceeded},
			},
			deleteGracePeriod: &defaultGracePeriod,
			gracePeriod:       0,
		},
		{
			name: "no phase",
			pod: &api.Pod{
				Spec:   api.PodSpec{},
				Status: api.PodStatus{},
			},
			deleteGracePeriod: &defaultGracePeriod,
			gracePeriod:       0,
		},
		{
			name: "has negative grace period",
			pod: &api.Pod{
				Spec: api.PodSpec{
					NodeName:                      "something",
					TerminationGracePeriodSeconds: utilpointer.Int64(-1),
				},
				Status: api.PodStatus{},
			},
			gracePeriod: 1,
		},
	}
	for _, tc := range tcs {
		t.Run(tc.name, func(t *testing.T) {
			out := &metav1.DeleteOptions{}
			if tc.deleteGracePeriod != nil {
				out.GracePeriodSeconds = utilpointer.Int64(*tc.deleteGracePeriod)
			}
			Strategy.CheckGracefulDelete(genericapirequest.NewContext(), tc.pod, out)
			if out.GracePeriodSeconds == nil {
				t.Errorf("out grace period was nil but supposed to be %v", tc.gracePeriod)
			}
			if *(out.GracePeriodSeconds) != tc.gracePeriod {
				t.Errorf("out grace period was %v but was expected to be %v", *out, tc.gracePeriod)
			}
		})
	}
}

type mockPodGetter struct {
	pod *api.Pod
}

func (g mockPodGetter) Get(context.Context, string, *metav1.GetOptions) (runtime.Object, error) {
	return g.pod, nil
}

func TestCheckLogLocation(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.EphemeralContainers, true)()
	ctx := genericapirequest.NewDefaultContext()
	fakePodName := "test"
	tcs := []struct {
		name              string
		in                *api.Pod
		opts              *api.PodLogOptions
		expectedErr       error
		expectedTransport http.RoundTripper
	}{
		{
			name: "simple",
			in: &api.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: fakePodName},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{Name: "mycontainer"},
					},
					NodeName: "foo",
				},
				Status: api.PodStatus{},
			},
			opts:              &api.PodLogOptions{},
			expectedErr:       nil,
			expectedTransport: fakeSecureRoundTripper,
		},
		{
			name: "insecure",
			in: &api.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: fakePodName},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{Name: "mycontainer"},
					},
					NodeName: "foo",
				},
				Status: api.PodStatus{},
			},
			opts: &api.PodLogOptions{
				InsecureSkipTLSVerifyBackend: true,
			},
			expectedErr:       nil,
			expectedTransport: fakeInsecureRoundTripper,
		},
		{
			name: "missing container",
			in: &api.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: fakePodName},
				Spec:       api.PodSpec{},
				Status:     api.PodStatus{},
			},
			opts:              &api.PodLogOptions{},
			expectedErr:       errors.NewBadRequest("a container name must be specified for pod test"),
			expectedTransport: nil,
		},
		{
			name: "choice of two containers",
			in: &api.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: fakePodName},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{Name: "container1"},
						{Name: "container2"},
					},
				},
				Status: api.PodStatus{},
			},
			opts:              &api.PodLogOptions{},
			expectedErr:       errors.NewBadRequest("a container name must be specified for pod test, choose one of: [container1 container2]"),
			expectedTransport: nil,
		},
		{
			name: "initcontainers",
			in: &api.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: fakePodName},
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
			opts:              &api.PodLogOptions{},
			expectedErr:       errors.NewBadRequest("a container name must be specified for pod test, choose one of: [initcontainer1 container1 container2]"),
			expectedTransport: nil,
		},
		{
			in: &api.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: fakePodName},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{Name: "container1"},
						{Name: "container2"},
					},
					InitContainers: []api.Container{
						{Name: "initcontainer1"},
					},
					EphemeralContainers: []api.EphemeralContainer{
						{EphemeralContainerCommon: api.EphemeralContainerCommon{Name: "debugger"}},
					},
				},
				Status: api.PodStatus{},
			},
			opts:              &api.PodLogOptions{},
			expectedErr:       errors.NewBadRequest("a container name must be specified for pod test, choose one of: [initcontainer1 container1 container2 debugger]"),
			expectedTransport: nil,
		},
		{
			name: "bad container",
			in: &api.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: fakePodName},
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
			expectedErr:       errors.NewBadRequest("container unknown is not valid for pod test"),
			expectedTransport: nil,
		},
		{
			name: "good with two containers",
			in: &api.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: fakePodName},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{Name: "container1"},
						{Name: "container2"},
					},
					NodeName: "foo",
				},
				Status: api.PodStatus{},
			},
			opts: &api.PodLogOptions{
				Container: "container2",
			},
			expectedErr:       nil,
			expectedTransport: fakeSecureRoundTripper,
		},
	}
	for _, tc := range tcs {
		t.Run(tc.name, func(t *testing.T) {
			getter := &mockPodGetter{tc.in}
			connectionGetter := &mockConnectionInfoGetter{&client.ConnectionInfo{
				Transport:                      fakeSecureRoundTripper,
				InsecureSkipTLSVerifyTransport: fakeInsecureRoundTripper,
			}}

			_, actualTransport, err := LogLocation(ctx, getter, connectionGetter, fakePodName, tc.opts)
			if !reflect.DeepEqual(err, tc.expectedErr) {
				t.Errorf("expected %q, got %q", tc.expectedErr, err)
			}
			if actualTransport != tc.expectedTransport {
				t.Errorf("expected %q, got %q", tc.expectedTransport, actualTransport)
			}
		})
	}
}

func TestSelectableFieldLabelConversions(t *testing.T) {
	apitesting.TestSelectableFieldLabelConversionsOfKind(t,
		"v1",
		"Pod",
		ToSelectableFields(&api.Pod{}),
		nil,
	)
}

type mockConnectionInfoGetter struct {
	info *client.ConnectionInfo
}

func (g mockConnectionInfoGetter) GetConnectionInfo(ctx context.Context, nodeName types.NodeName) (*client.ConnectionInfo, error) {
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
		loc, _, err := PortForwardLocation(ctx, getter, connectionGetter, "test", tc.opts)
		if !reflect.DeepEqual(err, tc.expectedErr) {
			t.Errorf("expected %v, got %v", tc.expectedErr, err)
		}
		if !reflect.DeepEqual(loc, tc.expectedURL) {
			t.Errorf("expected %v, got %v", tc.expectedURL, loc)
		}
	}
}

func TestGetPodIP(t *testing.T) {
	testCases := []struct {
		name       string
		pod        *api.Pod
		expectedIP string
	}{
		{
			name:       "nil pod",
			pod:        nil,
			expectedIP: "",
		},
		{
			name: "no status object",
			pod: &api.Pod{
				ObjectMeta: metav1.ObjectMeta{Namespace: "ns", Name: "pod1"},
				Spec:       api.PodSpec{},
			},
			expectedIP: "",
		},
		{
			name: "no pod ips",
			pod: &api.Pod{
				ObjectMeta: metav1.ObjectMeta{Namespace: "ns", Name: "pod1"},
				Spec:       api.PodSpec{},
				Status:     api.PodStatus{},
			},
			expectedIP: "",
		},
		{
			name: "empty list",
			pod: &api.Pod{
				ObjectMeta: metav1.ObjectMeta{Namespace: "ns", Name: "pod1"},
				Spec:       api.PodSpec{},
				Status: api.PodStatus{
					PodIPs: []api.PodIP{},
				},
			},
			expectedIP: "",
		},
		{
			name: "1 ip",
			pod: &api.Pod{
				ObjectMeta: metav1.ObjectMeta{Namespace: "ns", Name: "pod1"},
				Spec:       api.PodSpec{},
				Status: api.PodStatus{
					PodIPs: []api.PodIP{
						{IP: "10.0.0.10"},
					},
				},
			},
			expectedIP: "10.0.0.10",
		},
		{
			name: "multiple ips",
			pod: &api.Pod{
				ObjectMeta: metav1.ObjectMeta{Namespace: "ns", Name: "pod1"},
				Spec:       api.PodSpec{},
				Status: api.PodStatus{
					PodIPs: []api.PodIP{
						{IP: "10.0.0.10"},
						{IP: "10.0.0.20"},
					},
				},
			},
			expectedIP: "10.0.0.10",
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			podIP := getPodIP(tc.pod)
			if podIP != tc.expectedIP {
				t.Errorf("expected pod ip:%v does not match actual %v", tc.expectedIP, podIP)
			}
		})
	}
}

type fakeTransport struct {
	val string
}

func (f fakeTransport) RoundTrip(*http.Request) (*http.Response, error) {
	return nil, nil
}

var (
	fakeSecureRoundTripper   = fakeTransport{val: "secure"}
	fakeInsecureRoundTripper = fakeTransport{val: "insecure"}
)

func TestPodIndexFunc(t *testing.T) {
	tcs := []struct {
		name          string
		indexFunc     cache.IndexFunc
		pod           interface{}
		expectedValue string
		expectedErr   error
	}{
		{
			name:      "node name index",
			indexFunc: NodeNameIndexFunc,
			pod: &api.Pod{
				Spec: api.PodSpec{
					NodeName: "test-pod",
				},
			},
			expectedValue: "test-pod",
			expectedErr:   nil,
		},
		{
			name:          "not a pod failed",
			indexFunc:     NodeNameIndexFunc,
			pod:           "not a pod object",
			expectedValue: "test-pod",
			expectedErr:   fmt.Errorf("not a pod"),
		},
	}

	for _, tc := range tcs {
		indexValues, err := tc.indexFunc(tc.pod)
		if !reflect.DeepEqual(err, tc.expectedErr) {
			t.Errorf("name %v, expected %v, got %v", tc.name, tc.expectedErr, err)
		}
		if err == nil && len(indexValues) != 1 && indexValues[0] != tc.expectedValue {
			t.Errorf("name %v, expected %v, got %v", tc.name, tc.expectedValue, indexValues)
		}

	}
}
func TestApplySeccompVersionSkew(t *testing.T) {
	const containerName = "container"
	testProfile := "test"

	for _, test := range []struct {
		description string
		pod         *api.Pod
		validation  func(*testing.T, *api.Pod)
	}{
		{
			description: "Security context nil",
			pod:         &api.Pod{},
			validation: func(t *testing.T, pod *api.Pod) {
				require.NotNil(t, pod)
			},
		},
		{
			description: "Security context not nil",
			pod: &api.Pod{
				Spec: api.PodSpec{SecurityContext: &api.PodSecurityContext{}},
			},
			validation: func(t *testing.T, pod *api.Pod) {
				require.NotNil(t, pod)
			},
		},
		{
			description: "Field type unconfined and no annotation present",
			pod: &api.Pod{
				Spec: api.PodSpec{
					SecurityContext: &api.PodSecurityContext{
						SeccompProfile: &api.SeccompProfile{
							Type: api.SeccompProfileTypeUnconfined,
						},
					},
				},
			},
			validation: func(t *testing.T, pod *api.Pod) {
				require.Len(t, pod.Annotations, 1)
				require.Equal(t, v1.SeccompProfileNameUnconfined, pod.Annotations[api.SeccompPodAnnotationKey])
			},
		},
		{
			description: "Field type default and no annotation present",
			pod: &api.Pod{
				Spec: api.PodSpec{
					SecurityContext: &api.PodSecurityContext{
						SeccompProfile: &api.SeccompProfile{
							Type: api.SeccompProfileTypeRuntimeDefault,
						},
					},
				},
			},
			validation: func(t *testing.T, pod *api.Pod) {
				require.Len(t, pod.Annotations, 1)
				require.Equal(t, v1.SeccompProfileRuntimeDefault, pod.Annotations[v1.SeccompPodAnnotationKey])
			},
		},
		{
			description: "Field type localhost and no annotation present",
			pod: &api.Pod{
				Spec: api.PodSpec{
					SecurityContext: &api.PodSecurityContext{
						SeccompProfile: &api.SeccompProfile{
							Type:             api.SeccompProfileTypeLocalhost,
							LocalhostProfile: &testProfile,
						},
					},
				},
			},
			validation: func(t *testing.T, pod *api.Pod) {
				require.Len(t, pod.Annotations, 1)
				require.Equal(t, "localhost/test", pod.Annotations[v1.SeccompPodAnnotationKey])
			},
		},
		{
			description: "Field type localhost but profile is nil",
			pod: &api.Pod{
				Spec: api.PodSpec{
					SecurityContext: &api.PodSecurityContext{
						SeccompProfile: &api.SeccompProfile{
							Type: api.SeccompProfileTypeLocalhost,
						},
					},
				},
			},
			validation: func(t *testing.T, pod *api.Pod) {
				require.Len(t, pod.Annotations, 0)
			},
		},
		{
			description: "Annotation 'unconfined' and no field present",
			pod: &api.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						v1.SeccompPodAnnotationKey: v1.SeccompProfileNameUnconfined,
					},
				},
				Spec: api.PodSpec{},
			},
			validation: func(t *testing.T, pod *api.Pod) {
				require.Equal(t, api.SeccompProfileTypeUnconfined, pod.Spec.SecurityContext.SeccompProfile.Type)
				require.Nil(t, pod.Spec.SecurityContext.SeccompProfile.LocalhostProfile)
			},
		},
		{
			description: "Annotation 'runtime/default' and no field present",
			pod: &api.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						v1.SeccompPodAnnotationKey: v1.SeccompProfileRuntimeDefault,
					},
				},
				Spec: api.PodSpec{SecurityContext: &api.PodSecurityContext{}},
			},
			validation: func(t *testing.T, pod *api.Pod) {
				require.Equal(t, api.SeccompProfileTypeRuntimeDefault, pod.Spec.SecurityContext.SeccompProfile.Type)
				require.Nil(t, pod.Spec.SecurityContext.SeccompProfile.LocalhostProfile)
			},
		},
		{
			description: "Annotation 'docker/default' and no field present",
			pod: &api.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						v1.SeccompPodAnnotationKey: v1.DeprecatedSeccompProfileDockerDefault,
					},
				},
				Spec: api.PodSpec{SecurityContext: &api.PodSecurityContext{}},
			},
			validation: func(t *testing.T, pod *api.Pod) {
				require.Equal(t, api.SeccompProfileTypeRuntimeDefault, pod.Spec.SecurityContext.SeccompProfile.Type)
				require.Nil(t, pod.Spec.SecurityContext.SeccompProfile.LocalhostProfile)
			},
		},
		{
			description: "Annotation 'localhost/test' and no field present",
			pod: &api.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						v1.SeccompPodAnnotationKey: v1.SeccompLocalhostProfileNamePrefix + testProfile,
					},
				},
				Spec: api.PodSpec{SecurityContext: &api.PodSecurityContext{}},
			},
			validation: func(t *testing.T, pod *api.Pod) {
				require.Equal(t, api.SeccompProfileTypeLocalhost, pod.Spec.SecurityContext.SeccompProfile.Type)
				require.Equal(t, testProfile, *pod.Spec.SecurityContext.SeccompProfile.LocalhostProfile)
			},
		},
		{
			description: "Annotation 'localhost/' has zero length",
			pod: &api.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						v1.SeccompPodAnnotationKey: v1.SeccompLocalhostProfileNamePrefix,
					},
				},
				Spec: api.PodSpec{SecurityContext: &api.PodSecurityContext{}},
			},
			validation: func(t *testing.T, pod *api.Pod) {
				require.Nil(t, pod.Spec.SecurityContext.SeccompProfile)
			},
		},
		{
			description: "Security context nil (container)",
			pod: &api.Pod{
				Spec: api.PodSpec{
					Containers: []api.Container{{}},
				},
			},
			validation: func(t *testing.T, pod *api.Pod) {
				require.NotNil(t, pod)
			},
		},
		{
			description: "Security context not nil (container)",
			pod: &api.Pod{
				Spec: api.PodSpec{
					Containers: []api.Container{{
						SecurityContext: &api.SecurityContext{},
					}},
				},
			},
			validation: func(t *testing.T, pod *api.Pod) {
				require.NotNil(t, pod)
			},
		},
		{
			description: "Field type unconfined and no annotation present (container)",
			pod: &api.Pod{
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Name: containerName,
							SecurityContext: &api.SecurityContext{
								SeccompProfile: &api.SeccompProfile{
									Type: api.SeccompProfileTypeUnconfined,
								},
							},
						},
					},
				},
			},
			validation: func(t *testing.T, pod *api.Pod) {
				require.Len(t, pod.Annotations, 1)
				require.Equal(t, v1.SeccompProfileNameUnconfined, pod.Annotations[v1.SeccompContainerAnnotationKeyPrefix+containerName])
			},
		},
		{
			description: "Field type runtime/default and no annotation present (container)",
			pod: &api.Pod{
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Name: containerName,
							SecurityContext: &api.SecurityContext{
								SeccompProfile: &api.SeccompProfile{
									Type: api.SeccompProfileTypeRuntimeDefault,
								},
							},
						},
					},
				},
			},
			validation: func(t *testing.T, pod *api.Pod) {
				require.Len(t, pod.Annotations, 1)
				require.Equal(t, v1.SeccompProfileRuntimeDefault, pod.Annotations[v1.SeccompContainerAnnotationKeyPrefix+containerName])
			},
		},
		{
			description: "Field type localhost and no annotation present (container)",
			pod: &api.Pod{
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Name: containerName,
							SecurityContext: &api.SecurityContext{
								SeccompProfile: &api.SeccompProfile{
									Type:             api.SeccompProfileTypeLocalhost,
									LocalhostProfile: &testProfile,
								},
							},
						},
					},
				},
			},
			validation: func(t *testing.T, pod *api.Pod) {
				require.Len(t, pod.Annotations, 1)
				require.Equal(t, "localhost/test", pod.Annotations[v1.SeccompContainerAnnotationKeyPrefix+containerName])
			},
		},
		{
			description: "Multiple containers with fields (container)",
			pod: &api.Pod{
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Name: containerName + "1",
							SecurityContext: &api.SecurityContext{
								SeccompProfile: &api.SeccompProfile{
									Type: api.SeccompProfileTypeUnconfined,
								},
							},
						},
						{
							Name: containerName + "2",
						},
						{
							Name: containerName + "3",
							SecurityContext: &api.SecurityContext{
								SeccompProfile: &api.SeccompProfile{
									Type: api.SeccompProfileTypeRuntimeDefault,
								},
							},
						},
					},
				},
			},
			validation: func(t *testing.T, pod *api.Pod) {
				require.Len(t, pod.Annotations, 2)
				require.Equal(t, v1.SeccompProfileNameUnconfined, pod.Annotations[v1.SeccompContainerAnnotationKeyPrefix+containerName+"1"])
				require.Equal(t, v1.SeccompProfileRuntimeDefault, pod.Annotations[v1.SeccompContainerAnnotationKeyPrefix+containerName+"3"])
			},
		},
		{
			description: "Annotation 'unconfined' and no field present (container)",
			pod: &api.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						v1.SeccompContainerAnnotationKeyPrefix + containerName: v1.SeccompProfileNameUnconfined,
					},
				},
				Spec: api.PodSpec{
					Containers: []api.Container{{
						Name: containerName,
					}},
				},
			},
			validation: func(t *testing.T, pod *api.Pod) {
				require.Equal(t, api.SeccompProfileTypeUnconfined, pod.Spec.Containers[0].SecurityContext.SeccompProfile.Type)
				require.Nil(t, pod.Spec.Containers[0].SecurityContext.SeccompProfile.LocalhostProfile)
			},
		},
		{
			description: "Annotation 'runtime/default' and no field present (container)",
			pod: &api.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						v1.SeccompContainerAnnotationKeyPrefix + containerName: v1.SeccompProfileRuntimeDefault,
					},
				},
				Spec: api.PodSpec{
					Containers: []api.Container{{
						Name:            containerName,
						SecurityContext: &api.SecurityContext{},
					}},
				},
			},
			validation: func(t *testing.T, pod *api.Pod) {
				require.Equal(t, api.SeccompProfileTypeRuntimeDefault, pod.Spec.Containers[0].SecurityContext.SeccompProfile.Type)
				require.Nil(t, pod.Spec.Containers[0].SecurityContext.SeccompProfile.LocalhostProfile)
			},
		},
		{
			description: "Annotation 'docker/default' and no field present (container)",
			pod: &api.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						v1.SeccompContainerAnnotationKeyPrefix + containerName: v1.DeprecatedSeccompProfileDockerDefault,
					},
				},
				Spec: api.PodSpec{
					Containers: []api.Container{{
						Name:            containerName,
						SecurityContext: &api.SecurityContext{},
					}},
				},
			},
			validation: func(t *testing.T, pod *api.Pod) {
				require.Equal(t, api.SeccompProfileTypeRuntimeDefault, pod.Spec.Containers[0].SecurityContext.SeccompProfile.Type)
				require.Nil(t, pod.Spec.Containers[0].SecurityContext.SeccompProfile.LocalhostProfile)
			},
		},
		{
			description: "Multiple containers by annotations (container)",
			pod: &api.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						v1.SeccompContainerAnnotationKeyPrefix + containerName + "1": v1.SeccompLocalhostProfileNamePrefix + testProfile,
						v1.SeccompContainerAnnotationKeyPrefix + containerName + "3": v1.SeccompProfileRuntimeDefault,
					},
				},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{Name: containerName + "1"},
						{Name: containerName + "2"},
						{Name: containerName + "3"},
					},
				},
			},
			validation: func(t *testing.T, pod *api.Pod) {
				require.Equal(t, api.SeccompProfileTypeLocalhost, pod.Spec.Containers[0].SecurityContext.SeccompProfile.Type)
				require.Equal(t, testProfile, *pod.Spec.Containers[0].SecurityContext.SeccompProfile.LocalhostProfile)
				require.Equal(t, api.SeccompProfileTypeRuntimeDefault, pod.Spec.Containers[2].SecurityContext.SeccompProfile.Type)
			},
		},
		{
			description: "Annotation 'localhost/test' and no field present (container)",
			pod: &api.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						v1.SeccompContainerAnnotationKeyPrefix + containerName: v1.SeccompLocalhostProfileNamePrefix + testProfile,
					},
				},
				Spec: api.PodSpec{
					Containers: []api.Container{{
						Name:            containerName,
						SecurityContext: &api.SecurityContext{},
					}},
				},
			},
			validation: func(t *testing.T, pod *api.Pod) {
				require.Equal(t, api.SeccompProfileTypeLocalhost, pod.Spec.Containers[0].SecurityContext.SeccompProfile.Type)
				require.Equal(t, testProfile, *pod.Spec.Containers[0].SecurityContext.SeccompProfile.LocalhostProfile)
			},
		},
	} {
		output := &api.Pod{
			ObjectMeta: metav1.ObjectMeta{Annotations: map[string]string{}},
		}
		for i, ctr := range test.pod.Spec.Containers {
			output.Spec.Containers = append(output.Spec.Containers, api.Container{})
			if ctr.SecurityContext != nil && ctr.SecurityContext.SeccompProfile != nil {
				output.Spec.Containers[i].SecurityContext = &api.SecurityContext{
					SeccompProfile: &api.SeccompProfile{
						Type:             api.SeccompProfileType(ctr.SecurityContext.SeccompProfile.Type),
						LocalhostProfile: ctr.SecurityContext.SeccompProfile.LocalhostProfile,
					},
				}
			}
		}
		applySeccompVersionSkew(test.pod)
		test.validation(t, test.pod)
	}
}

func testUpdatePod(t *testing.T, oldPod *api.Pod, labelValue string) *api.Pod {
	updatedPod := oldPod.DeepCopy()
	updatedPod.Labels = map[string]string{"XYZ": labelValue}
	expectedPod := updatedPod.DeepCopy()
	Strategy.PrepareForUpdate(context.Background(), updatedPod, oldPod)
	require.Equal(t, expectedPod.Spec, updatedPod.Spec, "updated pod spec")
	errs := Strategy.Validate(context.Background(), updatedPod)
	require.Empty(t, errs, "errors from validation")
	return updatedPod
}

func createPodWithGenericEphemeralVolume() *api.Pod {
	return &api.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "ns",
			Name:      "pod",
		},
		Spec: api.PodSpec{
			RestartPolicy: api.RestartPolicyAlways,
			DNSPolicy:     api.DNSClusterFirst,
			Containers: []api.Container{{
				Name:                     "foo",
				Image:                    "example",
				TerminationMessagePolicy: api.TerminationMessageReadFile,
				ImagePullPolicy:          api.PullAlways,
			}},
			Volumes: []api.Volume{
				{
					Name: "ephemeral",
					VolumeSource: api.VolumeSource{
						Ephemeral: &api.EphemeralVolumeSource{
							VolumeClaimTemplate: &api.PersistentVolumeClaimTemplate{
								Spec: api.PersistentVolumeClaimSpec{
									AccessModes: []api.PersistentVolumeAccessMode{
										api.ReadWriteOnce,
									},
									Resources: api.ResourceRequirements{
										Requests: api.ResourceList{
											api.ResourceStorage: resource.MustParse("1Gi"),
										},
									},
								},
							},
						},
					},
				},
			},
		},
	}
}

func newPodWithHugePageValue(resourceName api.ResourceName, value resource.Quantity) *api.Pod {
	return &api.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Namespace:       "default",
			Name:            "foo",
			ResourceVersion: "1",
		},
		Spec: api.PodSpec{
			RestartPolicy: api.RestartPolicyAlways,
			DNSPolicy:     api.DNSDefault,
			Containers: []api.Container{{
				Name:                     "foo",
				Image:                    "image",
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePolicy: "File",
				Resources: api.ResourceRequirements{
					Requests: api.ResourceList{
						api.ResourceCPU: resource.MustParse("10"),
						resourceName:    value,
					},
					Limits: api.ResourceList{
						api.ResourceCPU: resource.MustParse("10"),
						resourceName:    value,
					},
				}},
			},
		},
	}
}

func TestPodStrategyValidate(t *testing.T) {
	const containerName = "container"
	errTest := []struct {
		name string
		pod  *api.Pod
	}{
		{
			name: "a new pod setting container with indivisible hugepages values",
			pod:  newPodWithHugePageValue(api.ResourceHugePagesPrefix+"1Mi", resource.MustParse("1.1Mi")),
		},
		{
			name: "a new pod setting init-container with indivisible hugepages values",
			pod: &api.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: "default",
					Name:      "foo",
				},
				Spec: api.PodSpec{
					RestartPolicy: api.RestartPolicyAlways,
					DNSPolicy:     api.DNSDefault,
					InitContainers: []api.Container{{
						Name:                     containerName,
						Image:                    "image",
						ImagePullPolicy:          "IfNotPresent",
						TerminationMessagePolicy: "File",
						Resources: api.ResourceRequirements{
							Requests: api.ResourceList{
								api.ResourceName(api.ResourceHugePagesPrefix + "64Ki"): resource.MustParse("127Ki"),
							},
							Limits: api.ResourceList{
								api.ResourceName(api.ResourceHugePagesPrefix + "64Ki"): resource.MustParse("127Ki"),
							},
						}},
					},
				},
			},
		},
		{
			name: "a new pod setting init-container with indivisible hugepages values while container with divisible hugepages values",
			pod: &api.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: "default",
					Name:      "foo",
				},
				Spec: api.PodSpec{
					RestartPolicy: api.RestartPolicyAlways,
					DNSPolicy:     api.DNSDefault,
					InitContainers: []api.Container{{
						Name:                     containerName,
						Image:                    "image",
						ImagePullPolicy:          "IfNotPresent",
						TerminationMessagePolicy: "File",
						Resources: api.ResourceRequirements{
							Requests: api.ResourceList{
								api.ResourceName(api.ResourceHugePagesPrefix + "2Mi"): resource.MustParse("5.1Mi"),
							},
							Limits: api.ResourceList{
								api.ResourceName(api.ResourceHugePagesPrefix + "2Mi"): resource.MustParse("5.1Mi"),
							},
						}},
					},
					Containers: []api.Container{{
						Name:                     containerName,
						Image:                    "image",
						ImagePullPolicy:          "IfNotPresent",
						TerminationMessagePolicy: "File",
						Resources: api.ResourceRequirements{
							Requests: api.ResourceList{
								api.ResourceName(api.ResourceHugePagesPrefix + "1Gi"): resource.MustParse("2Gi"),
							},
							Limits: api.ResourceList{
								api.ResourceName(api.ResourceHugePagesPrefix + "1Gi"): resource.MustParse("2Gi"),
							},
						}},
					},
				},
			},
		},
	}

	for _, tc := range errTest {
		t.Run(tc.name, func(t *testing.T) {
			if errs := Strategy.Validate(genericapirequest.NewContext(), tc.pod); len(errs) == 0 {
				t.Error("expected failure")
			}
		})
	}

	tests := []struct {
		name string
		pod  *api.Pod
	}{
		{
			name: "a new pod setting container with divisible hugepages values",
			pod: &api.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: "default",
					Name:      "foo",
				},
				Spec: api.PodSpec{
					RestartPolicy: api.RestartPolicyAlways,
					DNSPolicy:     api.DNSDefault,
					Containers: []api.Container{{
						Name:                     containerName,
						Image:                    "image",
						ImagePullPolicy:          "IfNotPresent",
						TerminationMessagePolicy: "File",
						Resources: api.ResourceRequirements{
							Requests: api.ResourceList{
								api.ResourceName(api.ResourceCPU):                     resource.MustParse("10"),
								api.ResourceName(api.ResourceHugePagesPrefix + "1Mi"): resource.MustParse("2Mi"),
							},
							Limits: api.ResourceList{
								api.ResourceName(api.ResourceCPU):                     resource.MustParse("10"),
								api.ResourceName(api.ResourceHugePagesPrefix + "1Mi"): resource.MustParse("2Mi"),
							},
						}},
					},
				},
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if errs := Strategy.Validate(genericapirequest.NewContext(), tc.pod); len(errs) != 0 {
				t.Errorf("unexpected error:%v", errs)
			}
		})
	}
}

func TestPodStrategyValidateUpdate(t *testing.T) {
	test := []struct {
		name   string
		newPod *api.Pod
		oldPod *api.Pod
	}{
		{
			name:   "an existing pod with indivisible hugepages values to a new pod with indivisible hugepages values",
			newPod: newPodWithHugePageValue(api.ResourceHugePagesPrefix+"2Mi", resource.MustParse("2.1Mi")),
			oldPod: newPodWithHugePageValue(api.ResourceHugePagesPrefix+"2Mi", resource.MustParse("2.1Mi")),
		},
	}

	for _, tc := range test {
		t.Run(tc.name, func(t *testing.T) {
			if errs := Strategy.ValidateUpdate(genericapirequest.NewContext(), tc.newPod, tc.oldPod); len(errs) != 0 {
				t.Errorf("unexpected error:%v", errs)
			}
		})
	}
}

func TestDropNonEphemeralContainerUpdates(t *testing.T) {
	tests := []struct {
		name                    string
		oldPod, newPod, wantPod *api.Pod
	}{
		{
			name: "simple ephemeral container append",
			oldPod: &api.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "test-pod",
					Namespace:       "test-ns",
					ResourceVersion: "1",
				},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Name:  "container",
							Image: "image",
						},
					},
				},
			},
			newPod: &api.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "test-pod",
					Namespace:       "test-ns",
					ResourceVersion: "1",
				},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Name:  "container",
							Image: "image",
						},
					},
					EphemeralContainers: []api.EphemeralContainer{
						{
							EphemeralContainerCommon: api.EphemeralContainerCommon{
								Name:  "container",
								Image: "image",
							},
						},
					},
				},
			},
			wantPod: &api.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "test-pod",
					Namespace:       "test-ns",
					ResourceVersion: "1",
				},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Name:  "container",
							Image: "image",
						},
					},
					EphemeralContainers: []api.EphemeralContainer{
						{
							EphemeralContainerCommon: api.EphemeralContainerCommon{
								Name:  "container",
								Image: "image",
							},
						},
					},
				},
			},
		},
		{
			name: "whoops wrong pod",
			oldPod: &api.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "test-pod",
					Namespace:       "test-ns",
					ResourceVersion: "1",
					UID:             "blue",
				},
			},
			newPod: &api.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "new-pod",
					Namespace:       "new-ns",
					ResourceVersion: "1",
					UID:             "green",
				},
			},
			wantPod: &api.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "new-pod",
					Namespace:       "new-ns",
					ResourceVersion: "1",
					UID:             "green",
				},
			},
		},
		{
			name: "resource conflict during update",
			oldPod: &api.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "test-pod",
					Namespace:       "test-ns",
					ResourceVersion: "2",
					UID:             "blue",
				},
			},
			newPod: &api.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "test-pod",
					Namespace:       "test-ns",
					ResourceVersion: "1",
					UID:             "blue",
				},
			},
			wantPod: &api.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "test-pod",
					Namespace:       "test-ns",
					ResourceVersion: "1",
					UID:             "blue",
				},
			},
		},
		{
			name: "drop non-ephemeral container changes",
			oldPod: &api.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "test-pod",
					Namespace:       "test-ns",
					ResourceVersion: "1",
					Annotations:     map[string]string{"foo": "bar"},
				},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Name:  "container",
							Image: "image",
						},
					},
				},
			},
			newPod: &api.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "test-pod",
					Namespace:       "test-ns",
					ResourceVersion: "1",
					Annotations:     map[string]string{"foo": "bar", "whiz": "pop"},
					ClusterName:     "milo",
				},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Name:  "container",
							Image: "newimage",
						},
					},
					EphemeralContainers: []api.EphemeralContainer{
						{
							EphemeralContainerCommon: api.EphemeralContainerCommon{
								Name:  "container1",
								Image: "image",
							},
						},
						{
							EphemeralContainerCommon: api.EphemeralContainerCommon{
								Name:  "container2",
								Image: "image",
							},
						},
					},
				},
				Status: api.PodStatus{
					Message: "hi.",
				},
			},
			wantPod: &api.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "test-pod",
					Namespace:       "test-ns",
					ResourceVersion: "1",
					Annotations:     map[string]string{"foo": "bar"},
				},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Name:  "container",
							Image: "image",
						},
					},
					EphemeralContainers: []api.EphemeralContainer{
						{
							EphemeralContainerCommon: api.EphemeralContainerCommon{
								Name:  "container1",
								Image: "image",
							},
						},
						{
							EphemeralContainerCommon: api.EphemeralContainerCommon{
								Name:  "container2",
								Image: "image",
							},
						},
					},
				},
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			gotPod := dropNonEphemeralContainerUpdates(tc.newPod, tc.oldPod)
			if diff := cmp.Diff(tc.wantPod, gotPod); diff != "" {
				t.Errorf("unexpected diff when dropping fields (-want, +got):\n%s", diff)
			}
		})
	}
}

func Test_podStrategy_CheckGracefulDelete(t *testing.T) {
	int64p := func(i int64) *int64 { return &i }
	tests := []struct {
		name      string
		obj       *core.Pod
		options   func(*metav1.DeleteOptions)
		wantGrace int64
	}{
		{
			name:      "nil grace period (impossible) is zero",
			obj:       &core.Pod{Spec: core.PodSpec{NodeName: "test", TerminationGracePeriodSeconds: nil}},
			wantGrace: 0,
		},
		{
			// Grace period zero has special semantics - it means "delete without waiting for the kubelet".
			// Since this means the same pod can be created with the same name on a different node, honoring
			// this value as zero on a pod can result in safety guarantee violations when used incorrectly.
			// The kubelet already ignores 0 grace period, giving containers an opportunity to shutdown before
			// force termination, and the apiserver should not allow a zero value default.
			name:      "nil grace period (impossible) is overridden",
			obj:       &core.Pod{Spec: core.PodSpec{NodeName: "test", TerminationGracePeriodSeconds: nil}},
			options:   func(opt *metav1.DeleteOptions) { opt.GracePeriodSeconds = int64p(1) },
			wantGrace: 1,
		},
		{
			name:      "zero grace period becomes one, ensuring kubelet shutdown is not bypassed",
			obj:       &core.Pod{Spec: core.PodSpec{NodeName: "test", TerminationGracePeriodSeconds: int64p(0)}},
			wantGrace: 1,
		},
		{
			name:      "negative grace period becomes one, ensuring kubelet shutdown is not bypassed",
			obj:       &core.Pod{Spec: core.PodSpec{NodeName: "test", TerminationGracePeriodSeconds: int64p(-1)}},
			wantGrace: 1,
		},
		{
			name:      "non-zero grace period default is used",
			obj:       &core.Pod{Spec: core.PodSpec{NodeName: "test", TerminationGracePeriodSeconds: int64p(1)}},
			wantGrace: 1,
		},
		{
			name:      "large non-zero grace period default is used",
			obj:       &core.Pod{Spec: core.PodSpec{NodeName: "test", TerminationGracePeriodSeconds: int64p(30)}},
			wantGrace: 30,
		},

		{
			name: "terminal (succeeded) pod is deleted immediately",
			obj: &core.Pod{
				Spec:   core.PodSpec{NodeName: "test", TerminationGracePeriodSeconds: int64p(1)},
				Status: core.PodStatus{Phase: core.PodSucceeded},
			},
			wantGrace: 0,
		},
		{
			name: "terminal (failed) pod is deleted immediately",
			obj: &core.Pod{
				Spec:   core.PodSpec{NodeName: "test", TerminationGracePeriodSeconds: int64p(1)},
				Status: core.PodStatus{Phase: core.PodFailed},
			},
			wantGrace: 0,
		},
		{
			name:      "unscheduled pod is deleted immediately",
			obj:       &core.Pod{Spec: core.PodSpec{TerminationGracePeriodSeconds: int64p(1)}},
			wantGrace: 0,
		},

		{
			name: "non-terminal (unknown) pod is not deleted immediately",
			obj: &core.Pod{
				Spec:   core.PodSpec{NodeName: "test", TerminationGracePeriodSeconds: int64p(1)},
				Status: core.PodStatus{Phase: core.PodUnknown},
			},
			wantGrace: 1,
		},
		{
			name: "non-terminal (running) pod is not deleted immediately",
			obj: &core.Pod{
				Spec:   core.PodSpec{NodeName: "test", TerminationGracePeriodSeconds: int64p(1)},
				Status: core.PodStatus{Phase: core.PodRunning},
			},
			wantGrace: 1,
		},
		{
			name: "non-terminal (pending) pod is not deleted immediately",
			obj: &core.Pod{
				Spec:   core.PodSpec{NodeName: "test", TerminationGracePeriodSeconds: int64p(1)},
				Status: core.PodStatus{Phase: core.PodPending},
			},
			wantGrace: 1,
		},

		{
			name: "delete options overrides default grace period (shorter)",
			obj: &core.Pod{
				Spec:   core.PodSpec{NodeName: "test", TerminationGracePeriodSeconds: int64p(5)},
				Status: core.PodStatus{Phase: core.PodPending},
			},
			options:   func(opt *metav1.DeleteOptions) { opt.GracePeriodSeconds = int64p(1) },
			wantGrace: 1,
		},
		{
			name: "delete options overrides default grace period (longer)",
			obj: &core.Pod{
				Spec:   core.PodSpec{NodeName: "test", TerminationGracePeriodSeconds: int64p(5)},
				Status: core.PodStatus{Phase: core.PodPending},
			},
			options:   func(opt *metav1.DeleteOptions) { opt.GracePeriodSeconds = int64p(10) },
			wantGrace: 10,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			opts := &metav1.DeleteOptions{}
			if tt.options != nil {
				tt.options(opts)
			}
			if got := Strategy.CheckGracefulDelete(context.Background(), tt.obj, opts); !got {
				t.Errorf("all pods should be gracefully terminated")
			}
			if opts.GracePeriodSeconds == nil {
				t.Fatalf("unexpected grace period nil")
			}
			if tt.wantGrace != *opts.GracePeriodSeconds {
				t.Errorf("unexpected grace period: expected %d, got %d", tt.wantGrace, *opts.GracePeriodSeconds)
			}
		})
	}
}
