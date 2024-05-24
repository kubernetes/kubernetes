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

package describe

import (
	"bytes"
	"fmt"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/lithammer/dedent"
	"github.com/stretchr/testify/assert"

	appsv1 "k8s.io/api/apps/v1"
	autoscalingv1 "k8s.io/api/autoscaling/v1"
	autoscalingv2 "k8s.io/api/autoscaling/v2"
	batchv1 "k8s.io/api/batch/v1"
	coordinationv1 "k8s.io/api/coordination/v1"
	corev1 "k8s.io/api/core/v1"
	discoveryv1 "k8s.io/api/discovery/v1"
	discoveryv1beta1 "k8s.io/api/discovery/v1beta1"
	networkingv1 "k8s.io/api/networking/v1"
	networkingv1alpha1 "k8s.io/api/networking/v1alpha1"
	networkingv1beta1 "k8s.io/api/networking/v1beta1"
	policyv1 "k8s.io/api/policy/v1"
	policyv1beta1 "k8s.io/api/policy/v1beta1"
	schedulingv1 "k8s.io/api/scheduling/v1"
	storagev1 "k8s.io/api/storage/v1"
	storagev1alpha1 "k8s.io/api/storage/v1alpha1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/utils/ptr"
)

type describeClient struct {
	T         *testing.T
	Namespace string
	Err       error
	kubernetes.Interface
}

func TestDescribePod(t *testing.T) {
	deletionTimestamp := metav1.Time{Time: time.Now().UTC().AddDate(-10, 0, 0)}
	gracePeriod := int64(1234)
	condition1 := corev1.PodConditionType("condition1")
	condition2 := corev1.PodConditionType("condition2")
	fake := fake.NewSimpleClientset(&corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:                       "bar",
			Namespace:                  "foo",
			DeletionTimestamp:          &deletionTimestamp,
			DeletionGracePeriodSeconds: &gracePeriod,
		},
		Spec: corev1.PodSpec{
			ReadinessGates: []corev1.PodReadinessGate{
				{
					ConditionType: condition1,
				},
				{
					ConditionType: condition2,
				},
			},
		},
		Status: corev1.PodStatus{
			Conditions: []corev1.PodCondition{
				{
					Type:   condition1,
					Status: corev1.ConditionTrue,
				},
			},
		},
	})
	c := &describeClient{T: t, Namespace: "foo", Interface: fake}
	d := PodDescriber{c}
	out, err := d.Describe("foo", "bar", DescriberSettings{ShowEvents: true})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if !strings.Contains(out, "bar") || !strings.Contains(out, "Status:") {
		t.Errorf("unexpected out: %s", out)
	}
	if !strings.Contains(out, "Terminating (lasts 10y)") || !strings.Contains(out, "1234s") {
		t.Errorf("unexpected out: %s", out)
	}
}

func TestDescribePodServiceAccount(t *testing.T) {
	fake := fake.NewSimpleClientset(&corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "bar",
			Namespace: "foo",
		},
		Spec: corev1.PodSpec{
			ServiceAccountName: "fooaccount",
		},
	})
	c := &describeClient{T: t, Namespace: "foo", Interface: fake}
	d := PodDescriber{c}
	out, err := d.Describe("foo", "bar", DescriberSettings{ShowEvents: true})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if !strings.Contains(out, "Service Account:") {
		t.Errorf("unexpected out: %s", out)
	}
	if !strings.Contains(out, "fooaccount") {
		t.Errorf("unexpected out: %s", out)
	}
}

func TestDescribePodEphemeralContainers(t *testing.T) {
	fake := fake.NewSimpleClientset(&corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "bar",
			Namespace: "foo",
		},
		Spec: corev1.PodSpec{
			EphemeralContainers: []corev1.EphemeralContainer{
				{
					EphemeralContainerCommon: corev1.EphemeralContainerCommon{
						Name:  "debugger",
						Image: "busybox",
					},
				},
			},
		},
		Status: corev1.PodStatus{
			EphemeralContainerStatuses: []corev1.ContainerStatus{
				{
					Name: "debugger",
					State: corev1.ContainerState{
						Running: &corev1.ContainerStateRunning{
							StartedAt: metav1.NewTime(time.Now()),
						},
					},
					Ready:        false,
					RestartCount: 0,
				},
			},
		},
	})
	c := &describeClient{T: t, Namespace: "foo", Interface: fake}
	d := PodDescriber{c}
	out, err := d.Describe("foo", "bar", DescriberSettings{ShowEvents: true})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if !strings.Contains(out, "debugger:") {
		t.Errorf("unexpected out: %s", out)
	}
	if !strings.Contains(out, "busybox") {
		t.Errorf("unexpected out: %s", out)
	}
}

func TestDescribePodNode(t *testing.T) {
	fake := fake.NewSimpleClientset(&corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "bar",
			Namespace: "foo",
		},
		Spec: corev1.PodSpec{
			NodeName: "all-in-one",
		},
		Status: corev1.PodStatus{
			HostIP:            "127.0.0.1",
			NominatedNodeName: "nodeA",
		},
	})
	c := &describeClient{T: t, Namespace: "foo", Interface: fake}
	d := PodDescriber{c}
	out, err := d.Describe("foo", "bar", DescriberSettings{ShowEvents: true})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if !strings.Contains(out, "all-in-one/127.0.0.1") {
		t.Errorf("unexpected out: %s", out)
	}
	if !strings.Contains(out, "nodeA") {
		t.Errorf("unexpected out: %s", out)
	}
}

func TestDescribePodTolerations(t *testing.T) {
	fake := fake.NewSimpleClientset(&corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "bar",
			Namespace: "foo",
		},
		Spec: corev1.PodSpec{
			Tolerations: []corev1.Toleration{
				{Operator: corev1.TolerationOpExists},
				{Effect: corev1.TaintEffectNoSchedule, Operator: corev1.TolerationOpExists},
				{Key: "key0", Operator: corev1.TolerationOpExists},
				{Key: "key1", Value: "value1"},
				{Key: "key2", Operator: corev1.TolerationOpEqual, Value: "value2", Effect: corev1.TaintEffectNoSchedule},
				{Key: "key3", Value: "value3", Effect: corev1.TaintEffectNoExecute, TolerationSeconds: &[]int64{300}[0]},
				{Key: "key4", Effect: corev1.TaintEffectNoExecute, TolerationSeconds: &[]int64{60}[0]},
			},
		},
	})
	c := &describeClient{T: t, Namespace: "foo", Interface: fake}
	d := PodDescriber{c}
	out, err := d.Describe("foo", "bar", DescriberSettings{})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if !strings.Contains(out, "  op=Exists\n") ||
		!strings.Contains(out, ":NoSchedule op=Exists\n") ||
		!strings.Contains(out, "key0 op=Exists\n") ||
		!strings.Contains(out, "key1=value1\n") ||
		!strings.Contains(out, "key2=value2:NoSchedule\n") ||
		!strings.Contains(out, "key3=value3:NoExecute for 300s\n") ||
		!strings.Contains(out, "key4:NoExecute for 60s\n") ||
		!strings.Contains(out, "Tolerations:") {
		t.Errorf("unexpected out:\n%s", out)
	}
}

func TestDescribeTopologySpreadConstraints(t *testing.T) {
	fake := fake.NewSimpleClientset(&corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "bar",
			Namespace: "foo",
		},
		Spec: corev1.PodSpec{
			TopologySpreadConstraints: []corev1.TopologySpreadConstraint{
				{
					MaxSkew:           3,
					TopologyKey:       "topology.kubernetes.io/test1",
					WhenUnsatisfiable: "DoNotSchedule",
					LabelSelector:     &metav1.LabelSelector{MatchLabels: map[string]string{"key1": "val1", "key2": "val2"}},
				},
				{
					MaxSkew:           1,
					TopologyKey:       "topology.kubernetes.io/test2",
					WhenUnsatisfiable: "ScheduleAnyway",
				},
			},
		},
	})
	c := &describeClient{T: t, Namespace: "foo", Interface: fake}
	d := PodDescriber{c}
	out, err := d.Describe("foo", "bar", DescriberSettings{})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if !strings.Contains(out, "topology.kubernetes.io/test1:DoNotSchedule when max skew 3 is exceeded for selector key1=val1,key2=val2\n") ||
		!strings.Contains(out, "topology.kubernetes.io/test2:ScheduleAnyway when max skew 1 is exceeded\n") {
		t.Errorf("unexpected out:\n%s", out)
	}
}

func TestDescribeSecret(t *testing.T) {
	fake := fake.NewSimpleClientset(&corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "bar",
			Namespace: "foo",
		},
		Data: map[string][]byte{
			"username": []byte("YWRtaW4="),
			"password": []byte("MWYyZDFlMmU2N2Rm"),
		},
	})
	c := &describeClient{T: t, Namespace: "foo", Interface: fake}
	d := SecretDescriber{c}
	out, err := d.Describe("foo", "bar", DescriberSettings{})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if !strings.Contains(out, "bar") || !strings.Contains(out, "foo") || !strings.Contains(out, "username") || !strings.Contains(out, "8 bytes") || !strings.Contains(out, "password") || !strings.Contains(out, "16 bytes") {
		t.Errorf("unexpected out: %s", out)
	}
	if strings.Contains(out, "YWRtaW4=") || strings.Contains(out, "MWYyZDFlMmU2N2Rm") {
		t.Errorf("sensitive data should not be shown, unexpected out: %s", out)
	}
}

func TestDescribeNamespace(t *testing.T) {
	exampleNamespaceName := "example"

	testCases := []struct {
		name      string
		namespace *corev1.Namespace
		expect    []string
	}{
		{
			name: "no quotas or limit ranges",
			namespace: &corev1.Namespace{
				ObjectMeta: metav1.ObjectMeta{
					Name: exampleNamespaceName,
				},
				Status: corev1.NamespaceStatus{
					Phase: corev1.NamespaceActive,
				},
			},
			expect: []string{
				"Name",
				exampleNamespaceName,
				"Status",
				string(corev1.NamespaceActive),
				"No resource quota",
				"No LimitRange resource.",
			},
		},
		{
			name: "has conditions",
			namespace: &corev1.Namespace{
				ObjectMeta: metav1.ObjectMeta{
					Name: exampleNamespaceName,
				},
				Status: corev1.NamespaceStatus{
					Phase: corev1.NamespaceTerminating,
					Conditions: []corev1.NamespaceCondition{
						{
							LastTransitionTime: metav1.NewTime(time.Date(2014, time.January, 15, 0, 0, 0, 0, time.UTC)),
							Message:            "example message",
							Reason:             "example reason",
							Status:             corev1.ConditionTrue,
							Type:               corev1.NamespaceDeletionContentFailure,
						},
					},
				},
			},
			expect: []string{
				"Name",
				exampleNamespaceName,
				"Status",
				string(corev1.NamespaceTerminating),
				"Conditions",
				"Type",
				string(corev1.NamespaceDeletionContentFailure),
				"Status",
				string(corev1.ConditionTrue),
				"Reason",
				"example reason",
				"Message",
				"example message",
				"No resource quota",
				"No LimitRange resource.",
			},
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			fake := fake.NewSimpleClientset(testCase.namespace)
			c := &describeClient{T: t, Namespace: "", Interface: fake}
			d := NamespaceDescriber{c}

			out, err := d.Describe("", testCase.namespace.Name, DescriberSettings{ShowEvents: true})
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}

			for _, expected := range testCase.expect {
				if !strings.Contains(out, expected) {
					t.Errorf("expected to find %q in output: %q", expected, out)
				}
			}
		})
	}
}

func TestDescribePodPriority(t *testing.T) {
	priority := int32(1000)
	fake := fake.NewSimpleClientset(&corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "bar",
		},
		Spec: corev1.PodSpec{
			PriorityClassName: "high-priority",
			Priority:          &priority,
		},
	})
	c := &describeClient{T: t, Namespace: "", Interface: fake}
	d := PodDescriber{c}
	out, err := d.Describe("", "bar", DescriberSettings{ShowEvents: true})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if !strings.Contains(out, "high-priority") || !strings.Contains(out, "1000") {
		t.Errorf("unexpected out: %s", out)
	}
}

func TestDescribePodRuntimeClass(t *testing.T) {
	runtimeClassNames := []string{"test1", ""}
	testCases := []struct {
		name     string
		pod      *corev1.Pod
		expect   []string
		unexpect []string
	}{
		{
			name: "test1",
			pod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "bar",
				},
				Spec: corev1.PodSpec{
					RuntimeClassName: &runtimeClassNames[0],
				},
			},
			expect: []string{
				"Name", "bar",
				"Runtime Class Name", "test1",
			},
			unexpect: []string{},
		},
		{
			name: "test2",
			pod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "bar",
				},
				Spec: corev1.PodSpec{
					RuntimeClassName: &runtimeClassNames[1],
				},
			},
			expect: []string{
				"Name", "bar",
			},
			unexpect: []string{
				"Runtime Class Name",
			},
		},
		{
			name: "test3",
			pod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "bar",
				},
				Spec: corev1.PodSpec{},
			},
			expect: []string{
				"Name", "bar",
			},
			unexpect: []string{
				"Runtime Class Name",
			},
		},
	}
	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			fake := fake.NewSimpleClientset(testCase.pod)
			c := &describeClient{T: t, Interface: fake}
			d := PodDescriber{c}
			out, err := d.Describe("", "bar", DescriberSettings{ShowEvents: true})
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			for _, expected := range testCase.expect {
				if !strings.Contains(out, expected) {
					t.Errorf("expected to find %q in output: %q", expected, out)
				}
			}
			for _, unexpected := range testCase.unexpect {
				if strings.Contains(out, unexpected) {
					t.Errorf("unexpected to find %q in output: %q", unexpected, out)
				}
			}
		})
	}
}

func TestDescribePriorityClass(t *testing.T) {
	preemptLowerPriority := corev1.PreemptLowerPriority
	preemptNever := corev1.PreemptNever

	testCases := []struct {
		name          string
		priorityClass *schedulingv1.PriorityClass
		expect        []string
	}{
		{
			name: "test1",
			priorityClass: &schedulingv1.PriorityClass{
				ObjectMeta: metav1.ObjectMeta{
					Name: "bar",
				},
				Value:            10,
				GlobalDefault:    false,
				PreemptionPolicy: &preemptLowerPriority,
				Description:      "test1",
			},
			expect: []string{
				"Name", "bar",
				"Value", "10",
				"GlobalDefault", "false",
				"PreemptionPolicy", "PreemptLowerPriority",
				"Description", "test1",
				"Annotations", "",
			},
		},
		{
			name: "test2",
			priorityClass: &schedulingv1.PriorityClass{
				ObjectMeta: metav1.ObjectMeta{
					Name: "bar",
				},
				Value:            100,
				GlobalDefault:    true,
				PreemptionPolicy: &preemptNever,
				Description:      "test2",
			},
			expect: []string{
				"Name", "bar",
				"Value", "100",
				"GlobalDefault", "true",
				"PreemptionPolicy", "Never",
				"Description", "test2",
				"Annotations", "",
			},
		},
	}
	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			fake := fake.NewSimpleClientset(testCase.priorityClass)
			c := &describeClient{T: t, Interface: fake}
			d := PriorityClassDescriber{c}
			out, err := d.Describe("", "bar", DescriberSettings{ShowEvents: true})
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			for _, expected := range testCase.expect {
				if !strings.Contains(out, expected) {
					t.Errorf("expected to find %q in output: %q", expected, out)
				}
			}
		})
	}
}

func TestDescribeConfigMap(t *testing.T) {
	fake := fake.NewSimpleClientset(&corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "mycm",
			Namespace: "foo",
		},
		Data: map[string]string{
			"key1": "value1",
			"key2": "value2",
		},
		BinaryData: map[string][]byte{
			"binarykey1": {0xFF, 0xFE, 0xFD, 0xFC, 0xFB},
			"binarykey2": {0xFF, 0xFE, 0xFD, 0xFC, 0xFB, 0xFA},
		},
	})
	c := &describeClient{T: t, Namespace: "foo", Interface: fake}
	d := ConfigMapDescriber{c}
	out, err := d.Describe("foo", "mycm", DescriberSettings{ShowEvents: true})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if !strings.Contains(out, "foo") || !strings.Contains(out, "mycm") {
		t.Errorf("unexpected out: %s", out)
	}
	if !strings.Contains(out, "key1") || !strings.Contains(out, "value1") || !strings.Contains(out, "key2") || !strings.Contains(out, "value2") {
		t.Errorf("unexpected out: %s", out)
	}
	if !strings.Contains(out, "binarykey1") || !strings.Contains(out, "5 bytes") || !strings.Contains(out, "binarykey2") || !strings.Contains(out, "6 bytes") {
		t.Errorf("unexpected out: %s", out)
	}
}

func TestDescribeLimitRange(t *testing.T) {
	fake := fake.NewSimpleClientset(&corev1.LimitRange{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "mylr",
			Namespace: "foo",
		},
		Spec: corev1.LimitRangeSpec{
			Limits: []corev1.LimitRangeItem{
				{
					Type:                 corev1.LimitTypePod,
					Max:                  getResourceList("100m", "10000Mi"),
					Min:                  getResourceList("5m", "100Mi"),
					MaxLimitRequestRatio: getResourceList("10", ""),
				},
				{
					Type:                 corev1.LimitTypeContainer,
					Max:                  getResourceList("100m", "10000Mi"),
					Min:                  getResourceList("5m", "100Mi"),
					Default:              getResourceList("50m", "500Mi"),
					DefaultRequest:       getResourceList("10m", "200Mi"),
					MaxLimitRequestRatio: getResourceList("10", ""),
				},
				{
					Type: corev1.LimitTypePersistentVolumeClaim,
					Max:  getStorageResourceList("10Gi"),
					Min:  getStorageResourceList("5Gi"),
				},
			},
		},
	})
	c := &describeClient{T: t, Namespace: "foo", Interface: fake}
	d := LimitRangeDescriber{c}
	out, err := d.Describe("foo", "mylr", DescriberSettings{ShowEvents: true})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	checks := []string{"foo", "mylr", "Pod", "cpu", "5m", "100m", "memory", "100Mi", "10000Mi", "10", "Container", "cpu", "10m", "50m", "200Mi", "500Mi", "PersistentVolumeClaim", "storage", "5Gi", "10Gi"}
	for _, check := range checks {
		if !strings.Contains(out, check) {
			t.Errorf("unexpected out: %s", out)
		}
	}
}

func getStorageResourceList(storage string) corev1.ResourceList {
	res := corev1.ResourceList{}
	if storage != "" {
		res[corev1.ResourceStorage] = resource.MustParse(storage)
	}
	return res
}

func getResourceList(cpu, memory string) corev1.ResourceList {
	res := corev1.ResourceList{}
	if cpu != "" {
		res[corev1.ResourceCPU] = resource.MustParse(cpu)
	}
	if memory != "" {
		res[corev1.ResourceMemory] = resource.MustParse(memory)
	}
	return res
}

func TestDescribeService(t *testing.T) {
	singleStack := corev1.IPFamilyPolicySingleStack
	testCases := []struct {
		name           string
		service        *corev1.Service
		endpointSlices []*discoveryv1.EndpointSlice
		expected       string
	}{
		{
			name: "test1",
			service: &corev1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "bar",
					Namespace: "foo",
				},
				Spec: corev1.ServiceSpec{
					Type: corev1.ServiceTypeLoadBalancer,
					Ports: []corev1.ServicePort{{
						Name:       "port-tcp",
						Port:       8080,
						Protocol:   corev1.ProtocolTCP,
						TargetPort: intstr.FromInt32(9527),
						NodePort:   31111,
					}},
					Selector:              map[string]string{"blah": "heh"},
					ClusterIP:             "1.2.3.4",
					IPFamilies:            []corev1.IPFamily{corev1.IPv4Protocol},
					LoadBalancerIP:        "5.6.7.8",
					SessionAffinity:       corev1.ServiceAffinityNone,
					ExternalTrafficPolicy: corev1.ServiceExternalTrafficPolicyLocal,
					InternalTrafficPolicy: ptr.To(corev1.ServiceInternalTrafficPolicyCluster),
					HealthCheckNodePort:   32222,
				},
			},
			endpointSlices: []*discoveryv1.EndpointSlice{{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "bar-abcde",
					Namespace: "foo",
					Labels: map[string]string{
						"kubernetes.io/service-name": "bar",
					},
				},
				Endpoints: []discoveryv1.Endpoint{
					{Addresses: []string{"10.244.0.1"}},
					{Addresses: []string{"10.244.0.2"}},
					{Addresses: []string{"10.244.0.3"}},
				},
				Ports: []discoveryv1.EndpointPort{{
					Name:     ptr.To("port-tcp"),
					Port:     ptr.To[int32](9527),
					Protocol: ptr.To(corev1.ProtocolTCP),
				}},
			}},
			expected: dedent.Dedent(`
				Name:                     bar
				Namespace:                foo
				Labels:                   <none>
				Annotations:              <none>
				Selector:                 blah=heh
				Type:                     LoadBalancer
				IP Families:              IPv4
				IP:                       1.2.3.4
				IPs:                      <none>
				IP:                       5.6.7.8
				Port:                     port-tcp  8080/TCP
				TargetPort:               9527/TCP
				NodePort:                 port-tcp  31111/TCP
				Endpoints:                10.244.0.1:9527,10.244.0.2:9527,10.244.0.3:9527
				Session Affinity:         None
				External Traffic Policy:  Local
				Internal Traffic Policy:  Cluster
				HealthCheck NodePort:     32222
				Events:                   <none>
			`)[1:],
		},
		{
			name: "test2",
			service: &corev1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "bar",
					Namespace: "foo",
				},
				Spec: corev1.ServiceSpec{
					Type: corev1.ServiceTypeLoadBalancer,
					Ports: []corev1.ServicePort{{
						Name:       "port-tcp",
						Port:       8080,
						Protocol:   corev1.ProtocolTCP,
						TargetPort: intstr.FromString("targetPort"),
						NodePort:   31111,
					}},
					Selector:              map[string]string{"blah": "heh"},
					ClusterIP:             "1.2.3.4",
					IPFamilies:            []corev1.IPFamily{corev1.IPv4Protocol},
					LoadBalancerIP:        "5.6.7.8",
					SessionAffinity:       corev1.ServiceAffinityNone,
					ExternalTrafficPolicy: corev1.ServiceExternalTrafficPolicyLocal,
					InternalTrafficPolicy: ptr.To(corev1.ServiceInternalTrafficPolicyLocal),
					HealthCheckNodePort:   32222,
				},
			},
			endpointSlices: []*discoveryv1.EndpointSlice{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "bar-12345",
						Namespace: "foo",
						Labels: map[string]string{
							"kubernetes.io/service-name": "bar",
						},
					},
					Endpoints: []discoveryv1.Endpoint{
						{Addresses: []string{"10.244.0.1"}},
						{Addresses: []string{"10.244.0.2"}},
					},
					Ports: []discoveryv1.EndpointPort{{
						Name:     ptr.To("port-tcp"),
						Port:     ptr.To[int32](9527),
						Protocol: ptr.To(corev1.ProtocolUDP),
					}},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "bar-54321",
						Namespace: "foo",
						Labels: map[string]string{
							"kubernetes.io/service-name": "bar",
						},
					},
					Endpoints: []discoveryv1.Endpoint{
						{Addresses: []string{"10.244.0.3"}},
						{Addresses: []string{"10.244.0.4"}},
						{Addresses: []string{"10.244.0.5"}},
					},
					Ports: []discoveryv1.EndpointPort{{
						Name:     ptr.To("port-tcp"),
						Port:     ptr.To[int32](9527),
						Protocol: ptr.To(corev1.ProtocolUDP),
					}},
				},
			},
			expected: dedent.Dedent(`
				Name:                     bar
				Namespace:                foo
				Labels:                   <none>
				Annotations:              <none>
				Selector:                 blah=heh
				Type:                     LoadBalancer
				IP Families:              IPv4
				IP:                       1.2.3.4
				IPs:                      <none>
				IP:                       5.6.7.8
				Port:                     port-tcp  8080/TCP
				TargetPort:               targetPort/TCP
				NodePort:                 port-tcp  31111/TCP
				Endpoints:                10.244.0.1:9527,10.244.0.2:9527,10.244.0.3:9527 + 2 more...
				Session Affinity:         None
				External Traffic Policy:  Local
				Internal Traffic Policy:  Local
				HealthCheck NodePort:     32222
				Events:                   <none>
			`)[1:],
		},
		{
			name: "test-ServiceIPFamily",
			service: &corev1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "bar",
					Namespace: "foo",
				},
				Spec: corev1.ServiceSpec{
					Type: corev1.ServiceTypeLoadBalancer,
					Ports: []corev1.ServicePort{{
						Name:       "port-tcp",
						Port:       8080,
						Protocol:   corev1.ProtocolTCP,
						TargetPort: intstr.FromString("targetPort"),
						NodePort:   31111,
					}},
					Selector:              map[string]string{"blah": "heh"},
					ClusterIP:             "1.2.3.4",
					IPFamilies:            []corev1.IPFamily{corev1.IPv4Protocol},
					LoadBalancerIP:        "5.6.7.8",
					SessionAffinity:       corev1.ServiceAffinityNone,
					ExternalTrafficPolicy: corev1.ServiceExternalTrafficPolicyLocal,
					HealthCheckNodePort:   32222,
				},
			},
			endpointSlices: []*discoveryv1.EndpointSlice{{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "bar-123ab",
					Namespace: "foo",
					Labels: map[string]string{
						"kubernetes.io/service-name": "bar",
					},
				},
				Endpoints: []discoveryv1.Endpoint{
					{Addresses: []string{"10.244.0.1"}},
				},
				Ports: []discoveryv1.EndpointPort{{
					Name:     ptr.To("port-tcp"),
					Port:     ptr.To[int32](9527),
					Protocol: ptr.To(corev1.ProtocolTCP),
				}},
			}},
			expected: dedent.Dedent(`
				Name:                     bar
				Namespace:                foo
				Labels:                   <none>
				Annotations:              <none>
				Selector:                 blah=heh
				Type:                     LoadBalancer
				IP Families:              IPv4
				IP:                       1.2.3.4
				IPs:                      <none>
				IP:                       5.6.7.8
				Port:                     port-tcp  8080/TCP
				TargetPort:               targetPort/TCP
				NodePort:                 port-tcp  31111/TCP
				Endpoints:                10.244.0.1:9527
				Session Affinity:         None
				External Traffic Policy:  Local
				HealthCheck NodePort:     32222
				Events:                   <none>
			`)[1:],
		},
		{
			name: "test-ServiceIPFamilyPolicy+ClusterIPs",
			service: &corev1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "bar",
					Namespace: "foo",
				},
				Spec: corev1.ServiceSpec{
					Type: corev1.ServiceTypeLoadBalancer,
					Ports: []corev1.ServicePort{{
						Name:       "port-tcp",
						Port:       8080,
						Protocol:   corev1.ProtocolTCP,
						TargetPort: intstr.FromString("targetPort"),
						NodePort:   31111,
					}},
					Selector:              map[string]string{"blah": "heh"},
					ClusterIP:             "1.2.3.4",
					IPFamilies:            []corev1.IPFamily{corev1.IPv4Protocol},
					IPFamilyPolicy:        &singleStack,
					ClusterIPs:            []string{"1.2.3.4"},
					LoadBalancerIP:        "5.6.7.8",
					SessionAffinity:       corev1.ServiceAffinityNone,
					ExternalTrafficPolicy: corev1.ServiceExternalTrafficPolicyLocal,
					HealthCheckNodePort:   32222,
				},
			},
			expected: dedent.Dedent(`
				Name:                     bar
				Namespace:                foo
				Labels:                   <none>
				Annotations:              <none>
				Selector:                 blah=heh
				Type:                     LoadBalancer
				IP Family Policy:         SingleStack
				IP Families:              IPv4
				IP:                       1.2.3.4
				IPs:                      1.2.3.4
				IP:                       5.6.7.8
				Port:                     port-tcp  8080/TCP
				TargetPort:               targetPort/TCP
				NodePort:                 port-tcp  31111/TCP
				Endpoints:                <none>
				Session Affinity:         None
				External Traffic Policy:  Local
				HealthCheck NodePort:     32222
				Events:                   <none>
			`)[1:],
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			objects := []runtime.Object{tc.service}
			for i := range tc.endpointSlices {
				objects = append(objects, tc.endpointSlices[i])
			}
			fakeClient := fake.NewSimpleClientset(objects...)
			c := &describeClient{T: t, Namespace: "foo", Interface: fakeClient}
			d := ServiceDescriber{c}
			out, err := d.Describe("foo", "bar", DescriberSettings{ShowEvents: true})
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}

			assert.Equal(t, tc.expected, out)
		})
	}
}

func TestPodDescribeResultsSorted(t *testing.T) {
	// Arrange
	fake := fake.NewSimpleClientset(
		&corev1.EventList{
			Items: []corev1.Event{
				{
					ObjectMeta:     metav1.ObjectMeta{Name: "one"},
					Source:         corev1.EventSource{Component: "kubelet"},
					Message:        "Item 1",
					FirstTimestamp: metav1.NewTime(time.Date(2014, time.January, 15, 0, 0, 0, 0, time.UTC)),
					LastTimestamp:  metav1.NewTime(time.Date(2014, time.January, 15, 0, 0, 0, 0, time.UTC)),
					Count:          1,
					Type:           corev1.EventTypeNormal,
				},
				{
					ObjectMeta:     metav1.ObjectMeta{Name: "two"},
					Source:         corev1.EventSource{Component: "scheduler"},
					Message:        "Item 2",
					FirstTimestamp: metav1.NewTime(time.Date(1987, time.June, 17, 0, 0, 0, 0, time.UTC)),
					LastTimestamp:  metav1.NewTime(time.Date(1987, time.June, 17, 0, 0, 0, 0, time.UTC)),
					Count:          1,
					Type:           corev1.EventTypeNormal,
				},
				{
					ObjectMeta:     metav1.ObjectMeta{Name: "three"},
					Source:         corev1.EventSource{Component: "kubelet"},
					Message:        "Item 3",
					FirstTimestamp: metav1.NewTime(time.Date(2002, time.December, 25, 0, 0, 0, 0, time.UTC)),
					LastTimestamp:  metav1.NewTime(time.Date(2002, time.December, 25, 0, 0, 0, 0, time.UTC)),
					Count:          1,
					Type:           corev1.EventTypeNormal,
				},
			},
		},
		&corev1.Pod{ObjectMeta: metav1.ObjectMeta{Namespace: "foo", Name: "bar"}},
	)
	c := &describeClient{T: t, Namespace: "foo", Interface: fake}
	d := PodDescriber{c}

	// Act
	out, err := d.Describe("foo", "bar", DescriberSettings{ShowEvents: true})

	// Assert
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	VerifyDatesInOrder(out, "\n" /* rowDelimiter */, "\t" /* columnDelimiter */, t)
}

// VerifyDatesInOrder checks the start of each line for a RFC1123Z date
// and posts error if all subsequent dates are not equal or increasing
func VerifyDatesInOrder(
	resultToTest, rowDelimiter, columnDelimiter string, t *testing.T) {
	lines := strings.Split(resultToTest, rowDelimiter)
	var previousTime time.Time
	for _, str := range lines {
		columns := strings.Split(str, columnDelimiter)
		if len(columns) > 0 {
			currentTime, err := time.Parse(time.RFC1123Z, columns[0])
			if err == nil {
				if previousTime.After(currentTime) {
					t.Errorf(
						"Output is not sorted by time. %s should be listed after %s. Complete output: %s",
						previousTime.Format(time.RFC1123Z),
						currentTime.Format(time.RFC1123Z),
						resultToTest)
				}
				previousTime = currentTime
			}
		}
	}
}

func TestDescribeContainers(t *testing.T) {
	trueVal := true
	testCases := []struct {
		container        corev1.Container
		status           corev1.ContainerStatus
		expectedElements []string
	}{
		// Running state.
		{
			container: corev1.Container{Name: "test", Image: "image"},
			status: corev1.ContainerStatus{
				Name: "test",
				State: corev1.ContainerState{
					Running: &corev1.ContainerStateRunning{
						StartedAt: metav1.NewTime(time.Now()),
					},
				},
				Ready:        true,
				RestartCount: 7,
			},
			expectedElements: []string{"test", "State", "Running", "Ready", "True", "Restart Count", "7", "Image", "image", "Started"},
		},
		// Waiting state.
		{
			container: corev1.Container{Name: "test", Image: "image"},
			status: corev1.ContainerStatus{
				Name: "test",
				State: corev1.ContainerState{
					Waiting: &corev1.ContainerStateWaiting{
						Reason: "potato",
					},
				},
				Ready:        true,
				RestartCount: 7,
			},
			expectedElements: []string{"test", "State", "Waiting", "Ready", "True", "Restart Count", "7", "Image", "image", "Reason", "potato"},
		},
		// Terminated state.
		{
			container: corev1.Container{Name: "test", Image: "image"},
			status: corev1.ContainerStatus{
				Name: "test",
				State: corev1.ContainerState{
					Terminated: &corev1.ContainerStateTerminated{
						StartedAt:  metav1.NewTime(time.Now()),
						FinishedAt: metav1.NewTime(time.Now()),
						Reason:     "potato",
						ExitCode:   2,
					},
				},
				Ready:        true,
				RestartCount: 7,
			},
			expectedElements: []string{"test", "State", "Terminated", "Ready", "True", "Restart Count", "7", "Image", "image", "Reason", "potato", "Started", "Finished", "Exit Code", "2"},
		},
		// Last Terminated
		{
			container: corev1.Container{Name: "test", Image: "image"},
			status: corev1.ContainerStatus{
				Name: "test",
				State: corev1.ContainerState{
					Running: &corev1.ContainerStateRunning{
						StartedAt: metav1.NewTime(time.Now()),
					},
				},
				LastTerminationState: corev1.ContainerState{
					Terminated: &corev1.ContainerStateTerminated{
						StartedAt:  metav1.NewTime(time.Now().Add(time.Second * 3)),
						FinishedAt: metav1.NewTime(time.Now()),
						Reason:     "crashing",
						ExitCode:   3,
					},
				},
				Ready:        true,
				RestartCount: 7,
			},
			expectedElements: []string{"test", "State", "Terminated", "Ready", "True", "Restart Count", "7", "Image", "image", "Started", "Finished", "Exit Code", "2", "crashing", "3"},
		},
		// No state defaults to waiting.
		{
			container: corev1.Container{Name: "test", Image: "image"},
			status: corev1.ContainerStatus{
				Name:         "test",
				Ready:        true,
				RestartCount: 7,
			},
			expectedElements: []string{"test", "State", "Waiting", "Ready", "True", "Restart Count", "7", "Image", "image"},
		},
		// Env
		{
			container: corev1.Container{Name: "test", Image: "image", Env: []corev1.EnvVar{{Name: "envname", Value: "xyz"}}, EnvFrom: []corev1.EnvFromSource{{ConfigMapRef: &corev1.ConfigMapEnvSource{LocalObjectReference: corev1.LocalObjectReference{Name: "a123"}}}}},
			status: corev1.ContainerStatus{
				Name:         "test",
				Ready:        true,
				RestartCount: 7,
			},
			expectedElements: []string{"test", "State", "Waiting", "Ready", "True", "Restart Count", "7", "Image", "image", "envname", "xyz", "a123\tConfigMap\tOptional: false"},
		},
		{
			container: corev1.Container{Name: "test", Image: "image", Env: []corev1.EnvVar{{Name: "envname", Value: "xyz"}}, EnvFrom: []corev1.EnvFromSource{{Prefix: "p_", ConfigMapRef: &corev1.ConfigMapEnvSource{LocalObjectReference: corev1.LocalObjectReference{Name: "a123"}}}}},
			status: corev1.ContainerStatus{
				Name:         "test",
				Ready:        true,
				RestartCount: 7,
			},
			expectedElements: []string{"test", "State", "Waiting", "Ready", "True", "Restart Count", "7", "Image", "image", "envname", "xyz", "a123\tConfigMap with prefix 'p_'\tOptional: false"},
		},
		{
			container: corev1.Container{Name: "test", Image: "image", Env: []corev1.EnvVar{{Name: "envname", Value: "xyz"}}, EnvFrom: []corev1.EnvFromSource{{ConfigMapRef: &corev1.ConfigMapEnvSource{Optional: &trueVal, LocalObjectReference: corev1.LocalObjectReference{Name: "a123"}}}}},
			status: corev1.ContainerStatus{
				Name:         "test",
				Ready:        true,
				RestartCount: 7,
			},
			expectedElements: []string{"test", "State", "Waiting", "Ready", "True", "Restart Count", "7", "Image", "image", "envname", "xyz", "a123\tConfigMap\tOptional: true"},
		},
		{
			container: corev1.Container{Name: "test", Image: "image", Env: []corev1.EnvVar{{Name: "envname", Value: "xyz"}}, EnvFrom: []corev1.EnvFromSource{{SecretRef: &corev1.SecretEnvSource{LocalObjectReference: corev1.LocalObjectReference{Name: "a123"}, Optional: &trueVal}}}},
			status: corev1.ContainerStatus{
				Name:         "test",
				Ready:        true,
				RestartCount: 7,
			},
			expectedElements: []string{"test", "State", "Waiting", "Ready", "True", "Restart Count", "7", "Image", "image", "envname", "xyz", "a123\tSecret\tOptional: true"},
		},
		{
			container: corev1.Container{Name: "test", Image: "image", Env: []corev1.EnvVar{{Name: "envname", Value: "xyz"}}, EnvFrom: []corev1.EnvFromSource{{Prefix: "p_", SecretRef: &corev1.SecretEnvSource{LocalObjectReference: corev1.LocalObjectReference{Name: "a123"}}}}},
			status: corev1.ContainerStatus{
				Name:         "test",
				Ready:        true,
				RestartCount: 7,
			},
			expectedElements: []string{"test", "State", "Waiting", "Ready", "True", "Restart Count", "7", "Image", "image", "envname", "xyz", "a123\tSecret with prefix 'p_'\tOptional: false"},
		},
		// Command
		{
			container: corev1.Container{Name: "test", Image: "image", Command: []string{"sleep", "1000"}},
			status: corev1.ContainerStatus{
				Name:         "test",
				Ready:        true,
				RestartCount: 7,
			},
			expectedElements: []string{"test", "State", "Waiting", "Ready", "True", "Restart Count", "7", "Image", "image", "sleep", "1000"},
		},
		// Command with newline
		{
			container: corev1.Container{Name: "test", Image: "image", Command: []string{"sleep", "1000\n2000"}},
			status: corev1.ContainerStatus{
				Name:         "test",
				Ready:        true,
				RestartCount: 7,
			},
			expectedElements: []string{"1000\n      2000"},
		},
		// Args
		{
			container: corev1.Container{Name: "test", Image: "image", Args: []string{"time", "1000"}},
			status: corev1.ContainerStatus{
				Name:         "test",
				Ready:        true,
				RestartCount: 7,
			},
			expectedElements: []string{"test", "State", "Waiting", "Ready", "True", "Restart Count", "7", "Image", "image", "time", "1000"},
		},
		// Args with newline
		{
			container: corev1.Container{Name: "test", Image: "image", Args: []string{"time", "1000\n2000"}},
			status: corev1.ContainerStatus{
				Name:         "test",
				Ready:        true,
				RestartCount: 7,
			},
			expectedElements: []string{"1000\n      2000"},
		},
		// Using limits.
		{
			container: corev1.Container{
				Name:  "test",
				Image: "image",
				Resources: corev1.ResourceRequirements{
					Limits: corev1.ResourceList{
						corev1.ResourceName(corev1.ResourceCPU):     resource.MustParse("1000"),
						corev1.ResourceName(corev1.ResourceMemory):  resource.MustParse("4G"),
						corev1.ResourceName(corev1.ResourceStorage): resource.MustParse("20G"),
					},
				},
			},
			status: corev1.ContainerStatus{
				Name:         "test",
				Ready:        true,
				RestartCount: 7,
			},
			expectedElements: []string{"cpu", "1k", "memory", "4G", "storage", "20G"},
		},
		// Using requests.
		{
			container: corev1.Container{
				Name:  "test",
				Image: "image",
				Resources: corev1.ResourceRequirements{
					Requests: corev1.ResourceList{
						corev1.ResourceName(corev1.ResourceCPU):     resource.MustParse("1000"),
						corev1.ResourceName(corev1.ResourceMemory):  resource.MustParse("4G"),
						corev1.ResourceName(corev1.ResourceStorage): resource.MustParse("20G"),
					},
				},
			},
			expectedElements: []string{"cpu", "1k", "memory", "4G", "storage", "20G"},
		},
		// volumeMounts read/write
		{
			container: corev1.Container{
				Name:  "test",
				Image: "image",
				VolumeMounts: []corev1.VolumeMount{
					{
						Name:      "mounted-volume",
						MountPath: "/opt/",
					},
				},
			},
			expectedElements: []string{"mounted-volume", "/opt/", "(rw)"},
		},
		// volumeMounts readonly
		{
			container: corev1.Container{
				Name:  "test",
				Image: "image",
				VolumeMounts: []corev1.VolumeMount{
					{
						Name:      "mounted-volume",
						MountPath: "/opt/",
						ReadOnly:  true,
					},
				},
			},
			expectedElements: []string{"Mounts", "mounted-volume", "/opt/", "(ro)"},
		},

		// volumeMounts subPath
		{
			container: corev1.Container{
				Name:  "test",
				Image: "image",
				VolumeMounts: []corev1.VolumeMount{
					{
						Name:      "mounted-volume",
						MountPath: "/opt/",
						SubPath:   "foo",
					},
				},
			},
			expectedElements: []string{"Mounts", "mounted-volume", "/opt/", "(rw,path=\"foo\")"},
		},

		// volumeDevices
		{
			container: corev1.Container{
				Name:  "test",
				Image: "image",
				VolumeDevices: []corev1.VolumeDevice{
					{
						Name:       "volume-device",
						DevicePath: "/dev/xvda",
					},
				},
			},
			expectedElements: []string{"Devices", "volume-device", "/dev/xvda"},
		},
	}

	for i, testCase := range testCases {
		t.Run(fmt.Sprintf("%d", i), func(t *testing.T) {
			out := new(bytes.Buffer)
			pod := corev1.Pod{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{testCase.container},
				},
				Status: corev1.PodStatus{
					ContainerStatuses: []corev1.ContainerStatus{testCase.status},
				},
			}
			writer := NewPrefixWriter(out)
			describeContainers("Containers", pod.Spec.Containers, pod.Status.ContainerStatuses, EnvValueRetriever(&pod), writer, "")
			output := out.String()
			for _, expected := range testCase.expectedElements {
				if !strings.Contains(output, expected) {
					t.Errorf("Test case %d: expected to find %q in output: %q", i, expected, output)
				}
			}
		})
	}
}

func TestDescribers(t *testing.T) {
	first := &corev1.Event{}
	second := &corev1.Pod{}
	var third *corev1.Pod
	testErr := fmt.Errorf("test")
	d := Describers{}
	d.Add(
		func(e *corev1.Event, p *corev1.Pod) (string, error) {
			if e != first {
				t.Errorf("first argument not equal: %#v", e)
			}
			if p != second {
				t.Errorf("second argument not equal: %#v", p)
			}
			return "test", testErr
		},
	)
	if out, err := d.DescribeObject(first, second); out != "test" || err != testErr {
		t.Errorf("unexpected result: %s %v", out, err)
	}

	if out, err := d.DescribeObject(first, second, third); out != "" || err == nil {
		t.Errorf("unexpected result: %s %v", out, err)
	} else {
		if noDescriber, ok := err.(ErrNoDescriber); ok {
			if !reflect.DeepEqual(noDescriber.Types, []string{"*v1.Event", "*v1.Pod", "*v1.Pod"}) {
				t.Errorf("unexpected describer: %v", err)
			}
		} else {
			t.Errorf("unexpected error type: %v", err)
		}
	}

	d.Add(
		func(e *corev1.Event) (string, error) {
			if e != first {
				t.Errorf("first argument not equal: %#v", e)
			}
			return "simpler", testErr
		},
	)
	if out, err := d.DescribeObject(first); out != "simpler" || err != testErr {
		t.Errorf("unexpected result: %s %v", out, err)
	}
}

func TestDefaultDescribers(t *testing.T) {
	out, err := DefaultObjectDescriber.DescribeObject(&corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo"}})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(out, "foo") {
		t.Errorf("missing Pod `foo` in output: %s", out)
	}

	out, err = DefaultObjectDescriber.DescribeObject(&corev1.Service{ObjectMeta: metav1.ObjectMeta{Name: "foo"}})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(out, "foo") {
		t.Errorf("missing Service `foo` in output: %s", out)
	}

	out, err = DefaultObjectDescriber.DescribeObject(&corev1.ReplicationController{
		ObjectMeta: metav1.ObjectMeta{Name: "foo"},
		Spec:       corev1.ReplicationControllerSpec{Replicas: ptr.To[int32](1)},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(out, "foo") {
		t.Errorf("missing Replication Controller `foo` in output: %s", out)
	}

	out, err = DefaultObjectDescriber.DescribeObject(&corev1.Node{ObjectMeta: metav1.ObjectMeta{Name: "foo"}})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(out, "foo") {
		t.Errorf("missing Node `foo` output: %s", out)
	}

	out, err = DefaultObjectDescriber.DescribeObject(&appsv1.StatefulSet{
		ObjectMeta: metav1.ObjectMeta{Name: "foo"},
		Spec:       appsv1.StatefulSetSpec{Replicas: ptr.To[int32](1)},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(out, "foo") {
		t.Errorf("missing StatefulSet `foo` in output: %s", out)
	}
}

func TestGetPodsTotalRequests(t *testing.T) {
	testCases := []struct {
		name         string
		pods         *corev1.PodList
		expectedReqs map[corev1.ResourceName]resource.Quantity
	}{
		{
			name: "test1",
			pods: &corev1.PodList{
				Items: []corev1.Pod{
					{
						Spec: corev1.PodSpec{
							Containers: []corev1.Container{
								{
									Resources: corev1.ResourceRequirements{
										Requests: corev1.ResourceList{
											corev1.ResourceName(corev1.ResourceCPU):     resource.MustParse("1"),
											corev1.ResourceName(corev1.ResourceMemory):  resource.MustParse("300Mi"),
											corev1.ResourceName(corev1.ResourceStorage): resource.MustParse("1G"),
										},
									},
								},
								{
									Resources: corev1.ResourceRequirements{
										Requests: corev1.ResourceList{
											corev1.ResourceName(corev1.ResourceCPU):     resource.MustParse("90m"),
											corev1.ResourceName(corev1.ResourceMemory):  resource.MustParse("120Mi"),
											corev1.ResourceName(corev1.ResourceStorage): resource.MustParse("200M"),
										},
									},
								},
							},
						},
					},
					{
						Spec: corev1.PodSpec{
							Containers: []corev1.Container{
								{
									Resources: corev1.ResourceRequirements{
										Requests: corev1.ResourceList{
											corev1.ResourceName(corev1.ResourceCPU):     resource.MustParse("60m"),
											corev1.ResourceName(corev1.ResourceMemory):  resource.MustParse("43Mi"),
											corev1.ResourceName(corev1.ResourceStorage): resource.MustParse("500M"),
										},
									},
								},
								{
									Resources: corev1.ResourceRequirements{
										Requests: corev1.ResourceList{
											corev1.ResourceName(corev1.ResourceCPU):     resource.MustParse("34m"),
											corev1.ResourceName(corev1.ResourceMemory):  resource.MustParse("83Mi"),
											corev1.ResourceName(corev1.ResourceStorage): resource.MustParse("700M"),
										},
									},
								},
							},
						},
					},
				},
			},
			expectedReqs: map[corev1.ResourceName]resource.Quantity{
				corev1.ResourceName(corev1.ResourceCPU):     resource.MustParse("1.184"),
				corev1.ResourceName(corev1.ResourceMemory):  resource.MustParse("546Mi"),
				corev1.ResourceName(corev1.ResourceStorage): resource.MustParse("2.4G"),
			},
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			reqs, _ := getPodsTotalRequestsAndLimits(testCase.pods)
			if !apiequality.Semantic.DeepEqual(reqs, testCase.expectedReqs) {
				t.Errorf("Expected %v, got %v", testCase.expectedReqs, reqs)
			}
		})
	}
}

func TestPersistentVolumeDescriber(t *testing.T) {
	block := corev1.PersistentVolumeBlock
	file := corev1.PersistentVolumeFilesystem
	foo := "glusterfsendpointname"
	deletionTimestamp := metav1.Time{Time: time.Now().UTC().AddDate(-10, 0, 0)}
	testCases := []struct {
		name               string
		plugin             string
		pv                 *corev1.PersistentVolume
		expectedElements   []string
		unexpectedElements []string
	}{
		{
			name:   "test0",
			plugin: "hostpath",
			pv: &corev1.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{Name: "bar"},
				Spec: corev1.PersistentVolumeSpec{
					PersistentVolumeSource: corev1.PersistentVolumeSource{
						HostPath: &corev1.HostPathVolumeSource{Type: new(corev1.HostPathType)},
					},
				},
			},
			unexpectedElements: []string{"VolumeMode", "Filesystem"},
		},
		{
			name:   "test1",
			plugin: "gce",
			pv: &corev1.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{Name: "bar"},
				Spec: corev1.PersistentVolumeSpec{
					PersistentVolumeSource: corev1.PersistentVolumeSource{
						GCEPersistentDisk: &corev1.GCEPersistentDiskVolumeSource{},
					},
					VolumeMode: &file,
				},
			},
			expectedElements: []string{"VolumeMode", "Filesystem"},
		},
		{
			name:   "test2",
			plugin: "ebs",
			pv: &corev1.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{Name: "bar"},
				Spec: corev1.PersistentVolumeSpec{
					PersistentVolumeSource: corev1.PersistentVolumeSource{
						AWSElasticBlockStore: &corev1.AWSElasticBlockStoreVolumeSource{},
					},
				},
			},
			unexpectedElements: []string{"VolumeMode", "Filesystem"},
		},
		{
			name:   "test3",
			plugin: "nfs",
			pv: &corev1.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{Name: "bar"},
				Spec: corev1.PersistentVolumeSpec{
					PersistentVolumeSource: corev1.PersistentVolumeSource{
						NFS: &corev1.NFSVolumeSource{},
					},
				},
			},
			unexpectedElements: []string{"VolumeMode", "Filesystem"},
		},
		{
			name:   "test4",
			plugin: "iscsi",
			pv: &corev1.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{Name: "bar"},
				Spec: corev1.PersistentVolumeSpec{
					PersistentVolumeSource: corev1.PersistentVolumeSource{
						ISCSI: &corev1.ISCSIPersistentVolumeSource{},
					},
					VolumeMode: &block,
				},
			},
			expectedElements: []string{"VolumeMode", "Block"},
		},
		{
			name:   "test5",
			plugin: "gluster",
			pv: &corev1.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{Name: "bar"},
				Spec: corev1.PersistentVolumeSpec{
					PersistentVolumeSource: corev1.PersistentVolumeSource{
						Glusterfs: &corev1.GlusterfsPersistentVolumeSource{},
					},
				},
			},
			expectedElements:   []string{"EndpointsNamespace", "<unset>"},
			unexpectedElements: []string{"VolumeMode", "Filesystem"},
		},
		{
			name:   "test6",
			plugin: "rbd",
			pv: &corev1.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{Name: "bar"},
				Spec: corev1.PersistentVolumeSpec{
					PersistentVolumeSource: corev1.PersistentVolumeSource{
						RBD: &corev1.RBDPersistentVolumeSource{},
					},
				},
			},
			unexpectedElements: []string{"VolumeMode", "Filesystem"},
		},
		{
			name:   "test7",
			plugin: "quobyte",
			pv: &corev1.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{Name: "bar"},
				Spec: corev1.PersistentVolumeSpec{
					PersistentVolumeSource: corev1.PersistentVolumeSource{
						Quobyte: &corev1.QuobyteVolumeSource{},
					},
				},
			},
			unexpectedElements: []string{"VolumeMode", "Filesystem"},
		},
		{
			name:   "test8",
			plugin: "cinder",
			pv: &corev1.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{Name: "bar"},
				Spec: corev1.PersistentVolumeSpec{
					PersistentVolumeSource: corev1.PersistentVolumeSource{
						Cinder: &corev1.CinderPersistentVolumeSource{},
					},
				},
			},
			unexpectedElements: []string{"VolumeMode", "Filesystem"},
		},
		{
			name:   "test9",
			plugin: "fc",
			pv: &corev1.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{Name: "bar"},
				Spec: corev1.PersistentVolumeSpec{
					PersistentVolumeSource: corev1.PersistentVolumeSource{
						FC: &corev1.FCVolumeSource{},
					},
					VolumeMode: &block,
				},
			},
			expectedElements: []string{"VolumeMode", "Block"},
		},
		{
			name:   "test10",
			plugin: "local",
			pv: &corev1.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{Name: "bar"},
				Spec: corev1.PersistentVolumeSpec{
					PersistentVolumeSource: corev1.PersistentVolumeSource{
						Local: &corev1.LocalVolumeSource{},
					},
				},
			},
			expectedElements:   []string{"Node Affinity:   <none>"},
			unexpectedElements: []string{"Required Terms", "Term "},
		},
		{
			name:   "test11",
			plugin: "local",
			pv: &corev1.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{Name: "bar"},
				Spec: corev1.PersistentVolumeSpec{
					PersistentVolumeSource: corev1.PersistentVolumeSource{
						Local: &corev1.LocalVolumeSource{},
					},
					NodeAffinity: &corev1.VolumeNodeAffinity{},
				},
			},
			expectedElements:   []string{"Node Affinity:   <none>"},
			unexpectedElements: []string{"Required Terms", "Term "},
		},
		{
			name:   "test12",
			plugin: "local",
			pv: &corev1.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{Name: "bar"},
				Spec: corev1.PersistentVolumeSpec{
					PersistentVolumeSource: corev1.PersistentVolumeSource{
						Local: &corev1.LocalVolumeSource{},
					},
					NodeAffinity: &corev1.VolumeNodeAffinity{
						Required: &corev1.NodeSelector{},
					},
				},
			},
			expectedElements:   []string{"Node Affinity", "Required Terms:  <none>"},
			unexpectedElements: []string{"Term "},
		},
		{
			name:   "test13",
			plugin: "local",
			pv: &corev1.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{Name: "bar"},
				Spec: corev1.PersistentVolumeSpec{
					PersistentVolumeSource: corev1.PersistentVolumeSource{
						Local: &corev1.LocalVolumeSource{},
					},
					NodeAffinity: &corev1.VolumeNodeAffinity{
						Required: &corev1.NodeSelector{
							NodeSelectorTerms: []corev1.NodeSelectorTerm{
								{
									MatchExpressions: []corev1.NodeSelectorRequirement{},
								},
								{
									MatchExpressions: []corev1.NodeSelectorRequirement{},
								},
							},
						},
					},
				},
			},
			expectedElements: []string{"Node Affinity", "Required Terms", "Term 0", "Term 1"},
		},
		{
			name:   "test14",
			plugin: "local",
			pv: &corev1.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{Name: "bar"},
				Spec: corev1.PersistentVolumeSpec{
					PersistentVolumeSource: corev1.PersistentVolumeSource{
						Local: &corev1.LocalVolumeSource{},
					},
					NodeAffinity: &corev1.VolumeNodeAffinity{
						Required: &corev1.NodeSelector{
							NodeSelectorTerms: []corev1.NodeSelectorTerm{
								{
									MatchExpressions: []corev1.NodeSelectorRequirement{
										{
											Key:      "foo",
											Operator: "In",
											Values:   []string{"val1", "val2"},
										},
										{
											Key:      "foo",
											Operator: "Exists",
										},
									},
								},
							},
						},
					},
				},
			},
			expectedElements: []string{"Node Affinity", "Required Terms", "Term 0",
				"foo in [val1, val2]",
				"foo exists"},
		},
		{
			name:   "test15",
			plugin: "local",
			pv: &corev1.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "bar",
					DeletionTimestamp: &deletionTimestamp,
				},
				Spec: corev1.PersistentVolumeSpec{
					PersistentVolumeSource: corev1.PersistentVolumeSource{
						Local: &corev1.LocalVolumeSource{},
					},
				},
			},
			expectedElements: []string{"Terminating (lasts 10y)"},
		},
		{
			name:   "test16",
			plugin: "local",
			pv: &corev1.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name:                       "bar",
					GenerateName:               "test-GenerateName",
					UID:                        "test-UID",
					CreationTimestamp:          metav1.Time{Time: time.Now()},
					DeletionTimestamp:          &metav1.Time{Time: time.Now()},
					DeletionGracePeriodSeconds: new(int64),
					Labels:                     map[string]string{"label1": "label1", "label2": "label2", "label3": "label3"},
					Annotations:                map[string]string{"annotation1": "annotation1", "annotation2": "annotation2", "annotation3": "annotation3"},
				},
				Spec: corev1.PersistentVolumeSpec{
					PersistentVolumeSource: corev1.PersistentVolumeSource{
						Local: &corev1.LocalVolumeSource{},
					},
					NodeAffinity: &corev1.VolumeNodeAffinity{
						Required: &corev1.NodeSelector{
							NodeSelectorTerms: []corev1.NodeSelectorTerm{
								{
									MatchExpressions: []corev1.NodeSelectorRequirement{
										{
											Key:      "foo",
											Operator: "In",
											Values:   []string{"val1", "val2"},
										},
										{
											Key:      "foo",
											Operator: "Exists",
										},
									},
								},
							},
						},
					},
				},
			},
			expectedElements: []string{"Node Affinity", "Required Terms", "Term 0",
				"foo in [val1, val2]",
				"foo exists"},
		},
		{
			name:   "test17",
			plugin: "local",
			pv: &corev1.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name:                       "bar",
					GenerateName:               "test-GenerateName",
					UID:                        "test-UID",
					CreationTimestamp:          metav1.Time{Time: time.Now()},
					DeletionTimestamp:          &metav1.Time{Time: time.Now()},
					DeletionGracePeriodSeconds: new(int64),
					Labels:                     map[string]string{"label1": "label1", "label2": "label2", "label3": "label3"},
					Annotations:                map[string]string{"annotation1": "annotation1", "annotation2": "annotation2", "annotation3": "annotation3"},
				},
				Spec: corev1.PersistentVolumeSpec{
					PersistentVolumeSource: corev1.PersistentVolumeSource{
						CSI: &corev1.CSIPersistentVolumeSource{
							Driver:       "drive",
							VolumeHandle: "handler",
							ReadOnly:     true,
							VolumeAttributes: map[string]string{
								"Attribute1": "Value1",
								"Attribute2": "Value2",
								"Attribute3": "Value3",
							},
						},
					},
				},
			},
			expectedElements: []string{"Driver", "VolumeHandle", "ReadOnly", "VolumeAttributes"},
		},
		{
			name:   "test19",
			plugin: "gluster",
			pv: &corev1.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{Name: "bar"},
				Spec: corev1.PersistentVolumeSpec{
					PersistentVolumeSource: corev1.PersistentVolumeSource{
						Glusterfs: &corev1.GlusterfsPersistentVolumeSource{
							EndpointsNamespace: &foo,
						},
					},
				},
			},
			expectedElements:   []string{"EndpointsNamespace", "glusterfsendpointname"},
			unexpectedElements: []string{"VolumeMode", "Filesystem"},
		},
	}

	for _, test := range testCases {
		t.Run(test.name, func(t *testing.T) {
			fake := fake.NewSimpleClientset(test.pv)
			c := PersistentVolumeDescriber{fake}
			str, err := c.Describe("foo", "bar", DescriberSettings{ShowEvents: true})
			if err != nil {
				t.Errorf("Unexpected error for test %s: %v", test.plugin, err)
			}
			if str == "" {
				t.Errorf("Unexpected empty string for test %s.  Expected PV Describer output", test.plugin)
			}
			for _, expected := range test.expectedElements {
				if !strings.Contains(str, expected) {
					t.Errorf("expected to find %q in output: %q", expected, str)
				}
			}
			for _, unexpected := range test.unexpectedElements {
				if strings.Contains(str, unexpected) {
					t.Errorf("unexpected to find %q in output: %q", unexpected, str)
				}
			}
		})
	}
}

func TestPersistentVolumeClaimDescriber(t *testing.T) {
	block := corev1.PersistentVolumeBlock
	file := corev1.PersistentVolumeFilesystem
	goldClassName := "gold"
	now := time.Now()
	deletionTimestamp := metav1.Time{Time: time.Now().UTC().AddDate(-10, 0, 0)}
	snapshotAPIGroup := "snapshot.storage.k8s.io"
	defaultDescriberSettings := &DescriberSettings{ShowEvents: true}
	testCases := []struct {
		name               string
		pvc                *corev1.PersistentVolumeClaim
		describerSettings  *DescriberSettings
		expectedElements   []string
		unexpectedElements []string
	}{
		{
			name: "default",
			pvc: &corev1.PersistentVolumeClaim{
				ObjectMeta: metav1.ObjectMeta{Namespace: "foo", Name: "bar"},
				Spec: corev1.PersistentVolumeClaimSpec{
					VolumeName:       "volume1",
					StorageClassName: &goldClassName,
				},
				Status: corev1.PersistentVolumeClaimStatus{
					Phase: corev1.ClaimBound,
				},
			},
			expectedElements:   []string{"Events"},
			unexpectedElements: []string{"VolumeMode", "Filesystem"},
		},
		{
			name: "filesystem",
			pvc: &corev1.PersistentVolumeClaim{
				ObjectMeta: metav1.ObjectMeta{Namespace: "foo", Name: "bar"},
				Spec: corev1.PersistentVolumeClaimSpec{
					VolumeName:       "volume2",
					StorageClassName: &goldClassName,
					VolumeMode:       &file,
				},
				Status: corev1.PersistentVolumeClaimStatus{
					Phase: corev1.ClaimBound,
				},
			},
			expectedElements: []string{"VolumeMode", "Filesystem"},
		},
		{
			name: "block",
			pvc: &corev1.PersistentVolumeClaim{
				ObjectMeta: metav1.ObjectMeta{Namespace: "foo", Name: "bar"},
				Spec: corev1.PersistentVolumeClaimSpec{
					VolumeName:       "volume3",
					StorageClassName: &goldClassName,
					VolumeMode:       &block,
				},
				Status: corev1.PersistentVolumeClaimStatus{
					Phase: corev1.ClaimBound,
				},
			},
			expectedElements: []string{"VolumeMode", "Block"},
		},
		// Tests for Status.Condition.
		{
			name: "condition-type",
			pvc: &corev1.PersistentVolumeClaim{
				ObjectMeta: metav1.ObjectMeta{Namespace: "foo", Name: "bar"},
				Spec: corev1.PersistentVolumeClaimSpec{
					VolumeName:       "volume4",
					StorageClassName: &goldClassName,
				},
				Status: corev1.PersistentVolumeClaimStatus{
					Conditions: []corev1.PersistentVolumeClaimCondition{
						{Type: corev1.PersistentVolumeClaimResizing},
					},
				},
			},
			expectedElements: []string{"Conditions", "Type", "Resizing"},
		},
		{
			name: "condition-status",
			pvc: &corev1.PersistentVolumeClaim{
				ObjectMeta: metav1.ObjectMeta{Namespace: "foo", Name: "bar"},
				Spec: corev1.PersistentVolumeClaimSpec{
					VolumeName:       "volume5",
					StorageClassName: &goldClassName,
				},
				Status: corev1.PersistentVolumeClaimStatus{
					Conditions: []corev1.PersistentVolumeClaimCondition{
						{Status: corev1.ConditionTrue},
					},
				},
			},
			expectedElements: []string{"Conditions", "Status", "True"},
		},
		{
			name: "condition-last-probe-time",
			pvc: &corev1.PersistentVolumeClaim{
				ObjectMeta: metav1.ObjectMeta{Namespace: "foo", Name: "bar"},
				Spec: corev1.PersistentVolumeClaimSpec{
					VolumeName:       "volume6",
					StorageClassName: &goldClassName,
				},
				Status: corev1.PersistentVolumeClaimStatus{
					Conditions: []corev1.PersistentVolumeClaimCondition{
						{LastProbeTime: metav1.Time{Time: now}},
					},
				},
			},
			expectedElements: []string{"Conditions", "LastProbeTime", now.Format(time.RFC1123Z)},
		},
		{
			name: "condition-last-transition-time",
			pvc: &corev1.PersistentVolumeClaim{
				ObjectMeta: metav1.ObjectMeta{Namespace: "foo", Name: "bar"},
				Spec: corev1.PersistentVolumeClaimSpec{
					VolumeName:       "volume7",
					StorageClassName: &goldClassName,
				},
				Status: corev1.PersistentVolumeClaimStatus{
					Conditions: []corev1.PersistentVolumeClaimCondition{
						{LastTransitionTime: metav1.Time{Time: now}},
					},
				},
			},
			expectedElements: []string{"Conditions", "LastTransitionTime", now.Format(time.RFC1123Z)},
		},
		{
			name: "condition-reason",
			pvc: &corev1.PersistentVolumeClaim{
				ObjectMeta: metav1.ObjectMeta{Namespace: "foo", Name: "bar"},
				Spec: corev1.PersistentVolumeClaimSpec{
					VolumeName:       "volume8",
					StorageClassName: &goldClassName,
				},
				Status: corev1.PersistentVolumeClaimStatus{
					Conditions: []corev1.PersistentVolumeClaimCondition{
						{Reason: "OfflineResize"},
					},
				},
			},
			expectedElements: []string{"Conditions", "Reason", "OfflineResize"},
		},
		{
			name: "condition-message",
			pvc: &corev1.PersistentVolumeClaim{
				ObjectMeta: metav1.ObjectMeta{Namespace: "foo", Name: "bar"},
				Spec: corev1.PersistentVolumeClaimSpec{
					VolumeName:       "volume9",
					StorageClassName: &goldClassName,
				},
				Status: corev1.PersistentVolumeClaimStatus{
					Conditions: []corev1.PersistentVolumeClaimCondition{
						{Message: "User request resize"},
					},
				},
			},
			expectedElements: []string{"Conditions", "Message", "User request resize"},
		},
		{
			name: "deletion-timestamp",
			pvc: &corev1.PersistentVolumeClaim{
				ObjectMeta: metav1.ObjectMeta{
					Namespace:         "foo",
					Name:              "bar",
					DeletionTimestamp: &deletionTimestamp,
				},
				Spec: corev1.PersistentVolumeClaimSpec{
					VolumeName:       "volume10",
					StorageClassName: &goldClassName,
				},
				Status: corev1.PersistentVolumeClaimStatus{},
			},
			expectedElements: []string{"Terminating (lasts 10y)"},
		},
		{
			name: "pvc-datasource",
			pvc: &corev1.PersistentVolumeClaim{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: "foo",
					Name:      "bar",
				},
				Spec: corev1.PersistentVolumeClaimSpec{
					VolumeName:       "volume10",
					StorageClassName: &goldClassName,
					DataSource: &corev1.TypedLocalObjectReference{
						Name: "srcpvc",
						Kind: "PersistentVolumeClaim",
					},
				},
				Status: corev1.PersistentVolumeClaimStatus{},
			},
			expectedElements: []string{"\nDataSource:\n  Kind:   PersistentVolumeClaim\n  Name:   srcpvc"},
		},
		{
			name: "snapshot-datasource",
			pvc: &corev1.PersistentVolumeClaim{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: "foo",
					Name:      "bar",
				},
				Spec: corev1.PersistentVolumeClaimSpec{
					VolumeName:       "volume10",
					StorageClassName: &goldClassName,
					DataSource: &corev1.TypedLocalObjectReference{
						Name:     "src-snapshot",
						Kind:     "VolumeSnapshot",
						APIGroup: &snapshotAPIGroup,
					},
				},
				Status: corev1.PersistentVolumeClaimStatus{},
			},
			expectedElements: []string{"DataSource:\n  APIGroup:  snapshot.storage.k8s.io\n  Kind:      VolumeSnapshot\n  Name:      src-snapshot\n"},
		},
		{
			name: "no-show-events",
			pvc: &corev1.PersistentVolumeClaim{
				ObjectMeta: metav1.ObjectMeta{Namespace: "foo", Name: "bar"},
				Spec: corev1.PersistentVolumeClaimSpec{
					VolumeName:       "volume1",
					StorageClassName: &goldClassName,
				},
				Status: corev1.PersistentVolumeClaimStatus{
					Phase: corev1.ClaimBound,
				},
			},
			unexpectedElements: []string{"Events"},
			describerSettings:  &DescriberSettings{ShowEvents: false},
		},
	}

	for _, test := range testCases {
		t.Run(test.name, func(t *testing.T) {
			fake := fake.NewSimpleClientset(test.pvc)
			c := PersistentVolumeClaimDescriber{fake}

			var describerSettings DescriberSettings
			if test.describerSettings != nil {
				describerSettings = *test.describerSettings
			} else {
				describerSettings = *defaultDescriberSettings
			}

			str, err := c.Describe("foo", "bar", describerSettings)
			if err != nil {
				t.Errorf("Unexpected error for test %s: %v", test.name, err)
			}
			if str == "" {
				t.Errorf("Unexpected empty string for test %s.  Expected PVC Describer output", test.name)
			}
			for _, expected := range test.expectedElements {
				if !strings.Contains(str, expected) {
					t.Errorf("expected to find %q in output: %q", expected, str)
				}
			}
			for _, unexpected := range test.unexpectedElements {
				if strings.Contains(str, unexpected) {
					t.Errorf("unexpected to find %q in output: %q", unexpected, str)
				}
			}
		})
	}
}

func TestGetPodsForPVC(t *testing.T) {
	goldClassName := "gold"
	testCases := []struct {
		name            string
		pvc             *corev1.PersistentVolumeClaim
		requiredObjects []runtime.Object
		expectedPods    []string
	}{
		{
			name: "pvc-unused",
			pvc: &corev1.PersistentVolumeClaim{
				ObjectMeta: metav1.ObjectMeta{Namespace: "ns", Name: "pvc-name"},
				Spec: corev1.PersistentVolumeClaimSpec{
					VolumeName:       "volume1",
					StorageClassName: &goldClassName,
				},
				Status: corev1.PersistentVolumeClaimStatus{
					Phase: corev1.ClaimBound,
				},
			},
			expectedPods: []string{},
		},
		{
			name: "pvc-in-pods-volumes-list",
			pvc: &corev1.PersistentVolumeClaim{
				ObjectMeta: metav1.ObjectMeta{Namespace: "ns", Name: "pvc-name"},
				Spec: corev1.PersistentVolumeClaimSpec{
					VolumeName:       "volume1",
					StorageClassName: &goldClassName,
				},
				Status: corev1.PersistentVolumeClaimStatus{
					Phase: corev1.ClaimBound,
				},
			},
			requiredObjects: []runtime.Object{
				&corev1.Pod{
					ObjectMeta: metav1.ObjectMeta{Namespace: "ns", Name: "pod-name"},
					Spec: corev1.PodSpec{
						Volumes: []corev1.Volume{
							{
								Name: "volume",
								VolumeSource: corev1.VolumeSource{
									PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{
										ClaimName: "pvc-name",
									},
								},
							},
						},
					},
				},
			},
			expectedPods: []string{"pod-name"},
		},
		{
			name: "pvc-owned-by-pod",
			pvc: &corev1.PersistentVolumeClaim{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: "ns",
					Name:      "pvc-name",
					OwnerReferences: []metav1.OwnerReference{
						{
							Kind: "Pod",
							Name: "pod-name",
							UID:  "pod-uid",
						},
					},
				},
				Spec: corev1.PersistentVolumeClaimSpec{
					VolumeName:       "volume1",
					StorageClassName: &goldClassName,
				},
				Status: corev1.PersistentVolumeClaimStatus{
					Phase: corev1.ClaimBound,
				},
			},
			requiredObjects: []runtime.Object{
				&corev1.Pod{
					ObjectMeta: metav1.ObjectMeta{Namespace: "ns", Name: "pod-name", UID: "pod-uid"},
				},
			},
			expectedPods: []string{"pod-name"},
		},
	}

	for _, test := range testCases {
		t.Run(test.name, func(t *testing.T) {
			var objects []runtime.Object
			objects = append(objects, test.requiredObjects...)
			objects = append(objects, test.pvc)
			fake := fake.NewSimpleClientset(objects...)

			pods, err := getPodsForPVC(fake.CoreV1().Pods(test.pvc.ObjectMeta.Namespace), test.pvc, DescriberSettings{})
			if err != nil {
				t.Errorf("Unexpected error for test %s: %v", test.name, err)
			}

			for _, expectedPod := range test.expectedPods {
				foundPod := false
				for _, pod := range pods {
					if pod.Name == expectedPod {
						foundPod = true
						break
					}
				}

				if !foundPod {
					t.Errorf("Expected pod %s, but it was not returned: %v", expectedPod, pods)
				}
			}

			if len(test.expectedPods) != len(pods) {
				t.Errorf("Expected %d pods, but got %d pods", len(test.expectedPods), len(pods))
			}
		})
	}
}

func TestDescribeDeployment(t *testing.T) {
	labels := map[string]string{"k8s-app": "bar"}
	testCases := []struct {
		name    string
		objects []runtime.Object
		expects []string
	}{
		{
			name: "deployment with two mounted volumes",
			objects: []runtime.Object{
				&appsv1.Deployment{
					ObjectMeta: metav1.ObjectMeta{
						Name:              "bar",
						Namespace:         "foo",
						Labels:            labels,
						UID:               "00000000-0000-0000-0000-000000000001",
						CreationTimestamp: metav1.NewTime(time.Date(2021, time.Month(1), 1, 0, 0, 0, 0, time.UTC)),
					},
					Spec: appsv1.DeploymentSpec{
						Replicas: ptr.To[int32](1),
						Selector: &metav1.LabelSelector{
							MatchLabels: labels,
						},
						Template: corev1.PodTemplateSpec{
							ObjectMeta: metav1.ObjectMeta{
								Name:      "bar",
								Namespace: "foo",
								Labels:    labels,
							},
							Spec: corev1.PodSpec{
								Containers: []corev1.Container{
									{
										Image: "mytest-image:latest",
										VolumeMounts: []corev1.VolumeMount{
											{
												Name:      "vol-foo",
												MountPath: "/tmp/vol-foo",
											}, {
												Name:      "vol-bar",
												MountPath: "/tmp/vol-bar",
											},
										},
									},
								},
								Volumes: []corev1.Volume{
									{
										Name:         "vol-foo",
										VolumeSource: corev1.VolumeSource{EmptyDir: &corev1.EmptyDirVolumeSource{}},
									},
									{
										Name:         "vol-bar",
										VolumeSource: corev1.VolumeSource{EmptyDir: &corev1.EmptyDirVolumeSource{}},
									},
								},
							},
						},
					},
				}, &appsv1.ReplicaSet{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "bar-001",
						Namespace: "foo",
						Labels:    labels,
						OwnerReferences: []metav1.OwnerReference{
							{
								Controller: ptr.To(true),
								UID:        "00000000-0000-0000-0000-000000000001",
							},
						},
					},
					Spec: appsv1.ReplicaSetSpec{
						Replicas: ptr.To[int32](1),
						Selector: &metav1.LabelSelector{
							MatchLabels: labels,
						},
						Template: corev1.PodTemplateSpec{
							ObjectMeta: metav1.ObjectMeta{
								Name:      "bar",
								Namespace: "foo",
								Labels:    labels,
							},
							Spec: corev1.PodSpec{
								Containers: []corev1.Container{
									{
										Image: "mytest-image:latest",
										VolumeMounts: []corev1.VolumeMount{
											{
												Name:      "vol-foo",
												MountPath: "/tmp/vol-foo",
											}, {
												Name:      "vol-bar",
												MountPath: "/tmp/vol-bar",
											},
										},
									},
								},
								Volumes: []corev1.Volume{
									{
										Name:         "vol-foo",
										VolumeSource: corev1.VolumeSource{EmptyDir: &corev1.EmptyDirVolumeSource{}},
									},
									{
										Name:         "vol-bar",
										VolumeSource: corev1.VolumeSource{EmptyDir: &corev1.EmptyDirVolumeSource{}},
									},
								},
							},
						},
					},
					Status: appsv1.ReplicaSetStatus{
						Replicas:          1,
						ReadyReplicas:     1,
						AvailableReplicas: 1,
					},
				},
			},
			expects: []string{
				"Name:               bar\nNamespace:          foo",
				"CreationTimestamp:  Fri, 01 Jan 2021 00:00:00 +0000",
				"Labels:             k8s-app=bar",
				"Selector:           k8s-app=bar",
				"Replicas:           1 desired | 0 updated | 0 total | 0 available | 0 unavailable",
				"Image:        mytest-image:latest",
				"Mounts:\n      /tmp/vol-bar from vol-bar (rw)\n      /tmp/vol-foo from vol-foo (rw)",
				"OldReplicaSets:    <none>",
				"NewReplicaSet:     bar-001 (1/1 replicas created)",
				"Events:            <none>",
				"Node-Selectors:  <none>",
				"Tolerations:     <none>",
			},
		},
		{
			name: "deployment during the process of rolling out",
			objects: []runtime.Object{
				&appsv1.Deployment{
					ObjectMeta: metav1.ObjectMeta{
						Name:              "bar",
						Namespace:         "foo",
						Labels:            labels,
						UID:               "00000000-0000-0000-0000-000000000001",
						CreationTimestamp: metav1.NewTime(time.Date(2021, time.Month(1), 1, 0, 0, 0, 0, time.UTC)),
					},
					Spec: appsv1.DeploymentSpec{
						Replicas: ptr.To[int32](2),
						Selector: &metav1.LabelSelector{
							MatchLabels: labels,
						},
						Template: corev1.PodTemplateSpec{
							ObjectMeta: metav1.ObjectMeta{
								Name:      "bar",
								Namespace: "foo",
								Labels:    labels,
							},
							Spec: corev1.PodSpec{
								Containers: []corev1.Container{
									{
										Image: "mytest-image:v2.0",
										VolumeMounts: []corev1.VolumeMount{
											{
												Name:      "vol-foo",
												MountPath: "/tmp/vol-foo",
											}, {
												Name:      "vol-bar",
												MountPath: "/tmp/vol-bar",
											},
										},
									},
								},
								Volumes: []corev1.Volume{
									{
										Name:         "vol-foo",
										VolumeSource: corev1.VolumeSource{EmptyDir: &corev1.EmptyDirVolumeSource{}},
									},
									{
										Name:         "vol-bar",
										VolumeSource: corev1.VolumeSource{EmptyDir: &corev1.EmptyDirVolumeSource{}},
									},
								},
							},
						},
					},
					Status: appsv1.DeploymentStatus{
						Replicas:            3,
						UpdatedReplicas:     1,
						AvailableReplicas:   2,
						UnavailableReplicas: 1,
					},
				}, &appsv1.ReplicaSet{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "bar-001",
						Namespace: "foo",
						Labels:    labels,
						UID:       "00000000-0000-0000-0000-000000000001",
						OwnerReferences: []metav1.OwnerReference{
							{
								Controller: ptr.To(true),
								UID:        "00000000-0000-0000-0000-000000000001",
							},
						},
					},
					Spec: appsv1.ReplicaSetSpec{
						Replicas: ptr.To[int32](2),
						Selector: &metav1.LabelSelector{
							MatchLabels: labels,
						},
						Template: corev1.PodTemplateSpec{
							ObjectMeta: metav1.ObjectMeta{
								Name:      "bar",
								Namespace: "foo",
								Labels:    labels,
							},
							Spec: corev1.PodSpec{
								Containers: []corev1.Container{
									{
										Image: "mytest-image:v1.0",
										VolumeMounts: []corev1.VolumeMount{
											{
												Name:      "vol-foo",
												MountPath: "/tmp/vol-foo",
											}, {
												Name:      "vol-bar",
												MountPath: "/tmp/vol-bar",
											},
										},
									},
								},
								Volumes: []corev1.Volume{
									{
										Name:         "vol-foo",
										VolumeSource: corev1.VolumeSource{EmptyDir: &corev1.EmptyDirVolumeSource{}},
									},
									{
										Name:         "vol-bar",
										VolumeSource: corev1.VolumeSource{EmptyDir: &corev1.EmptyDirVolumeSource{}},
									},
								},
							},
						},
					},
					Status: appsv1.ReplicaSetStatus{
						Replicas:          2,
						ReadyReplicas:     2,
						AvailableReplicas: 2,
					},
				}, &appsv1.ReplicaSet{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "bar-002",
						Namespace: "foo",
						Labels:    labels,
						UID:       "00000000-0000-0000-0000-000000000002",
						OwnerReferences: []metav1.OwnerReference{
							{
								Controller: ptr.To(true),
								UID:        "00000000-0000-0000-0000-000000000001",
							},
						},
					},
					Spec: appsv1.ReplicaSetSpec{
						Replicas: ptr.To[int32](1),
						Selector: &metav1.LabelSelector{
							MatchLabels: labels,
						},
						Template: corev1.PodTemplateSpec{
							ObjectMeta: metav1.ObjectMeta{
								Name:      "bar",
								Namespace: "foo",
								Labels:    labels,
							},
							Spec: corev1.PodSpec{
								Containers: []corev1.Container{
									{
										Image: "mytest-image:v2.0",
										VolumeMounts: []corev1.VolumeMount{
											{
												Name:      "vol-foo",
												MountPath: "/tmp/vol-foo",
											}, {
												Name:      "vol-bar",
												MountPath: "/tmp/vol-bar",
											},
										},
									},
								},
								Volumes: []corev1.Volume{
									{
										Name:         "vol-foo",
										VolumeSource: corev1.VolumeSource{EmptyDir: &corev1.EmptyDirVolumeSource{}},
									},
									{
										Name:         "vol-bar",
										VolumeSource: corev1.VolumeSource{EmptyDir: &corev1.EmptyDirVolumeSource{}},
									},
								},
							},
						},
					},
					Status: appsv1.ReplicaSetStatus{
						Replicas:          1,
						ReadyReplicas:     0,
						AvailableReplicas: 1,
					},
				}, &corev1.Event{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "bar-000",
						Namespace: "foo",
					},
					InvolvedObject: corev1.ObjectReference{
						APIVersion: "apps/v1",
						Kind:       "Deployment",
						Name:       "bar",
						Namespace:  "foo",
						UID:        "00000000-0000-0000-0000-000000000001",
					},
					Type:                corev1.EventTypeNormal,
					Reason:              "ScalingReplicaSet",
					Message:             "Scaled up replica set bar-002 to 1",
					ReportingController: "deployment-controller",
					EventTime:           metav1.NewMicroTime(time.Now().Add(-20 * time.Minute)),
					Series: &corev1.EventSeries{
						Count:            3,
						LastObservedTime: metav1.NewMicroTime(time.Now().Add(-12 * time.Minute)),
					},
				}, &corev1.Event{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "bar-001",
						Namespace: "foo",
					},
					InvolvedObject: corev1.ObjectReference{
						APIVersion: "apps/v1",
						Kind:       "Deployment",
						Name:       "bar",
						Namespace:  "foo",
						UID:        "00000000-0000-0000-0000-000000000001",
					},
					Type:    corev1.EventTypeNormal,
					Reason:  "ScalingReplicaSet",
					Message: "Scaled up replica set bar-001 to 2",
					Source: corev1.EventSource{
						Component: "deployment-controller",
					},
					FirstTimestamp: metav1.NewTime(time.Now().Add(-10 * time.Minute)),
				}, &corev1.Event{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "bar-002",
						Namespace: "foo",
					},
					InvolvedObject: corev1.ObjectReference{
						APIVersion: "apps/v1",
						Kind:       "Deployment",
						Name:       "bar",
						Namespace:  "foo",
						UID:        "00000000-0000-0000-0000-000000000001",
					},
					Type:    corev1.EventTypeNormal,
					Reason:  "ScalingReplicaSet",
					Message: "Scaled up replica set bar-002 to 1",
					Source: corev1.EventSource{
						Component: "deployment-controller",
					},
					FirstTimestamp: metav1.NewTime(time.Now().Add(-2 * time.Minute)),
				}, &corev1.Event{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "bar-003",
						Namespace: "foo",
					},
					InvolvedObject: corev1.ObjectReference{
						APIVersion: "apps/v1",
						Kind:       "Deployment",
						Name:       "bar",
						Namespace:  "foo",
						UID:        "00000000-0000-0000-0000-000000000001",
					},
					Type:                corev1.EventTypeNormal,
					Reason:              "ScalingReplicaSet",
					Message:             "Scaled down replica set bar-002 to 1",
					ReportingController: "deployment-controller",
					EventTime:           metav1.NewMicroTime(time.Now().Add(-1 * time.Minute)),
				},
			},
			expects: []string{
				"Replicas:           2 desired | 1 updated | 3 total | 2 available | 1 unavailable",
				"Image:        mytest-image:v2.0",
				"OldReplicaSets:    bar-001 (2/2 replicas created)",
				"NewReplicaSet:     bar-002 (1/1 replicas created)",
				"Events:\n",
				"Normal  ScalingReplicaSet  12m (x3 over 20m)  deployment-controller  Scaled up replica set bar-002 to 1",
				"Normal  ScalingReplicaSet  10m                deployment-controller  Scaled up replica set bar-001 to 2",
				"Normal  ScalingReplicaSet  2m                 deployment-controller  Scaled up replica set bar-002 to 1",
				"Normal  ScalingReplicaSet  60s                deployment-controller  Scaled down replica set bar-002 to 1",
			},
		},
		{
			name: "deployment after successful rollout",
			objects: []runtime.Object{
				&appsv1.Deployment{
					ObjectMeta: metav1.ObjectMeta{
						Name:              "bar",
						Namespace:         "foo",
						Labels:            labels,
						UID:               "00000000-0000-0000-0000-000000000001",
						CreationTimestamp: metav1.NewTime(time.Date(2021, time.Month(1), 1, 0, 0, 0, 0, time.UTC)),
					},
					Spec: appsv1.DeploymentSpec{
						Replicas: ptr.To[int32](2),
						Selector: &metav1.LabelSelector{
							MatchLabels: labels,
						},
						Template: corev1.PodTemplateSpec{
							ObjectMeta: metav1.ObjectMeta{
								Name:      "bar",
								Namespace: "foo",
								Labels:    labels,
							},
							Spec: corev1.PodSpec{
								Containers: []corev1.Container{
									{
										Image: "mytest-image:v2.0",
										VolumeMounts: []corev1.VolumeMount{
											{
												Name:      "vol-foo",
												MountPath: "/tmp/vol-foo",
											}, {
												Name:      "vol-bar",
												MountPath: "/tmp/vol-bar",
											},
										},
									},
								},
								Volumes: []corev1.Volume{
									{
										Name:         "vol-foo",
										VolumeSource: corev1.VolumeSource{EmptyDir: &corev1.EmptyDirVolumeSource{}},
									},
									{
										Name:         "vol-bar",
										VolumeSource: corev1.VolumeSource{EmptyDir: &corev1.EmptyDirVolumeSource{}},
									},
								},
							},
						},
					},
					Status: appsv1.DeploymentStatus{
						Replicas:            2,
						UpdatedReplicas:     2,
						AvailableReplicas:   2,
						UnavailableReplicas: 0,
					},
				}, &appsv1.ReplicaSet{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "bar-001",
						Namespace: "foo",
						Labels:    labels,
						UID:       "00000000-0000-0000-0000-000000000001",
						OwnerReferences: []metav1.OwnerReference{
							{
								Controller: ptr.To(true),
								UID:        "00000000-0000-0000-0000-000000000001",
							},
						},
					},
					Spec: appsv1.ReplicaSetSpec{
						Replicas: ptr.To[int32](0),
						Selector: &metav1.LabelSelector{
							MatchLabels: labels,
						},
						Template: corev1.PodTemplateSpec{
							ObjectMeta: metav1.ObjectMeta{
								Name:      "bar",
								Namespace: "foo",
								Labels:    labels,
							},
							Spec: corev1.PodSpec{
								Containers: []corev1.Container{
									{
										Image: "mytest-image:v1.0",
										VolumeMounts: []corev1.VolumeMount{
											{
												Name:      "vol-foo",
												MountPath: "/tmp/vol-foo",
											}, {
												Name:      "vol-bar",
												MountPath: "/tmp/vol-bar",
											},
										},
									},
								},
								Volumes: []corev1.Volume{
									{
										Name:         "vol-foo",
										VolumeSource: corev1.VolumeSource{EmptyDir: &corev1.EmptyDirVolumeSource{}},
									},
									{
										Name:         "vol-bar",
										VolumeSource: corev1.VolumeSource{EmptyDir: &corev1.EmptyDirVolumeSource{}},
									},
								},
							},
						},
					},
					Status: appsv1.ReplicaSetStatus{
						Replicas:          0,
						ReadyReplicas:     0,
						AvailableReplicas: 0,
					},
				}, &appsv1.ReplicaSet{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "bar-002",
						Namespace: "foo",
						Labels:    labels,
						UID:       "00000000-0000-0000-0000-000000000002",
						OwnerReferences: []metav1.OwnerReference{
							{
								Controller: ptr.To(true),
								UID:        "00000000-0000-0000-0000-000000000001",
							},
						},
					},
					Spec: appsv1.ReplicaSetSpec{
						Replicas: ptr.To[int32](2),
						Selector: &metav1.LabelSelector{
							MatchLabels: labels,
						},
						Template: corev1.PodTemplateSpec{
							ObjectMeta: metav1.ObjectMeta{
								Name:      "bar",
								Namespace: "foo",
								Labels:    labels,
							},
							Spec: corev1.PodSpec{
								Containers: []corev1.Container{
									{
										Image: "mytest-image:v2.0",
										VolumeMounts: []corev1.VolumeMount{
											{
												Name:      "vol-foo",
												MountPath: "/tmp/vol-foo",
											}, {
												Name:      "vol-bar",
												MountPath: "/tmp/vol-bar",
											},
										},
									},
								},
								Volumes: []corev1.Volume{
									{
										Name:         "vol-foo",
										VolumeSource: corev1.VolumeSource{EmptyDir: &corev1.EmptyDirVolumeSource{}},
									},
									{
										Name:         "vol-bar",
										VolumeSource: corev1.VolumeSource{EmptyDir: &corev1.EmptyDirVolumeSource{}},
									},
								},
							},
						},
					},
					Status: appsv1.ReplicaSetStatus{
						Replicas:          2,
						ReadyReplicas:     2,
						AvailableReplicas: 2,
					},
				}, &corev1.Event{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "bar-000",
						Namespace: "foo",
					},
					InvolvedObject: corev1.ObjectReference{
						APIVersion: "apps/v1",
						Kind:       "Deployment",
						Name:       "bar",
						Namespace:  "foo",
						UID:        "00000000-0000-0000-0000-000000000001",
					},
					Type:                corev1.EventTypeNormal,
					Reason:              "ScalingReplicaSet",
					Message:             "Scaled up replica set bar-002 to 1",
					ReportingController: "deployment-controller",
					EventTime:           metav1.NewMicroTime(time.Now().Add(-20 * time.Minute)),
					Series: &corev1.EventSeries{
						Count:            3,
						LastObservedTime: metav1.NewMicroTime(time.Now().Add(-12 * time.Minute)),
					},
				}, &corev1.Event{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "bar-001",
						Namespace: "foo",
					},
					InvolvedObject: corev1.ObjectReference{
						APIVersion: "apps/v1",
						Kind:       "Deployment",
						Name:       "bar",
						Namespace:  "foo",
						UID:        "00000000-0000-0000-0000-000000000001",
					},
					Type:    corev1.EventTypeNormal,
					Reason:  "ScalingReplicaSet",
					Message: "Scaled up replica set bar-001 to 2",
					Source: corev1.EventSource{
						Component: "deployment-controller",
					},
					FirstTimestamp: metav1.NewTime(time.Now().Add(-10 * time.Minute)),
				}, &corev1.Event{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "bar-002",
						Namespace: "foo",
					},
					InvolvedObject: corev1.ObjectReference{
						APIVersion: "apps/v1",
						Kind:       "Deployment",
						Name:       "bar",
						Namespace:  "foo",
						UID:        "00000000-0000-0000-0000-000000000001",
					},
					Type:    corev1.EventTypeNormal,
					Reason:  "ScalingReplicaSet",
					Message: "Scaled up replica set bar-002 to 1",
					Source: corev1.EventSource{
						Component: "deployment-controller",
					},
					FirstTimestamp: metav1.NewTime(time.Now().Add(-2 * time.Minute)),
				}, &corev1.Event{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "bar-003",
						Namespace: "foo",
					},
					InvolvedObject: corev1.ObjectReference{
						APIVersion: "apps/v1",
						Kind:       "Deployment",
						Name:       "bar",
						Namespace:  "foo",
						UID:        "00000000-0000-0000-0000-000000000001",
					},
					Type:                corev1.EventTypeNormal,
					Reason:              "ScalingReplicaSet",
					Message:             "Scaled down replica set bar-002 to 1",
					ReportingController: "deployment-controller",
					EventTime:           metav1.NewMicroTime(time.Now().Add(-1 * time.Minute)),
				}, &corev1.Event{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "bar-004",
						Namespace: "foo",
					},
					InvolvedObject: corev1.ObjectReference{
						APIVersion: "apps/v1",
						Kind:       "Deployment",
						Name:       "bar",
						Namespace:  "foo",
						UID:        "00000000-0000-0000-0000-000000000001",
					},
					Type:                corev1.EventTypeNormal,
					Reason:              "ScalingReplicaSet",
					Message:             "Scaled up replica set bar-002 to 2",
					ReportingController: "deployment-controller",
					EventTime:           metav1.NewMicroTime(time.Now().Add(-15 * time.Second)),
				}, &corev1.Event{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "bar-005",
						Namespace: "foo",
					},
					InvolvedObject: corev1.ObjectReference{
						APIVersion: "apps/v1",
						Kind:       "Deployment",
						Name:       "bar",
						Namespace:  "foo",
						UID:        "00000000-0000-0000-0000-000000000001",
					},
					Type:                corev1.EventTypeNormal,
					Reason:              "ScalingReplicaSet",
					Message:             "Scaled down replica set bar-001 to 0",
					ReportingController: "deployment-controller",
					EventTime:           metav1.NewMicroTime(time.Now().Add(-3 * time.Second)),
				},
			},
			expects: []string{
				"Replicas:           2 desired | 2 updated | 2 total | 2 available | 0 unavailable",
				"Image:        mytest-image:v2.0",
				"OldReplicaSets:    bar-001 (0/0 replicas created)",
				"NewReplicaSet:     bar-002 (2/2 replicas created)",
				"Events:\n",
				"Normal  ScalingReplicaSet  12m (x3 over 20m)  deployment-controller  Scaled up replica set bar-002 to 1",
				"Normal  ScalingReplicaSet  10m                deployment-controller  Scaled up replica set bar-001 to 2",
				"Normal  ScalingReplicaSet  2m                 deployment-controller  Scaled up replica set bar-002 to 1",
				"Normal  ScalingReplicaSet  60s                deployment-controller  Scaled down replica set bar-002 to 1",
				"Normal  ScalingReplicaSet  15s                deployment-controller  Scaled up replica set bar-002 to 2",
				"Normal  ScalingReplicaSet  3s                 deployment-controller  Scaled down replica set bar-001 to 0",
			},
		},
	}
	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			fakeClient := fake.NewSimpleClientset(testCase.objects...)
			d := DeploymentDescriber{fakeClient}
			out, err := d.Describe("foo", "bar", DescriberSettings{ShowEvents: true})
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}

			for _, expect := range testCase.expects {
				if !strings.Contains(out, expect) {
					t.Errorf("expected to find \"%s\" in:\n %s", expect, out)
				}
			}

		})
	}
}

func TestDescribeJob(t *testing.T) {
	indexedCompletion := batchv1.IndexedCompletion
	cases := map[string]struct {
		job              *batchv1.Job
		wantElements     []string
		dontWantElements []string
	}{
		"empty job": {
			job: &batchv1.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "bar",
					Namespace: "foo",
				},
				Spec: batchv1.JobSpec{},
			},
			dontWantElements: []string{"Completed Indexes:", "Suspend:", "Backoff Limit:", "TTL Seconds After Finished:"},
		},
		"no completed indexes": {
			job: &batchv1.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "bar",
					Namespace: "foo",
				},
				Spec: batchv1.JobSpec{
					CompletionMode: &indexedCompletion,
				},
			},
			wantElements: []string{"Completed Indexes:  <none>"},
		},
		"few completed indexes": {
			job: &batchv1.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "bar",
					Namespace: "foo",
				},
				Spec: batchv1.JobSpec{
					CompletionMode: &indexedCompletion,
				},
				Status: batchv1.JobStatus{
					CompletedIndexes: "0-5,7,9,10,12,13,15,16,18,20,21,23,24,26,27,29,30,32",
				},
			},
			wantElements: []string{"Completed Indexes:  0-5,7,9,10,12,13,15,16,18,20,21,23,24,26,27,29,30,32"},
		},
		"too many completed indexes": {
			job: &batchv1.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "bar",
					Namespace: "foo",
				},
				Spec: batchv1.JobSpec{
					CompletionMode: &indexedCompletion,
				},
				Status: batchv1.JobStatus{
					CompletedIndexes: "0-5,7,9,10,12,13,15,16,18,20,21,23,24,26,27,29,30,32-34,36,37",
				},
			},
			wantElements: []string{"Completed Indexes:  0-5,7,9,10,12,13,15,16,18,20,21,23,24,26,27,29,30,32-34,..."},
		},
		"suspend set to true": {
			job: &batchv1.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "bar",
					Namespace: "foo",
				},
				Spec: batchv1.JobSpec{
					Suspend:                 ptr.To(true),
					TTLSecondsAfterFinished: ptr.To[int32](123),
					BackoffLimit:            ptr.To[int32](1),
				},
			},
			wantElements: []string{
				"Suspend:                     true",
				"TTL Seconds After Finished:  123",
				"Backoff Limit:               1",
			},
		},
		"suspend set to false": {
			job: &batchv1.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "bar",
					Namespace: "foo",
				},
				Spec: batchv1.JobSpec{
					Suspend: ptr.To(false),
				},
			},
			wantElements: []string{"Suspend:        false"},
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			client := &describeClient{
				T:         t,
				Namespace: tc.job.Namespace,
				Interface: fake.NewSimpleClientset(tc.job),
			}
			describer := JobDescriber{Interface: client}
			out, err := describer.Describe(tc.job.Namespace, tc.job.Name, DescriberSettings{ShowEvents: true})
			if err != nil {
				t.Fatalf("unexpected error describing object: %v", err)
			}

			for _, expected := range tc.wantElements {
				if !strings.Contains(out, expected) {
					t.Errorf("expected to find %q in output:\n %s", expected, out)
				}
			}

			for _, unexpected := range tc.dontWantElements {
				if strings.Contains(out, unexpected) {
					t.Errorf("unexpected to find %q in output:\n %s", unexpected, out)
				}
			}
		})
	}
}

func TestDescribeIngress(t *testing.T) {
	ingresClassName := "test"
	backendV1beta1 := networkingv1beta1.IngressBackend{
		ServiceName: "default-backend",
		ServicePort: intstr.FromInt32(80),
	}
	v1beta1 := fake.NewSimpleClientset(&networkingv1beta1.Ingress{
		ObjectMeta: metav1.ObjectMeta{
			Name: "bar",
			Labels: map[string]string{
				"id1": "app1",
				"id2": "app2",
			},
			Namespace: "foo",
		},
		Spec: networkingv1beta1.IngressSpec{
			IngressClassName: &ingresClassName,
			Rules: []networkingv1beta1.IngressRule{
				{
					Host: "foo.bar.com",
					IngressRuleValue: networkingv1beta1.IngressRuleValue{
						HTTP: &networkingv1beta1.HTTPIngressRuleValue{
							Paths: []networkingv1beta1.HTTPIngressPath{
								{
									Path:    "/foo",
									Backend: backendV1beta1,
								},
							},
						},
					},
				},
			},
		},
	})
	backendV1 := networkingv1.IngressBackend{
		Service: &networkingv1.IngressServiceBackend{
			Name: "default-backend",
			Port: networkingv1.ServiceBackendPort{
				Number: 80,
			},
		},
	}

	netv1 := fake.NewSimpleClientset(&networkingv1.Ingress{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "bar",
			Namespace: "foo",
		},
		Spec: networkingv1.IngressSpec{
			IngressClassName: &ingresClassName,
			Rules: []networkingv1.IngressRule{
				{
					Host: "foo.bar.com",
					IngressRuleValue: networkingv1.IngressRuleValue{
						HTTP: &networkingv1.HTTPIngressRuleValue{
							Paths: []networkingv1.HTTPIngressPath{
								{
									Path:    "/foo",
									Backend: backendV1,
								},
							},
						},
					},
				},
			},
		},
	})

	backendResource := networkingv1.IngressBackend{
		Resource: &corev1.TypedLocalObjectReference{
			APIGroup: ptr.To("example.com"),
			Kind:     "foo",
			Name:     "bar",
		},
	}
	backendResourceNoAPIGroup := networkingv1.IngressBackend{
		Resource: &corev1.TypedLocalObjectReference{
			Kind: "foo",
			Name: "bar",
		},
	}

	tests := map[string]struct {
		input  *fake.Clientset
		output string
	}{
		"IngressRule.HTTP.Paths.Backend.Service v1beta1": {
			input: v1beta1,
			output: `Name:             bar
Labels:           id1=app1
                  id2=app2
Namespace:        foo
Address:          
Ingress Class:    test
Default backend:  <default>
Rules:
  Host         Path  Backends
  ----         ----  --------
  foo.bar.com  
               /foo   default-backend:80 (<error: services "default-backend" not found>)
Annotations:   <none>
Events:        <none>` + "\n",
		},
		"IngressRule.HTTP.Paths.Backend.Service v1": {
			input: netv1,
			output: `Name:             bar
Labels:           <none>
Namespace:        foo
Address:          
Ingress Class:    test
Default backend:  <default>
Rules:
  Host         Path  Backends
  ----         ----  --------
  foo.bar.com  
               /foo   default-backend:80 (<error: services "default-backend" not found>)
Annotations:   <none>
Events:        <none>` + "\n",
		},
		"IngressRule.HTTP.Paths.Backend.Resource v1": {
			input: fake.NewSimpleClientset(&networkingv1.Ingress{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "bar",
					Namespace: "foo",
				},
				Spec: networkingv1.IngressSpec{
					IngressClassName: &ingresClassName,
					Rules: []networkingv1.IngressRule{
						{
							Host: "foo.bar.com",
							IngressRuleValue: networkingv1.IngressRuleValue{
								HTTP: &networkingv1.HTTPIngressRuleValue{
									Paths: []networkingv1.HTTPIngressPath{
										{
											Path:    "/foo",
											Backend: backendResource,
										},
									},
								},
							},
						},
					},
				},
			}),
			output: `Name:             bar
Labels:           <none>
Namespace:        foo
Address:          
Ingress Class:    test
Default backend:  <default>
Rules:
  Host         Path  Backends
  ----         ----  --------
  foo.bar.com  
               /foo   APIGroup: example.com, Kind: foo, Name: bar
Annotations:   <none>
Events:        <none>` + "\n",
		},
		"IngressRule.HTTP.Paths.Backend.Resource v1 Without APIGroup": {
			input: fake.NewSimpleClientset(&networkingv1.Ingress{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "bar",
					Namespace: "foo",
				},
				Spec: networkingv1.IngressSpec{
					IngressClassName: &ingresClassName,
					Rules: []networkingv1.IngressRule{
						{
							Host: "foo.bar.com",
							IngressRuleValue: networkingv1.IngressRuleValue{
								HTTP: &networkingv1.HTTPIngressRuleValue{
									Paths: []networkingv1.HTTPIngressPath{
										{
											Path:    "/foo",
											Backend: backendResourceNoAPIGroup,
										},
									},
								},
							},
						},
					},
				},
			}),
			output: `Name:             bar
Labels:           <none>
Namespace:        foo
Address:          
Ingress Class:    test
Default backend:  <default>
Rules:
  Host         Path  Backends
  ----         ----  --------
  foo.bar.com  
               /foo   APIGroup: <none>, Kind: foo, Name: bar
Annotations:   <none>
Events:        <none>` + "\n",
		},
		"Spec.DefaultBackend.Service & IngressRule.HTTP.Paths.Backend.Service v1": {
			input: fake.NewSimpleClientset(&networkingv1.Ingress{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "bar",
					Namespace: "foo",
				},
				Spec: networkingv1.IngressSpec{
					DefaultBackend:   &backendV1,
					IngressClassName: &ingresClassName,
					Rules: []networkingv1.IngressRule{
						{
							Host: "foo.bar.com",
							IngressRuleValue: networkingv1.IngressRuleValue{
								HTTP: &networkingv1.HTTPIngressRuleValue{
									Paths: []networkingv1.HTTPIngressPath{
										{
											Path:    "/foo",
											Backend: backendV1,
										},
									},
								},
							},
						},
					},
				},
			}),
			output: `Name:             bar
Labels:           <none>
Namespace:        foo
Address:          
Ingress Class:    test
Default backend:  default-backend:80 (<error: services "default-backend" not found>)
Rules:
  Host         Path  Backends
  ----         ----  --------
  foo.bar.com  
               /foo   default-backend:80 (<error: services "default-backend" not found>)
Annotations:   <none>
Events:        <none>` + "\n",
		},
		"Spec.DefaultBackend.Resource & IngressRule.HTTP.Paths.Backend.Resource v1": {
			input: fake.NewSimpleClientset(&networkingv1.Ingress{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "bar",
					Namespace: "foo",
				},
				Spec: networkingv1.IngressSpec{
					DefaultBackend:   &backendResource,
					IngressClassName: &ingresClassName,
					Rules: []networkingv1.IngressRule{
						{
							Host: "foo.bar.com",
							IngressRuleValue: networkingv1.IngressRuleValue{
								HTTP: &networkingv1.HTTPIngressRuleValue{
									Paths: []networkingv1.HTTPIngressPath{
										{
											Path:    "/foo",
											Backend: backendResource,
										},
									},
								},
							},
						},
					},
				},
			}),
			output: `Name:             bar
Labels:           <none>
Namespace:        foo
Address:          
Ingress Class:    test
Default backend:  APIGroup: example.com, Kind: foo, Name: bar
Rules:
  Host         Path  Backends
  ----         ----  --------
  foo.bar.com  
               /foo   APIGroup: example.com, Kind: foo, Name: bar
Annotations:   <none>
Events:        <none>` + "\n",
		},
		"Spec.DefaultBackend.Resource & IngressRule.HTTP.Paths.Backend.Service v1": {
			input: fake.NewSimpleClientset(&networkingv1.Ingress{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "bar",
					Namespace: "foo",
				},
				Spec: networkingv1.IngressSpec{
					DefaultBackend:   &backendResource,
					IngressClassName: &ingresClassName,
					Rules: []networkingv1.IngressRule{
						{
							Host: "foo.bar.com",
							IngressRuleValue: networkingv1.IngressRuleValue{
								HTTP: &networkingv1.HTTPIngressRuleValue{
									Paths: []networkingv1.HTTPIngressPath{
										{
											Path:    "/foo",
											Backend: backendV1,
										},
									},
								},
							},
						},
					},
				},
			}),
			output: `Name:             bar
Labels:           <none>
Namespace:        foo
Address:          
Ingress Class:    test
Default backend:  APIGroup: example.com, Kind: foo, Name: bar
Rules:
  Host         Path  Backends
  ----         ----  --------
  foo.bar.com  
               /foo   default-backend:80 (<error: services "default-backend" not found>)
Annotations:   <none>
Events:        <none>` + "\n",
		},
		"DefaultBackend": {
			input: fake.NewSimpleClientset(&networkingv1.Ingress{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "bar",
					Namespace: "foo",
				},
				Spec: networkingv1.IngressSpec{
					DefaultBackend:   &backendV1,
					IngressClassName: &ingresClassName,
				},
			}),
			output: `Name:             bar
Labels:           <none>
Namespace:        foo
Address:          
Ingress Class:    test
Default backend:  default-backend:80 (<error: services "default-backend" not found>)
Rules:
  Host        Path  Backends
  ----        ----  --------
  *           *     default-backend:80 (<error: services "default-backend" not found>)
Annotations:  <none>
Events:       <none>
`,
		},
		"EmptyBackend": {
			input: fake.NewSimpleClientset(&networkingv1.Ingress{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "bar",
					Namespace: "foo",
				},
				Spec: networkingv1.IngressSpec{
					IngressClassName: &ingresClassName,
				},
			}),
			output: `Name:             bar
Labels:           <none>
Namespace:        foo
Address:          
Ingress Class:    test
Default backend:  <default>
Rules:
  Host        Path  Backends
  ----        ----  --------
  *           *     <default>
Annotations:  <none>
Events:       <none>
`,
		},
		"EmptyIngressClassName": {
			input: fake.NewSimpleClientset(&networkingv1.Ingress{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "bar",
					Namespace: "foo",
				},
				Spec: networkingv1.IngressSpec{
					DefaultBackend: &backendV1,
				},
			}),
			output: `Name:             bar
Labels:           <none>
Namespace:        foo
Address:          
Ingress Class:    <none>
Default backend:  default-backend:80 (<error: services "default-backend" not found>)
Rules:
  Host        Path  Backends
  ----        ----  --------
  *           *     default-backend:80 (<error: services "default-backend" not found>)
Annotations:  <none>
Events:       <none>
`,
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			c := &describeClient{T: t, Namespace: "foo", Interface: test.input}
			i := IngressDescriber{c}
			out, err := i.Describe("foo", "bar", DescriberSettings{ShowEvents: true})
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			if out != test.output {
				t.Logf(out)
				t.Logf(test.output)
				t.Errorf("expected: \n%q\n but got output: \n%q\n", test.output, out)
			}
		})
	}
}

func TestDescribeIngressV1(t *testing.T) {
	ingresClassName := "test"
	defaultBackend := networkingv1.IngressBackend{
		Service: &networkingv1.IngressServiceBackend{
			Name: "default-backend",
			Port: networkingv1.ServiceBackendPort{
				Number: 80,
			},
		},
	}

	fakeClient := fake.NewSimpleClientset(&networkingv1.Ingress{
		ObjectMeta: metav1.ObjectMeta{
			Name: "bar",
			Labels: map[string]string{
				"id1": "app1",
				"id2": "app2",
			},
			Namespace: "foo",
		},
		Spec: networkingv1.IngressSpec{
			IngressClassName: &ingresClassName,
			Rules: []networkingv1.IngressRule{
				{
					Host: "foo.bar.com",
					IngressRuleValue: networkingv1.IngressRuleValue{
						HTTP: &networkingv1.HTTPIngressRuleValue{
							Paths: []networkingv1.HTTPIngressPath{
								{
									Path:    "/foo",
									Backend: defaultBackend,
								},
							},
						},
					},
				},
			},
		},
	})
	i := IngressDescriber{fakeClient}
	out, err := i.Describe("foo", "bar", DescriberSettings{ShowEvents: true})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if !strings.Contains(out, "bar") ||
		!strings.Contains(out, "foo") ||
		!strings.Contains(out, "foo.bar.com") ||
		!strings.Contains(out, "/foo") ||
		!strings.Contains(out, "app1") ||
		!strings.Contains(out, "app2") {
		t.Errorf("unexpected out: %s", out)
	}
}

func TestDescribeStorageClass(t *testing.T) {
	reclaimPolicy := corev1.PersistentVolumeReclaimRetain
	bindingMode := storagev1.VolumeBindingMode("bindingmode")
	f := fake.NewSimpleClientset(&storagev1.StorageClass{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "foo",
			ResourceVersion: "4",
			Annotations: map[string]string{
				"name": "foo",
			},
		},
		Provisioner: "my-provisioner",
		Parameters: map[string]string{
			"param1": "value1",
			"param2": "value2",
		},
		ReclaimPolicy:     &reclaimPolicy,
		VolumeBindingMode: &bindingMode,
		AllowedTopologies: []corev1.TopologySelectorTerm{
			{
				MatchLabelExpressions: []corev1.TopologySelectorLabelRequirement{
					{
						Key:    "failure-domain.beta.kubernetes.io/zone",
						Values: []string{"zone1"},
					},
					{
						Key:    "kubernetes.io/hostname",
						Values: []string{"node1"},
					},
				},
			},
			{
				MatchLabelExpressions: []corev1.TopologySelectorLabelRequirement{
					{
						Key:    "failure-domain.beta.kubernetes.io/zone",
						Values: []string{"zone2"},
					},
					{
						Key:    "kubernetes.io/hostname",
						Values: []string{"node2"},
					},
				},
			},
		},
	})
	s := StorageClassDescriber{f}
	out, err := s.Describe("", "foo", DescriberSettings{ShowEvents: true})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if !strings.Contains(out, "foo") ||
		!strings.Contains(out, "my-provisioner") ||
		!strings.Contains(out, "param1") ||
		!strings.Contains(out, "param2") ||
		!strings.Contains(out, "value1") ||
		!strings.Contains(out, "value2") ||
		!strings.Contains(out, "Retain") ||
		!strings.Contains(out, "bindingmode") ||
		!strings.Contains(out, "failure-domain.beta.kubernetes.io/zone") ||
		!strings.Contains(out, "zone1") ||
		!strings.Contains(out, "kubernetes.io/hostname") ||
		!strings.Contains(out, "node1") ||
		!strings.Contains(out, "zone2") ||
		!strings.Contains(out, "node2") {
		t.Errorf("unexpected out: %s", out)
	}
}

func TestDescribeVolumeAttributesClass(t *testing.T) {
	expectedOut := `Name:         foo
Annotations:  name=bar
DriverName:   my-driver
Parameters:   param1=value1,param2=value2
Events:       <none>
`

	f := fake.NewSimpleClientset(&storagev1alpha1.VolumeAttributesClass{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "foo",
			ResourceVersion: "4",
			Annotations: map[string]string{
				"name": "bar",
			},
		},
		DriverName: "my-driver",
		Parameters: map[string]string{
			"param1": "value1",
			"param2": "value2",
		},
	})
	s := VolumeAttributesClassDescriber{f}
	out, err := s.Describe("", "foo", DescriberSettings{ShowEvents: true})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if out != expectedOut {
		t.Errorf("expected:\n %s\n but got output:\n %s diff:\n%s", expectedOut, out, cmp.Diff(out, expectedOut))
	}
}

func TestDescribeCSINode(t *testing.T) {
	limit := ptr.To[int32](2)
	f := fake.NewSimpleClientset(&storagev1.CSINode{
		ObjectMeta: metav1.ObjectMeta{Name: "foo"},
		Spec: storagev1.CSINodeSpec{
			Drivers: []storagev1.CSINodeDriver{
				{
					Name:   "driver1",
					NodeID: "node1",
				},
				{
					Name:        "driver2",
					NodeID:      "node2",
					Allocatable: &storagev1.VolumeNodeResources{Count: limit},
				},
			},
		},
	})
	s := CSINodeDescriber{f}
	out, err := s.Describe("", "foo", DescriberSettings{ShowEvents: true})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if !strings.Contains(out, "foo") ||
		!strings.Contains(out, "driver1") ||
		!strings.Contains(out, "node1") ||
		!strings.Contains(out, "driver2") ||
		!strings.Contains(out, "node2") {
		t.Errorf("unexpected out: %s", out)
	}
}

func TestDescribePodDisruptionBudgetV1beta1(t *testing.T) {
	minAvailable := intstr.FromInt32(22)
	f := fake.NewSimpleClientset(&policyv1beta1.PodDisruptionBudget{
		ObjectMeta: metav1.ObjectMeta{
			Namespace:         "ns1",
			Name:              "pdb1",
			CreationTimestamp: metav1.Time{Time: time.Now().Add(1.9e9)},
		},
		Spec: policyv1beta1.PodDisruptionBudgetSpec{
			MinAvailable: &minAvailable,
		},
		Status: policyv1beta1.PodDisruptionBudgetStatus{
			DisruptionsAllowed: 5,
		},
	})
	s := PodDisruptionBudgetDescriber{f}
	out, err := s.Describe("ns1", "pdb1", DescriberSettings{ShowEvents: true})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if !strings.Contains(out, "pdb1") ||
		!strings.Contains(out, "ns1") ||
		!strings.Contains(out, "22") ||
		!strings.Contains(out, "5") {
		t.Errorf("unexpected out: %s", out)
	}
}

func TestDescribePodDisruptionBudgetV1(t *testing.T) {
	minAvailable := intstr.FromInt32(22)
	f := fake.NewSimpleClientset(&policyv1.PodDisruptionBudget{
		ObjectMeta: metav1.ObjectMeta{
			Namespace:         "ns1",
			Name:              "pdb1",
			CreationTimestamp: metav1.Time{Time: time.Now().Add(1.9e9)},
		},
		Spec: policyv1.PodDisruptionBudgetSpec{
			MinAvailable: &minAvailable,
		},
		Status: policyv1.PodDisruptionBudgetStatus{
			DisruptionsAllowed: 5,
		},
	})
	s := PodDisruptionBudgetDescriber{f}
	out, err := s.Describe("ns1", "pdb1", DescriberSettings{ShowEvents: true})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if !strings.Contains(out, "pdb1") ||
		!strings.Contains(out, "ns1") ||
		!strings.Contains(out, "22") ||
		!strings.Contains(out, "5") {
		t.Errorf("unexpected out: %s", out)
	}
}

func TestDescribeHorizontalPodAutoscaler(t *testing.T) {
	minReplicasVal := int32(2)
	targetUtilizationVal := int32(80)
	currentUtilizationVal := int32(50)
	maxSelectPolicy := autoscalingv2.MaxChangePolicySelect
	metricLabelSelector, err := metav1.ParseToLabelSelector("label=value")
	if err != nil {
		t.Errorf("unable to parse label selector: %v", err)
	}
	testsv2 := []struct {
		name string
		hpa  autoscalingv2.HorizontalPodAutoscaler
	}{
		{
			"minReplicas unset",
			autoscalingv2.HorizontalPodAutoscaler{
				Spec: autoscalingv2.HorizontalPodAutoscalerSpec{
					ScaleTargetRef: autoscalingv2.CrossVersionObjectReference{
						Name: "some-rc",
						Kind: "ReplicationController",
					},
					MaxReplicas: 10,
				},
				Status: autoscalingv2.HorizontalPodAutoscalerStatus{
					CurrentReplicas: 4,
					DesiredReplicas: 5,
				},
			},
		},
		{
			"external source type, target average value (no current)",
			autoscalingv2.HorizontalPodAutoscaler{
				Spec: autoscalingv2.HorizontalPodAutoscalerSpec{
					ScaleTargetRef: autoscalingv2.CrossVersionObjectReference{
						Name: "some-rc",
						Kind: "ReplicationController",
					},
					MinReplicas: &minReplicasVal,
					MaxReplicas: 10,
					Metrics: []autoscalingv2.MetricSpec{
						{
							Type: autoscalingv2.ExternalMetricSourceType,
							External: &autoscalingv2.ExternalMetricSource{
								Metric: autoscalingv2.MetricIdentifier{
									Name:     "some-external-metric",
									Selector: metricLabelSelector,
								},
								Target: autoscalingv2.MetricTarget{
									Type:         autoscalingv2.AverageValueMetricType,
									AverageValue: resource.NewMilliQuantity(100, resource.DecimalSI),
								},
							},
						},
					},
				},
				Status: autoscalingv2.HorizontalPodAutoscalerStatus{
					CurrentReplicas: 4,
					DesiredReplicas: 5,
				},
			},
		},
		{
			"external source type, target average value (with current)",
			autoscalingv2.HorizontalPodAutoscaler{
				Spec: autoscalingv2.HorizontalPodAutoscalerSpec{
					ScaleTargetRef: autoscalingv2.CrossVersionObjectReference{
						Name: "some-rc",
						Kind: "ReplicationController",
					},
					MinReplicas: &minReplicasVal,
					MaxReplicas: 10,
					Metrics: []autoscalingv2.MetricSpec{
						{
							Type: autoscalingv2.ExternalMetricSourceType,
							External: &autoscalingv2.ExternalMetricSource{
								Metric: autoscalingv2.MetricIdentifier{
									Name:     "some-external-metric",
									Selector: metricLabelSelector,
								},
								Target: autoscalingv2.MetricTarget{
									Type:         autoscalingv2.AverageValueMetricType,
									AverageValue: resource.NewMilliQuantity(100, resource.DecimalSI),
								},
							},
						},
					},
				},
				Status: autoscalingv2.HorizontalPodAutoscalerStatus{
					CurrentReplicas: 4,
					DesiredReplicas: 5,
					CurrentMetrics: []autoscalingv2.MetricStatus{
						{
							Type: autoscalingv2.ExternalMetricSourceType,
							External: &autoscalingv2.ExternalMetricStatus{
								Metric: autoscalingv2.MetricIdentifier{
									Name:     "some-external-metric",
									Selector: metricLabelSelector,
								},
								Current: autoscalingv2.MetricValueStatus{
									AverageValue: resource.NewMilliQuantity(50, resource.DecimalSI),
								},
							},
						},
					},
				},
			},
		},
		{
			"external source type, target value (no current)",
			autoscalingv2.HorizontalPodAutoscaler{
				Spec: autoscalingv2.HorizontalPodAutoscalerSpec{
					ScaleTargetRef: autoscalingv2.CrossVersionObjectReference{
						Name: "some-rc",
						Kind: "ReplicationController",
					},
					MinReplicas: &minReplicasVal,
					MaxReplicas: 10,
					Metrics: []autoscalingv2.MetricSpec{
						{
							Type: autoscalingv2.ExternalMetricSourceType,
							External: &autoscalingv2.ExternalMetricSource{
								Metric: autoscalingv2.MetricIdentifier{
									Name:     "some-external-metric",
									Selector: metricLabelSelector,
								},
								Target: autoscalingv2.MetricTarget{
									Type:  autoscalingv2.ValueMetricType,
									Value: resource.NewMilliQuantity(100, resource.DecimalSI),
								},
							},
						},
					},
				},
				Status: autoscalingv2.HorizontalPodAutoscalerStatus{
					CurrentReplicas: 4,
					DesiredReplicas: 5,
				},
			},
		},
		{
			"external source type, target value (with current)",
			autoscalingv2.HorizontalPodAutoscaler{
				Spec: autoscalingv2.HorizontalPodAutoscalerSpec{
					ScaleTargetRef: autoscalingv2.CrossVersionObjectReference{
						Name: "some-rc",
						Kind: "ReplicationController",
					},
					MinReplicas: &minReplicasVal,
					MaxReplicas: 10,
					Metrics: []autoscalingv2.MetricSpec{
						{
							Type: autoscalingv2.ExternalMetricSourceType,
							External: &autoscalingv2.ExternalMetricSource{
								Metric: autoscalingv2.MetricIdentifier{
									Name:     "some-external-metric",
									Selector: metricLabelSelector,
								},
								Target: autoscalingv2.MetricTarget{
									Type:  autoscalingv2.ValueMetricType,
									Value: resource.NewMilliQuantity(100, resource.DecimalSI),
								},
							},
						},
					},
				},
				Status: autoscalingv2.HorizontalPodAutoscalerStatus{
					CurrentReplicas: 4,
					DesiredReplicas: 5,
					CurrentMetrics: []autoscalingv2.MetricStatus{
						{
							Type: autoscalingv2.ExternalMetricSourceType,
							External: &autoscalingv2.ExternalMetricStatus{
								Metric: autoscalingv2.MetricIdentifier{
									Name:     "some-external-metric",
									Selector: metricLabelSelector,
								},
								Current: autoscalingv2.MetricValueStatus{
									Value: resource.NewMilliQuantity(50, resource.DecimalSI),
								},
							},
						},
					},
				},
			},
		},
		{
			"pods source type (no current)",
			autoscalingv2.HorizontalPodAutoscaler{
				Spec: autoscalingv2.HorizontalPodAutoscalerSpec{
					ScaleTargetRef: autoscalingv2.CrossVersionObjectReference{
						Name: "some-rc",
						Kind: "ReplicationController",
					},
					MinReplicas: &minReplicasVal,
					MaxReplicas: 10,
					Metrics: []autoscalingv2.MetricSpec{
						{
							Type: autoscalingv2.PodsMetricSourceType,
							Pods: &autoscalingv2.PodsMetricSource{
								Metric: autoscalingv2.MetricIdentifier{
									Name: "some-pods-metric",
								},
								Target: autoscalingv2.MetricTarget{
									Type:         autoscalingv2.AverageValueMetricType,
									AverageValue: resource.NewMilliQuantity(100, resource.DecimalSI),
								},
							},
						},
					},
				},
				Status: autoscalingv2.HorizontalPodAutoscalerStatus{
					CurrentReplicas: 4,
					DesiredReplicas: 5,
				},
			},
		},
		{
			"pods source type (with current)",
			autoscalingv2.HorizontalPodAutoscaler{
				Spec: autoscalingv2.HorizontalPodAutoscalerSpec{
					ScaleTargetRef: autoscalingv2.CrossVersionObjectReference{
						Name: "some-rc",
						Kind: "ReplicationController",
					},
					MinReplicas: &minReplicasVal,
					MaxReplicas: 10,
					Metrics: []autoscalingv2.MetricSpec{
						{
							Type: autoscalingv2.PodsMetricSourceType,
							Pods: &autoscalingv2.PodsMetricSource{
								Metric: autoscalingv2.MetricIdentifier{
									Name: "some-pods-metric",
								},
								Target: autoscalingv2.MetricTarget{
									Type:         autoscalingv2.AverageValueMetricType,
									AverageValue: resource.NewMilliQuantity(100, resource.DecimalSI),
								},
							},
						},
					},
				},
				Status: autoscalingv2.HorizontalPodAutoscalerStatus{
					CurrentReplicas: 4,
					DesiredReplicas: 5,
					CurrentMetrics: []autoscalingv2.MetricStatus{
						{
							Type: autoscalingv2.PodsMetricSourceType,
							Pods: &autoscalingv2.PodsMetricStatus{
								Metric: autoscalingv2.MetricIdentifier{
									Name: "some-pods-metric",
								},
								Current: autoscalingv2.MetricValueStatus{
									AverageValue: resource.NewMilliQuantity(50, resource.DecimalSI),
								},
							},
						},
					},
				},
			},
		},
		{
			"object source type target average value (no current)",
			autoscalingv2.HorizontalPodAutoscaler{
				Spec: autoscalingv2.HorizontalPodAutoscalerSpec{
					ScaleTargetRef: autoscalingv2.CrossVersionObjectReference{
						Name: "some-rc",
						Kind: "ReplicationController",
					},
					MinReplicas: &minReplicasVal,
					MaxReplicas: 10,
					Metrics: []autoscalingv2.MetricSpec{
						{
							Type: autoscalingv2.ObjectMetricSourceType,
							Object: &autoscalingv2.ObjectMetricSource{
								DescribedObject: autoscalingv2.CrossVersionObjectReference{
									Name: "some-service",
									Kind: "Service",
								},
								Metric: autoscalingv2.MetricIdentifier{
									Name: "some-service-metric",
								},
								Target: autoscalingv2.MetricTarget{
									Type:         autoscalingv2.AverageValueMetricType,
									AverageValue: resource.NewMilliQuantity(100, resource.DecimalSI),
								},
							},
						},
					},
				},
				Status: autoscalingv2.HorizontalPodAutoscalerStatus{
					CurrentReplicas: 4,
					DesiredReplicas: 5,
				},
			},
		},
		{
			"object source type target average value (with current)",
			autoscalingv2.HorizontalPodAutoscaler{
				Spec: autoscalingv2.HorizontalPodAutoscalerSpec{
					ScaleTargetRef: autoscalingv2.CrossVersionObjectReference{
						Name: "some-rc",
						Kind: "ReplicationController",
					},
					MinReplicas: &minReplicasVal,
					MaxReplicas: 10,
					Metrics: []autoscalingv2.MetricSpec{
						{
							Type: autoscalingv2.ObjectMetricSourceType,
							Object: &autoscalingv2.ObjectMetricSource{
								DescribedObject: autoscalingv2.CrossVersionObjectReference{
									Name: "some-service",
									Kind: "Service",
								},
								Metric: autoscalingv2.MetricIdentifier{
									Name: "some-service-metric",
								},
								Target: autoscalingv2.MetricTarget{
									Type:         autoscalingv2.AverageValueMetricType,
									AverageValue: resource.NewMilliQuantity(100, resource.DecimalSI),
								},
							},
						},
					},
				},
				Status: autoscalingv2.HorizontalPodAutoscalerStatus{
					CurrentReplicas: 4,
					DesiredReplicas: 5,
					CurrentMetrics: []autoscalingv2.MetricStatus{
						{
							Type: autoscalingv2.ObjectMetricSourceType,
							Object: &autoscalingv2.ObjectMetricStatus{
								DescribedObject: autoscalingv2.CrossVersionObjectReference{
									Name: "some-service",
									Kind: "Service",
								},
								Metric: autoscalingv2.MetricIdentifier{
									Name: "some-service-metric",
								},
								Current: autoscalingv2.MetricValueStatus{
									AverageValue: resource.NewMilliQuantity(50, resource.DecimalSI),
								},
							},
						},
					},
				},
			},
		},
		{
			"object source type target value (no current)",
			autoscalingv2.HorizontalPodAutoscaler{
				Spec: autoscalingv2.HorizontalPodAutoscalerSpec{
					ScaleTargetRef: autoscalingv2.CrossVersionObjectReference{
						Name: "some-rc",
						Kind: "ReplicationController",
					},
					MinReplicas: &minReplicasVal,
					MaxReplicas: 10,
					Metrics: []autoscalingv2.MetricSpec{
						{
							Type: autoscalingv2.ObjectMetricSourceType,
							Object: &autoscalingv2.ObjectMetricSource{
								DescribedObject: autoscalingv2.CrossVersionObjectReference{
									Name: "some-service",
									Kind: "Service",
								},
								Metric: autoscalingv2.MetricIdentifier{
									Name: "some-service-metric",
								},
								Target: autoscalingv2.MetricTarget{
									Type:  autoscalingv2.ValueMetricType,
									Value: resource.NewMilliQuantity(100, resource.DecimalSI),
								},
							},
						},
					},
				},
				Status: autoscalingv2.HorizontalPodAutoscalerStatus{
					CurrentReplicas: 4,
					DesiredReplicas: 5,
				},
			},
		},
		{
			"object source type target value (with current)",
			autoscalingv2.HorizontalPodAutoscaler{
				Spec: autoscalingv2.HorizontalPodAutoscalerSpec{
					ScaleTargetRef: autoscalingv2.CrossVersionObjectReference{
						Name: "some-rc",
						Kind: "ReplicationController",
					},
					MinReplicas: &minReplicasVal,
					MaxReplicas: 10,
					Metrics: []autoscalingv2.MetricSpec{
						{
							Type: autoscalingv2.ObjectMetricSourceType,
							Object: &autoscalingv2.ObjectMetricSource{
								DescribedObject: autoscalingv2.CrossVersionObjectReference{
									Name: "some-service",
									Kind: "Service",
								},
								Metric: autoscalingv2.MetricIdentifier{
									Name: "some-service-metric",
								},
								Target: autoscalingv2.MetricTarget{
									Type:  autoscalingv2.ValueMetricType,
									Value: resource.NewMilliQuantity(100, resource.DecimalSI),
								},
							},
						},
					},
				},
				Status: autoscalingv2.HorizontalPodAutoscalerStatus{
					CurrentReplicas: 4,
					DesiredReplicas: 5,
					CurrentMetrics: []autoscalingv2.MetricStatus{
						{
							Type: autoscalingv2.ObjectMetricSourceType,
							Object: &autoscalingv2.ObjectMetricStatus{
								DescribedObject: autoscalingv2.CrossVersionObjectReference{
									Name: "some-service",
									Kind: "Service",
								},
								Metric: autoscalingv2.MetricIdentifier{
									Name: "some-service-metric",
								},
								Current: autoscalingv2.MetricValueStatus{
									Value: resource.NewMilliQuantity(50, resource.DecimalSI),
								},
							},
						},
					},
				},
			},
		},
		{
			"resource source type, target average value (no current)",
			autoscalingv2.HorizontalPodAutoscaler{
				Spec: autoscalingv2.HorizontalPodAutoscalerSpec{
					ScaleTargetRef: autoscalingv2.CrossVersionObjectReference{
						Name: "some-rc",
						Kind: "ReplicationController",
					},
					MinReplicas: &minReplicasVal,
					MaxReplicas: 10,
					Metrics: []autoscalingv2.MetricSpec{
						{
							Type: autoscalingv2.ResourceMetricSourceType,
							Resource: &autoscalingv2.ResourceMetricSource{
								Name: corev1.ResourceCPU,
								Target: autoscalingv2.MetricTarget{
									Type:         autoscalingv2.AverageValueMetricType,
									AverageValue: resource.NewMilliQuantity(100, resource.DecimalSI),
								},
							},
						},
					},
				},
				Status: autoscalingv2.HorizontalPodAutoscalerStatus{
					CurrentReplicas: 4,
					DesiredReplicas: 5,
				},
			},
		},
		{
			"resource source type, target average value (with current)",
			autoscalingv2.HorizontalPodAutoscaler{
				Spec: autoscalingv2.HorizontalPodAutoscalerSpec{
					ScaleTargetRef: autoscalingv2.CrossVersionObjectReference{
						Name: "some-rc",
						Kind: "ReplicationController",
					},
					MinReplicas: &minReplicasVal,
					MaxReplicas: 10,
					Metrics: []autoscalingv2.MetricSpec{
						{
							Type: autoscalingv2.ResourceMetricSourceType,
							Resource: &autoscalingv2.ResourceMetricSource{
								Name: corev1.ResourceCPU,
								Target: autoscalingv2.MetricTarget{
									Type:         autoscalingv2.AverageValueMetricType,
									AverageValue: resource.NewMilliQuantity(100, resource.DecimalSI),
								},
							},
						},
					},
				},
				Status: autoscalingv2.HorizontalPodAutoscalerStatus{
					CurrentReplicas: 4,
					DesiredReplicas: 5,
					CurrentMetrics: []autoscalingv2.MetricStatus{
						{
							Type: autoscalingv2.ResourceMetricSourceType,
							Resource: &autoscalingv2.ResourceMetricStatus{
								Name: corev1.ResourceCPU,
								Current: autoscalingv2.MetricValueStatus{
									AverageValue: resource.NewMilliQuantity(50, resource.DecimalSI),
								},
							},
						},
					},
				},
			},
		},
		{
			"resource source type, target utilization (no current)",
			autoscalingv2.HorizontalPodAutoscaler{
				Spec: autoscalingv2.HorizontalPodAutoscalerSpec{
					ScaleTargetRef: autoscalingv2.CrossVersionObjectReference{
						Name: "some-rc",
						Kind: "ReplicationController",
					},
					MinReplicas: &minReplicasVal,
					MaxReplicas: 10,
					Metrics: []autoscalingv2.MetricSpec{
						{
							Type: autoscalingv2.ResourceMetricSourceType,
							Resource: &autoscalingv2.ResourceMetricSource{
								Name: corev1.ResourceCPU,
								Target: autoscalingv2.MetricTarget{
									Type:               autoscalingv2.UtilizationMetricType,
									AverageUtilization: &targetUtilizationVal,
								},
							},
						},
					},
				},
				Status: autoscalingv2.HorizontalPodAutoscalerStatus{
					CurrentReplicas: 4,
					DesiredReplicas: 5,
				},
			},
		},
		{
			"resource source type, target utilization (with current)",
			autoscalingv2.HorizontalPodAutoscaler{
				Spec: autoscalingv2.HorizontalPodAutoscalerSpec{
					ScaleTargetRef: autoscalingv2.CrossVersionObjectReference{
						Name: "some-rc",
						Kind: "ReplicationController",
					},
					MinReplicas: &minReplicasVal,
					MaxReplicas: 10,
					Metrics: []autoscalingv2.MetricSpec{
						{
							Type: autoscalingv2.ResourceMetricSourceType,
							Resource: &autoscalingv2.ResourceMetricSource{
								Name: corev1.ResourceCPU,
								Target: autoscalingv2.MetricTarget{
									Type:               autoscalingv2.UtilizationMetricType,
									AverageUtilization: &targetUtilizationVal,
								},
							},
						},
					},
				},
				Status: autoscalingv2.HorizontalPodAutoscalerStatus{
					CurrentReplicas: 4,
					DesiredReplicas: 5,
					CurrentMetrics: []autoscalingv2.MetricStatus{
						{
							Type: autoscalingv2.ResourceMetricSourceType,
							Resource: &autoscalingv2.ResourceMetricStatus{
								Name: corev1.ResourceCPU,
								Current: autoscalingv2.MetricValueStatus{
									AverageUtilization: &currentUtilizationVal,
									AverageValue:       resource.NewMilliQuantity(40, resource.DecimalSI),
								},
							},
						},
					},
				},
			},
		},
		{
			"container resource source type, target average value (no current)",
			autoscalingv2.HorizontalPodAutoscaler{
				Spec: autoscalingv2.HorizontalPodAutoscalerSpec{
					ScaleTargetRef: autoscalingv2.CrossVersionObjectReference{
						Name: "some-rc",
						Kind: "ReplicationController",
					},
					MinReplicas: &minReplicasVal,
					MaxReplicas: 10,
					Metrics: []autoscalingv2.MetricSpec{
						{
							Type: autoscalingv2.ContainerResourceMetricSourceType,
							ContainerResource: &autoscalingv2.ContainerResourceMetricSource{
								Name:      corev1.ResourceCPU,
								Container: "application",
								Target: autoscalingv2.MetricTarget{
									Type:         autoscalingv2.AverageValueMetricType,
									AverageValue: resource.NewMilliQuantity(100, resource.DecimalSI),
								},
							},
						},
					},
				},
				Status: autoscalingv2.HorizontalPodAutoscalerStatus{
					CurrentReplicas: 4,
					DesiredReplicas: 5,
				},
			},
		},
		{
			"container resource source type, target average value (with current)",
			autoscalingv2.HorizontalPodAutoscaler{
				Spec: autoscalingv2.HorizontalPodAutoscalerSpec{
					ScaleTargetRef: autoscalingv2.CrossVersionObjectReference{
						Name: "some-rc",
						Kind: "ReplicationController",
					},
					MinReplicas: &minReplicasVal,
					MaxReplicas: 10,
					Metrics: []autoscalingv2.MetricSpec{
						{
							Type: autoscalingv2.ContainerResourceMetricSourceType,
							ContainerResource: &autoscalingv2.ContainerResourceMetricSource{
								Name:      corev1.ResourceCPU,
								Container: "application",
								Target: autoscalingv2.MetricTarget{
									Type:         autoscalingv2.AverageValueMetricType,
									AverageValue: resource.NewMilliQuantity(100, resource.DecimalSI),
								},
							},
						},
					},
				},
				Status: autoscalingv2.HorizontalPodAutoscalerStatus{
					CurrentReplicas: 4,
					DesiredReplicas: 5,
					CurrentMetrics: []autoscalingv2.MetricStatus{
						{
							Type: autoscalingv2.ContainerResourceMetricSourceType,
							ContainerResource: &autoscalingv2.ContainerResourceMetricStatus{
								Name:      corev1.ResourceCPU,
								Container: "application",
								Current: autoscalingv2.MetricValueStatus{
									AverageValue: resource.NewMilliQuantity(50, resource.DecimalSI),
								},
							},
						},
					},
				},
			},
		},
		{
			"container resource source type, target utilization (no current)",
			autoscalingv2.HorizontalPodAutoscaler{
				Spec: autoscalingv2.HorizontalPodAutoscalerSpec{
					ScaleTargetRef: autoscalingv2.CrossVersionObjectReference{
						Name: "some-rc",
						Kind: "ReplicationController",
					},
					MinReplicas: &minReplicasVal,
					MaxReplicas: 10,
					Metrics: []autoscalingv2.MetricSpec{
						{
							Type: autoscalingv2.ContainerResourceMetricSourceType,
							ContainerResource: &autoscalingv2.ContainerResourceMetricSource{
								Name:      corev1.ResourceCPU,
								Container: "application",
								Target: autoscalingv2.MetricTarget{
									Type:               autoscalingv2.UtilizationMetricType,
									AverageUtilization: &targetUtilizationVal,
								},
							},
						},
					},
				},
				Status: autoscalingv2.HorizontalPodAutoscalerStatus{
					CurrentReplicas: 4,
					DesiredReplicas: 5,
				},
			},
		},
		{
			"container resource source type, target utilization (with current)",
			autoscalingv2.HorizontalPodAutoscaler{
				Spec: autoscalingv2.HorizontalPodAutoscalerSpec{
					ScaleTargetRef: autoscalingv2.CrossVersionObjectReference{
						Name: "some-rc",
						Kind: "ReplicationController",
					},
					MinReplicas: &minReplicasVal,
					MaxReplicas: 10,
					Metrics: []autoscalingv2.MetricSpec{
						{
							Type: autoscalingv2.ContainerResourceMetricSourceType,
							ContainerResource: &autoscalingv2.ContainerResourceMetricSource{
								Name:      corev1.ResourceCPU,
								Container: "application",
								Target: autoscalingv2.MetricTarget{
									Type:               autoscalingv2.UtilizationMetricType,
									AverageUtilization: &targetUtilizationVal,
								},
							},
						},
					},
				},
				Status: autoscalingv2.HorizontalPodAutoscalerStatus{
					CurrentReplicas: 4,
					DesiredReplicas: 5,
					CurrentMetrics: []autoscalingv2.MetricStatus{
						{
							Type: autoscalingv2.ContainerResourceMetricSourceType,
							ContainerResource: &autoscalingv2.ContainerResourceMetricStatus{
								Name:      corev1.ResourceCPU,
								Container: "application",
								Current: autoscalingv2.MetricValueStatus{
									AverageUtilization: &currentUtilizationVal,
									AverageValue:       resource.NewMilliQuantity(40, resource.DecimalSI),
								},
							},
						},
					},
				},
			},
		},

		{
			"multiple metrics",
			autoscalingv2.HorizontalPodAutoscaler{
				Spec: autoscalingv2.HorizontalPodAutoscalerSpec{
					ScaleTargetRef: autoscalingv2.CrossVersionObjectReference{
						Name: "some-rc",
						Kind: "ReplicationController",
					},
					MinReplicas: &minReplicasVal,
					MaxReplicas: 10,
					Metrics: []autoscalingv2.MetricSpec{
						{
							Type: autoscalingv2.PodsMetricSourceType,
							Pods: &autoscalingv2.PodsMetricSource{
								Metric: autoscalingv2.MetricIdentifier{
									Name: "some-pods-metric",
								},
								Target: autoscalingv2.MetricTarget{
									Type:         autoscalingv2.AverageValueMetricType,
									AverageValue: resource.NewMilliQuantity(100, resource.DecimalSI),
								},
							},
						},
						{
							Type: autoscalingv2.ResourceMetricSourceType,
							Resource: &autoscalingv2.ResourceMetricSource{
								Name: corev1.ResourceCPU,
								Target: autoscalingv2.MetricTarget{
									Type:               autoscalingv2.UtilizationMetricType,
									AverageUtilization: &targetUtilizationVal,
								},
							},
						},
						{
							Type: autoscalingv2.PodsMetricSourceType,
							Pods: &autoscalingv2.PodsMetricSource{
								Metric: autoscalingv2.MetricIdentifier{
									Name: "other-pods-metric",
								},
								Target: autoscalingv2.MetricTarget{
									Type:         autoscalingv2.AverageValueMetricType,
									AverageValue: resource.NewMilliQuantity(400, resource.DecimalSI),
								},
							},
						},
					},
				},
				Status: autoscalingv2.HorizontalPodAutoscalerStatus{
					CurrentReplicas: 4,
					DesiredReplicas: 5,
					CurrentMetrics: []autoscalingv2.MetricStatus{
						{
							Type: autoscalingv2.PodsMetricSourceType,
							Pods: &autoscalingv2.PodsMetricStatus{
								Metric: autoscalingv2.MetricIdentifier{
									Name: "some-pods-metric",
								},
								Current: autoscalingv2.MetricValueStatus{
									AverageValue: resource.NewMilliQuantity(50, resource.DecimalSI),
								},
							},
						},
						{
							Type: autoscalingv2.ResourceMetricSourceType,
							Resource: &autoscalingv2.ResourceMetricStatus{
								Name: corev1.ResourceCPU,
								Current: autoscalingv2.MetricValueStatus{
									AverageUtilization: &currentUtilizationVal,
									AverageValue:       resource.NewMilliQuantity(40, resource.DecimalSI),
								},
							},
						},
					},
				},
			},
		},
		{
			"scale up behavior specified",
			autoscalingv2.HorizontalPodAutoscaler{
				Spec: autoscalingv2.HorizontalPodAutoscalerSpec{
					ScaleTargetRef: autoscalingv2.CrossVersionObjectReference{
						Name: "behavior-target",
						Kind: "Deployment",
					},
					MinReplicas: &minReplicasVal,
					MaxReplicas: 10,
					Metrics: []autoscalingv2.MetricSpec{
						{
							Type: autoscalingv2.ResourceMetricSourceType,
							Resource: &autoscalingv2.ResourceMetricSource{
								Name: corev1.ResourceCPU,
								Target: autoscalingv2.MetricTarget{
									Type:               autoscalingv2.UtilizationMetricType,
									AverageUtilization: &targetUtilizationVal,
								},
							},
						},
					},
					Behavior: &autoscalingv2.HorizontalPodAutoscalerBehavior{
						ScaleUp: &autoscalingv2.HPAScalingRules{
							StabilizationWindowSeconds: ptr.To[int32](30),
							SelectPolicy:               &maxSelectPolicy,
							Policies: []autoscalingv2.HPAScalingPolicy{
								{Type: autoscalingv2.PodsScalingPolicy, Value: 10, PeriodSeconds: 10},
								{Type: autoscalingv2.PercentScalingPolicy, Value: 10, PeriodSeconds: 10},
							},
						},
					},
				},
			},
		},
		{
			"scale down behavior specified",
			autoscalingv2.HorizontalPodAutoscaler{
				Spec: autoscalingv2.HorizontalPodAutoscalerSpec{
					ScaleTargetRef: autoscalingv2.CrossVersionObjectReference{
						Name: "behavior-target",
						Kind: "Deployment",
					},
					MinReplicas: &minReplicasVal,
					MaxReplicas: 10,
					Metrics: []autoscalingv2.MetricSpec{
						{
							Type: autoscalingv2.ResourceMetricSourceType,
							Resource: &autoscalingv2.ResourceMetricSource{
								Name: corev1.ResourceCPU,
								Target: autoscalingv2.MetricTarget{
									Type:               autoscalingv2.UtilizationMetricType,
									AverageUtilization: &targetUtilizationVal,
								},
							},
						},
					},
					Behavior: &autoscalingv2.HorizontalPodAutoscalerBehavior{
						ScaleDown: &autoscalingv2.HPAScalingRules{
							StabilizationWindowSeconds: ptr.To[int32](30),
							Policies: []autoscalingv2.HPAScalingPolicy{
								{Type: autoscalingv2.PodsScalingPolicy, Value: 10, PeriodSeconds: 10},
								{Type: autoscalingv2.PercentScalingPolicy, Value: 10, PeriodSeconds: 10},
							},
						},
					},
				},
			},
		},
	}

	for _, test := range testsv2 {
		t.Run(test.name, func(t *testing.T) {
			test.hpa.ObjectMeta = metav1.ObjectMeta{
				Name:      "bar",
				Namespace: "foo",
			}
			fake := fake.NewSimpleClientset(&test.hpa)
			desc := HorizontalPodAutoscalerDescriber{fake}
			str, err := desc.Describe("foo", "bar", DescriberSettings{ShowEvents: true})
			if err != nil {
				t.Errorf("Unexpected error for test %s: %v", test.name, err)
			}
			if str == "" {
				t.Errorf("Unexpected empty string for test %s.  Expected HPA Describer output", test.name)
			}
			t.Logf("Description for %q:\n%s", test.name, str)
		})
	}

	testsV1 := []struct {
		name string
		hpa  autoscalingv1.HorizontalPodAutoscaler
	}{
		{
			"minReplicas unset",
			autoscalingv1.HorizontalPodAutoscaler{
				Spec: autoscalingv1.HorizontalPodAutoscalerSpec{
					ScaleTargetRef: autoscalingv1.CrossVersionObjectReference{
						Name: "some-rc",
						Kind: "ReplicationController",
					},
					MaxReplicas: 10,
				},
				Status: autoscalingv1.HorizontalPodAutoscalerStatus{
					CurrentReplicas: 4,
					DesiredReplicas: 5,
				},
			},
		},
		{
			"minReplicas set",
			autoscalingv1.HorizontalPodAutoscaler{
				Spec: autoscalingv1.HorizontalPodAutoscalerSpec{
					ScaleTargetRef: autoscalingv1.CrossVersionObjectReference{
						Name: "some-rc",
						Kind: "ReplicationController",
					},
					MinReplicas: &minReplicasVal,
					MaxReplicas: 10,
				},
				Status: autoscalingv1.HorizontalPodAutoscalerStatus{
					CurrentReplicas: 4,
					DesiredReplicas: 5,
				},
			},
		},
		{
			"with target no current",
			autoscalingv1.HorizontalPodAutoscaler{
				Spec: autoscalingv1.HorizontalPodAutoscalerSpec{
					ScaleTargetRef: autoscalingv1.CrossVersionObjectReference{
						Name: "some-rc",
						Kind: "ReplicationController",
					},
					MinReplicas:                    &minReplicasVal,
					MaxReplicas:                    10,
					TargetCPUUtilizationPercentage: &targetUtilizationVal,
				},
				Status: autoscalingv1.HorizontalPodAutoscalerStatus{
					CurrentReplicas: 4,
					DesiredReplicas: 5,
				},
			},
		},
		{
			"with target and current",
			autoscalingv1.HorizontalPodAutoscaler{
				Spec: autoscalingv1.HorizontalPodAutoscalerSpec{
					ScaleTargetRef: autoscalingv1.CrossVersionObjectReference{
						Name: "some-rc",
						Kind: "ReplicationController",
					},
					MinReplicas:                    &minReplicasVal,
					MaxReplicas:                    10,
					TargetCPUUtilizationPercentage: &targetUtilizationVal,
				},
				Status: autoscalingv1.HorizontalPodAutoscalerStatus{
					CurrentReplicas:                 4,
					DesiredReplicas:                 5,
					CurrentCPUUtilizationPercentage: &currentUtilizationVal,
				},
			},
		},
	}

	for _, test := range testsV1 {
		t.Run(test.name, func(t *testing.T) {
			test.hpa.ObjectMeta = metav1.ObjectMeta{
				Name:      "bar",
				Namespace: "foo",
			}
			fake := fake.NewSimpleClientset(&test.hpa)
			desc := HorizontalPodAutoscalerDescriber{fake}
			str, err := desc.Describe("foo", "bar", DescriberSettings{ShowEvents: true})
			if err != nil {
				t.Errorf("Unexpected error for test %s: %v", test.name, err)
			}
			if str == "" {
				t.Errorf("Unexpected empty string for test %s.  Expected HPA Describer output", test.name)
			}
			t.Logf("Description for %q:\n%s", test.name, str)
		})
	}
}

func TestDescribeEvents(t *testing.T) {

	events := &corev1.EventList{
		Items: []corev1.Event{
			{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "event-1",
					Namespace: "foo",
				},
				Source:         corev1.EventSource{Component: "kubelet"},
				Message:        "Item 1",
				FirstTimestamp: metav1.NewTime(time.Date(2014, time.January, 15, 0, 0, 0, 0, time.UTC)),
				LastTimestamp:  metav1.NewTime(time.Date(2014, time.January, 15, 0, 0, 0, 0, time.UTC)),
				Count:          1,
				Type:           corev1.EventTypeNormal,
			},
			{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "event-2",
					Namespace: "foo",
				},
				Source:    corev1.EventSource{Component: "kubelet"},
				Message:   "Item 1",
				EventTime: metav1.NewMicroTime(time.Date(2014, time.January, 15, 0, 0, 0, 0, time.UTC)),
				Series: &corev1.EventSeries{
					Count:            1,
					LastObservedTime: metav1.NewMicroTime(time.Date(2014, time.January, 15, 0, 0, 0, 0, time.UTC)),
				},
				Type: corev1.EventTypeNormal,
			},
		},
	}

	m := map[string]ResourceDescriber{
		"DaemonSetDescriber": &DaemonSetDescriber{
			fake.NewSimpleClientset(&appsv1.DaemonSet{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "bar",
					Namespace: "foo",
				},
			}, events),
		},
		"DeploymentDescriber": &DeploymentDescriber{
			fake.NewSimpleClientset(&appsv1.Deployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "bar",
					Namespace: "foo",
				},
				Spec: appsv1.DeploymentSpec{
					Replicas: ptr.To[int32](1),
					Selector: &metav1.LabelSelector{},
				},
			}, events),
		},
		"EndpointsDescriber": &EndpointsDescriber{
			fake.NewSimpleClientset(&corev1.Endpoints{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "bar",
					Namespace: "foo",
				},
			}, events),
		},
		"EndpointSliceDescriber": &EndpointSliceDescriber{
			fake.NewSimpleClientset(&discoveryv1beta1.EndpointSlice{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "bar",
					Namespace: "foo",
				},
			}, events),
		},
		"JobDescriber": &JobDescriber{
			fake.NewSimpleClientset(&batchv1.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "bar",
					Namespace: "foo",
				},
			}, events),
		},
		"IngressDescriber": &IngressDescriber{
			fake.NewSimpleClientset(&networkingv1beta1.Ingress{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "bar",
					Namespace: "foo",
				},
			}, events),
		},
		"NodeDescriber": &NodeDescriber{
			fake.NewSimpleClientset(&corev1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name: "bar",
				},
			}, events),
		},
		"PersistentVolumeDescriber": &PersistentVolumeDescriber{
			fake.NewSimpleClientset(&corev1.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name: "bar",
				},
			}, events),
		},
		"PodDescriber": &PodDescriber{
			fake.NewSimpleClientset(&corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "bar",
					Namespace: "foo",
				},
			}, events),
		},
		"ReplicaSetDescriber": &ReplicaSetDescriber{
			fake.NewSimpleClientset(&appsv1.ReplicaSet{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "bar",
					Namespace: "foo",
				},
				Spec: appsv1.ReplicaSetSpec{
					Replicas: ptr.To[int32](1),
				},
			}, events),
		},
		"ReplicationControllerDescriber": &ReplicationControllerDescriber{
			fake.NewSimpleClientset(&corev1.ReplicationController{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "bar",
					Namespace: "foo",
				},
				Spec: corev1.ReplicationControllerSpec{
					Replicas: ptr.To[int32](1),
				},
			}, events),
		},
		"Service": &ServiceDescriber{
			fake.NewSimpleClientset(&corev1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "bar",
					Namespace: "foo",
				},
			}, events),
		},
		"StorageClass": &StorageClassDescriber{
			fake.NewSimpleClientset(&storagev1.StorageClass{
				ObjectMeta: metav1.ObjectMeta{
					Name: "bar",
				},
			}, events),
		},
		"HorizontalPodAutoscaler": &HorizontalPodAutoscalerDescriber{
			fake.NewSimpleClientset(&autoscalingv2.HorizontalPodAutoscaler{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "bar",
					Namespace: "foo",
				},
			}, events),
		},
		"ConfigMap": &ConfigMapDescriber{
			fake.NewSimpleClientset(&corev1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "bar",
					Namespace: "foo",
				},
			}, events),
		},
	}

	for name, d := range m {
		t.Run(name, func(t *testing.T) {
			out, err := d.Describe("foo", "bar", DescriberSettings{ShowEvents: true})
			if err != nil {
				t.Errorf("unexpected error for %q: %v", name, err)
			}
			if !strings.Contains(out, "bar") {
				t.Errorf("unexpected out for %q: %s", name, out)
			}
			if !strings.Contains(out, "Events:") {
				t.Errorf("events not found for %q when ShowEvents=true: %s", name, out)
			}

			out, err = d.Describe("foo", "bar", DescriberSettings{ShowEvents: false})
			if err != nil {
				t.Errorf("unexpected error for %q: %s", name, err)
			}
			if !strings.Contains(out, "bar") {
				t.Errorf("unexpected out for %q: %s", name, out)
			}
			if strings.Contains(out, "Events:") {
				t.Errorf("events found for %q when ShowEvents=false: %s", name, out)
			}
		})
	}
}

func TestPrintLabelsMultiline(t *testing.T) {
	key := "MaxLenAnnotation"
	value := strings.Repeat("a", maxAnnotationLen-len(key)-2)
	testCases := []struct {
		annotations map[string]string
		expectPrint string
	}{
		{
			annotations: map[string]string{"col1": "asd", "COL2": "zxc"},
			expectPrint: "Annotations:\tCOL2: zxc\n\tcol1: asd\n",
		},
		{
			annotations: map[string]string{"MaxLenAnnotation": value},
			expectPrint: fmt.Sprintf("Annotations:\t%s: %s\n", key, value),
		},
		{
			annotations: map[string]string{"MaxLenAnnotation": value + "1"},
			expectPrint: fmt.Sprintf("Annotations:\t%s:\n\t  %s\n", key, value+"1"),
		},
		{
			annotations: map[string]string{"MaxLenAnnotation": value + value},
			expectPrint: fmt.Sprintf("Annotations:\t%s:\n\t  %s\n", key, strings.Repeat("a", maxAnnotationLen-2)+"..."),
		},
		{
			annotations: map[string]string{"key": "value\nwith\nnewlines\n"},
			expectPrint: "Annotations:\tkey:\n\t  value\n\t  with\n\t  newlines\n",
		},
		{
			annotations: map[string]string{},
			expectPrint: "Annotations:\t<none>\n",
		},
	}
	for i, testCase := range testCases {
		t.Run(fmt.Sprintf("%d", i), func(t *testing.T) {
			out := new(bytes.Buffer)
			writer := NewPrefixWriter(out)
			printAnnotationsMultiline(writer, "Annotations", testCase.annotations)
			output := out.String()
			if output != testCase.expectPrint {
				t.Errorf("Test case %d: expected to match:\n%q\nin output:\n%q", i, testCase.expectPrint, output)
			}
		})
	}
}

func TestDescribeUnstructuredContent(t *testing.T) {
	testCases := []struct {
		expected   string
		unexpected string
	}{
		{
			expected: `API Version:	v1
Dummy - Dummy:	present
dummy-dummy@dummy:	present
dummy/dummy:	present
dummy2:	present
Dummy Dummy:	present
Items:
  Item Bool:	true
  Item Int:	42
Kind:	Test
Metadata:
  Creation Timestamp:	2017-04-01T00:00:00Z
  Name:	MyName
  Namespace:	MyNamespace
  Resource Version:	123
  UID:	00000000-0000-0000-0000-000000000001
Status:	ok
URL:	http://localhost
`,
		},
		{
			unexpected: "\nDummy 1:\tpresent\n",
		},
		{
			unexpected: "Dummy 1",
		},
		{
			unexpected: "Dummy 3",
		},
		{
			unexpected: "Dummy3",
		},
		{
			unexpected: "dummy3",
		},
		{
			unexpected: "dummy 3",
		},
	}
	out := new(bytes.Buffer)
	w := NewPrefixWriter(out)
	obj := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion":        "v1",
			"kind":              "Test",
			"dummyDummy":        "present",
			"dummy/dummy":       "present",
			"dummy-dummy@dummy": "present",
			"dummy-dummy":       "present",
			"dummy1":            "present",
			"dummy2":            "present",
			"metadata": map[string]interface{}{
				"name":              "MyName",
				"namespace":         "MyNamespace",
				"creationTimestamp": "2017-04-01T00:00:00Z",
				"resourceVersion":   123,
				"uid":               "00000000-0000-0000-0000-000000000001",
				"dummy3":            "present",
			},
			"items": []interface{}{
				map[string]interface{}{
					"itemBool": true,
					"itemInt":  42,
				},
			},
			"url":    "http://localhost",
			"status": "ok",
		},
	}
	printUnstructuredContent(w, LEVEL_0, obj.UnstructuredContent(), "", ".dummy1", ".metadata.dummy3")
	output := out.String()

	for _, test := range testCases {
		if len(test.expected) > 0 {
			if !strings.Contains(output, test.expected) {
				t.Errorf("Expected to find %q in: %q", test.expected, output)
			}
		}
		if len(test.unexpected) > 0 {
			if strings.Contains(output, test.unexpected) {
				t.Errorf("Didn't expect to find %q in: %q", test.unexpected, output)
			}
		}
	}
}

func TestDescribeResourceQuota(t *testing.T) {
	fake := fake.NewSimpleClientset(&corev1.ResourceQuota{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "bar",
			Namespace: "foo",
		},
		Status: corev1.ResourceQuotaStatus{
			Hard: corev1.ResourceList{
				corev1.ResourceName(corev1.ResourceCPU):            resource.MustParse("1"),
				corev1.ResourceName(corev1.ResourceLimitsCPU):      resource.MustParse("2"),
				corev1.ResourceName(corev1.ResourceLimitsMemory):   resource.MustParse("2G"),
				corev1.ResourceName(corev1.ResourceMemory):         resource.MustParse("1G"),
				corev1.ResourceName(corev1.ResourceRequestsCPU):    resource.MustParse("1"),
				corev1.ResourceName(corev1.ResourceRequestsMemory): resource.MustParse("1G"),
			},
			Used: corev1.ResourceList{
				corev1.ResourceName(corev1.ResourceCPU):            resource.MustParse("0"),
				corev1.ResourceName(corev1.ResourceLimitsCPU):      resource.MustParse("0"),
				corev1.ResourceName(corev1.ResourceLimitsMemory):   resource.MustParse("0G"),
				corev1.ResourceName(corev1.ResourceMemory):         resource.MustParse("0G"),
				corev1.ResourceName(corev1.ResourceRequestsCPU):    resource.MustParse("0"),
				corev1.ResourceName(corev1.ResourceRequestsMemory): resource.MustParse("1000Ki"),
			},
		},
	})
	c := &describeClient{T: t, Namespace: "foo", Interface: fake}
	d := ResourceQuotaDescriber{c}
	out, err := d.Describe("foo", "bar", DescriberSettings{ShowEvents: true})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	expectedOut := []string{"bar", "foo", "limits.cpu", "2", "limits.memory", "2G", "requests.cpu", "1", "requests.memory", "1024k", "1G"}
	for _, expected := range expectedOut {
		if !strings.Contains(out, expected) {
			t.Errorf("expected to find %q in output: %q", expected, out)
		}
	}
}

func TestDescribeIngressClass(t *testing.T) {
	expectedOut := `Name:         example-class
Labels:       <none>
Annotations:  <none>
Controller:   example.com/controller
Parameters:
  APIGroup:  v1
  Kind:      ConfigMap
  Name:      example-parameters` + "\n"

	tests := map[string]struct {
		input  *fake.Clientset
		output string
	}{
		"basic IngressClass (v1beta1)": {
			input: fake.NewSimpleClientset(&networkingv1beta1.IngressClass{
				ObjectMeta: metav1.ObjectMeta{
					Name: "example-class",
				},
				Spec: networkingv1beta1.IngressClassSpec{
					Controller: "example.com/controller",
					Parameters: &networkingv1beta1.IngressClassParametersReference{
						APIGroup: ptr.To("v1"),
						Kind:     "ConfigMap",
						Name:     "example-parameters",
					},
				},
			}),
			output: expectedOut,
		},
		"basic IngressClass (v1)": {
			input: fake.NewSimpleClientset(&networkingv1.IngressClass{
				ObjectMeta: metav1.ObjectMeta{
					Name: "example-class",
				},
				Spec: networkingv1.IngressClassSpec{
					Controller: "example.com/controller",
					Parameters: &networkingv1.IngressClassParametersReference{
						APIGroup: ptr.To("v1"),
						Kind:     "ConfigMap",
						Name:     "example-parameters",
					},
				},
			}),
			output: expectedOut,
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			c := &describeClient{T: t, Namespace: "foo", Interface: test.input}
			d := IngressClassDescriber{c}
			out, err := d.Describe("", "example-class", DescriberSettings{})
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			if out != expectedOut {
				t.Logf(out)
				t.Errorf("expected : %q\n but got output:\n %q", test.output, out)
			}
		})
	}
}

func TestDescribeNetworkPolicies(t *testing.T) {
	expectedTime, err := time.Parse("2006-01-02 15:04:05 Z0700 MST", "2017-06-04 21:45:56 -0700 PDT")
	if err != nil {
		t.Errorf("unable to parse time %q error: %s", "2017-06-04 21:45:56 -0700 PDT", err)
	}
	expectedOut := `Name:         network-policy-1
Namespace:    default
Created on:   2017-06-04 21:45:56 -0700 PDT
Labels:       <none>
Annotations:  <none>
Spec:
  PodSelector:     foo in (bar1,bar2),foo2 notin (bar1,bar2),id1=app1,id2=app2
  Allowing ingress traffic:
    To Port: 80/TCP
    To Port: 82/TCP
    From:
      NamespaceSelector: id=ns1,id2=ns2
      PodSelector: id=pod1,id2=pod2
    From:
      PodSelector: id=app2,id2=app3
    From:
      NamespaceSelector: id=app2,id2=app3
    From:
      NamespaceSelector: foo in (bar1,bar2),id=app2,id2=app3
    From:
      IPBlock:
        CIDR: 192.168.0.0/16
        Except: 192.168.3.0/24, 192.168.4.0/24
    ----------
    To Port: <any> (traffic allowed to all ports)
    From: <any> (traffic not restricted by source)
  Allowing egress traffic:
    To Port: 80/TCP
    To Port: 82/TCP
    To:
      NamespaceSelector: id=ns1,id2=ns2
      PodSelector: id=pod1,id2=pod2
    To:
      PodSelector: id=app2,id2=app3
    To:
      NamespaceSelector: id=app2,id2=app3
    To:
      NamespaceSelector: foo in (bar1,bar2),id=app2,id2=app3
    To:
      IPBlock:
        CIDR: 192.168.0.0/16
        Except: 192.168.3.0/24, 192.168.4.0/24
    ----------
    To Port: <any> (traffic allowed to all ports)
    To: <any> (traffic not restricted by destination)
  Policy Types: Ingress, Egress
`

	port80 := intstr.FromInt32(80)
	port82 := intstr.FromInt32(82)
	protoTCP := corev1.ProtocolTCP

	versionedFake := fake.NewSimpleClientset(&networkingv1.NetworkPolicy{
		ObjectMeta: metav1.ObjectMeta{
			Name:              "network-policy-1",
			Namespace:         "default",
			CreationTimestamp: metav1.NewTime(expectedTime),
		},
		Spec: networkingv1.NetworkPolicySpec{
			PodSelector: metav1.LabelSelector{
				MatchLabels: map[string]string{
					"id1": "app1",
					"id2": "app2",
				},
				MatchExpressions: []metav1.LabelSelectorRequirement{
					{Key: "foo", Operator: "In", Values: []string{"bar1", "bar2"}},
					{Key: "foo2", Operator: "NotIn", Values: []string{"bar1", "bar2"}},
				},
			},
			Ingress: []networkingv1.NetworkPolicyIngressRule{
				{
					Ports: []networkingv1.NetworkPolicyPort{
						{Port: &port80},
						{Port: &port82, Protocol: &protoTCP},
					},
					From: []networkingv1.NetworkPolicyPeer{
						{
							PodSelector: &metav1.LabelSelector{
								MatchLabels: map[string]string{
									"id":  "pod1",
									"id2": "pod2",
								},
							},
							NamespaceSelector: &metav1.LabelSelector{
								MatchLabels: map[string]string{
									"id":  "ns1",
									"id2": "ns2",
								},
							},
						},
						{
							PodSelector: &metav1.LabelSelector{
								MatchLabels: map[string]string{
									"id":  "app2",
									"id2": "app3",
								},
							},
						},
						{
							NamespaceSelector: &metav1.LabelSelector{
								MatchLabels: map[string]string{
									"id":  "app2",
									"id2": "app3",
								},
							},
						},
						{
							NamespaceSelector: &metav1.LabelSelector{
								MatchLabels: map[string]string{
									"id":  "app2",
									"id2": "app3",
								},
								MatchExpressions: []metav1.LabelSelectorRequirement{
									{Key: "foo", Operator: "In", Values: []string{"bar1", "bar2"}},
								},
							},
						},
						{
							IPBlock: &networkingv1.IPBlock{
								CIDR:   "192.168.0.0/16",
								Except: []string{"192.168.3.0/24", "192.168.4.0/24"},
							},
						},
					},
				},
				{},
			},
			Egress: []networkingv1.NetworkPolicyEgressRule{
				{
					Ports: []networkingv1.NetworkPolicyPort{
						{Port: &port80},
						{Port: &port82, Protocol: &protoTCP},
					},
					To: []networkingv1.NetworkPolicyPeer{
						{
							PodSelector: &metav1.LabelSelector{
								MatchLabels: map[string]string{
									"id":  "pod1",
									"id2": "pod2",
								},
							},
							NamespaceSelector: &metav1.LabelSelector{
								MatchLabels: map[string]string{
									"id":  "ns1",
									"id2": "ns2",
								},
							},
						},
						{
							PodSelector: &metav1.LabelSelector{
								MatchLabels: map[string]string{
									"id":  "app2",
									"id2": "app3",
								},
							},
						},
						{
							NamespaceSelector: &metav1.LabelSelector{
								MatchLabels: map[string]string{
									"id":  "app2",
									"id2": "app3",
								},
							},
						},
						{
							NamespaceSelector: &metav1.LabelSelector{
								MatchLabels: map[string]string{
									"id":  "app2",
									"id2": "app3",
								},
								MatchExpressions: []metav1.LabelSelectorRequirement{
									{Key: "foo", Operator: "In", Values: []string{"bar1", "bar2"}},
								},
							},
						},
						{
							IPBlock: &networkingv1.IPBlock{
								CIDR:   "192.168.0.0/16",
								Except: []string{"192.168.3.0/24", "192.168.4.0/24"},
							},
						},
					},
				},
				{},
			},
			PolicyTypes: []networkingv1.PolicyType{networkingv1.PolicyTypeIngress, networkingv1.PolicyTypeEgress},
		},
	})
	d := NetworkPolicyDescriber{versionedFake}
	out, err := d.Describe("default", "network-policy-1", DescriberSettings{})
	if err != nil {
		t.Errorf("unexpected error: %s", err)
	}
	if out != expectedOut {
		t.Errorf("want:\n%s\ngot:\n%s", expectedOut, out)
	}
}

func TestDescribeIngressNetworkPolicies(t *testing.T) {
	expectedTime, err := time.Parse("2006-01-02 15:04:05 Z0700 MST", "2017-06-04 21:45:56 -0700 PDT")
	if err != nil {
		t.Errorf("unable to parse time %q error: %s", "2017-06-04 21:45:56 -0700 PDT", err)
	}
	expectedOut := `Name:         network-policy-1
Namespace:    default
Created on:   2017-06-04 21:45:56 -0700 PDT
Labels:       <none>
Annotations:  <none>
Spec:
  PodSelector:     foo in (bar1,bar2),foo2 notin (bar1,bar2),id1=app1,id2=app2
  Allowing ingress traffic:
    To Port: 80/TCP
    To Port: 82/TCP
    From:
      NamespaceSelector: id=ns1,id2=ns2
      PodSelector: id=pod1,id2=pod2
    From:
      PodSelector: id=app2,id2=app3
    From:
      NamespaceSelector: id=app2,id2=app3
    From:
      NamespaceSelector: foo in (bar1,bar2),id=app2,id2=app3
    From:
      IPBlock:
        CIDR: 192.168.0.0/16
        Except: 192.168.3.0/24, 192.168.4.0/24
    ----------
    To Port: <any> (traffic allowed to all ports)
    From: <any> (traffic not restricted by source)
  Not affecting egress traffic
  Policy Types: Ingress
`

	port80 := intstr.FromInt32(80)
	port82 := intstr.FromInt32(82)
	protoTCP := corev1.ProtocolTCP

	versionedFake := fake.NewSimpleClientset(&networkingv1.NetworkPolicy{
		ObjectMeta: metav1.ObjectMeta{
			Name:              "network-policy-1",
			Namespace:         "default",
			CreationTimestamp: metav1.NewTime(expectedTime),
		},
		Spec: networkingv1.NetworkPolicySpec{
			PodSelector: metav1.LabelSelector{
				MatchLabels: map[string]string{
					"id1": "app1",
					"id2": "app2",
				},
				MatchExpressions: []metav1.LabelSelectorRequirement{
					{Key: "foo", Operator: "In", Values: []string{"bar1", "bar2"}},
					{Key: "foo2", Operator: "NotIn", Values: []string{"bar1", "bar2"}},
				},
			},
			Ingress: []networkingv1.NetworkPolicyIngressRule{
				{
					Ports: []networkingv1.NetworkPolicyPort{
						{Port: &port80},
						{Port: &port82, Protocol: &protoTCP},
					},
					From: []networkingv1.NetworkPolicyPeer{
						{
							PodSelector: &metav1.LabelSelector{
								MatchLabels: map[string]string{
									"id":  "pod1",
									"id2": "pod2",
								},
							},
							NamespaceSelector: &metav1.LabelSelector{
								MatchLabels: map[string]string{
									"id":  "ns1",
									"id2": "ns2",
								},
							},
						},
						{
							PodSelector: &metav1.LabelSelector{
								MatchLabels: map[string]string{
									"id":  "app2",
									"id2": "app3",
								},
							},
						},
						{
							NamespaceSelector: &metav1.LabelSelector{
								MatchLabels: map[string]string{
									"id":  "app2",
									"id2": "app3",
								},
							},
						},
						{
							NamespaceSelector: &metav1.LabelSelector{
								MatchLabels: map[string]string{
									"id":  "app2",
									"id2": "app3",
								},
								MatchExpressions: []metav1.LabelSelectorRequirement{
									{Key: "foo", Operator: "In", Values: []string{"bar1", "bar2"}},
								},
							},
						},
						{
							IPBlock: &networkingv1.IPBlock{
								CIDR:   "192.168.0.0/16",
								Except: []string{"192.168.3.0/24", "192.168.4.0/24"},
							},
						},
					},
				},
				{},
			},
			PolicyTypes: []networkingv1.PolicyType{networkingv1.PolicyTypeIngress},
		},
	})
	d := NetworkPolicyDescriber{versionedFake}
	out, err := d.Describe("default", "network-policy-1", DescriberSettings{})
	if err != nil {
		t.Errorf("unexpected error: %s", err)
	}
	if out != expectedOut {
		t.Errorf("want:\n%s\ngot:\n%s", expectedOut, out)
	}
}

func TestDescribeIsolatedEgressNetworkPolicies(t *testing.T) {
	expectedTime, err := time.Parse("2006-01-02 15:04:05 Z0700 MST", "2017-06-04 21:45:56 -0700 PDT")
	if err != nil {
		t.Errorf("unable to parse time %q error: %s", "2017-06-04 21:45:56 -0700 PDT", err)
	}
	expectedOut := `Name:         network-policy-1
Namespace:    default
Created on:   2017-06-04 21:45:56 -0700 PDT
Labels:       <none>
Annotations:  <none>
Spec:
  PodSelector:     foo in (bar1,bar2),foo2 notin (bar1,bar2),id1=app1,id2=app2
  Allowing ingress traffic:
    To Port: 80/TCP
    To Port: 82/TCP
    From:
      NamespaceSelector: id=ns1,id2=ns2
      PodSelector: id=pod1,id2=pod2
    From:
      PodSelector: id=app2,id2=app3
    From:
      NamespaceSelector: id=app2,id2=app3
    From:
      NamespaceSelector: foo in (bar1,bar2),id=app2,id2=app3
    From:
      IPBlock:
        CIDR: 192.168.0.0/16
        Except: 192.168.3.0/24, 192.168.4.0/24
    ----------
    To Port: <any> (traffic allowed to all ports)
    From: <any> (traffic not restricted by source)
  Allowing egress traffic:
    <none> (Selected pods are isolated for egress connectivity)
  Policy Types: Ingress, Egress
`

	port80 := intstr.FromInt32(80)
	port82 := intstr.FromInt32(82)
	protoTCP := corev1.ProtocolTCP

	versionedFake := fake.NewSimpleClientset(&networkingv1.NetworkPolicy{
		ObjectMeta: metav1.ObjectMeta{
			Name:              "network-policy-1",
			Namespace:         "default",
			CreationTimestamp: metav1.NewTime(expectedTime),
		},
		Spec: networkingv1.NetworkPolicySpec{
			PodSelector: metav1.LabelSelector{
				MatchLabels: map[string]string{
					"id1": "app1",
					"id2": "app2",
				},
				MatchExpressions: []metav1.LabelSelectorRequirement{
					{Key: "foo", Operator: "In", Values: []string{"bar1", "bar2"}},
					{Key: "foo2", Operator: "NotIn", Values: []string{"bar1", "bar2"}},
				},
			},
			Ingress: []networkingv1.NetworkPolicyIngressRule{
				{
					Ports: []networkingv1.NetworkPolicyPort{
						{Port: &port80},
						{Port: &port82, Protocol: &protoTCP},
					},
					From: []networkingv1.NetworkPolicyPeer{
						{
							PodSelector: &metav1.LabelSelector{
								MatchLabels: map[string]string{
									"id":  "pod1",
									"id2": "pod2",
								},
							},
							NamespaceSelector: &metav1.LabelSelector{
								MatchLabels: map[string]string{
									"id":  "ns1",
									"id2": "ns2",
								},
							},
						},
						{
							PodSelector: &metav1.LabelSelector{
								MatchLabels: map[string]string{
									"id":  "app2",
									"id2": "app3",
								},
							},
						},
						{
							NamespaceSelector: &metav1.LabelSelector{
								MatchLabels: map[string]string{
									"id":  "app2",
									"id2": "app3",
								},
							},
						},
						{
							NamespaceSelector: &metav1.LabelSelector{
								MatchLabels: map[string]string{
									"id":  "app2",
									"id2": "app3",
								},
								MatchExpressions: []metav1.LabelSelectorRequirement{
									{Key: "foo", Operator: "In", Values: []string{"bar1", "bar2"}},
								},
							},
						},
						{
							IPBlock: &networkingv1.IPBlock{
								CIDR:   "192.168.0.0/16",
								Except: []string{"192.168.3.0/24", "192.168.4.0/24"},
							},
						},
					},
				},
				{},
			},
			PolicyTypes: []networkingv1.PolicyType{networkingv1.PolicyTypeIngress, networkingv1.PolicyTypeEgress},
		},
	})
	d := NetworkPolicyDescriber{versionedFake}
	out, err := d.Describe("default", "network-policy-1", DescriberSettings{})
	if err != nil {
		t.Errorf("unexpected error: %s", err)
	}
	if out != expectedOut {
		t.Errorf("want:\n%s\ngot:\n%s", expectedOut, out)
	}
}

func TestDescribeNetworkPoliciesWithPortRange(t *testing.T) {
	expectedTime, err := time.Parse("2006-01-02 15:04:05 Z0700 MST", "2017-06-04 21:45:56 -0700 PDT")
	if err != nil {
		t.Errorf("unable to parse time %q error: %s", "2017-06-04 21:45:56 -0700 PDT", err)
	}
	expectedOut := `Name:         network-policy-1
Namespace:    default
Created on:   2017-06-04 21:45:56 -0700 PDT
Labels:       <none>
Annotations:  <none>
Spec:
  PodSelector:     foo in (bar1,bar2),foo2 notin (bar1,bar2),id1=app1,id2=app2
  Allowing ingress traffic:
    To Port Range: 80-82/TCP
    From:
      NamespaceSelector: id=ns1,id2=ns2
      PodSelector: id=pod1,id2=pod2
    From:
      PodSelector: id=app2,id2=app3
    From:
      NamespaceSelector: id=app2,id2=app3
    From:
      NamespaceSelector: foo in (bar1,bar2),id=app2,id2=app3
    From:
      IPBlock:
        CIDR: 192.168.0.0/16
        Except: 192.168.3.0/24, 192.168.4.0/24
    ----------
    To Port: <any> (traffic allowed to all ports)
    From: <any> (traffic not restricted by source)
  Allowing egress traffic:
    To Port Range: 80-82/TCP
    To:
      NamespaceSelector: id=ns1,id2=ns2
      PodSelector: id=pod1,id2=pod2
    To:
      PodSelector: id=app2,id2=app3
    To:
      NamespaceSelector: id=app2,id2=app3
    To:
      NamespaceSelector: foo in (bar1,bar2),id=app2,id2=app3
    To:
      IPBlock:
        CIDR: 192.168.0.0/16
        Except: 192.168.3.0/24, 192.168.4.0/24
    ----------
    To Port: <any> (traffic allowed to all ports)
    To: <any> (traffic not restricted by destination)
  Policy Types: Ingress, Egress
`

	port80 := intstr.FromInt(80)
	port82 := int32(82)
	protoTCP := corev1.ProtocolTCP

	versionedFake := fake.NewSimpleClientset(&networkingv1.NetworkPolicy{
		ObjectMeta: metav1.ObjectMeta{
			Name:              "network-policy-1",
			Namespace:         "default",
			CreationTimestamp: metav1.NewTime(expectedTime),
		},
		Spec: networkingv1.NetworkPolicySpec{
			PodSelector: metav1.LabelSelector{
				MatchLabels: map[string]string{
					"id1": "app1",
					"id2": "app2",
				},
				MatchExpressions: []metav1.LabelSelectorRequirement{
					{Key: "foo", Operator: "In", Values: []string{"bar1", "bar2"}},
					{Key: "foo2", Operator: "NotIn", Values: []string{"bar1", "bar2"}},
				},
			},
			Ingress: []networkingv1.NetworkPolicyIngressRule{
				{
					Ports: []networkingv1.NetworkPolicyPort{
						{Port: &port80, EndPort: &port82, Protocol: &protoTCP},
					},
					From: []networkingv1.NetworkPolicyPeer{
						{
							PodSelector: &metav1.LabelSelector{
								MatchLabels: map[string]string{
									"id":  "pod1",
									"id2": "pod2",
								},
							},
							NamespaceSelector: &metav1.LabelSelector{
								MatchLabels: map[string]string{
									"id":  "ns1",
									"id2": "ns2",
								},
							},
						},
						{
							PodSelector: &metav1.LabelSelector{
								MatchLabels: map[string]string{
									"id":  "app2",
									"id2": "app3",
								},
							},
						},
						{
							NamespaceSelector: &metav1.LabelSelector{
								MatchLabels: map[string]string{
									"id":  "app2",
									"id2": "app3",
								},
							},
						},
						{
							NamespaceSelector: &metav1.LabelSelector{
								MatchLabels: map[string]string{
									"id":  "app2",
									"id2": "app3",
								},
								MatchExpressions: []metav1.LabelSelectorRequirement{
									{Key: "foo", Operator: "In", Values: []string{"bar1", "bar2"}},
								},
							},
						},
						{
							IPBlock: &networkingv1.IPBlock{
								CIDR:   "192.168.0.0/16",
								Except: []string{"192.168.3.0/24", "192.168.4.0/24"},
							},
						},
					},
				},
				{},
			},
			Egress: []networkingv1.NetworkPolicyEgressRule{
				{
					Ports: []networkingv1.NetworkPolicyPort{
						{Port: &port80, EndPort: &port82, Protocol: &protoTCP},
					},
					To: []networkingv1.NetworkPolicyPeer{
						{
							PodSelector: &metav1.LabelSelector{
								MatchLabels: map[string]string{
									"id":  "pod1",
									"id2": "pod2",
								},
							},
							NamespaceSelector: &metav1.LabelSelector{
								MatchLabels: map[string]string{
									"id":  "ns1",
									"id2": "ns2",
								},
							},
						},
						{
							PodSelector: &metav1.LabelSelector{
								MatchLabels: map[string]string{
									"id":  "app2",
									"id2": "app3",
								},
							},
						},
						{
							NamespaceSelector: &metav1.LabelSelector{
								MatchLabels: map[string]string{
									"id":  "app2",
									"id2": "app3",
								},
							},
						},
						{
							NamespaceSelector: &metav1.LabelSelector{
								MatchLabels: map[string]string{
									"id":  "app2",
									"id2": "app3",
								},
								MatchExpressions: []metav1.LabelSelectorRequirement{
									{Key: "foo", Operator: "In", Values: []string{"bar1", "bar2"}},
								},
							},
						},
						{
							IPBlock: &networkingv1.IPBlock{
								CIDR:   "192.168.0.0/16",
								Except: []string{"192.168.3.0/24", "192.168.4.0/24"},
							},
						},
					},
				},
				{},
			},
			PolicyTypes: []networkingv1.PolicyType{networkingv1.PolicyTypeIngress, networkingv1.PolicyTypeEgress},
		},
	})
	d := NetworkPolicyDescriber{versionedFake}
	out, err := d.Describe("default", "network-policy-1", DescriberSettings{})
	if err != nil {
		t.Errorf("unexpected error: %s", err)
	}
	if out != expectedOut {
		t.Errorf("want:\n%s\ngot:\n%s", expectedOut, out)
	}
}

func TestDescribeServiceAccount(t *testing.T) {
	fake := fake.NewSimpleClientset(&corev1.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "bar",
			Namespace: "foo",
		},
		Secrets: []corev1.ObjectReference{
			{
				Name: "test-objectref",
			},
		},
		ImagePullSecrets: []corev1.LocalObjectReference{
			{
				Name: "test-local-ref",
			},
		},
	})
	c := &describeClient{T: t, Namespace: "foo", Interface: fake}
	d := ServiceAccountDescriber{c}
	out, err := d.Describe("foo", "bar", DescriberSettings{ShowEvents: true})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	expectedOut := `Name:                bar
Namespace:           foo
Labels:              <none>
Annotations:         <none>
Image pull secrets:  test-local-ref (not found)
Mountable secrets:   test-objectref (not found)
Tokens:              <none>
Events:              <none>` + "\n"
	if out != expectedOut {
		t.Errorf("expected : %q\n but got output:\n %q", expectedOut, out)
	}

}
func getHugePageResourceList(pageSize, value string) corev1.ResourceList {
	res := corev1.ResourceList{}
	if pageSize != "" && value != "" {
		res[corev1.ResourceName(corev1.ResourceHugePagesPrefix+pageSize)] = resource.MustParse(value)
	}
	return res
}

// mergeResourceLists will merge resoure lists. When two lists have the same resourece, the value from
// the last list will be present in the result
func mergeResourceLists(resourceLists ...corev1.ResourceList) corev1.ResourceList {
	result := corev1.ResourceList{}
	for _, rl := range resourceLists {
		for resource, quantity := range rl {
			result[resource] = quantity
		}
	}
	return result
}

func TestDescribeNode(t *testing.T) {
	holderIdentity := "holder"
	nodeCapacity := mergeResourceLists(
		getHugePageResourceList("2Mi", "4Gi"),
		getResourceList("8", "24Gi"),
		getHugePageResourceList("1Gi", "0"),
	)
	nodeAllocatable := mergeResourceLists(
		getHugePageResourceList("2Mi", "2Gi"),
		getResourceList("4", "12Gi"),
		getHugePageResourceList("1Gi", "0"),
	)

	fake := fake.NewSimpleClientset(
		&corev1.Node{
			ObjectMeta: metav1.ObjectMeta{
				Name: "bar",
				UID:  "uid",
			},
			Spec: corev1.NodeSpec{
				Unschedulable: true,
			},
			Status: corev1.NodeStatus{
				Capacity:    nodeCapacity,
				Allocatable: nodeAllocatable,
			},
		},
		&coordinationv1.Lease{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "bar",
				Namespace: corev1.NamespaceNodeLease,
			},
			Spec: coordinationv1.LeaseSpec{
				HolderIdentity: &holderIdentity,
				AcquireTime:    &metav1.MicroTime{Time: time.Now().Add(-time.Hour)},
				RenewTime:      &metav1.MicroTime{Time: time.Now()},
			},
		},
		&corev1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "pod-with-resources",
				Namespace: "foo",
			},
			TypeMeta: metav1.TypeMeta{
				Kind: "Pod",
			},
			Spec: corev1.PodSpec{
				Containers: []corev1.Container{
					{
						Name:  "cpu-mem",
						Image: "image:latest",
						Resources: corev1.ResourceRequirements{
							Requests: getResourceList("1", "1Gi"),
							Limits:   getResourceList("2", "2Gi"),
						},
					},
					{
						Name:  "hugepages",
						Image: "image:latest",
						Resources: corev1.ResourceRequirements{
							Requests: getHugePageResourceList("2Mi", "512Mi"),
							Limits:   getHugePageResourceList("2Mi", "512Mi"),
						},
					},
				},
			},
			Status: corev1.PodStatus{
				Phase: corev1.PodRunning,
			},
		},
		&corev1.EventList{
			Items: []corev1.Event{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "event-1",
						Namespace: "default",
					},
					InvolvedObject: corev1.ObjectReference{
						Kind: "Node",
						Name: "bar",
						UID:  "bar",
					},
					Message:        "Node bar status is now: NodeHasNoDiskPressure",
					FirstTimestamp: metav1.NewTime(time.Date(2014, time.January, 15, 0, 0, 0, 0, time.UTC)),
					LastTimestamp:  metav1.NewTime(time.Date(2014, time.January, 15, 0, 0, 0, 0, time.UTC)),
					Count:          1,
					Type:           corev1.EventTypeNormal,
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "event-2",
						Namespace: "default",
					},
					InvolvedObject: corev1.ObjectReference{
						Kind: "Node",
						Name: "bar",
						UID:  "0ceac5fb-a393-49d7-b04f-9ea5f18de5e9",
					},
					Message:        "Node bar status is now: NodeReady",
					FirstTimestamp: metav1.NewTime(time.Date(2014, time.January, 15, 0, 0, 0, 0, time.UTC)),
					LastTimestamp:  metav1.NewTime(time.Date(2014, time.January, 15, 0, 0, 0, 0, time.UTC)),
					Count:          2,
					Type:           corev1.EventTypeNormal,
				},
			},
		},
	)
	c := &describeClient{T: t, Namespace: "foo", Interface: fake}
	d := NodeDescriber{c}
	out, err := d.Describe("foo", "bar", DescriberSettings{ShowEvents: true})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	expectedOut := []string{"Unschedulable", "true", "holder",
		`Allocated resources:
  (Total limits may be over 100 percent, i.e., overcommitted.)
  Resource           Requests     Limits
  --------           --------     ------
  cpu                1 (25%)      2 (50%)
  memory             1Gi (8%)     2Gi (16%)
  ephemeral-storage  0 (0%)       0 (0%)
  hugepages-1Gi      0 (0%)       0 (0%)
  hugepages-2Mi      512Mi (25%)  512Mi (25%)`,
		`Node bar status is now: NodeHasNoDiskPressure`,
		`Node bar status is now: NodeReady`}
	for _, expected := range expectedOut {
		if !strings.Contains(out, expected) {
			t.Errorf("expected to find %q in output: %q", expected, out)
		}
	}
}

func TestDescribeNodeWithSidecar(t *testing.T) {
	holderIdentity := "holder"
	nodeCapacity := mergeResourceLists(
		getHugePageResourceList("2Mi", "4Gi"),
		getResourceList("8", "24Gi"),
		getHugePageResourceList("1Gi", "0"),
	)
	nodeAllocatable := mergeResourceLists(
		getHugePageResourceList("2Mi", "2Gi"),
		getResourceList("4", "12Gi"),
		getHugePageResourceList("1Gi", "0"),
	)

	restartPolicy := corev1.ContainerRestartPolicyAlways
	fake := fake.NewSimpleClientset(
		&corev1.Node{
			ObjectMeta: metav1.ObjectMeta{
				Name: "bar",
				UID:  "uid",
			},
			Spec: corev1.NodeSpec{
				Unschedulable: true,
			},
			Status: corev1.NodeStatus{
				Capacity:    nodeCapacity,
				Allocatable: nodeAllocatable,
			},
		},
		&coordinationv1.Lease{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "bar",
				Namespace: corev1.NamespaceNodeLease,
			},
			Spec: coordinationv1.LeaseSpec{
				HolderIdentity: &holderIdentity,
				AcquireTime:    &metav1.MicroTime{Time: time.Now().Add(-time.Hour)},
				RenewTime:      &metav1.MicroTime{Time: time.Now()},
			},
		},
		&corev1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "pod-with-resources",
				Namespace: "foo",
			},
			TypeMeta: metav1.TypeMeta{
				Kind: "Pod",
			},
			Spec: corev1.PodSpec{
				InitContainers: []corev1.Container{
					// sidecar, should sum into the total resources
					{
						Name:          "init-container-1",
						RestartPolicy: &restartPolicy,
						Resources: corev1.ResourceRequirements{
							Requests: getResourceList("1", "1Gi"),
						},
					},
					// non-sidecar
					{
						Name: "init-container-2",
						Resources: corev1.ResourceRequirements{
							Requests: getResourceList("1", "1Gi"),
						},
					},
				},
				Containers: []corev1.Container{
					{
						Name:  "cpu-mem",
						Image: "image:latest",
						Resources: corev1.ResourceRequirements{
							Requests: getResourceList("1", "1Gi"),
							Limits:   getResourceList("2", "2Gi"),
						},
					},
					{
						Name:  "hugepages",
						Image: "image:latest",
						Resources: corev1.ResourceRequirements{
							Requests: getHugePageResourceList("2Mi", "512Mi"),
							Limits:   getHugePageResourceList("2Mi", "512Mi"),
						},
					},
				},
			},
			Status: corev1.PodStatus{
				Phase: corev1.PodRunning,
			},
		},
		&corev1.EventList{
			Items: []corev1.Event{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "event-1",
						Namespace: "default",
					},
					InvolvedObject: corev1.ObjectReference{
						Kind: "Node",
						Name: "bar",
						UID:  "bar",
					},
					Message:        "Node bar status is now: NodeHasNoDiskPressure",
					FirstTimestamp: metav1.NewTime(time.Date(2014, time.January, 15, 0, 0, 0, 0, time.UTC)),
					LastTimestamp:  metav1.NewTime(time.Date(2014, time.January, 15, 0, 0, 0, 0, time.UTC)),
					Count:          1,
					Type:           corev1.EventTypeNormal,
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "event-2",
						Namespace: "default",
					},
					InvolvedObject: corev1.ObjectReference{
						Kind: "Node",
						Name: "bar",
						UID:  "0ceac5fb-a393-49d7-b04f-9ea5f18de5e9",
					},
					Message:        "Node bar status is now: NodeReady",
					FirstTimestamp: metav1.NewTime(time.Date(2014, time.January, 15, 0, 0, 0, 0, time.UTC)),
					LastTimestamp:  metav1.NewTime(time.Date(2014, time.January, 15, 0, 0, 0, 0, time.UTC)),
					Count:          2,
					Type:           corev1.EventTypeNormal,
				},
			},
		},
	)
	c := &describeClient{T: t, Namespace: "foo", Interface: fake}
	d := NodeDescriber{c}
	out, err := d.Describe("foo", "bar", DescriberSettings{ShowEvents: true})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	expectedOut := []string{"Unschedulable", "true", "holder",
		`Allocated resources:
  (Total limits may be over 100 percent, i.e., overcommitted.)
  Resource           Requests     Limits
  --------           --------     ------
  cpu                2 (50%)      2 (50%)
  memory             2Gi (16%)    2Gi (16%)
  ephemeral-storage  0 (0%)       0 (0%)
  hugepages-1Gi      0 (0%)       0 (0%)
  hugepages-2Mi      512Mi (25%)  512Mi (25%)`,
		`Node bar status is now: NodeHasNoDiskPressure`,
		`Node bar status is now: NodeReady`}
	for _, expected := range expectedOut {
		if !strings.Contains(out, expected) {
			t.Errorf("expected to find %s in output: %s", expected, out)
		}
	}
}
func TestDescribeStatefulSet(t *testing.T) {
	var partition int32 = 2672
	var replicas int32 = 1
	fake := fake.NewSimpleClientset(&appsv1.StatefulSet{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "bar",
			Namespace: "foo",
		},
		Spec: appsv1.StatefulSetSpec{
			Replicas: &replicas,
			Selector: &metav1.LabelSelector{},
			Template: corev1.PodTemplateSpec{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{Image: "mytest-image:latest"},
					},
				},
			},
			UpdateStrategy: appsv1.StatefulSetUpdateStrategy{
				Type: appsv1.RollingUpdateStatefulSetStrategyType,
				RollingUpdate: &appsv1.RollingUpdateStatefulSetStrategy{
					Partition: &partition,
				},
			},
		},
	})
	d := StatefulSetDescriber{fake}
	out, err := d.Describe("foo", "bar", DescriberSettings{ShowEvents: true})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	expectedOutputs := []string{
		"bar", "foo", "Containers:", "mytest-image:latest", "Update Strategy", "RollingUpdate", "Partition", "2672",
	}
	for _, o := range expectedOutputs {
		if !strings.Contains(out, o) {
			t.Errorf("unexpected out: %s", out)
			break
		}
	}
}

func TestDescribeEndpointSlice(t *testing.T) {
	protocolTCP := corev1.ProtocolTCP
	port80 := int32(80)

	testcases := map[string]struct {
		input  *fake.Clientset
		output string
	}{
		"EndpointSlices v1beta1": {
			input: fake.NewSimpleClientset(&discoveryv1beta1.EndpointSlice{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "foo.123",
					Namespace: "bar",
				},
				AddressType: discoveryv1beta1.AddressTypeIPv4,
				Endpoints: []discoveryv1beta1.Endpoint{
					{
						Addresses:  []string{"1.2.3.4", "1.2.3.5"},
						Conditions: discoveryv1beta1.EndpointConditions{Ready: ptr.To(true)},
						TargetRef:  &corev1.ObjectReference{Kind: "Pod", Name: "test-123"},
						Topology: map[string]string{
							"topology.kubernetes.io/zone":   "us-central1-a",
							"topology.kubernetes.io/region": "us-central1",
						},
					}, {
						Addresses:  []string{"1.2.3.6", "1.2.3.7"},
						Conditions: discoveryv1beta1.EndpointConditions{Ready: ptr.To(true)},
						TargetRef:  &corev1.ObjectReference{Kind: "Pod", Name: "test-124"},
						Topology: map[string]string{
							"topology.kubernetes.io/zone":   "us-central1-b",
							"topology.kubernetes.io/region": "us-central1",
						},
					},
				},
				Ports: []discoveryv1beta1.EndpointPort{
					{
						Protocol: &protocolTCP,
						Port:     &port80,
					},
				},
			}),

			output: `Name:         foo.123
Namespace:    bar
Labels:       <none>
Annotations:  <none>
AddressType:  IPv4
Ports:
  Name     Port  Protocol
  ----     ----  --------
  <unset>  80    TCP
Endpoints:
  - Addresses:  1.2.3.4,1.2.3.5
    Conditions:
      Ready:    true
    Hostname:   <unset>
    TargetRef:  Pod/test-123
    Topology:   topology.kubernetes.io/region=us-central1
                topology.kubernetes.io/zone=us-central1-a
  - Addresses:  1.2.3.6,1.2.3.7
    Conditions:
      Ready:    true
    Hostname:   <unset>
    TargetRef:  Pod/test-124
    Topology:   topology.kubernetes.io/region=us-central1
                topology.kubernetes.io/zone=us-central1-b
Events:         <none>` + "\n",
		},
		"EndpointSlices v1": {
			input: fake.NewSimpleClientset(&discoveryv1.EndpointSlice{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "foo.123",
					Namespace: "bar",
				},
				AddressType: discoveryv1.AddressTypeIPv4,
				Endpoints: []discoveryv1.Endpoint{
					{
						Addresses:  []string{"1.2.3.4", "1.2.3.5"},
						Conditions: discoveryv1.EndpointConditions{Ready: ptr.To(true)},
						TargetRef:  &corev1.ObjectReference{Kind: "Pod", Name: "test-123"},
						Zone:       ptr.To("us-central1-a"),
						NodeName:   ptr.To("node-1"),
					}, {
						Addresses:  []string{"1.2.3.6", "1.2.3.7"},
						Conditions: discoveryv1.EndpointConditions{Ready: ptr.To(true)},
						TargetRef:  &corev1.ObjectReference{Kind: "Pod", Name: "test-124"},
						NodeName:   ptr.To("node-2"),
					},
				},
				Ports: []discoveryv1.EndpointPort{
					{
						Protocol: &protocolTCP,
						Port:     &port80,
					},
				},
			}),

			output: `Name:         foo.123
Namespace:    bar
Labels:       <none>
Annotations:  <none>
AddressType:  IPv4
Ports:
  Name     Port  Protocol
  ----     ----  --------
  <unset>  80    TCP
Endpoints:
  - Addresses:  1.2.3.4, 1.2.3.5
    Conditions:
      Ready:    true
    Hostname:   <unset>
    TargetRef:  Pod/test-123
    NodeName:   node-1
    Zone:       us-central1-a
  - Addresses:  1.2.3.6, 1.2.3.7
    Conditions:
      Ready:    true
    Hostname:   <unset>
    TargetRef:  Pod/test-124
    NodeName:   node-2
    Zone:       <unset>
Events:         <none>` + "\n",
		},
	}

	for name, tc := range testcases {
		t.Run(name, func(t *testing.T) {
			c := &describeClient{T: t, Namespace: "foo", Interface: tc.input}
			d := EndpointSliceDescriber{c}
			out, err := d.Describe("bar", "foo.123", DescriberSettings{ShowEvents: true})
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			if out != tc.output {
				t.Logf(out)
				t.Errorf("expected :\n%s\nbut got output:\n%s", tc.output, out)
			}
		})
	}
}

func TestDescribeServiceCIDR(t *testing.T) {

	testcases := map[string]struct {
		input  *fake.Clientset
		output string
	}{
		"ServiceCIDR v1alpha1": {
			input: fake.NewSimpleClientset(&networkingv1alpha1.ServiceCIDR{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo.123",
				},
				Spec: networkingv1alpha1.ServiceCIDRSpec{
					CIDRs: []string{"10.1.0.0/16", "fd00:1:1::/64"},
				},
			}),

			output: `Name:         foo.123
Labels:       <none>
Annotations:  <none>
CIDRs:        10.1.0.0/16, fd00:1:1::/64
Events:       <none>` + "\n",
		},
		"ServiceCIDR v1alpha1 IPv4": {
			input: fake.NewSimpleClientset(&networkingv1alpha1.ServiceCIDR{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo.123",
				},
				Spec: networkingv1alpha1.ServiceCIDRSpec{
					CIDRs: []string{"10.1.0.0/16"},
				},
			}),

			output: `Name:         foo.123
Labels:       <none>
Annotations:  <none>
CIDRs:        10.1.0.0/16
Events:       <none>` + "\n",
		},
		"ServiceCIDR v1alpha1 IPv6": {
			input: fake.NewSimpleClientset(&networkingv1alpha1.ServiceCIDR{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo.123",
				},
				Spec: networkingv1alpha1.ServiceCIDRSpec{
					CIDRs: []string{"fd00:1:1::/64"},
				},
			}),

			output: `Name:         foo.123
Labels:       <none>
Annotations:  <none>
CIDRs:        fd00:1:1::/64
Events:       <none>` + "\n",
		},
	}

	for name, tc := range testcases {
		t.Run(name, func(t *testing.T) {
			c := &describeClient{T: t, Namespace: "foo", Interface: tc.input}
			d := ServiceCIDRDescriber{c}
			out, err := d.Describe("bar", "foo.123", DescriberSettings{ShowEvents: true})
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			if out != tc.output {
				t.Errorf("expected :\n%s\nbut got output:\n%s diff:\n%s", tc.output, out, cmp.Diff(tc.output, out))
			}
		})
	}
}

func TestDescribeIPAddress(t *testing.T) {

	testcases := map[string]struct {
		input  *fake.Clientset
		output string
	}{
		"IPAddress v1alpha1": {
			input: fake.NewSimpleClientset(&networkingv1alpha1.IPAddress{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo.123",
				},
				Spec: networkingv1alpha1.IPAddressSpec{
					ParentRef: &networkingv1alpha1.ParentReference{
						Group:     "mygroup",
						Resource:  "myresource",
						Namespace: "mynamespace",
						Name:      "myname",
					},
				},
			}),

			output: `Name:         foo.123
Labels:       <none>
Annotations:  <none>
Parent Reference:
  Group:      mygroup
  Resource:   myresource
  Namespace:  mynamespace
  Name:       myname
Events:       <none>` + "\n",
		},
	}

	for name, tc := range testcases {
		t.Run(name, func(t *testing.T) {
			c := &describeClient{T: t, Namespace: "foo", Interface: tc.input}
			d := IPAddressDescriber{c}
			out, err := d.Describe("bar", "foo.123", DescriberSettings{ShowEvents: true})
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			if out != tc.output {
				t.Errorf("expected :\n%s\nbut got output:\n%s diff:\n%s", tc.output, out, cmp.Diff(tc.output, out))
			}
		})
	}
}

func TestControllerRef(t *testing.T) {
	var replicas int32 = 1
	f := fake.NewSimpleClientset(
		&corev1.ReplicationController{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "bar",
				Namespace: "foo",
				UID:       "123456",
			},
			TypeMeta: metav1.TypeMeta{
				Kind: "ReplicationController",
			},
			Spec: corev1.ReplicationControllerSpec{
				Replicas: &replicas,
				Selector: map[string]string{"abc": "xyz"},
				Template: &corev1.PodTemplateSpec{
					Spec: corev1.PodSpec{
						Containers: []corev1.Container{
							{Image: "mytest-image:latest"},
						},
					},
				},
			},
		},
		&corev1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:            "barpod",
				Namespace:       "foo",
				Labels:          map[string]string{"abc": "xyz"},
				OwnerReferences: []metav1.OwnerReference{{Name: "bar", UID: "123456", Controller: ptr.To(true)}},
			},
			TypeMeta: metav1.TypeMeta{
				Kind: "Pod",
			},
			Spec: corev1.PodSpec{
				Containers: []corev1.Container{
					{Image: "mytest-image:latest"},
				},
			},
			Status: corev1.PodStatus{
				Phase: corev1.PodRunning,
			},
		},
		&corev1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "orphan",
				Namespace: "foo",
				Labels:    map[string]string{"abc": "xyz"},
			},
			TypeMeta: metav1.TypeMeta{
				Kind: "Pod",
			},
			Spec: corev1.PodSpec{
				Containers: []corev1.Container{
					{Image: "mytest-image:latest"},
				},
			},
			Status: corev1.PodStatus{
				Phase: corev1.PodRunning,
			},
		},
		&corev1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:            "buzpod",
				Namespace:       "foo",
				Labels:          map[string]string{"abc": "xyz"},
				OwnerReferences: []metav1.OwnerReference{{Name: "buz", UID: "654321", Controller: ptr.To(true)}},
			},
			TypeMeta: metav1.TypeMeta{
				Kind: "Pod",
			},
			Spec: corev1.PodSpec{
				Containers: []corev1.Container{
					{Image: "mytest-image:latest"},
				},
			},
			Status: corev1.PodStatus{
				Phase: corev1.PodRunning,
			},
		})
	d := ReplicationControllerDescriber{f}
	out, err := d.Describe("foo", "bar", DescriberSettings{ShowEvents: false})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if !strings.Contains(out, "1 Running") {
		t.Errorf("unexpected out: %s", out)
	}
}

func TestDescribeTerminalEscape(t *testing.T) {
	fake := fake.NewSimpleClientset(&corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:        "mycm",
			Namespace:   "foo",
			Annotations: map[string]string{"annotation1": "terminal escape: \x1b"},
		},
	})
	c := &describeClient{T: t, Namespace: "foo", Interface: fake}
	d := ConfigMapDescriber{c}
	out, err := d.Describe("foo", "mycm", DescriberSettings{ShowEvents: true})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if strings.Contains(out, "\x1b") || !strings.Contains(out, "^[") {
		t.Errorf("unexpected out: %s", out)
	}
}

func TestDescribeSeccompProfile(t *testing.T) {
	testLocalhostProfiles := []string{"lauseafoodpod", "tikkamasalaconatiner", "dropshotephemeral"}

	testCases := []struct {
		name   string
		pod    *corev1.Pod
		expect []string
	}{
		{
			name: "podLocalhostSeccomp",
			pod: &corev1.Pod{
				Spec: corev1.PodSpec{
					SecurityContext: &corev1.PodSecurityContext{
						SeccompProfile: &corev1.SeccompProfile{
							Type:             corev1.SeccompProfileTypeLocalhost,
							LocalhostProfile: &testLocalhostProfiles[0],
						},
					},
				},
			},
			expect: []string{
				"SeccompProfile", "Localhost",
				"LocalhostProfile", testLocalhostProfiles[0],
			},
		},
		{
			name: "podOther",
			pod: &corev1.Pod{
				Spec: corev1.PodSpec{
					SecurityContext: &corev1.PodSecurityContext{
						SeccompProfile: &corev1.SeccompProfile{
							Type: corev1.SeccompProfileTypeRuntimeDefault,
						},
					},
				},
			},
			expect: []string{
				"SeccompProfile", "RuntimeDefault",
			},
		},
		{
			name: "containerLocalhostSeccomp",
			pod: &corev1.Pod{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							SecurityContext: &corev1.SecurityContext{
								SeccompProfile: &corev1.SeccompProfile{
									Type:             corev1.SeccompProfileTypeLocalhost,
									LocalhostProfile: &testLocalhostProfiles[1],
								},
							},
						},
					},
				},
			},
			expect: []string{
				"SeccompProfile", "Localhost",
				"LocalhostProfile", testLocalhostProfiles[1],
			},
		},
		{
			name: "containerOther",
			pod: &corev1.Pod{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							SecurityContext: &corev1.SecurityContext{
								SeccompProfile: &corev1.SeccompProfile{
									Type: corev1.SeccompProfileTypeUnconfined,
								},
							},
						},
					},
				},
			},
			expect: []string{
				"SeccompProfile", "Unconfined",
			},
		},
		{
			name: "ephemeralLocalhostSeccomp",
			pod: &corev1.Pod{
				Spec: corev1.PodSpec{
					EphemeralContainers: []corev1.EphemeralContainer{
						{
							EphemeralContainerCommon: corev1.EphemeralContainerCommon{
								SecurityContext: &corev1.SecurityContext{
									SeccompProfile: &corev1.SeccompProfile{
										Type:             corev1.SeccompProfileTypeLocalhost,
										LocalhostProfile: &testLocalhostProfiles[2],
									},
								},
							},
						},
					},
				},
			},
			expect: []string{
				"SeccompProfile", "Localhost",
				"LocalhostProfile", testLocalhostProfiles[2],
			},
		},
		{
			name: "ephemeralOther",
			pod: &corev1.Pod{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							SecurityContext: &corev1.SecurityContext{
								SeccompProfile: &corev1.SeccompProfile{
									Type: corev1.SeccompProfileTypeUnconfined,
								},
							},
						},
					},
				},
			},
			expect: []string{
				"SeccompProfile", "Unconfined",
			},
		},
	}
	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			fake := fake.NewSimpleClientset(testCase.pod)
			c := &describeClient{T: t, Interface: fake}
			d := PodDescriber{c}
			out, err := d.Describe("", "", DescriberSettings{ShowEvents: true})
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			for _, expected := range testCase.expect {
				if !strings.Contains(out, expected) {
					t.Errorf("expected to find %q in output: %q", expected, out)
				}
			}
		})
	}
}
