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

package versioned

import (
	"bytes"
	"fmt"
	"reflect"
	"regexp"
	"strings"
	"testing"
	"time"

	appsv1 "k8s.io/api/apps/v1"
	autoscalingv2beta2 "k8s.io/api/autoscaling/v2beta2"
	"k8s.io/api/core/v1"
	corev1 "k8s.io/api/core/v1"
	networkingv1 "k8s.io/api/networking/v1"
	policyv1beta1 "k8s.io/api/policy/v1beta1"
	storagev1 "k8s.io/api/storage/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/kubernetes/pkg/kubectl/describe"
	utilpointer "k8s.io/utils/pointer"
)

type describeClient struct {
	T         *testing.T
	Namespace string
	Err       error
	kubernetes.Interface
}

func TestDescribePod(t *testing.T) {
	deletionTimestamp := metav1.Time{Time: time.Now().UTC().AddDate(10, 0, 0)}
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
	out, err := d.Describe("foo", "bar", describe.DescriberSettings{ShowEvents: true})
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
	out, err := d.Describe("foo", "bar", describe.DescriberSettings{ShowEvents: true})
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
	out, err := d.Describe("foo", "bar", describe.DescriberSettings{})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if !strings.Contains(out, "key0\n") ||
		!strings.Contains(out, "key1=value1\n") ||
		!strings.Contains(out, "key2=value2:NoSchedule\n") ||
		!strings.Contains(out, "key3=value3:NoExecute for 300s\n") ||
		!strings.Contains(out, "key4:NoExecute for 60s\n") ||
		!strings.Contains(out, "Tolerations:") {
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
	out, err := d.Describe("foo", "bar", describe.DescriberSettings{})
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
	fake := fake.NewSimpleClientset(&corev1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			Name: "myns",
		},
	})
	c := &describeClient{T: t, Namespace: "", Interface: fake}
	d := NamespaceDescriber{c}
	out, err := d.Describe("", "myns", describe.DescriberSettings{ShowEvents: true})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if !strings.Contains(out, "myns") {
		t.Errorf("unexpected out: %s", out)
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
	out, err := d.Describe("", "bar", describe.DescriberSettings{ShowEvents: true})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if !strings.Contains(out, "high-priority") || !strings.Contains(out, "1000") {
		t.Errorf("unexpected out: %s", out)
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
	})
	c := &describeClient{T: t, Namespace: "foo", Interface: fake}
	d := ConfigMapDescriber{c}
	out, err := d.Describe("foo", "mycm", describe.DescriberSettings{ShowEvents: true})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if !strings.Contains(out, "foo") || !strings.Contains(out, "mycm") || !strings.Contains(out, "key1") || !strings.Contains(out, "value1") || !strings.Contains(out, "key2") || !strings.Contains(out, "value2") {
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
	out, err := d.Describe("foo", "mylr", describe.DescriberSettings{ShowEvents: true})
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
	testCases := []struct {
		name    string
		service *corev1.Service
		expect  []string
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
						TargetPort: intstr.FromInt(9527),
						NodePort:   31111,
					}},
					Selector:              map[string]string{"blah": "heh"},
					ClusterIP:             "1.2.3.4",
					LoadBalancerIP:        "5.6.7.8",
					SessionAffinity:       "None",
					ExternalTrafficPolicy: "Local",
					HealthCheckNodePort:   32222,
				},
			},
			expect: []string{
				"Name", "bar",
				"Namespace", "foo",
				"Selector", "blah=heh",
				"Type", "LoadBalancer",
				"IP", "1.2.3.4",
				"Port", "port-tcp", "8080/TCP",
				"TargetPort", "9527/TCP",
				"NodePort", "port-tcp", "31111/TCP",
				"Session Affinity", "None",
				"External Traffic Policy", "Local",
				"HealthCheck NodePort", "32222",
			},
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
					LoadBalancerIP:        "5.6.7.8",
					SessionAffinity:       "None",
					ExternalTrafficPolicy: "Local",
					HealthCheckNodePort:   32222,
				},
			},
			expect: []string{
				"Name", "bar",
				"Namespace", "foo",
				"Selector", "blah=heh",
				"Type", "LoadBalancer",
				"IP", "1.2.3.4",
				"Port", "port-tcp", "8080/TCP",
				"TargetPort", "targetPort/TCP",
				"NodePort", "port-tcp", "31111/TCP",
				"Session Affinity", "None",
				"External Traffic Policy", "Local",
				"HealthCheck NodePort", "32222",
			},
		},
	}
	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			fake := fake.NewSimpleClientset(testCase.service)
			c := &describeClient{T: t, Namespace: "foo", Interface: fake}
			d := ServiceDescriber{c}
			out, err := d.Describe("foo", "bar", describe.DescriberSettings{ShowEvents: true})
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
	out, err := d.Describe("foo", "bar", describe.DescriberSettings{ShowEvents: true})

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
		if noDescriber, ok := err.(describe.ErrNoDescriber); ok {
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
		t.Errorf("unexpected output: %s", out)
	}

	out, err = DefaultObjectDescriber.DescribeObject(&corev1.Service{ObjectMeta: metav1.ObjectMeta{Name: "foo"}})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(out, "foo") {
		t.Errorf("unexpected output: %s", out)
	}

	out, err = DefaultObjectDescriber.DescribeObject(&corev1.ReplicationController{
		ObjectMeta: metav1.ObjectMeta{Name: "foo"},
		Spec:       corev1.ReplicationControllerSpec{Replicas: utilpointer.Int32Ptr(1)},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(out, "foo") {
		t.Errorf("unexpected output: %s", out)
	}

	out, err = DefaultObjectDescriber.DescribeObject(&corev1.Node{ObjectMeta: metav1.ObjectMeta{Name: "foo"}})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(out, "foo") {
		t.Errorf("unexpected output: %s", out)
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
	deletionTimestamp := metav1.Time{Time: time.Now().UTC().AddDate(10, 0, 0)}
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
	}

	for _, test := range testCases {
		t.Run(test.name, func(t *testing.T) {
			fake := fake.NewSimpleClientset(test.pv)
			c := PersistentVolumeDescriber{fake}
			str, err := c.Describe("foo", "bar", describe.DescriberSettings{ShowEvents: true})
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
	deletionTimestamp := metav1.Time{Time: time.Now().UTC().AddDate(10, 0, 0)}
	testCases := []struct {
		name               string
		pvc                *corev1.PersistentVolumeClaim
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
	}

	for _, test := range testCases {
		t.Run(test.name, func(t *testing.T) {
			fake := fake.NewSimpleClientset(test.pvc)
			c := PersistentVolumeClaimDescriber{fake}
			str, err := c.Describe("foo", "bar", describe.DescriberSettings{ShowEvents: true})
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

func TestDescribeDeployment(t *testing.T) {
	fakeClient := fake.NewSimpleClientset(&appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "bar",
			Namespace: "foo",
		},
		Spec: appsv1.DeploymentSpec{
			Replicas: utilpointer.Int32Ptr(1),
			Selector: &metav1.LabelSelector{},
			Template: v1.PodTemplateSpec{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{Image: "mytest-image:latest"},
					},
				},
			},
		},
	})
	d := DeploymentDescriber{fakeClient}
	out, err := d.Describe("foo", "bar", describe.DescriberSettings{ShowEvents: true})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if !strings.Contains(out, "bar") || !strings.Contains(out, "foo") || !strings.Contains(out, "Containers:") || !strings.Contains(out, "mytest-image:latest") {
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
	out, err := s.Describe("", "foo", describe.DescriberSettings{ShowEvents: true})
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

func TestDescribePodDisruptionBudget(t *testing.T) {
	minAvailable := intstr.FromInt(22)
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
			PodDisruptionsAllowed: 5,
		},
	})
	s := PodDisruptionBudgetDescriber{f}
	out, err := s.Describe("ns1", "pdb1", describe.DescriberSettings{ShowEvents: true})
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
	metricLabelSelector, err := metav1.ParseToLabelSelector("label=value")
	if err != nil {
		t.Errorf("unable to parse label selector: %v", err)
	}
	tests := []struct {
		name string
		hpa  autoscalingv2beta2.HorizontalPodAutoscaler
	}{
		{
			"minReplicas unset",
			autoscalingv2beta2.HorizontalPodAutoscaler{
				Spec: autoscalingv2beta2.HorizontalPodAutoscalerSpec{
					ScaleTargetRef: autoscalingv2beta2.CrossVersionObjectReference{
						Name: "some-rc",
						Kind: "ReplicationController",
					},
					MaxReplicas: 10,
				},
				Status: autoscalingv2beta2.HorizontalPodAutoscalerStatus{
					CurrentReplicas: 4,
					DesiredReplicas: 5,
				},
			},
		},
		{
			"external source type, target average value (no current)",
			autoscalingv2beta2.HorizontalPodAutoscaler{
				Spec: autoscalingv2beta2.HorizontalPodAutoscalerSpec{
					ScaleTargetRef: autoscalingv2beta2.CrossVersionObjectReference{
						Name: "some-rc",
						Kind: "ReplicationController",
					},
					MinReplicas: &minReplicasVal,
					MaxReplicas: 10,
					Metrics: []autoscalingv2beta2.MetricSpec{
						{
							Type: autoscalingv2beta2.ExternalMetricSourceType,
							External: &autoscalingv2beta2.ExternalMetricSource{
								Metric: autoscalingv2beta2.MetricIdentifier{
									Name:     "some-external-metric",
									Selector: metricLabelSelector,
								},
								Target: autoscalingv2beta2.MetricTarget{
									Type:         autoscalingv2beta2.AverageValueMetricType,
									AverageValue: resource.NewMilliQuantity(100, resource.DecimalSI),
								},
							},
						},
					},
				},
				Status: autoscalingv2beta2.HorizontalPodAutoscalerStatus{
					CurrentReplicas: 4,
					DesiredReplicas: 5,
				},
			},
		},
		{
			"external source type, target average value (with current)",
			autoscalingv2beta2.HorizontalPodAutoscaler{
				Spec: autoscalingv2beta2.HorizontalPodAutoscalerSpec{
					ScaleTargetRef: autoscalingv2beta2.CrossVersionObjectReference{
						Name: "some-rc",
						Kind: "ReplicationController",
					},
					MinReplicas: &minReplicasVal,
					MaxReplicas: 10,
					Metrics: []autoscalingv2beta2.MetricSpec{
						{
							Type: autoscalingv2beta2.ExternalMetricSourceType,
							External: &autoscalingv2beta2.ExternalMetricSource{
								Metric: autoscalingv2beta2.MetricIdentifier{
									Name:     "some-external-metric",
									Selector: metricLabelSelector,
								},
								Target: autoscalingv2beta2.MetricTarget{
									Type:         autoscalingv2beta2.AverageValueMetricType,
									AverageValue: resource.NewMilliQuantity(100, resource.DecimalSI),
								},
							},
						},
					},
				},
				Status: autoscalingv2beta2.HorizontalPodAutoscalerStatus{
					CurrentReplicas: 4,
					DesiredReplicas: 5,
					CurrentMetrics: []autoscalingv2beta2.MetricStatus{
						{
							Type: autoscalingv2beta2.ExternalMetricSourceType,
							External: &autoscalingv2beta2.ExternalMetricStatus{
								Metric: autoscalingv2beta2.MetricIdentifier{
									Name:     "some-external-metric",
									Selector: metricLabelSelector,
								},
								Current: autoscalingv2beta2.MetricValueStatus{
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
			autoscalingv2beta2.HorizontalPodAutoscaler{
				Spec: autoscalingv2beta2.HorizontalPodAutoscalerSpec{
					ScaleTargetRef: autoscalingv2beta2.CrossVersionObjectReference{
						Name: "some-rc",
						Kind: "ReplicationController",
					},
					MinReplicas: &minReplicasVal,
					MaxReplicas: 10,
					Metrics: []autoscalingv2beta2.MetricSpec{
						{
							Type: autoscalingv2beta2.ExternalMetricSourceType,
							External: &autoscalingv2beta2.ExternalMetricSource{
								Metric: autoscalingv2beta2.MetricIdentifier{
									Name:     "some-external-metric",
									Selector: metricLabelSelector,
								},
								Target: autoscalingv2beta2.MetricTarget{
									Type:  autoscalingv2beta2.ValueMetricType,
									Value: resource.NewMilliQuantity(100, resource.DecimalSI),
								},
							},
						},
					},
				},
				Status: autoscalingv2beta2.HorizontalPodAutoscalerStatus{
					CurrentReplicas: 4,
					DesiredReplicas: 5,
				},
			},
		},
		{
			"external source type, target value (with current)",
			autoscalingv2beta2.HorizontalPodAutoscaler{
				Spec: autoscalingv2beta2.HorizontalPodAutoscalerSpec{
					ScaleTargetRef: autoscalingv2beta2.CrossVersionObjectReference{
						Name: "some-rc",
						Kind: "ReplicationController",
					},
					MinReplicas: &minReplicasVal,
					MaxReplicas: 10,
					Metrics: []autoscalingv2beta2.MetricSpec{
						{
							Type: autoscalingv2beta2.ExternalMetricSourceType,
							External: &autoscalingv2beta2.ExternalMetricSource{
								Metric: autoscalingv2beta2.MetricIdentifier{
									Name:     "some-external-metric",
									Selector: metricLabelSelector,
								},
								Target: autoscalingv2beta2.MetricTarget{
									Type:  autoscalingv2beta2.ValueMetricType,
									Value: resource.NewMilliQuantity(100, resource.DecimalSI),
								},
							},
						},
					},
				},
				Status: autoscalingv2beta2.HorizontalPodAutoscalerStatus{
					CurrentReplicas: 4,
					DesiredReplicas: 5,
					CurrentMetrics: []autoscalingv2beta2.MetricStatus{
						{
							Type: autoscalingv2beta2.ExternalMetricSourceType,
							External: &autoscalingv2beta2.ExternalMetricStatus{
								Metric: autoscalingv2beta2.MetricIdentifier{
									Name:     "some-external-metric",
									Selector: metricLabelSelector,
								},
								Current: autoscalingv2beta2.MetricValueStatus{
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
			autoscalingv2beta2.HorizontalPodAutoscaler{
				Spec: autoscalingv2beta2.HorizontalPodAutoscalerSpec{
					ScaleTargetRef: autoscalingv2beta2.CrossVersionObjectReference{
						Name: "some-rc",
						Kind: "ReplicationController",
					},
					MinReplicas: &minReplicasVal,
					MaxReplicas: 10,
					Metrics: []autoscalingv2beta2.MetricSpec{
						{
							Type: autoscalingv2beta2.PodsMetricSourceType,
							Pods: &autoscalingv2beta2.PodsMetricSource{
								Metric: autoscalingv2beta2.MetricIdentifier{
									Name: "some-pods-metric",
								},
								Target: autoscalingv2beta2.MetricTarget{
									Type:         autoscalingv2beta2.AverageValueMetricType,
									AverageValue: resource.NewMilliQuantity(100, resource.DecimalSI),
								},
							},
						},
					},
				},
				Status: autoscalingv2beta2.HorizontalPodAutoscalerStatus{
					CurrentReplicas: 4,
					DesiredReplicas: 5,
				},
			},
		},
		{
			"pods source type (with current)",
			autoscalingv2beta2.HorizontalPodAutoscaler{
				Spec: autoscalingv2beta2.HorizontalPodAutoscalerSpec{
					ScaleTargetRef: autoscalingv2beta2.CrossVersionObjectReference{
						Name: "some-rc",
						Kind: "ReplicationController",
					},
					MinReplicas: &minReplicasVal,
					MaxReplicas: 10,
					Metrics: []autoscalingv2beta2.MetricSpec{
						{
							Type: autoscalingv2beta2.PodsMetricSourceType,
							Pods: &autoscalingv2beta2.PodsMetricSource{
								Metric: autoscalingv2beta2.MetricIdentifier{
									Name: "some-pods-metric",
								},
								Target: autoscalingv2beta2.MetricTarget{
									Type:         autoscalingv2beta2.AverageValueMetricType,
									AverageValue: resource.NewMilliQuantity(100, resource.DecimalSI),
								},
							},
						},
					},
				},
				Status: autoscalingv2beta2.HorizontalPodAutoscalerStatus{
					CurrentReplicas: 4,
					DesiredReplicas: 5,
					CurrentMetrics: []autoscalingv2beta2.MetricStatus{
						{
							Type: autoscalingv2beta2.PodsMetricSourceType,
							Pods: &autoscalingv2beta2.PodsMetricStatus{
								Metric: autoscalingv2beta2.MetricIdentifier{
									Name: "some-pods-metric",
								},
								Current: autoscalingv2beta2.MetricValueStatus{
									AverageValue: resource.NewMilliQuantity(50, resource.DecimalSI),
								},
							},
						},
					},
				},
			},
		},
		{
			"object source type (no current)",
			autoscalingv2beta2.HorizontalPodAutoscaler{
				Spec: autoscalingv2beta2.HorizontalPodAutoscalerSpec{
					ScaleTargetRef: autoscalingv2beta2.CrossVersionObjectReference{
						Name: "some-rc",
						Kind: "ReplicationController",
					},
					MinReplicas: &minReplicasVal,
					MaxReplicas: 10,
					Metrics: []autoscalingv2beta2.MetricSpec{
						{
							Type: autoscalingv2beta2.ObjectMetricSourceType,
							Object: &autoscalingv2beta2.ObjectMetricSource{
								DescribedObject: autoscalingv2beta2.CrossVersionObjectReference{
									Name: "some-service",
									Kind: "Service",
								},
								Metric: autoscalingv2beta2.MetricIdentifier{
									Name: "some-service-metric",
								},
								Target: autoscalingv2beta2.MetricTarget{
									Type:  autoscalingv2beta2.ValueMetricType,
									Value: resource.NewMilliQuantity(100, resource.DecimalSI),
								},
							},
						},
					},
				},
				Status: autoscalingv2beta2.HorizontalPodAutoscalerStatus{
					CurrentReplicas: 4,
					DesiredReplicas: 5,
				},
			},
		},
		{
			"object source type (with current)",
			autoscalingv2beta2.HorizontalPodAutoscaler{
				Spec: autoscalingv2beta2.HorizontalPodAutoscalerSpec{
					ScaleTargetRef: autoscalingv2beta2.CrossVersionObjectReference{
						Name: "some-rc",
						Kind: "ReplicationController",
					},
					MinReplicas: &minReplicasVal,
					MaxReplicas: 10,
					Metrics: []autoscalingv2beta2.MetricSpec{
						{
							Type: autoscalingv2beta2.ObjectMetricSourceType,
							Object: &autoscalingv2beta2.ObjectMetricSource{
								DescribedObject: autoscalingv2beta2.CrossVersionObjectReference{
									Name: "some-service",
									Kind: "Service",
								},
								Metric: autoscalingv2beta2.MetricIdentifier{
									Name: "some-service-metric",
								},
								Target: autoscalingv2beta2.MetricTarget{
									Type:  autoscalingv2beta2.ValueMetricType,
									Value: resource.NewMilliQuantity(100, resource.DecimalSI),
								},
							},
						},
					},
				},
				Status: autoscalingv2beta2.HorizontalPodAutoscalerStatus{
					CurrentReplicas: 4,
					DesiredReplicas: 5,
					CurrentMetrics: []autoscalingv2beta2.MetricStatus{
						{
							Type: autoscalingv2beta2.ObjectMetricSourceType,
							Object: &autoscalingv2beta2.ObjectMetricStatus{
								DescribedObject: autoscalingv2beta2.CrossVersionObjectReference{
									Name: "some-service",
									Kind: "Service",
								},
								Metric: autoscalingv2beta2.MetricIdentifier{
									Name: "some-service-metric",
								},
								Current: autoscalingv2beta2.MetricValueStatus{
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
			autoscalingv2beta2.HorizontalPodAutoscaler{
				Spec: autoscalingv2beta2.HorizontalPodAutoscalerSpec{
					ScaleTargetRef: autoscalingv2beta2.CrossVersionObjectReference{
						Name: "some-rc",
						Kind: "ReplicationController",
					},
					MinReplicas: &minReplicasVal,
					MaxReplicas: 10,
					Metrics: []autoscalingv2beta2.MetricSpec{
						{
							Type: autoscalingv2beta2.ResourceMetricSourceType,
							Resource: &autoscalingv2beta2.ResourceMetricSource{
								Name: corev1.ResourceCPU,
								Target: autoscalingv2beta2.MetricTarget{
									Type:         autoscalingv2beta2.AverageValueMetricType,
									AverageValue: resource.NewMilliQuantity(100, resource.DecimalSI),
								},
							},
						},
					},
				},
				Status: autoscalingv2beta2.HorizontalPodAutoscalerStatus{
					CurrentReplicas: 4,
					DesiredReplicas: 5,
				},
			},
		},
		{
			"resource source type, target average value (with current)",
			autoscalingv2beta2.HorizontalPodAutoscaler{
				Spec: autoscalingv2beta2.HorizontalPodAutoscalerSpec{
					ScaleTargetRef: autoscalingv2beta2.CrossVersionObjectReference{
						Name: "some-rc",
						Kind: "ReplicationController",
					},
					MinReplicas: &minReplicasVal,
					MaxReplicas: 10,
					Metrics: []autoscalingv2beta2.MetricSpec{
						{
							Type: autoscalingv2beta2.ResourceMetricSourceType,
							Resource: &autoscalingv2beta2.ResourceMetricSource{
								Name: corev1.ResourceCPU,
								Target: autoscalingv2beta2.MetricTarget{
									Type:         autoscalingv2beta2.AverageValueMetricType,
									AverageValue: resource.NewMilliQuantity(100, resource.DecimalSI),
								},
							},
						},
					},
				},
				Status: autoscalingv2beta2.HorizontalPodAutoscalerStatus{
					CurrentReplicas: 4,
					DesiredReplicas: 5,
					CurrentMetrics: []autoscalingv2beta2.MetricStatus{
						{
							Type: autoscalingv2beta2.ResourceMetricSourceType,
							Resource: &autoscalingv2beta2.ResourceMetricStatus{
								Name: corev1.ResourceCPU,
								Current: autoscalingv2beta2.MetricValueStatus{
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
			autoscalingv2beta2.HorizontalPodAutoscaler{
				Spec: autoscalingv2beta2.HorizontalPodAutoscalerSpec{
					ScaleTargetRef: autoscalingv2beta2.CrossVersionObjectReference{
						Name: "some-rc",
						Kind: "ReplicationController",
					},
					MinReplicas: &minReplicasVal,
					MaxReplicas: 10,
					Metrics: []autoscalingv2beta2.MetricSpec{
						{
							Type: autoscalingv2beta2.ResourceMetricSourceType,
							Resource: &autoscalingv2beta2.ResourceMetricSource{
								Name: corev1.ResourceCPU,
								Target: autoscalingv2beta2.MetricTarget{
									Type:               autoscalingv2beta2.UtilizationMetricType,
									AverageUtilization: &targetUtilizationVal,
								},
							},
						},
					},
				},
				Status: autoscalingv2beta2.HorizontalPodAutoscalerStatus{
					CurrentReplicas: 4,
					DesiredReplicas: 5,
				},
			},
		},
		{
			"resource source type, target utilization (with current)",
			autoscalingv2beta2.HorizontalPodAutoscaler{
				Spec: autoscalingv2beta2.HorizontalPodAutoscalerSpec{
					ScaleTargetRef: autoscalingv2beta2.CrossVersionObjectReference{
						Name: "some-rc",
						Kind: "ReplicationController",
					},
					MinReplicas: &minReplicasVal,
					MaxReplicas: 10,
					Metrics: []autoscalingv2beta2.MetricSpec{
						{
							Type: autoscalingv2beta2.ResourceMetricSourceType,
							Resource: &autoscalingv2beta2.ResourceMetricSource{
								Name: corev1.ResourceCPU,
								Target: autoscalingv2beta2.MetricTarget{
									Type:               autoscalingv2beta2.UtilizationMetricType,
									AverageUtilization: &targetUtilizationVal,
								},
							},
						},
					},
				},
				Status: autoscalingv2beta2.HorizontalPodAutoscalerStatus{
					CurrentReplicas: 4,
					DesiredReplicas: 5,
					CurrentMetrics: []autoscalingv2beta2.MetricStatus{
						{
							Type: autoscalingv2beta2.ResourceMetricSourceType,
							Resource: &autoscalingv2beta2.ResourceMetricStatus{
								Name: corev1.ResourceCPU,
								Current: autoscalingv2beta2.MetricValueStatus{
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
			autoscalingv2beta2.HorizontalPodAutoscaler{
				Spec: autoscalingv2beta2.HorizontalPodAutoscalerSpec{
					ScaleTargetRef: autoscalingv2beta2.CrossVersionObjectReference{
						Name: "some-rc",
						Kind: "ReplicationController",
					},
					MinReplicas: &minReplicasVal,
					MaxReplicas: 10,
					Metrics: []autoscalingv2beta2.MetricSpec{
						{
							Type: autoscalingv2beta2.PodsMetricSourceType,
							Pods: &autoscalingv2beta2.PodsMetricSource{
								Metric: autoscalingv2beta2.MetricIdentifier{
									Name: "some-pods-metric",
								},
								Target: autoscalingv2beta2.MetricTarget{
									Type:         autoscalingv2beta2.AverageValueMetricType,
									AverageValue: resource.NewMilliQuantity(100, resource.DecimalSI),
								},
							},
						},
						{
							Type: autoscalingv2beta2.ResourceMetricSourceType,
							Resource: &autoscalingv2beta2.ResourceMetricSource{
								Name: corev1.ResourceCPU,
								Target: autoscalingv2beta2.MetricTarget{
									Type:               autoscalingv2beta2.UtilizationMetricType,
									AverageUtilization: &targetUtilizationVal,
								},
							},
						},
						{
							Type: autoscalingv2beta2.PodsMetricSourceType,
							Pods: &autoscalingv2beta2.PodsMetricSource{
								Metric: autoscalingv2beta2.MetricIdentifier{
									Name: "other-pods-metric",
								},
								Target: autoscalingv2beta2.MetricTarget{
									Type:         autoscalingv2beta2.AverageValueMetricType,
									AverageValue: resource.NewMilliQuantity(400, resource.DecimalSI),
								},
							},
						},
					},
				},
				Status: autoscalingv2beta2.HorizontalPodAutoscalerStatus{
					CurrentReplicas: 4,
					DesiredReplicas: 5,
					CurrentMetrics: []autoscalingv2beta2.MetricStatus{
						{
							Type: autoscalingv2beta2.PodsMetricSourceType,
							Pods: &autoscalingv2beta2.PodsMetricStatus{
								Metric: autoscalingv2beta2.MetricIdentifier{
									Name: "some-pods-metric",
								},
								Current: autoscalingv2beta2.MetricValueStatus{
									AverageValue: resource.NewMilliQuantity(50, resource.DecimalSI),
								},
							},
						},
						{
							Type: autoscalingv2beta2.ResourceMetricSourceType,
							Resource: &autoscalingv2beta2.ResourceMetricStatus{
								Name: corev1.ResourceCPU,
								Current: autoscalingv2beta2.MetricValueStatus{
									AverageUtilization: &currentUtilizationVal,
									AverageValue:       resource.NewMilliQuantity(40, resource.DecimalSI),
								},
							},
						},
					},
				},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			test.hpa.ObjectMeta = metav1.ObjectMeta{
				Name:      "bar",
				Namespace: "foo",
			}
			fake := fake.NewSimpleClientset(&test.hpa)
			desc := HorizontalPodAutoscalerDescriber{fake}
			str, err := desc.Describe("foo", "bar", describe.DescriberSettings{ShowEvents: true})
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
					Namespace: "foo",
				},
				Source:         corev1.EventSource{Component: "kubelet"},
				Message:        "Item 1",
				FirstTimestamp: metav1.NewTime(time.Date(2014, time.January, 15, 0, 0, 0, 0, time.UTC)),
				LastTimestamp:  metav1.NewTime(time.Date(2014, time.January, 15, 0, 0, 0, 0, time.UTC)),
				Count:          1,
				Type:           corev1.EventTypeNormal,
			},
		},
	}

	m := map[string]describe.Describer{
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
					Replicas: utilpointer.Int32Ptr(1),
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
		// TODO(jchaloup): add tests for:
		// - IngressDescriber
		// - JobDescriber
		"NodeDescriber": &NodeDescriber{
			fake.NewSimpleClientset(&corev1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:     "bar",
					SelfLink: "url/url/url/url",
				},
			}, events),
		},
		"PersistentVolumeDescriber": &PersistentVolumeDescriber{
			fake.NewSimpleClientset(&corev1.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name:     "bar",
					SelfLink: "url/url/url/url",
				},
			}, events),
		},
		"PodDescriber": &PodDescriber{
			fake.NewSimpleClientset(&corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "bar",
					Namespace: "foo",
					SelfLink:  "url/url/url/url",
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
					Replicas: utilpointer.Int32Ptr(1),
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
					Replicas: utilpointer.Int32Ptr(1),
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
			fake.NewSimpleClientset(&autoscalingv2beta2.HorizontalPodAutoscaler{
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
			out, err := d.Describe("foo", "bar", describe.DescriberSettings{ShowEvents: true})
			if err != nil {
				t.Errorf("unexpected error for %q: %v", name, err)
			}
			if !strings.Contains(out, "bar") {
				t.Errorf("unexpected out for %q: %s", name, out)
			}
			if !strings.Contains(out, "Events:") {
				t.Errorf("events not found for %q when ShowEvents=true: %s", name, out)
			}

			out, err = d.Describe("foo", "bar", describe.DescriberSettings{ShowEvents: false})
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
Dummy 2:	present
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
			"apiVersion": "v1",
			"kind":       "Test",
			"dummy1":     "present",
			"dummy2":     "present",
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

func TestDescribePodSecurityPolicy(t *testing.T) {
	expected := []string{
		"Name:\\s*mypsp",
		"Allow Privileged:\\s*false",
		"Default Add Capabilities:\\s*<none>",
		"Required Drop Capabilities:\\s*<none>",
		"Allowed Capabilities:\\s*<none>",
		"Allowed Volume Types:\\s*<none>",
		"Allowed Unsafe Sysctls:\\s*kernel\\.\\*,net\\.ipv4.ip_local_port_range",
		"Forbidden Sysctls:\\s*net\\.ipv4\\.ip_default_ttl",
		"Allow Host Network:\\s*false",
		"Allow Host Ports:\\s*<none>",
		"Allow Host PID:\\s*false",
		"Allow Host IPC:\\s*false",
		"Read Only Root Filesystem:\\s*false",
		"SELinux Context Strategy: RunAsAny",
		"User:\\s*<none>",
		"Role:\\s*<none>",
		"Type:\\s*<none>",
		"Level:\\s*<none>",
		"Run As User Strategy: RunAsAny",
		"FSGroup Strategy: RunAsAny",
		"Supplemental Groups Strategy: RunAsAny",
	}

	fake := fake.NewSimpleClientset(&policyv1beta1.PodSecurityPolicy{
		ObjectMeta: metav1.ObjectMeta{
			Name: "mypsp",
		},
		Spec: policyv1beta1.PodSecurityPolicySpec{
			AllowedUnsafeSysctls: []string{"kernel.*", "net.ipv4.ip_local_port_range"},
			ForbiddenSysctls:     []string{"net.ipv4.ip_default_ttl"},
			SELinux: policyv1beta1.SELinuxStrategyOptions{
				Rule: policyv1beta1.SELinuxStrategyRunAsAny,
			},
			RunAsUser: policyv1beta1.RunAsUserStrategyOptions{
				Rule: policyv1beta1.RunAsUserStrategyRunAsAny,
			},
			FSGroup: policyv1beta1.FSGroupStrategyOptions{
				Rule: policyv1beta1.FSGroupStrategyRunAsAny,
			},
			SupplementalGroups: policyv1beta1.SupplementalGroupsStrategyOptions{
				Rule: policyv1beta1.SupplementalGroupsStrategyRunAsAny,
			},
		},
	})

	c := &describeClient{T: t, Namespace: "", Interface: fake}
	d := PodSecurityPolicyDescriber{c}
	out, err := d.Describe("", "mypsp", describe.DescriberSettings{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	for _, item := range expected {
		if matched, _ := regexp.MatchString(item, out); !matched {
			t.Errorf("Expected to find %q in: %q", item, out)
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
				corev1.ResourceName(corev1.ResourceRequestsMemory): resource.MustParse("0G"),
			},
		},
	})
	c := &describeClient{T: t, Namespace: "foo", Interface: fake}
	d := ResourceQuotaDescriber{c}
	out, err := d.Describe("foo", "bar", describe.DescriberSettings{ShowEvents: true})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	expectedOut := []string{"bar", "foo", "limits.cpu", "2", "limits.memory", "2G", "requests.cpu", "1", "requests.memory", "1G"}
	for _, expected := range expectedOut {
		if !strings.Contains(out, expected) {
			t.Errorf("expected to find %q in output: %q", expected, out)
		}
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
    To: <any> (traffic not restricted by source)
  Policy Types: Ingress, Egress
`

	port80 := intstr.FromInt(80)
	port82 := intstr.FromInt(82)
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
	out, err := d.Describe("", "network-policy-1", describe.DescriberSettings{})
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
	out, err := d.Describe("foo", "bar", describe.DescriberSettings{ShowEvents: true})
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

func TestDescribeNode(t *testing.T) {
	fake := fake.NewSimpleClientset(&corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "bar",
			Namespace: "foo",
		},
		Spec: corev1.NodeSpec{
			Unschedulable: true,
		},
	})
	c := &describeClient{T: t, Namespace: "foo", Interface: fake}
	d := NodeDescriber{c}
	out, err := d.Describe("foo", "bar", describe.DescriberSettings{ShowEvents: true})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	expectedOut := []string{"Unschedulable", "true"}
	for _, expected := range expectedOut {
		if !strings.Contains(out, expected) {
			t.Errorf("expected to find %q in output: %q", expected, out)
		}
	}

}

func TestDescribeStatefulSet(t *testing.T) {
	var partition int32 = 2
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
	out, err := d.Describe("foo", "bar", describe.DescriberSettings{ShowEvents: true})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	expectedOutputs := []string{
		"bar", "foo", "Containers:", "mytest-image:latest", "Update Strategy", "RollingUpdate", "Partition",
	}
	for _, o := range expectedOutputs {
		if !strings.Contains(out, o) {
			t.Errorf("unexpected out: %s", out)
			break
		}
	}
}

// boolPtr returns a pointer to a bool
func boolPtr(b bool) *bool {
	o := b
	return &o
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
				OwnerReferences: []metav1.OwnerReference{{Name: "bar", UID: "123456", Controller: boolPtr(true)}},
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
				OwnerReferences: []metav1.OwnerReference{{Name: "buz", UID: "654321", Controller: boolPtr(true)}},
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
	out, err := d.Describe("foo", "bar", describe.DescriberSettings{ShowEvents: false})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if !strings.Contains(out, "1 Running") {
		t.Errorf("unexpected out: %s", out)
	}
}
