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

package kubectl

import (
	"bytes"
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
	"testing"
	"time"

	"k8s.io/kubernetes/federation/apis/federation"
	fedfake "k8s.io/kubernetes/federation/client/clientset_generated/federation_internalclientset/fake"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	metav1 "k8s.io/kubernetes/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/apis/policy"
	"k8s.io/kubernetes/pkg/apis/storage"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/fake"
	versionedfake "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/fake"
	"k8s.io/kubernetes/pkg/util/intstr"
)

type describeClient struct {
	T         *testing.T
	Namespace string
	Err       error
	internalclientset.Interface
}

func TestDescribePod(t *testing.T) {
	fake := fake.NewSimpleClientset(&api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name:      "bar",
			Namespace: "foo",
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
}

func TestDescribePodTolerations(t *testing.T) {

	podTolerations := []api.Toleration{{Key: "key1", Value: "value1"},
		{Key: "key2", Value: "value2"}}
	pt, _ := json.Marshal(podTolerations)
	fake := fake.NewSimpleClientset(&api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name:      "bar",
			Namespace: "foo",
			Annotations: map[string]string{
				api.TolerationsAnnotationKey: string(pt),
			},
		},
	})
	c := &describeClient{T: t, Namespace: "foo", Interface: fake}
	d := PodDescriber{c}
	out, err := d.Describe("foo", "bar", DescriberSettings{})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if !strings.Contains(out, "key1=value1") || !strings.Contains(out, "key2=value2") || !strings.Contains(out, "Tolerations:") {
		t.Errorf("unexpected out: %s", out)
	}
}

func TestDescribeNamespace(t *testing.T) {
	fake := fake.NewSimpleClientset(&api.Namespace{
		ObjectMeta: api.ObjectMeta{
			Name: "myns",
		},
	})
	c := &describeClient{T: t, Namespace: "", Interface: fake}
	d := NamespaceDescriber{c}
	out, err := d.Describe("", "myns", DescriberSettings{ShowEvents: true})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if !strings.Contains(out, "myns") {
		t.Errorf("unexpected out: %s", out)
	}
}

func TestDescribeService(t *testing.T) {
	fake := fake.NewSimpleClientset(&api.Service{
		ObjectMeta: api.ObjectMeta{
			Name:      "bar",
			Namespace: "foo",
		},
	})
	c := &describeClient{T: t, Namespace: "foo", Interface: fake}
	d := ServiceDescriber{c}
	out, err := d.Describe("foo", "bar", DescriberSettings{ShowEvents: true})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if !strings.Contains(out, "Labels:") || !strings.Contains(out, "bar") {
		t.Errorf("unexpected out: %s", out)
	}
}

func TestPodDescribeResultsSorted(t *testing.T) {
	// Arrange
	fake := fake.NewSimpleClientset(
		&api.EventList{
			Items: []api.Event{
				{
					ObjectMeta:     api.ObjectMeta{Name: "one"},
					Source:         api.EventSource{Component: "kubelet"},
					Message:        "Item 1",
					FirstTimestamp: metav1.NewTime(time.Date(2014, time.January, 15, 0, 0, 0, 0, time.UTC)),
					LastTimestamp:  metav1.NewTime(time.Date(2014, time.January, 15, 0, 0, 0, 0, time.UTC)),
					Count:          1,
					Type:           api.EventTypeNormal,
				},
				{
					ObjectMeta:     api.ObjectMeta{Name: "two"},
					Source:         api.EventSource{Component: "scheduler"},
					Message:        "Item 2",
					FirstTimestamp: metav1.NewTime(time.Date(1987, time.June, 17, 0, 0, 0, 0, time.UTC)),
					LastTimestamp:  metav1.NewTime(time.Date(1987, time.June, 17, 0, 0, 0, 0, time.UTC)),
					Count:          1,
					Type:           api.EventTypeNormal,
				},
				{
					ObjectMeta:     api.ObjectMeta{Name: "three"},
					Source:         api.EventSource{Component: "kubelet"},
					Message:        "Item 3",
					FirstTimestamp: metav1.NewTime(time.Date(2002, time.December, 25, 0, 0, 0, 0, time.UTC)),
					LastTimestamp:  metav1.NewTime(time.Date(2002, time.December, 25, 0, 0, 0, 0, time.UTC)),
					Count:          1,
					Type:           api.EventTypeNormal,
				},
			},
		},
		&api.Pod{ObjectMeta: api.ObjectMeta{Namespace: "foo", Name: "bar"}},
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
	testCases := []struct {
		container        api.Container
		status           api.ContainerStatus
		expectedElements []string
	}{
		// Running state.
		{
			container: api.Container{Name: "test", Image: "image"},
			status: api.ContainerStatus{
				Name: "test",
				State: api.ContainerState{
					Running: &api.ContainerStateRunning{
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
			container: api.Container{Name: "test", Image: "image"},
			status: api.ContainerStatus{
				Name: "test",
				State: api.ContainerState{
					Waiting: &api.ContainerStateWaiting{
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
			container: api.Container{Name: "test", Image: "image"},
			status: api.ContainerStatus{
				Name: "test",
				State: api.ContainerState{
					Terminated: &api.ContainerStateTerminated{
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
			container: api.Container{Name: "test", Image: "image"},
			status: api.ContainerStatus{
				Name: "test",
				State: api.ContainerState{
					Running: &api.ContainerStateRunning{
						StartedAt: metav1.NewTime(time.Now()),
					},
				},
				LastTerminationState: api.ContainerState{
					Terminated: &api.ContainerStateTerminated{
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
			container: api.Container{Name: "test", Image: "image"},
			status: api.ContainerStatus{
				Name:         "test",
				Ready:        true,
				RestartCount: 7,
			},
			expectedElements: []string{"test", "State", "Waiting", "Ready", "True", "Restart Count", "7", "Image", "image"},
		},
		// Env
		{
			container: api.Container{Name: "test", Image: "image", Env: []api.EnvVar{{Name: "envname", Value: "xyz"}}},
			status: api.ContainerStatus{
				Name:         "test",
				Ready:        true,
				RestartCount: 7,
			},
			expectedElements: []string{"test", "State", "Waiting", "Ready", "True", "Restart Count", "7", "Image", "image", "envname", "xyz"},
		},
		// Command
		{
			container: api.Container{Name: "test", Image: "image", Command: []string{"sleep", "1000"}},
			status: api.ContainerStatus{
				Name:         "test",
				Ready:        true,
				RestartCount: 7,
			},
			expectedElements: []string{"test", "State", "Waiting", "Ready", "True", "Restart Count", "7", "Image", "image", "sleep", "1000"},
		},
		// Args
		{
			container: api.Container{Name: "test", Image: "image", Args: []string{"time", "1000"}},
			status: api.ContainerStatus{
				Name:         "test",
				Ready:        true,
				RestartCount: 7,
			},
			expectedElements: []string{"test", "State", "Waiting", "Ready", "True", "Restart Count", "7", "Image", "image", "time", "1000"},
		},
		// Using limits.
		{
			container: api.Container{
				Name:  "test",
				Image: "image",
				Resources: api.ResourceRequirements{
					Limits: api.ResourceList{
						api.ResourceName(api.ResourceCPU):     resource.MustParse("1000"),
						api.ResourceName(api.ResourceMemory):  resource.MustParse("4G"),
						api.ResourceName(api.ResourceStorage): resource.MustParse("20G"),
					},
				},
			},
			status: api.ContainerStatus{
				Name:         "test",
				Ready:        true,
				RestartCount: 7,
			},
			expectedElements: []string{"cpu", "1k", "memory", "4G", "storage", "20G"},
		},
		// Using requests.
		{
			container: api.Container{
				Name:  "test",
				Image: "image",
				Resources: api.ResourceRequirements{
					Requests: api.ResourceList{
						api.ResourceName(api.ResourceCPU):     resource.MustParse("1000"),
						api.ResourceName(api.ResourceMemory):  resource.MustParse("4G"),
						api.ResourceName(api.ResourceStorage): resource.MustParse("20G"),
					},
				},
			},
			expectedElements: []string{"cpu", "1k", "memory", "4G", "storage", "20G"},
		},
	}

	for i, testCase := range testCases {
		out := new(bytes.Buffer)
		pod := api.Pod{
			Spec: api.PodSpec{
				Containers: []api.Container{testCase.container},
			},
			Status: api.PodStatus{
				ContainerStatuses: []api.ContainerStatus{testCase.status},
			},
		}
		writer := &PrefixWriter{out}
		describeContainers("Containers", pod.Spec.Containers, pod.Status.ContainerStatuses, EnvValueRetriever(&pod), writer, "")
		output := out.String()
		for _, expected := range testCase.expectedElements {
			if !strings.Contains(output, expected) {
				t.Errorf("Test case %d: expected to find %q in output: %q", i, expected, output)
			}
		}
	}
}

func TestDescribers(t *testing.T) {
	first := &api.Event{}
	second := &api.Pod{}
	var third *api.Pod
	testErr := fmt.Errorf("test")
	d := Describers{}
	d.Add(
		func(e *api.Event, p *api.Pod) (string, error) {
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
			if !reflect.DeepEqual(noDescriber.Types, []string{"*api.Event", "*api.Pod", "*api.Pod"}) {
				t.Errorf("unexpected describer: %v", err)
			}
		} else {
			t.Errorf("unexpected error type: %v", err)
		}
	}

	d.Add(
		func(e *api.Event) (string, error) {
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
	out, err := DefaultObjectDescriber.DescribeObject(&api.Pod{ObjectMeta: api.ObjectMeta{Name: "foo"}})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(out, "foo") {
		t.Errorf("unexpected output: %s", out)
	}

	out, err = DefaultObjectDescriber.DescribeObject(&api.Service{ObjectMeta: api.ObjectMeta{Name: "foo"}})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(out, "foo") {
		t.Errorf("unexpected output: %s", out)
	}

	out, err = DefaultObjectDescriber.DescribeObject(&api.ReplicationController{ObjectMeta: api.ObjectMeta{Name: "foo"}})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(out, "foo") {
		t.Errorf("unexpected output: %s", out)
	}

	out, err = DefaultObjectDescriber.DescribeObject(&api.Node{ObjectMeta: api.ObjectMeta{Name: "foo"}})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(out, "foo") {
		t.Errorf("unexpected output: %s", out)
	}
}

func TestGetPodsTotalRequests(t *testing.T) {
	testCases := []struct {
		pods                         *api.PodList
		expectedReqs, expectedLimits map[api.ResourceName]resource.Quantity
	}{
		{
			pods: &api.PodList{
				Items: []api.Pod{
					{
						Spec: api.PodSpec{
							Containers: []api.Container{
								{
									Resources: api.ResourceRequirements{
										Requests: api.ResourceList{
											api.ResourceName(api.ResourceCPU):     resource.MustParse("1"),
											api.ResourceName(api.ResourceMemory):  resource.MustParse("300Mi"),
											api.ResourceName(api.ResourceStorage): resource.MustParse("1G"),
										},
									},
								},
								{
									Resources: api.ResourceRequirements{
										Requests: api.ResourceList{
											api.ResourceName(api.ResourceCPU):     resource.MustParse("90m"),
											api.ResourceName(api.ResourceMemory):  resource.MustParse("120Mi"),
											api.ResourceName(api.ResourceStorage): resource.MustParse("200M"),
										},
									},
								},
							},
						},
					},
					{
						Spec: api.PodSpec{
							Containers: []api.Container{
								{
									Resources: api.ResourceRequirements{
										Requests: api.ResourceList{
											api.ResourceName(api.ResourceCPU):     resource.MustParse("60m"),
											api.ResourceName(api.ResourceMemory):  resource.MustParse("43Mi"),
											api.ResourceName(api.ResourceStorage): resource.MustParse("500M"),
										},
									},
								},
								{
									Resources: api.ResourceRequirements{
										Requests: api.ResourceList{
											api.ResourceName(api.ResourceCPU):     resource.MustParse("34m"),
											api.ResourceName(api.ResourceMemory):  resource.MustParse("83Mi"),
											api.ResourceName(api.ResourceStorage): resource.MustParse("700M"),
										},
									},
								},
							},
						},
					},
				},
			},
			expectedReqs: map[api.ResourceName]resource.Quantity{
				api.ResourceName(api.ResourceCPU):     resource.MustParse("1.184"),
				api.ResourceName(api.ResourceMemory):  resource.MustParse("546Mi"),
				api.ResourceName(api.ResourceStorage): resource.MustParse("2.4G"),
			},
		},
	}

	for _, testCase := range testCases {
		reqs, _, err := getPodsTotalRequestsAndLimits(testCase.pods)
		if err != nil {
			t.Errorf("Unexpected error %v", err)
		}
		if !api.Semantic.DeepEqual(reqs, testCase.expectedReqs) {
			t.Errorf("Expected %v, got %v", testCase.expectedReqs, reqs)
		}
	}
}

func TestPersistentVolumeDescriber(t *testing.T) {
	tests := map[string]*api.PersistentVolume{

		"hostpath": {
			ObjectMeta: api.ObjectMeta{Name: "bar"},
			Spec: api.PersistentVolumeSpec{
				PersistentVolumeSource: api.PersistentVolumeSource{
					HostPath: &api.HostPathVolumeSource{},
				},
			},
		},
		"gce": {
			ObjectMeta: api.ObjectMeta{Name: "bar"},
			Spec: api.PersistentVolumeSpec{
				PersistentVolumeSource: api.PersistentVolumeSource{
					GCEPersistentDisk: &api.GCEPersistentDiskVolumeSource{},
				},
			},
		},
		"ebs": {
			ObjectMeta: api.ObjectMeta{Name: "bar"},
			Spec: api.PersistentVolumeSpec{
				PersistentVolumeSource: api.PersistentVolumeSource{
					AWSElasticBlockStore: &api.AWSElasticBlockStoreVolumeSource{},
				},
			},
		},
		"nfs": {
			ObjectMeta: api.ObjectMeta{Name: "bar"},
			Spec: api.PersistentVolumeSpec{
				PersistentVolumeSource: api.PersistentVolumeSource{
					NFS: &api.NFSVolumeSource{},
				},
			},
		},
		"iscsi": {
			ObjectMeta: api.ObjectMeta{Name: "bar"},
			Spec: api.PersistentVolumeSpec{
				PersistentVolumeSource: api.PersistentVolumeSource{
					ISCSI: &api.ISCSIVolumeSource{},
				},
			},
		},
		"gluster": {
			ObjectMeta: api.ObjectMeta{Name: "bar"},
			Spec: api.PersistentVolumeSpec{
				PersistentVolumeSource: api.PersistentVolumeSource{
					Glusterfs: &api.GlusterfsVolumeSource{},
				},
			},
		},
		"rbd": {
			ObjectMeta: api.ObjectMeta{Name: "bar"},
			Spec: api.PersistentVolumeSpec{
				PersistentVolumeSource: api.PersistentVolumeSource{
					RBD: &api.RBDVolumeSource{},
				},
			},
		},
		"quobyte": {
			ObjectMeta: api.ObjectMeta{Name: "bar"},
			Spec: api.PersistentVolumeSpec{
				PersistentVolumeSource: api.PersistentVolumeSource{
					Quobyte: &api.QuobyteVolumeSource{},
				},
			},
		},
		"cinder": {
			ObjectMeta: api.ObjectMeta{Name: "bar"},
			Spec: api.PersistentVolumeSpec{
				PersistentVolumeSource: api.PersistentVolumeSource{
					Cinder: &api.CinderVolumeSource{},
				},
			},
		},
	}

	for name, pv := range tests {
		fake := fake.NewSimpleClientset(pv)
		c := PersistentVolumeDescriber{fake}
		str, err := c.Describe("foo", "bar", DescriberSettings{ShowEvents: true})
		if err != nil {
			t.Errorf("Unexpected error for test %s: %v", name, err)
		}
		if str == "" {
			t.Errorf("Unexpected empty string for test %s.  Expected PV Describer output", name)
		}
	}
}

func TestDescribeDeployment(t *testing.T) {
	fake := fake.NewSimpleClientset()
	versionedFake := versionedfake.NewSimpleClientset(&v1beta1.Deployment{
		ObjectMeta: v1.ObjectMeta{
			Name:      "bar",
			Namespace: "foo",
		},
		Spec: v1beta1.DeploymentSpec{
			Selector: &metav1.LabelSelector{},
			Template: v1.PodTemplateSpec{},
		},
	})
	d := DeploymentDescriber{fake, versionedFake}
	out, err := d.Describe("foo", "bar", DescriberSettings{ShowEvents: true})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if !strings.Contains(out, "bar") || !strings.Contains(out, "foo") {
		t.Errorf("unexpected out: %s", out)
	}
}

func TestDescribeCluster(t *testing.T) {
	cluster := federation.Cluster{
		ObjectMeta: api.ObjectMeta{
			Name:            "foo",
			ResourceVersion: "4",
			Labels: map[string]string{
				"name": "foo",
			},
		},
		Spec: federation.ClusterSpec{
			ServerAddressByClientCIDRs: []federation.ServerAddressByClientCIDR{
				{
					ClientCIDR:    "0.0.0.0/0",
					ServerAddress: "localhost:8888",
				},
			},
		},
		Status: federation.ClusterStatus{
			Conditions: []federation.ClusterCondition{
				{Type: federation.ClusterReady, Status: api.ConditionTrue},
			},
		},
	}
	fake := fedfake.NewSimpleClientset(&cluster)
	d := ClusterDescriber{Interface: fake}
	out, err := d.Describe("any", "foo", DescriberSettings{ShowEvents: true})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if !strings.Contains(out, "foo") {
		t.Errorf("unexpected out: %s", out)
	}
}

func TestDescribeStorageClass(t *testing.T) {
	f := fake.NewSimpleClientset(&storage.StorageClass{
		ObjectMeta: api.ObjectMeta{
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
	})
	s := StorageClassDescriber{f}
	out, err := s.Describe("", "foo", DescriberSettings{ShowEvents: true})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if !strings.Contains(out, "foo") {
		t.Errorf("unexpected out: %s", out)
	}
}

func TestDescribePodDisruptionBudget(t *testing.T) {
	f := fake.NewSimpleClientset(&policy.PodDisruptionBudget{
		ObjectMeta: api.ObjectMeta{
			Namespace:         "ns1",
			Name:              "pdb1",
			CreationTimestamp: metav1.Time{Time: time.Now().Add(1.9e9)},
		},
		Spec: policy.PodDisruptionBudgetSpec{
			MinAvailable: intstr.FromInt(22),
		},
		Status: policy.PodDisruptionBudgetStatus{
			PodDisruptionsAllowed: 5,
		},
	})
	s := PodDisruptionBudgetDescriber{f}
	out, err := s.Describe("ns1", "pdb1", DescriberSettings{ShowEvents: true})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if !strings.Contains(out, "pdb1") {
		t.Errorf("unexpected out: %s", out)
	}
}

func TestDescribeEvents(t *testing.T) {

	events := &api.EventList{
		Items: []api.Event{
			{
				ObjectMeta: api.ObjectMeta{
					Namespace: "foo",
				},
				Source:         api.EventSource{Component: "kubelet"},
				Message:        "Item 1",
				FirstTimestamp: metav1.NewTime(time.Date(2014, time.January, 15, 0, 0, 0, 0, time.UTC)),
				LastTimestamp:  metav1.NewTime(time.Date(2014, time.January, 15, 0, 0, 0, 0, time.UTC)),
				Count:          1,
				Type:           api.EventTypeNormal,
			},
		},
	}

	m := map[string]Describer{
		"DaemonSetDescriber": &DaemonSetDescriber{
			fake.NewSimpleClientset(&extensions.DaemonSet{
				ObjectMeta: api.ObjectMeta{
					Name:      "bar",
					Namespace: "foo",
				},
			}, events),
		},
		"DeploymentDescriber": &DeploymentDescriber{
			fake.NewSimpleClientset(events),
			versionedfake.NewSimpleClientset(&v1beta1.Deployment{
				ObjectMeta: v1.ObjectMeta{
					Name:      "bar",
					Namespace: "foo",
				},
				Spec: v1beta1.DeploymentSpec{
					Selector: &metav1.LabelSelector{},
				},
			}),
		},
		"EndpointsDescriber": &EndpointsDescriber{
			fake.NewSimpleClientset(&api.Endpoints{
				ObjectMeta: api.ObjectMeta{
					Name:      "bar",
					Namespace: "foo",
				},
			}, events),
		},
		// TODO(jchaloup): add tests for:
		// - HorizontalPodAutoscalerDescriber
		// - IngressDescriber
		// - JobDescriber
		"NodeDescriber": &NodeDescriber{
			fake.NewSimpleClientset(&api.Node{
				ObjectMeta: api.ObjectMeta{
					Name:     "bar",
					SelfLink: "url/url/url",
				},
			}, events),
		},
		"PersistentVolumeDescriber": &PersistentVolumeDescriber{
			fake.NewSimpleClientset(&api.PersistentVolume{
				ObjectMeta: api.ObjectMeta{
					Name:     "bar",
					SelfLink: "url/url/url",
				},
			}, events),
		},
		"PodDescriber": &PodDescriber{
			fake.NewSimpleClientset(&api.Pod{
				ObjectMeta: api.ObjectMeta{
					Name:      "bar",
					Namespace: "foo",
					SelfLink:  "url/url/url",
				},
			}, events),
		},
		"ReplicaSetDescriber": &ReplicaSetDescriber{
			fake.NewSimpleClientset(&extensions.ReplicaSet{
				ObjectMeta: api.ObjectMeta{
					Name:      "bar",
					Namespace: "foo",
				},
			}, events),
		},
		"ReplicationControllerDescriber": &ReplicationControllerDescriber{
			fake.NewSimpleClientset(&api.ReplicationController{
				ObjectMeta: api.ObjectMeta{
					Name:      "bar",
					Namespace: "foo",
				},
			}, events),
		},
		"Service": &ServiceDescriber{
			fake.NewSimpleClientset(&api.Service{
				ObjectMeta: api.ObjectMeta{
					Name:      "bar",
					Namespace: "foo",
				},
			}, events),
		},
		"StorageClass": &StorageClassDescriber{
			fake.NewSimpleClientset(&storage.StorageClass{
				ObjectMeta: api.ObjectMeta{
					Name: "bar",
				},
			}, events),
		},
	}

	for name, d := range m {
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
	}
}
