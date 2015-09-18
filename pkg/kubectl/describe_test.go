/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"fmt"
	"reflect"
	"strings"
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apis/experimental"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/client/unversioned/testclient"
)

type describeClient struct {
	T         *testing.T
	Namespace string
	Err       error
	client.Interface
}

func init() {
	api.ForTesting_ReferencesAllowBlankSelfLinks = true
}

func TestDescribePod(t *testing.T) {
	fake := testclient.NewSimpleFake(&api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name:      "bar",
			Namespace: "foo",
		},
	})
	c := &describeClient{T: t, Namespace: "foo", Interface: fake}
	d := PodDescriber{c}
	out, err := d.Describe("foo", "bar")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if !strings.Contains(out, "bar") || !strings.Contains(out, "Status:") {
		t.Errorf("unexpected out: %s", out)
	}
}

func TestDescribeService(t *testing.T) {
	fake := testclient.NewSimpleFake(&api.Service{
		ObjectMeta: api.ObjectMeta{
			Name:      "bar",
			Namespace: "foo",
		},
	})
	c := &describeClient{T: t, Namespace: "foo", Interface: fake}
	d := ServiceDescriber{c}
	out, err := d.Describe("foo", "bar")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if !strings.Contains(out, "Labels:") || !strings.Contains(out, "bar") {
		t.Errorf("unexpected out: %s", out)
	}
}

func TestPodDescribeResultsSorted(t *testing.T) {
	// Arrange
	fake := testclient.NewSimpleFake(&api.EventList{
		Items: []api.Event{
			{
				Source:         api.EventSource{Component: "kubelet"},
				Message:        "Item 1",
				FirstTimestamp: unversioned.NewTime(time.Date(2014, time.January, 15, 0, 0, 0, 0, time.UTC)),
				LastTimestamp:  unversioned.NewTime(time.Date(2014, time.January, 15, 0, 0, 0, 0, time.UTC)),
				Count:          1,
			},
			{
				Source:         api.EventSource{Component: "scheduler"},
				Message:        "Item 2",
				FirstTimestamp: unversioned.NewTime(time.Date(1987, time.June, 17, 0, 0, 0, 0, time.UTC)),
				LastTimestamp:  unversioned.NewTime(time.Date(1987, time.June, 17, 0, 0, 0, 0, time.UTC)),
				Count:          1,
			},
			{
				Source:         api.EventSource{Component: "kubelet"},
				Message:        "Item 3",
				FirstTimestamp: unversioned.NewTime(time.Date(2002, time.December, 25, 0, 0, 0, 0, time.UTC)),
				LastTimestamp:  unversioned.NewTime(time.Date(2002, time.December, 25, 0, 0, 0, 0, time.UTC)),
				Count:          1,
			},
		},
	})
	c := &describeClient{T: t, Namespace: "foo", Interface: fake}
	d := PodDescriber{c}

	// Act
	out, err := d.Describe("foo", "bar")

	// Assert
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	VerifyDatesInOrder(out, "\n" /* rowDelimiter */, "\t" /* columnDelimiter */, t)
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
						StartedAt: unversioned.NewTime(time.Now()),
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
						StartedAt:  unversioned.NewTime(time.Now()),
						FinishedAt: unversioned.NewTime(time.Now()),
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
						StartedAt: unversioned.NewTime(time.Now()),
					},
				},
				LastTerminationState: api.ContainerState{
					Terminated: &api.ContainerStateTerminated{
						StartedAt:  unversioned.NewTime(time.Now().Add(time.Second * 3)),
						FinishedAt: unversioned.NewTime(time.Now()),
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
		//env
		{
			container: api.Container{Name: "test", Image: "image", Env: []api.EnvVar{{Name: "envname", Value: "xyz"}}},
			status: api.ContainerStatus{
				Name:         "test",
				Ready:        true,
				RestartCount: 7,
			},
			expectedElements: []string{"test", "State", "Waiting", "Ready", "True", "Restart Count", "7", "Image", "image", "envname", "xyz"},
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
		describeContainers(&pod, out)
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
		pods                         []*api.Pod
		expectedReqs, expectedLimits map[api.ResourceName]resource.Quantity
	}{
		{
			pods: []*api.Pod{
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
		if !reflect.DeepEqual(reqs, testCase.expectedReqs) {
			t.Errorf("Expected %v, got %v", testCase.expectedReqs, reqs)
		}
	}
}

func TestPersistentVolumeDescriber(t *testing.T) {
	tests := map[string]*api.PersistentVolume{

		"hostpath": {
			Spec: api.PersistentVolumeSpec{
				PersistentVolumeSource: api.PersistentVolumeSource{
					HostPath: &api.HostPathVolumeSource{},
				},
			},
		},
		"gce": {
			Spec: api.PersistentVolumeSpec{
				PersistentVolumeSource: api.PersistentVolumeSource{
					GCEPersistentDisk: &api.GCEPersistentDiskVolumeSource{},
				},
			},
		},
		"ebs": {
			Spec: api.PersistentVolumeSpec{
				PersistentVolumeSource: api.PersistentVolumeSource{
					AWSElasticBlockStore: &api.AWSElasticBlockStoreVolumeSource{},
				},
			},
		},
		"nfs": {
			Spec: api.PersistentVolumeSpec{
				PersistentVolumeSource: api.PersistentVolumeSource{
					NFS: &api.NFSVolumeSource{},
				},
			},
		},
		"iscsi": {
			Spec: api.PersistentVolumeSpec{
				PersistentVolumeSource: api.PersistentVolumeSource{
					ISCSI: &api.ISCSIVolumeSource{},
				},
			},
		},
		"gluster": {
			Spec: api.PersistentVolumeSpec{
				PersistentVolumeSource: api.PersistentVolumeSource{
					Glusterfs: &api.GlusterfsVolumeSource{},
				},
			},
		},
		"rbd": {
			Spec: api.PersistentVolumeSpec{
				PersistentVolumeSource: api.PersistentVolumeSource{
					RBD: &api.RBDVolumeSource{},
				},
			},
		},
	}

	for name, pv := range tests {
		fake := testclient.NewSimpleFake(pv)
		c := PersistentVolumeDescriber{fake}
		str, err := c.Describe("foo", "bar")
		if err != nil {
			t.Errorf("Unexpected error for test %s: %v", name, err)
		}
		if str == "" {
			t.Errorf("Unexpected empty string for test %s.  Expected PV Describer output", name)
		}
	}
}

func TestDescribeDeployment(t *testing.T) {
	fake := testclient.NewSimpleFake(&experimental.Deployment{
		ObjectMeta: api.ObjectMeta{
			Name:      "bar",
			Namespace: "foo",
		},
		Spec: experimental.DeploymentSpec{
			Template: &api.PodTemplateSpec{},
		},
	})
	c := &describeClient{T: t, Namespace: "foo", Interface: fake}
	d := DeploymentDescriber{c}
	out, err := d.Describe("foo", "bar")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if !strings.Contains(out, "bar") || !strings.Contains(out, "foo") {
		t.Errorf("unexpected out: %s", out)
	}
}
