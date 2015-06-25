/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package podtask

import (
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	mesos "github.com/mesos/mesos-go/mesosproto"
	mutil "github.com/mesos/mesos-go/mesosutil"
)

const (
	t_min_cpu = 128
	t_min_mem = 128
)

func fakePodTask(id string) (*T, error) {
	return New(api.NewDefaultContext(), "", api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name:      id,
			Namespace: api.NamespaceDefault,
		},
	}, &mesos.ExecutorInfo{})
}

func TestEmptyOffer(t *testing.T) {
	t.Parallel()
	task, err := fakePodTask("foo")
	if err != nil {
		t.Fatal(err)
	}
	if ok := task.AcceptOffer(nil); ok {
		t.Fatalf("accepted nil offer")
	}
	if ok := task.AcceptOffer(&mesos.Offer{}); ok {
		t.Fatalf("accepted empty offer")
	}
}

func TestNoPortsInPodOrOffer(t *testing.T) {
	t.Parallel()
	task, err := fakePodTask("foo")
	if err != nil || task == nil {
		t.Fatal(err)
	}

	offer := &mesos.Offer{
		Resources: []*mesos.Resource{
			mutil.NewScalarResource("cpus", 0.001),
			mutil.NewScalarResource("mem", 0.001),
		},
	}
	if ok := task.AcceptOffer(offer); ok {
		t.Fatalf("accepted offer %v:", offer)
	}

	offer = &mesos.Offer{
		Resources: []*mesos.Resource{
			mutil.NewScalarResource("cpus", t_min_cpu),
			mutil.NewScalarResource("mem", t_min_mem),
		},
	}
	if ok := task.AcceptOffer(offer); !ok {
		t.Fatalf("did not accepted offer %v:", offer)
	}
}

func TestAcceptOfferPorts(t *testing.T) {
	t.Parallel()
	task, _ := fakePodTask("foo")
	pod := &task.Pod

	offer := &mesos.Offer{
		Resources: []*mesos.Resource{
			mutil.NewScalarResource("cpus", t_min_cpu),
			mutil.NewScalarResource("mem", t_min_mem),
			rangeResource("ports", []uint64{1, 1}),
		},
	}
	if ok := task.AcceptOffer(offer); !ok {
		t.Fatalf("did not accepted offer %v:", offer)
	}

	pod.Spec = api.PodSpec{
		Containers: []api.Container{{
			Ports: []api.ContainerPort{{
				HostPort: 123,
			}},
		}},
	}
	if ok := task.AcceptOffer(offer); ok {
		t.Fatalf("accepted offer %v:", offer)
	}

	pod.Spec.Containers[0].Ports[0].HostPort = 1
	if ok := task.AcceptOffer(offer); !ok {
		t.Fatalf("did not accepted offer %v:", offer)
	}

	pod.Spec.Containers[0].Ports[0].HostPort = 0
	if ok := task.AcceptOffer(offer); !ok {
		t.Fatalf("did not accepted offer %v:", offer)
	}

	offer.Resources = []*mesos.Resource{
		mutil.NewScalarResource("cpus", t_min_cpu),
		mutil.NewScalarResource("mem", t_min_mem),
	}
	if ok := task.AcceptOffer(offer); ok {
		t.Fatalf("accepted offer %v:", offer)
	}

	pod.Spec.Containers[0].Ports[0].HostPort = 1
	if ok := task.AcceptOffer(offer); ok {
		t.Fatalf("accepted offer %v:", offer)
	}
}

func TestGeneratePodName(t *testing.T) {
	p := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name:      "foo",
			Namespace: "bar",
		},
	}
	name := generateTaskName(p)
	expected := "foo.bar.pods"
	if name != expected {
		t.Fatalf("expected %q instead of %q", expected, name)
	}

	p.Namespace = ""
	name = generateTaskName(p)
	expected = "foo.default.pods"
	if name != expected {
		t.Fatalf("expected %q instead of %q", expected, name)
	}
}
