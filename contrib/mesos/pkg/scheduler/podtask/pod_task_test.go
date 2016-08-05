/*
Copyright 2015 The Kubernetes Authors.

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
	"reflect"
	"testing"

	"github.com/gogo/protobuf/proto"
	mesos "github.com/mesos/mesos-go/mesosproto"
	mutil "github.com/mesos/mesos-go/mesosutil"
	"github.com/stretchr/testify/assert"
	"k8s.io/kubernetes/contrib/mesos/pkg/node"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/meta"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/podtask/hostport"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/resources"
	"k8s.io/kubernetes/pkg/api"
)

const (
	t_min_cpu = 128
	t_min_mem = 128
)

func fakePodTask(id string, allowedRoles, defaultRoles []string) *T {
	t, _ := New(
		api.NewDefaultContext(),
		Config{
			Prototype:        &mesos.ExecutorInfo{},
			FrameworkRoles:   allowedRoles,
			DefaultPodRoles:  defaultRoles,
			HostPortStrategy: hostport.StrategyWildcard,
		},
		&api.Pod{
			ObjectMeta: api.ObjectMeta{
				Name:      id,
				Namespace: api.NamespaceDefault,
			},
		},
	)

	return t
}

func TestRoles(t *testing.T) {
	assert := assert.New(t)

	for i, tt := range []struct {
		annotations    map[string]string
		frameworkRoles []string
		want           []string
	}{
		{
			map[string]string{},
			nil,
			starRole,
		},
		{
			map[string]string{"other": "label"},
			nil,
			starRole,
		},
		{
			map[string]string{meta.RolesKey: ""},
			nil,
			[]string{},
		},
		{
			map[string]string{
				"other":       "label",
				meta.RolesKey: ",  , ,",
			},
			nil,
			[]string{},
		},
		{
			map[string]string{meta.RolesKey: "forbiddenRole"},
			[]string{"allowedRole"},
			[]string{},
		},
		{
			map[string]string{meta.RolesKey: "*, , *, ,slave_public,"},
			[]string{"*", "slave_public"},
			[]string{"*", "slave_public"},
		},
		{
			map[string]string{meta.RolesKey: "role3,role2,role1"},
			[]string{"role1", "role4"},
			[]string{"role1"},
		},
		{
			map[string]string{},
			[]string{"role1"},
			[]string{"*"},
		},
	} {
		task := fakePodTask("test", tt.frameworkRoles, starRole)
		task.Pod.ObjectMeta.Annotations = tt.annotations
		assert.True(reflect.DeepEqual(task.Roles(), tt.want), "test #%d got %#v want %#v", i, task.Roles(), tt.want)
	}
}

type mockRegistry struct{}

func (mr mockRegistry) New(nodename string, resources []*mesos.Resource) *mesos.ExecutorInfo {
	return &mesos.ExecutorInfo{
		Resources: resources,
	}
}

func (mr mockRegistry) Get(nodename string) (*mesos.ExecutorInfo, error) {
	panic("N/A")
}

func (mr mockRegistry) Invalidate(hostname string) {
	panic("N/A")
}

func TestEmptyOffer(t *testing.T) {
	t.Parallel()
	task := fakePodTask("foo", nil, nil)

	task.Pod.Spec = api.PodSpec{
		Containers: []api.Container{{
			Name: "a",
		}},
	}

	defaultProc := NewDefaultProcurement(
		&mesos.ExecutorInfo{
			Resources: []*mesos.Resource{
				mutil.NewScalarResource("cpus", 1.0),
				mutil.NewScalarResource("mem", 64.0),
			},
		},
		mockRegistry{},
	)

	if err := defaultProc.Procure(
		task,
		&api.Node{},
		NewProcureState(&mesos.Offer{}),
	); err == nil {
		t.Fatalf("accepted empty offer")
	}
}

func TestNoPortsInPodOrOffer(t *testing.T) {
	t.Parallel()
	task := fakePodTask("foo", nil, nil)

	task.Pod.Spec = api.PodSpec{
		Containers: []api.Container{{
			Name: "a",
		}},
	}

	executor := &mesos.ExecutorInfo{
		Resources: []*mesos.Resource{
			mutil.NewScalarResource("cpus", 1.0),
			mutil.NewScalarResource("mem", 64.0),
		},
	}

	defaultProc := NewDefaultProcurement(executor, mockRegistry{})

	offer := &mesos.Offer{
		Resources: []*mesos.Resource{
			mutil.NewScalarResource("cpus", 0.001),
			mutil.NewScalarResource("mem", 0.001),
		},
	}

	if err := defaultProc.Procure(
		task,
		nil,
		NewProcureState(offer),
	); err == nil {
		t.Fatalf("accepted offer %v:", offer)
	}

	offer = &mesos.Offer{
		Resources: []*mesos.Resource{
			mutil.NewScalarResource("cpus", t_min_cpu),
			mutil.NewScalarResource("mem", t_min_mem),
		},
	}

	if err := defaultProc.Procure(
		task,
		nil,
		NewProcureState(offer),
	); err != nil {
		t.Fatalf("did not accepted offer %v:", offer)
	}
}

func TestAcceptOfferPorts(t *testing.T) {
	t.Parallel()
	task := fakePodTask("foo", nil, nil)
	pod := &task.Pod

	defaultProc := NewDefaultProcurement(
		&mesos.ExecutorInfo{},
		mockRegistry{},
	)

	offer := &mesos.Offer{
		Resources: []*mesos.Resource{
			mutil.NewScalarResource("cpus", t_min_cpu),
			mutil.NewScalarResource("mem", t_min_mem),
			resources.NewPorts("*", 1, 1),
		},
	}

	if err := defaultProc.Procure(
		task,
		&api.Node{},
		NewProcureState(offer),
	); err != nil {
		t.Fatalf("did not accepted offer %v:", offer)
	}

	pod.Spec = api.PodSpec{
		Containers: []api.Container{{
			Ports: []api.ContainerPort{{
				HostPort: 123,
			}},
		}},
	}

	if err := defaultProc.Procure(
		task,
		&api.Node{},
		NewProcureState(offer),
	); err == nil {
		t.Fatalf("accepted offer %v:", offer)
	}

	pod.Spec.Containers[0].Ports[0].HostPort = 1

	if err := defaultProc.Procure(
		task,
		&api.Node{},
		NewProcureState(offer),
	); err != nil {
		t.Fatalf("did not accepted offer %v:", offer)
	}

	pod.Spec.Containers[0].Ports[0].HostPort = 0

	if err := defaultProc.Procure(
		task,
		&api.Node{},
		NewProcureState(offer),
	); err != nil {
		t.Fatalf("did not accepted offer %v:", offer)
	}

	offer.Resources = []*mesos.Resource{
		mutil.NewScalarResource("cpus", t_min_cpu),
		mutil.NewScalarResource("mem", t_min_mem),
	}

	if err := defaultProc.Procure(
		task,
		&api.Node{},
		NewProcureState(offer),
	); err == nil {
		t.Fatalf("accepted offer %v:", offer)
	}

	pod.Spec.Containers[0].Ports[0].HostPort = 1

	if err := defaultProc.Procure(
		task,
		&api.Node{},
		NewProcureState(offer),
	); err == nil {
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
	expected := "foo.bar.pod"
	if name != expected {
		t.Fatalf("expected %q instead of %q", expected, name)
	}

	p.Namespace = ""
	name = generateTaskName(p)
	expected = "foo.default.pod"
	if name != expected {
		t.Fatalf("expected %q instead of %q", expected, name)
	}
}

func TestNodeSelector(t *testing.T) {
	t.Parallel()

	newNode := func(hostName string, l map[string]string) *api.Node {
		nodeLabels := map[string]string{"kubernetes.io/hostname": hostName}
		if l != nil {
			for k, v := range l {
				nodeLabels[k] = v
			}
		}
		return &api.Node{
			ObjectMeta: api.ObjectMeta{
				Name:   hostName,
				Labels: nodeLabels,
			},
			Spec: api.NodeSpec{
				ExternalID: hostName,
			},
		}
	}
	node1 := newNode("node1", node.SlaveAttributesToLabels([]*mesos.Attribute{
		newTextAttribute("rack", "a"),
		newTextAttribute("gen", "2014"),
		newScalarAttribute("num", 42.0),
	}))
	node2 := newNode("node2", node.SlaveAttributesToLabels([]*mesos.Attribute{
		newTextAttribute("rack", "b"),
		newTextAttribute("gen", "2015"),
		newScalarAttribute("num", 0.0),
	}))
	labels3 := node.SlaveAttributesToLabels([]*mesos.Attribute{
		newTextAttribute("rack", "c"),
		newTextAttribute("gen", "2015"),
		newScalarAttribute("old", 42),
	})
	labels3["some.other/label"] = "43"
	node3 := newNode("node3", labels3)

	tests := []struct {
		selector map[string]string
		node     *api.Node
		ok       bool
		desc     string
	}{
		{map[string]string{"k8s.mesosphere.io/attribute-rack": "a"}, node1, true, "label value matches"},
		{map[string]string{"k8s.mesosphere.io/attribute-rack": "b"}, node1, false, "label value does not match"},
		{map[string]string{"k8s.mesosphere.io/attribute-rack": "a", "k8s.mesosphere.io/attribute-gen": "2014"}, node1, true, "multiple required labels match"},
		{map[string]string{"k8s.mesosphere.io/attribute-rack": "a", "k8s.mesosphere.io/attribute-gen": "2015"}, node1, false, "one label does not match"},
		{map[string]string{"k8s.mesosphere.io/attribute-rack": "a", "k8s.mesosphere.io/attribute-num": "42"}, node1, true, "scalar label matches"},
		{map[string]string{"k8s.mesosphere.io/attribute-rack": "a", "k8s.mesosphere.io/attribute-num": "43"}, node1, false, "scalar label does not match"},

		{map[string]string{"kubernetes.io/hostname": "node1"}, node1, true, "hostname label matches"},
		{map[string]string{"kubernetes.io/hostname": "node2"}, node1, false, "hostname label does not match"},
		{map[string]string{"kubernetes.io/hostname": "node2"}, node2, true, "hostname label matches"},

		{map[string]string{"some.other/label": "43"}, node1, false, "non-slave attribute does not match"},
		{map[string]string{"some.other/label": "43"}, node3, true, "non-slave attribute matches"},
	}

	defaultProc := NewDefaultProcurement(
		&mesos.ExecutorInfo{},
		mockRegistry{},
	)

	for _, ts := range tests {
		task := fakePodTask("foo", nil, nil)
		task.Pod.Spec.NodeSelector = ts.selector
		offer := &mesos.Offer{
			Resources: []*mesos.Resource{
				mutil.NewScalarResource("cpus", t_min_cpu),
				mutil.NewScalarResource("mem", t_min_mem),
			},
			Hostname: &ts.node.Name,
		}

		err := defaultProc.Procure(
			task,
			ts.node,
			NewProcureState(offer),
		)

		ok := err == nil
		if ts.ok != ok {
			t.Fatalf("expected acceptance of offer for selector %v to be %v, got %v: %q", ts.selector, ts.ok, ok, ts.desc)
		}
	}
}

func newTextAttribute(name string, val string) *mesos.Attribute {
	return &mesos.Attribute{
		Name: proto.String(name),
		Type: mesos.Value_TEXT.Enum(),
		Text: &mesos.Value_Text{Value: &val},
	}
}

func newScalarAttribute(name string, val float64) *mesos.Attribute {
	return &mesos.Attribute{
		Name:   proto.String(name),
		Type:   mesos.Value_SCALAR.Enum(),
		Scalar: &mesos.Value_Scalar{Value: proto.Float64(val)},
	}
}
