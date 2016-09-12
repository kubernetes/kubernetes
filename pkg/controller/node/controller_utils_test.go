package node

import (
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/fake"
)

func TestNodeOutageTaintAddedRemoved(t *testing.T) {
	fakeNow := unversioned.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC)
	node := &api.Node{
		ObjectMeta: api.ObjectMeta{
			Name:              "node0",
			CreationTimestamp: fakeNow,
			Labels: map[string]string{
				unversioned.LabelZoneRegion:        "region1",
				unversioned.LabelZoneFailureDomain: "zone1",
			},
			Annotations: map[string]string{},
		},
	}

	nodeHandler := &FakeNodeHandler{Clientset: fake.NewSimpleClientset()}
	if err := addNodeOutageTaint(nodeHandler, node); err != nil {
		t.Errorf("unexpected error %v", err)
	}
	taints, _ := api.GetTaintsFromNodeAnnotations(node.Annotations)
	if len(taints) != 1 {
		t.Errorf("unexpected taints lengths %v", taints)
	}

	if err := removeNodeOutageTaint(nodeHandler, node); err != nil {
		t.Errorf("unexpected error %v", err)
	}
	taints, _ = api.GetTaintsFromNodeAnnotations(node.Annotations)
	if len(taints) != 0 {
		t.Errorf("taints should be empty %v", taints)
	}
}
