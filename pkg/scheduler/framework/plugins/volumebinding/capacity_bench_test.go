package volumebinding

import (
	"fmt"
	"sync"
	"testing"

	v1 "k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	storagelisters "k8s.io/client-go/listers/storage/v1"
	"k8s.io/klog/v2"
	"k8s.io/utils/ptr"
)

// copyingCapacityLister copies the slice the way an informer cache List does.
type copyingCapacityLister struct {
	items []*storagev1.CSIStorageCapacity
}

func (l *copyingCapacityLister) List(labels.Selector) ([]*storagev1.CSIStorageCapacity, error) {
	out := make([]*storagev1.CSIStorageCapacity, len(l.items))
	copy(out, l.items)
	return out, nil
}

func (l *copyingCapacityLister) CSIStorageCapacities(string) storagelisters.CSIStorageCapacityNamespaceLister {
	return nil
}

type oneCSIDriverLister struct {
	driver *storagev1.CSIDriver
}

func (l oneCSIDriverLister) List(labels.Selector) ([]*storagev1.CSIDriver, error) {
	return []*storagev1.CSIDriver{l.driver}, nil
}

func (l oneCSIDriverLister) Get(name string) (*storagev1.CSIDriver, error) {
	if l.driver != nil && l.driver.Name == name {
		return l.driver, nil
	}
	return nil, fmt.Errorf("csidriver %q not found", name)
}

func TestStorageCapacityIndexAgreesWithScan(t *testing.T) {
	logger := klog.Background()
	nodeA := makeNode("node-a").withLabel(nodeLabelKey, "node-a").Node
	nodeB := makeNode("node-b").withLabel(nodeLabelKey, "node-b").Node
	nodeC := makeNode("node-c").withLabel(nodeLabelKey, "node-c").Node
	nilTopo := makeCapacity("nil-topo", waitClassWithProvisioner, nil, "100Gi", "")
	nilTopo.NodeTopology = nil
	tooSmall := makeCapacity("too-small", waitClassWithProvisioner, nodeA, "10Gi", "512Mi")
	otherClass := makeCapacity("other", "other-class", nodeA, "10Gi", "")
	specific := makeCapacity("specific", waitClassWithProvisioner, nodeA, "10Gi", "")
	both := makeCapacity("both", waitClassWithProvisioner, nil, "8Gi", "")
	both.NodeTopology = &metav1.LabelSelector{MatchExpressions: []metav1.LabelSelectorRequirement{{
		Key: nodeLabelKey, Operator: metav1.LabelSelectorOpIn, Values: []string{"node-a", "node-b"},
	}}}
	notIn := makeCapacity("notin", waitClassWithProvisioner, nil, "9Gi", "")
	notIn.NodeTopology = &metav1.LabelSelector{MatchExpressions: []metav1.LabelSelectorRequirement{{
		Key: nodeLabelKey, Operator: metav1.LabelSelectorOpNotIn, Values: []string{"node-a"},
	}}}
	wide := makeCapacity("wide", waitClassWithProvisioner, nil, "7Gi", "")
	caps := []*storagev1.CSIStorageCapacity{nilTopo, tooSmall, otherClass, specific, both, notIn, wide}
	binder := &volumeBinder{
		csiDriverLister: oneCSIDriverLister{driver: &storagev1.CSIDriver{
			ObjectMeta: metav1.ObjectMeta{Name: provisioner},
			Spec:       storagev1.CSIDriverSpec{StorageCapacity: ptr.To(true)},
		}},
		csiStorageCapacityLister: &copyingCapacityLister{items: caps},
	}
	class := &storagev1.StorageClass{ObjectMeta: metav1.ObjectMeta{Name: waitClassWithProvisioner}, Provisioner: provisioner}
	claim := makeTestPVC("provisioned-pvc", "1Gi", "", pvcUnbound, "", "1", &waitClassWithProvisioner)
	claims := &PodVolumeClaims{capacityIndex: sync.OnceValues(func() (*storageCapacityIndex, error) {
		return binder.buildStorageCapacityIndex(logger, map[string][]*v1.PersistentVolume{class.Name: nil})
	})}
	for _, node := range []*v1.Node{nodeA, nodeB, nodeC} {
		_, linear, err := binder.hasEnoughCapacity(logger, provisioner, claim, class, node, nil)
		if err != nil {
			t.Fatal(err)
		}
		_, indexed, err := binder.hasEnoughCapacity(logger, provisioner, claim, class, node, claims)
		if err != nil {
			t.Fatal(err)
		}
		linearName, indexedName := "", ""
		if linear != nil {
			linearName = linear.Name
		}
		if indexed != nil {
			indexedName = indexed.Name
		}
		if linearName != indexedName {
			t.Errorf("node %s: linear %q, indexed %q", node.Name, linearName, indexedName)
		}
	}
}

// BenchmarkHasEnoughCapacity measures one scheduling cycle of storage-capacity checks.
func BenchmarkHasEnoughCapacity(b *testing.B) {
	logger := klog.Background()
	for _, n := range []int{1000, 4000} {
		for _, indexed := range []bool{false, true} {
			name := fmt.Sprintf("nodes=%d/capacities=%d/indexed=%t", n, n, indexed)
			b.Run(name, func(b *testing.B) {
				nodes := make([]*v1.Node, n)
				caps := make([]*storagev1.CSIStorageCapacity, n)
				for i := 0; i < n; i++ {
					name := fmt.Sprintf("node-%d", i)
					nodes[i] = &v1.Node{ObjectMeta: metav1.ObjectMeta{
						Name:   name,
						Labels: map[string]string{nodeLabelKey: name},
					}}
					caps[i] = makeCapacity(name, waitClassWithProvisioner, nodes[i], "100Gi", "")
				}
				binder := &volumeBinder{
					csiDriverLister: oneCSIDriverLister{driver: &storagev1.CSIDriver{
						ObjectMeta: metav1.ObjectMeta{Name: provisioner},
						Spec:       storagev1.CSIDriverSpec{StorageCapacity: ptr.To(true)},
					}},
					csiStorageCapacityLister: &copyingCapacityLister{items: caps},
				}
				class := &storagev1.StorageClass{
					ObjectMeta:  metav1.ObjectMeta{Name: waitClassWithProvisioner},
					Provisioner: provisioner,
				}
				classes := map[string][]*v1.PersistentVolume{class.Name: nil}
				claim := makeTestPVC("provisioned-pvc", "1Gi", "", pvcUnbound, "", "1", &waitClassWithProvisioner)
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					var claims *PodVolumeClaims
					if indexed {
						claims = &PodVolumeClaims{capacityIndex: sync.OnceValues(func() (*storageCapacityIndex, error) {
							return binder.buildStorageCapacityIndex(logger, classes)
						})}
					}
					for _, node := range nodes {
						ok, found, err := binder.hasEnoughCapacity(logger, provisioner, claim, class, node, claims)
						if err != nil || !ok || found == nil || found.Name != node.Name {
							b.Fatalf("node %s: ok=%v found=%v err=%v", node.Name, ok, found, err)
						}
					}
				}
			})
		}
	}
}
