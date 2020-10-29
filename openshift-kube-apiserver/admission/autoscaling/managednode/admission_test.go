package managednode

import (
	"context"
	"fmt"
	"testing"

	configv1 "github.com/openshift/api/config/v1"

	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/tools/cache"

	configv1listers "github.com/openshift/client-go/config/listers/config/v1"

	corev1 "k8s.io/api/core/v1"
	kapi "k8s.io/kubernetes/pkg/apis/core"
)

const (
	managedCapacityLabel = "management.workload.openshift.io/cores"
)

func TestAdmit(t *testing.T) {
	tests := []struct {
		name          string
		node          *corev1.Node
		infra         *configv1.Infrastructure
		expectedError error
	}{
		{
			name:  "should succeed when CPU partitioning is set to AllNodes",
			node:  testNodeWithManagementResource(true),
			infra: testClusterInfra(configv1.CPUPartitioningAllNodes),
		},
		{
			name:  "should succeed when CPU partitioning is set to None",
			node:  testNodeWithManagementResource(true),
			infra: testClusterInfra(configv1.CPUPartitioningNone),
		},
		{
			name:          "should fail when nodes don't have capacity",
			node:          testNodeWithManagementResource(false),
			infra:         testClusterInfra(configv1.CPUPartitioningAllNodes),
			expectedError: fmt.Errorf("node does not contain resource information, this is required for clusters with workload partitioning enabled"),
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			m, err := getMockNode(test.infra)
			if err != nil {
				t.Fatalf("%s: failed to get mock managementNode: %v", test.name, err)
			}

			attrs := admission.NewAttributesRecord(
				test.node, nil, schema.GroupVersionKind{},
				test.node.Namespace, test.node.Name, kapi.Resource("nodes").WithVersion("version"), "",
				admission.Create, nil, false, fakeUser())
			err = m.Validate(context.TODO(), attrs, nil)

			if err == nil && test.expectedError != nil {
				t.Fatalf("%s: the expected error %v, got nil", test.name, test.expectedError)
			}
		})
	}
}

func testNodeWithManagementResource(capacity bool) *corev1.Node {
	q := resource.NewQuantity(16000, resource.DecimalSI)
	node := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: "managed-node",
		},
	}
	if capacity {
		node.Status.Capacity = corev1.ResourceList{
			managedCapacityLabel: *q,
		}
	}
	return node
}

func testClusterInfra(mode configv1.CPUPartitioningMode) *configv1.Infrastructure {
	return &configv1.Infrastructure{
		ObjectMeta: metav1.ObjectMeta{
			Name: infraClusterName,
		},
		Status: configv1.InfrastructureStatus{
			APIServerURL:           "test",
			ControlPlaneTopology:   configv1.HighlyAvailableTopologyMode,
			InfrastructureTopology: configv1.HighlyAvailableTopologyMode,
			CPUPartitioning:        mode,
		},
	}
}

func fakeUser() user.Info {
	return &user.DefaultInfo{
		Name: "testuser",
	}
}

func getMockNode(infra *configv1.Infrastructure) (*managedNodeValidate, error) {
	m := &managedNodeValidate{
		Handler:               admission.NewHandler(admission.Create),
		client:                &fake.Clientset{},
		infraConfigLister:     fakeInfraConfigLister(infra),
		infraConfigListSynced: func() bool { return true },
	}
	if err := m.ValidateInitialization(); err != nil {
		return nil, err
	}

	return m, nil
}

func fakeInfraConfigLister(infra *configv1.Infrastructure) configv1listers.InfrastructureLister {
	indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{})
	if infra != nil {
		_ = indexer.Add(infra)
	}
	return configv1listers.NewInfrastructureLister(indexer)
}
