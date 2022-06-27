package cm

import (
	"github.com/golang/mock/gomock"
	"github.com/stretchr/testify/require"
	"k8s.io/api/core/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	kubefeatures "k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/cm/cpumanager"
	"k8s.io/kubernetes/pkg/kubelet/cm/memorymanager"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager"

	"testing"
)

func TestPreStartContainer(t *testing.T) {
	mockCtrl := gomock.NewController(t)

	var tests = []struct {
		name            string
		i               internalContainerLifecycleImpl
		cpuManager      cpumanager.MockManager
		memoryManager   memorymanager.MockManager
		topologyManager topologymanager.MockManager
	}{
		// setup vars
		//{
		//	name: "when cpu,memory,topology manager are nil",
		//},
		//{
		//	name: "when only cpu manager is not nil",
		//
		//	cpuManager:      *cpumanager.NewMockManager(mockCtrl),
		//	memoryManager:   nil,
		//	topologyManager: nil,
		//},
		//{
		//	name: "when cpu and memory manager are not nil",
		//	//i: internalContainerLifecycleImpl{
		//	cpuManager:      cpumanager.NewMockManager(mockCtrl),
		//	memoryManager:   memorymanager.NewMockManager(mockCtrl),
		//	topologyManager: nil,
		//	//},
		//},
		//{
		//	name: "when cpu and topology manager are not nil",
		//	//i: internalContainerLifecycleImpl{
		//	cpuManager:      cpumanager.NewMockManager(mockCtrl),
		//	memoryManager:   nil,
		//	topologyManager: topologymanager.NewMockManager(mockCtrl),
		//	//},
		//},
		//{
		//	name: "when only memory manager is not nil",
		//	//i: internalContainerLifecycleImpl{
		//	cpuManager:      nil,
		//	memoryManager:   memorymanager.NewMockManager(mockCtrl),
		//	topologyManager: nil,
		//	//},
		//},
		//{
		//	name: "when memory and topology manager is not nil",
		//	//i: internalContainerLifecycleImpl{
		//	cpuManager:      nil,
		//	memoryManager:   memorymanager.NewMockManager(mockCtrl),
		//	topologyManager: topologymanager.NewMockManager(mockCtrl),
		//	//},
		//},
		//{
		//	name: "when only topology manager is not nil",
		//	//i: internalContainerLifecycleImpl{
		//	cpuManager:      nil,
		//	memoryManager:   nil,
		//	topologyManager: topologymanager.NewMockManager(mockCtrl),
		//	//},
		//},
		{
			name: "when cpu, memory, topology manager are not nil",
			//i: internalContainerLifecycleImpl{
			cpuManager:      *cpumanager.NewMockManager(mockCtrl),
			memoryManager:   *memorymanager.NewMockManager(mockCtrl),
			topologyManager: *topologymanager.NewMockManager(mockCtrl),
			//},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			i := test.i
			i.cpuManager = &test.cpuManager
			i.memoryManager = &test.memoryManager
			i.topologyManager = &test.topologyManager

			//assert.NoError(t, i.PreStartContainer(pod, container, containerId), "No error expected")
			//assert.NoError(t, i.PreStartContainer(pod, container, containerId), "PreStartContainer throws error when no error expected")
			// run the function

			pod := &v1.Pod{}
			container := &v1.Container{}
			containerId := "someid"

			store := NewMockInternalContainerLifecycle(mockCtrl)

			store.EXPECT().PreStartContainer(gomock.Any(), gomock.Any(), gomock.Eq(containerId)).Times(1)
			test.cpuManager.EXPECT().AddContainer(gomock.Any(), gomock.Any(), gomock.Eq(containerId)).Times(1)
			test.memoryManager.EXPECT().AddContainer(gomock.Any(), gomock.Any(), gomock.Eq(containerId)).Times(1)
			test.topologyManager.EXPECT().AddContainer(gomock.Any(), gomock.Any(), gomock.Eq(containerId)).Times(1)

			err := test.i.PreStartContainer(pod, container, containerId)
			require.NoError(t, err, "No error expected")
			//if i.cpuManager != nil {
			//test.cpuManager.AddContainer(pod, container, containerId)
			//test.cpuManager.EXPECT().AddContainer(pod, container, containerId).Times(1)
			//} else {
			//mockCPUManager.EXPECT().AddContainer(pod, container, containerId).Times(0)
			//}
			//
			//if i.memoryManager != nil {
			//	test.memoryManager.EXPECT().AddContainer(pod, container, containerId).Times(1)
			//} else {
			//			mockMemoryManager.EXPECT().AddContainer(pod, container, containerId).Times(0)
			//}
			//
			//if i.topologyManager != nil {
			//	if utilfeature.DefaultFeatureGate.Enabled(kubefeatures.TopologyManager) {
			//		test.topologyManager.EXPECT().AddContainer(pod, container, containerId).Times(1)
			//	} else {
			//				mockTopologyManager.EXPECT().AddContainer(pod, container, containerId).Times(0)
			//}
			//} else {
			//	//			mockTopologyManager.EXPECT().AddContainer(pod, container, containerId).Times(0)
			//}
		})
	}
}

func TestPreStopContainer(t *testing.T) {

}

func TestPostStopContainer(t *testing.T) {
	tests := []struct {
		name string
		i    internalContainerLifecycleImpl
	}{
		{
			name: "when topology manager feature gate enabled",
			i: internalContainerLifecycleImpl{
				cpuManager:      cpumanager.NewFakeManager(),
				memoryManager:   nil,
				topologyManager: topologymanager.NewFakeManager(),
			},
		},
		{
			name: "when topology manager feature gate disabled",
			i: internalContainerLifecycleImpl{
				cpuManager:      nil,
				memoryManager:   nil,
				topologyManager: topologymanager.NewFakeManager(),
			},
		},
	}

	mockCtrl := gomock.NewController(t)
	containerId := "test-id"

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			mockTopologyManager := topologymanager.NewMockManager(mockCtrl)
			test.i.PostStopContainer(containerId)

			if utilfeature.DefaultFeatureGate.Enabled(kubefeatures.TopologyManager) {
				mockTopologyManager.EXPECT().RemoveContainer(containerId).Times(1).Return(nil)
			} else {
				mockTopologyManager.EXPECT().RemoveContainer(containerId).Times(0)
			}
		})

	}
}
