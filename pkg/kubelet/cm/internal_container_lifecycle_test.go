package cm

import (
	"github.com/golang/mock/gomock"
	"k8s.io/api/core/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	kubefeatures "k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/cm/cpumanager"
	"k8s.io/kubernetes/pkg/kubelet/cm/memorymanager"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager"
	"testing"
)

func TestPreStartContainer(t *testing.T) {
	// setup vars

	var tests = []struct {
		name string
		i    internalContainerLifecycleImpl
	}{
		{
			name: "when cpu,memory,topology manager are nil",
			i: internalContainerLifecycleImpl{
				cpuManager:      nil,
				memoryManager:   nil,
				topologyManager: nil,
			},
		},
		{
			name: "when only cpu manager is not nil",
			i: internalContainerLifecycleImpl{
				cpuManager:      cpumanager.NewFakeManager(),
				memoryManager:   nil,
				topologyManager: nil,
			},
		},
		{
			name: "when cpu and memory manager are not nil",
			i: internalContainerLifecycleImpl{
				cpuManager:      cpumanager.NewFakeManager(),
				memoryManager:   memorymanager.NewFakeManager(),
				topologyManager: nil,
			},
		},
		{
			name: "when cpu and topology manager are not nil",
			i: internalContainerLifecycleImpl{
				cpuManager:      cpumanager.NewFakeManager(),
				memoryManager:   nil,
				topologyManager: topologymanager.NewFakeManager(),
			},
		},
		{
			name: "when only memory manager is not nil",
			i: internalContainerLifecycleImpl{
				cpuManager:      nil,
				memoryManager:   memorymanager.NewFakeManager(),
				topologyManager: nil,
			},
		},
		{
			name: "when memory and topology manager is not nil",
			i: internalContainerLifecycleImpl{
				cpuManager:      nil,
				memoryManager:   memorymanager.NewFakeManager(),
				topologyManager: topologymanager.NewFakeManager(),
			},
		},
		{
			name: "when only topology manager is not nil",
			i: internalContainerLifecycleImpl{
				cpuManager:      nil,
				memoryManager:   nil,
				topologyManager: topologymanager.NewFakeManager(),
			},
		},
		{
			name: "when cpu , memory, topology manager are not nil",
			i: internalContainerLifecycleImpl{
				cpuManager:      cpumanager.NewFakeManager(),
				memoryManager:   memorymanager.NewFakeManager(),
				topologyManager: topologymanager.NewFakeManager(),
			},
		},
	}

	for _, test := range tests {
		mockCtrl := gomock.NewController(t)

		pod := &v1.Pod{}
		container := &v1.Container{}
		containerId := "test-id"
		i := test.i

		t.Run(test.name, func(t *testing.T) {

			//("Running PreStart Container",i.PreStartContainer())
			err := i.PreStartContainer(pod, container, containerId)
			if err != nil {
				t.Error("Error Occured!")
			}
			mockCPUManager := cpumanager.NewMockManager(mockCtrl)
			mockMemoryManager := memorymanager.NewMockManager(mockCtrl)
			mockTopologyManager := topologymanager.NewMockManager(mockCtrl)

			if i.cpuManager != nil {
				//i.cpuManager.AddContainer(pod, container, containerId)
				mockCPUManager.EXPECT().AddContainer(pod, container, containerId).Return(nil).AnyTimes()
			} else {
				mockCPUManager.EXPECT().AddContainer(pod, container, containerId).Return(nil).Times(0)
			}

			if i.memoryManager != nil {
				mockMemoryManager.EXPECT().AddContainer(pod, container, containerId).Return(nil).Times(1)
			} else {
				mockMemoryManager.EXPECT().AddContainer(pod, container, containerId).Return(nil).Times(0)
			}

			if i.topologyManager != nil {
				if utilfeature.DefaultFeatureGate.Enabled(kubefeatures.TopologyManager) {
					mockTopologyManager.EXPECT().AddContainer(pod, container, containerId).Return(nil).Times(1)
				} else {
					mockTopologyManager.EXPECT().AddContainer(pod, container, containerId).Return(nil).Times(0)
				}
			} else {
				mockTopologyManager.EXPECT().AddContainer(pod, container, containerId).Return(nil).Times(0)
			}
		})
	}
}

func TestPreStopContainer(t *testing.T) {

}

func TestPostStopContainer(t *testing.T) {

}
