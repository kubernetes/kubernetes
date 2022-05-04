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
		i internalContainerLifecycleImpl
	}{
		{
			i: internalContainerLifecycleImpl{
				cpuManager:      nil,
				memoryManager:   nil,
				topologyManager: nil,
			},
		},
		{
			i: internalContainerLifecycleImpl{
				cpuManager:      cpumanager.NewFakeManager(),
				memoryManager:   nil,
				topologyManager: nil,
			},
		},
		{
			i: internalContainerLifecycleImpl{
				cpuManager:      cpumanager.NewFakeManager(),
				memoryManager:   memorymanager.NewFakeManager(),
				topologyManager: nil,
			},
		},
		{
			i: internalContainerLifecycleImpl{
				cpuManager:      cpumanager.NewFakeManager(),
				memoryManager:   nil,
				topologyManager: topologymanager.NewFakeManager(),
			},
		},
		{
			i: internalContainerLifecycleImpl{
				cpuManager:      nil,
				memoryManager:   memorymanager.NewFakeManager(),
				topologyManager: nil,
			},
		},
		{
			i: internalContainerLifecycleImpl{
				cpuManager:      nil,
				memoryManager:   memorymanager.NewFakeManager(),
				topologyManager: topologymanager.NewFakeManager(),
			},
		},
		{
			i: internalContainerLifecycleImpl{
				cpuManager:      nil,
				memoryManager:   nil,
				topologyManager: topologymanager.NewFakeManager(),
			},
		},
		{
			i: internalContainerLifecycleImpl{
				cpuManager:      cpumanager.NewFakeManager(),
				memoryManager:   memorymanager.NewFakeManager(),
				topologyManager: topologymanager.NewFakeManager(),
			},
		},
	}

	for _, test := range tests {
		mockCtrl := gomock.NewController(t)
		defer mockCtrl.Finish()

		pod := &v1.Pod{}
		container := &v1.Container{}
		containerId := "test-id"
		i := test.i

		//("Running PreStart Container",i.PreStartContainer())
		i.PreStartContainer(pod, container, containerId)

		mockCPUManager := cpumanager.NewMockManager(mockCtrl)
		mockMemoryManager := memorymanager.NewMockManager(mockCtrl)
		mockTopologyManager := topologymanager.NewMockManager(mockCtrl)

		if i.cpuManager != nil {
			mockCPUManager.EXPECT().AddContainer(pod, container, containerId).Return(nil).Times(1)
		} else {
			mockCPUManager.EXPECT().AddContainer(pod, container, containerId).Return(nil).Times(0)
		}

		if i.memoryManager != nil {
			mockMemoryManager.EXPECT().AddContainer(pod, container, containerId).Return(nil).Times(1)
		} else {
			mockMemoryManager.EXPECT().AddContainer(pod, container, containerId).Return(nil).Times(0)
		}

		if i.topologyManager != nil && utilfeature.DefaultFeatureGate.Enabled(kubefeatures.TopologyManager) {
			mockTopologyManager.EXPECT().AddContainer(pod, container, containerId).Return(nil).Times(1)
		} else {
			mockTopologyManager.EXPECT().AddContainer(pod, container, containerId).Return(nil).Times(0)
		}

	}
}

func TestPreStopContainer(t *testing.T) {

}

func TestPostStopContainer(t *testing.T) {

}
