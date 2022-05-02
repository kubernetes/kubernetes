package cm

import (
	"github.com/stretchr/testify/assert"
	"k8s.io/api/core/v1"
	"k8s.io/kubernetes/pkg/kubelet/cm/cpumanager"
	"k8s.io/kubernetes/pkg/kubelet/cm/memorymanager"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager"
	"testing"
)

func TestPreStartContainer(t *testing.T) {
	// setup vars
	i := internalContainerLifecycleImpl{
		cpuManager:      cpumanager.NewFakeManager(),
		memoryManager:   memorymanager.NewFakeManager(),
		topologyManager: topologymanager.NewFakeManager(),
	}
	fakePod := &v1.Pod{}
	fakeContainer := &v1.Container{}
	containerId := "testid"
	// execute and match expected
	err := i.PreStartContainer(fakePod, fakeContainer, containerId)
	if err != nil {
		assert.Error(t, err, "Error Occured when trying to prestart container")
	}
	assert.NotNil(t, i.cpuManager, "cpu manager Container should not be null %v", i.cpuManager)
	assert.NotNil(t, i.memoryManager, "memory manager Container should not be null %v", i.cpuManager)
	assert.NotNil(t, i.topologyManager, "topology manager Container should not be null %v", i.cpuManager)
}
