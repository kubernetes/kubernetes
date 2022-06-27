package cm

import (
	"github.com/golang/mock/gomock"
	v1 "k8s.io/api/core/v1"
	"testing"
)

func TestPreStartContainer(t *testing.T) {
	// setup vars
	mockCtrl := gomock.NewController(t)

	var tests = []struct {
		name string
		i    *MockInternalContainerLifecycle
	}{
		{
			name: "When Internal Container Lifecycle is nil",
			i:    nil,
		},
		{
			name: "Internal Container Lifecycle is not nil",
			i:    NewMockInternalContainerLifecycle(mockCtrl),
		},
	}

	for _, test := range tests {
		pod := &v1.Pod{}
		container := &v1.Container{}
		containerId := "test-id"

		t.Run(test.name, func(t *testing.T) {
			if test.i != nil {
				test.i.EXPECT().PreStartContainer(pod, container, containerId).Times(1)
				test.i.PreStartContainer(pod, container, containerId)
			}
		})
	}
}

func TestPreStopContainer(t *testing.T) {

}

func TestPostStopContainer(t *testing.T) {
	mockCtrl := gomock.NewController(t)
	tests := []struct {
		name string
		i    *MockInternalContainerLifecycle
	}{
		{
			name: "when topology manager feature gate enabled",
			i:    nil,
		},
		{
			name: "when topology manager feature gate disabled",
			i:    NewMockInternalContainerLifecycle(mockCtrl),
		},
	}

	containerId := "test-id"

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if test.i != nil {
				test.i.EXPECT().PostStopContainer(containerId).Times(1)
				test.i.PostStopContainer(containerId)
			}
		})
	}
}
