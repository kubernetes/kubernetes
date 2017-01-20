package petset

import (
	"fmt"
	"testing"
)

func TestNewStatefulPod(t *testing.T) {
	set := newStatefulSet(3)
	pod := newStatefulPod(set, 0)
	if pod.exists {
		t.Error("New statefulPod should not exist")
	}
	if pod.dirty {
		t.Error("New statefulPod should not be dirty")
	}
	if !pod.hasValidId() {
		t.Error("New statefulPod has an invalid identity")
	}
	if !pod.hasValidVolumes() {
		t.Error("new statefulPod has invalid volumes")
	}
	if pod.terminating() {
		t.Error("new statefulPod should not be terminating")
	}
	if pod.runningAndReady() {
		t.Error("New statefulPod not be running and ready")
	}
}

func TestFromV1Pod(t *testing.T) {
	set := newStatefulSet(3)
	pod := runningPod(set.Namespace, fmt.Sprintf("%s-%d", set.Name, 0))
	stateful := fromV1Pod(set, pod)
	if stateful.ordinal != 0 {
		t.Errorf("Expected %d found %d", 0, stateful.ordinal)
	}
	if !stateful.exists {
		t.Error("statefulPod from pod does not exist")
	}
	if !stateful.dirty {
		t.Error("statefulPod from pod is not dirty")
	}
}

