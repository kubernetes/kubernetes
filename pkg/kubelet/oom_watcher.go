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

package kubelet

import (
	"github.com/golang/glog"
	"github.com/google/cadvisor/events"
	cadvisorapi "github.com/google/cadvisor/info/v1"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/client/record"
	"k8s.io/kubernetes/pkg/kubelet/cadvisor"
	"k8s.io/kubernetes/pkg/util/runtime"
)

type OOMWatcher interface {
	Start(ref *api.ObjectReference) error
}

type realOOMWatcher struct {
	cadvisor cadvisor.Interface
	recorder record.EventRecorder
}

func NewOOMWatcher(cadvisor cadvisor.Interface, recorder record.EventRecorder) OOMWatcher {
	return &realOOMWatcher{
		cadvisor: cadvisor,
		recorder: recorder,
	}
}

const systemOOMEvent = "SystemOOM"

// Watches cadvisor for system oom's and records an event for every system oom encountered.
func (ow *realOOMWatcher) Start(ref *api.ObjectReference) error {
	request := events.Request{
		EventType: map[cadvisorapi.EventType]bool{
			cadvisorapi.EventOom: true,
		},
		ContainerName:        "/",
		IncludeSubcontainers: false,
	}
	eventChannel, err := ow.cadvisor.WatchEvents(&request)
	if err != nil {
		return err
	}

	go func() {
		defer runtime.HandleCrash()

		for event := range eventChannel.GetChannel() {
			glog.V(2).Infof("Got sys oom event from cadvisor: %v", event)
			ow.recorder.PastEventf(ref, unversioned.Time{Time: event.Timestamp}, api.EventTypeWarning, systemOOMEvent, "System OOM encountered")
		}
		glog.Errorf("Unexpectedly stopped receiving OOM notifications from cAdvisor")
	}()
	return nil
}
