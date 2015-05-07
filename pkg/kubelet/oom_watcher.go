/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"fmt"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/record"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/cadvisor"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/golang/glog"
	"github.com/google/cadvisor/events"
	cadvisorApi "github.com/google/cadvisor/info/v1"
)

type OOMWatcher interface {
	RecordSysOOMs(ref *api.ObjectReference) error
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
func (ow *realOOMWatcher) RecordSysOOMs(ref *api.ObjectReference) error {
	request := events.Request{
		EventType: map[cadvisorApi.EventType]bool{
			cadvisorApi.EventOom: true,
		},
		ContainerName:        "/",
		IncludeSubcontainers: false,
	}
	eventChannel, err := ow.cadvisor.WatchEvents(&request)
	if err != nil {
		return err
	}
	for event := range eventChannel.GetChannel() {
		glog.V(2).Infof("got sys oom event from cadvisor: %v", event)
		ow.recorder.PastEventf(ref, util.Time{event.Timestamp}, systemOOMEvent, "System OOM encountered")
	}
	return fmt.Errorf("failed to watch cadvisor for sys oom events")
}
