// +build linux

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

package oom

import (
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/client-go/tools/record"
	"k8s.io/klog"

	"github.com/google/cadvisor/utils/oomparser"
)

type realWatcher struct {
	recorder record.EventRecorder
}

var _ Watcher = &realWatcher{}

// NewWatcher creates and initializes a OOMWatcher based on parameters.
func NewWatcher(recorder record.EventRecorder) Watcher {
	return &realWatcher{
		recorder: recorder,
	}
}

const systemOOMEvent = "SystemOOM"

// Start watches for system oom's and records an event for every system oom encountered.
func (ow *realWatcher) Start(ref *v1.ObjectReference) error {
	oomLog, err := oomparser.New()
	if err != nil {
		return err
	}
	outStream := make(chan *oomparser.OomInstance, 10)
	go oomLog.StreamOoms(outStream)

	go func() {
		defer runtime.HandleCrash()

		for event := range outStream {
			if event.ContainerName == "/" {
				klog.V(1).Infof("Got sys oom event: %v", event)
				ow.recorder.PastEventf(ref, metav1.Time{Time: event.TimeOfDeath}, v1.EventTypeWarning, systemOOMEvent, "System OOM encountered")
			}
		}
		klog.Errorf("Unexpectedly stopped receiving OOM notifications")
	}()
	return nil
}
