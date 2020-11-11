// +build windows

/*
Copyright 2018 The Kubernetes Authors.

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

package app

import (
	"fmt"

	"golang.org/x/sys/windows"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/windows/service"
)

const (
	serviceName = "kubelet"
)

// getPriorityValue returns the value associated with a Windows process priorityClass
// Ref: https://docs.microsoft.com/en-us/windows/win32/cimwin32prov/setpriority-method-in-class-win32-process
func getPriorityValue(priorityClassName string) uint32 {
	var priorityClassMap = map[string]uint32{
		"IDLE_PRIORITY_CLASS":         uint32(64),
		"BELOW_NORMAL_PRIORITY_CLASS": uint32(16384),
		"NORMAL_PRIORITY_CLASS":       uint32(32),
		"ABOVE_NORMAL_PRIORITY_CLASS": uint32(32768),
		"HIGH_PRIORITY_CLASS":         uint32(128),
		"REALTIME_PRIORITY_CLASS":     uint32(256),
	}
	return priorityClassMap[priorityClassName]
}

func initForOS(windowsService bool, windowsPriorityClass string) error {
	priority := getPriorityValue(windowsPriorityClass)
	if priority == 0 {
		return fmt.Errorf("unknown priority class %s, valid ones are available at "+
			"https://docs.microsoft.com/en-us/windows/win32/procthread/scheduling-priorities", windowsPriorityClass)
	}
	kubeletProcessHandle := windows.CurrentProcess()
	// Set the priority of the kubelet process to given priority
	klog.Infof("Setting the priority of kubelet process to %s", windowsPriorityClass)
	if err := windows.SetPriorityClass(kubeletProcessHandle, priority); err != nil {
		return err
	}

	if windowsService {
		return service.InitService(serviceName)
	}
	return nil
}
