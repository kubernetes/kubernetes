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

	"github.com/spf13/pflag"
	"golang.org/x/sys/windows"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/windows/service"
)

const (
	serviceName = "kube-proxy"
)

// getPriorityValue returns the value associated with a Windows process priorityClass
// Ref: https://docs.microsoft.com/en-us/windows/win32/cimwin32prov/setpriority-method-in-class-win32-process
// TODO: Think of a better way to share code between kubelet and kube-proxy as this function definition is same as what
// 		 we have cmd/kubelet/app
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
	kubeProxyProcessHandle := windows.CurrentProcess()
	// Set the priority of the kube-proxy process to given priority
	klog.Infof("Setting the priority of kube-proxy process to %s", windowsPriorityClass)
	if err := windows.SetPriorityClass(kubeProxyProcessHandle, priority); err != nil {
		return err
	}
	if windowsService {
		return service.InitService(serviceName)
	}
	return nil
}

func (o *Options) addOSFlags(fs *pflag.FlagSet) {
	fs.BoolVar(&o.WindowsService, "windows-service", o.WindowsService, "Enable Windows Service Control Manager API integration")
	// The default priority class associated with any process in Windows is NORMAL_PRIORITY_CLASS. Keeping it as is
	// to maintain backwards compatibility.
	// Source: https://docs.microsoft.com/en-us/windows/win32/procthread/scheduling-priorities
	fs.StringVar(&o.WindowsPriorityClass, "windows-priorityclass", "NORMAL_PRIORITY_CLASS",
		"Set the PriorityClass associated with kube-proxy process, the default ones are available at "+
			"https://docs.microsoft.com/en-us/windows/win32/procthread/scheduling-priorities")
	fs.StringVar(&o.config.Winkernel.SourceVip, "source-vip", o.config.Winkernel.SourceVip, "The IP address of the source VIP for non-DSR.")
	fs.StringVar(&o.config.Winkernel.NetworkName, "network-name", o.config.Winkernel.NetworkName, "The name of the cluster network.")
	fs.BoolVar(&o.config.Winkernel.EnableDSR, "enable-dsr", o.config.Winkernel.EnableDSR, "If true make kube-proxy apply DSR policies for service VIP")
}
