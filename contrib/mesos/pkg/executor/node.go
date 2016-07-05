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

package executor

import mesos "github.com/mesos/mesos-go/mesosproto"

type NodeInfo struct {
	Cores int
	Mem   uint64 // in bytes
}

func nodeInfo(si *mesos.SlaveInfo, ei *mesos.ExecutorInfo) NodeInfo {
	var executorCPU, executorMem float64

	// get executor resources
	if ei != nil {
		for _, r := range ei.GetResources() {
			if r == nil || r.GetType() != mesos.Value_SCALAR {
				continue
			}
			switch r.GetName() {
			case "cpus":
				executorCPU += r.GetScalar().GetValue()
			case "mem":
				executorMem += r.GetScalar().GetValue()
			}
		}
	}

	// get resource capacity of the node
	ni := NodeInfo{}
	for _, r := range si.GetResources() {
		if r == nil || r.GetType() != mesos.Value_SCALAR {
			continue
		}

		switch r.GetName() {
		case "cpus":
			// We intentionally take the floor of executorCPU because cores are integers
			// and we would loose a complete cpu here if the value is <1.
			// TODO(sttts): switch to float64 when "Machine Allocables" are implemented
			ni.Cores += int(r.GetScalar().GetValue())
		case "mem":
			ni.Mem += uint64(r.GetScalar().GetValue()) * 1024 * 1024
		}
	}

	// TODO(sttts): subtract executorCPU/Mem from static pod resources before subtracting them from the capacity
	ni.Cores -= int(executorCPU)
	ni.Mem -= uint64(executorMem) * 1024 * 1024

	return ni
}
