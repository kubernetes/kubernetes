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

package examples

import (
	"fmt"
	"k8s.io/klog"

	"k8s.io/api/core/v1"
	plugins "k8s.io/kubernetes/pkg/scheduler/plugins/v1alpha1"
)

// StatefulMultipointExample is an example plugin that is executed at multiple extension points.
// This plugin is stateful. It receives arguments at initialization (NewMultipointPlugin)
// and changes its state when it is executed.
type StatefulMultipointExample struct {
	mpState map[int]string
	numRuns int
}

var _ = plugins.ReservePlugin(&StatefulMultipointExample{})
var _ = plugins.PrebindPlugin(&StatefulMultipointExample{})

// Name returns name of the plugin. It is used in logs, etc.
func (mp *StatefulMultipointExample) Name() string {
	return "multipoint-plugin-example"
}

// Reserve is the functions invoked by the framework at "reserve" extension point.
func (mp *StatefulMultipointExample) Reserve(ps plugins.PluginSet, pod *v1.Pod, nodeName string) error {
	mp.numRuns++
	return nil
}

// Prebind is the functions invoked by the framework at "prebind" extension point.
func (mp *StatefulMultipointExample) Prebind(ps plugins.PluginSet, pod *v1.Pod, nodeName string) (bool, error) {
	mp.numRuns++
	if pod == nil {
		return false, fmt.Errorf("pod must not be nil")
	}
	return true, nil
}

// NewStatefulMultipointExample initializes a new plugin and returns it.
func NewStatefulMultipointExample(initState ...interface{}) *StatefulMultipointExample {
	if len(initState) == 0 {
		klog.Error("StatefulMultipointExample needs exactly one argument for initialization")
		return nil
	}
	mp := StatefulMultipointExample{
		mpState: initState[0].(map[int]string),
	}
	return &mp
}
