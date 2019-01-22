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

	"k8s.io/api/core/v1"
	plugins "k8s.io/kubernetes/pkg/scheduler/plugins/v1alpha1"
)

// StatelessPrebindExample is an example of a simple plugin that has no state
// and implements only one hook for prebind.
type StatelessPrebindExample struct{}

var _ = plugins.PrebindPlugin(StatelessPrebindExample{})

// Name returns name of the plugin. It is used in logs, etc.
func (sr StatelessPrebindExample) Name() string {
	return "stateless-prebind-plugin-example"
}

// Prebind is the functions invoked by the framework at "prebind" extension point.
func (sr StatelessPrebindExample) Prebind(ps plugins.PluginSet, pod *v1.Pod, nodeName string) (bool, error) {
	if pod == nil {
		return false, fmt.Errorf("pod cannot be nil")
	}
	return true, nil
}

// NewStatelessPrebindExample initializes a new plugin and returns it.
func NewStatelessPrebindExample() *StatelessPrebindExample {
	return &StatelessPrebindExample{}
}
