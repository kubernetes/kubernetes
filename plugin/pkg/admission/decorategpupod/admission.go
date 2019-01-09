/*
Copyright 2016 The Kubernetes Authors.
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
package decorategpupod

import (
	"io"

	"github.com/golang/glog"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apiserver/pkg/admission"
	api "k8s.io/kubernetes/pkg/apis/core"
)

const (
	// PluginName indicates name of admission plugin.
	PluginName         = "DecorateGPUPod"
	defaultGPUTaintKey = "node-role.qiniu.com/gpu"
)

var (
	toleration = api.Toleration{
		Key:      defaultGPUTaintKey,
		Operator: api.TolerationOpExists,
	}
)

// Register registers a plugin
func Register(plugins *admission.Plugins) {
	plugins.Register(PluginName, func(config io.Reader) (admission.Interface, error) {
		return NewDecorateGPUPodPlugin(), nil
	})
}

// plugin contains the client used by the admission controller
type plugin struct {
	*admission.Handler
}

// NewDecorateGPUPodPlugin creates a new instance of the DecorateGPUPod admission controller
func NewDecorateGPUPodPlugin() admission.MutationInterface {
	return &plugin{
		Handler: admission.NewHandler(admission.Create),
	}
}

// check if a pod is required GPU resource
func checkNeedGPU(pod *api.Pod) bool {
	isNeed := false
	// check if pod spec has defined gpu limitation
	containers := pod.Spec.Containers
	for _, container := range containers {
		gpuLimit, ok := container.Resources.Limits.NvidiaGPU().AsInt64()
		if ok && gpuLimit > 0 {
			isNeed = true
			break
		}
	}
	return isNeed
}

// add a toleration to pod's spec
func addToleration(toleration *api.Toleration, pod *api.Pod) {
	found := -1
	for i, t := range pod.Spec.Tolerations {
		if t.Key == toleration.Key && t.Operator == toleration.Operator && t.Value == toleration.Value {
			found = i
			break
		}
	}
	// append if not exists
	// update if exists
	if found != -1 {
		glog.Infof("toleration found <%v>, update", found)
		pod.Spec.Tolerations[found].Effect = toleration.Effect
		pod.Spec.Tolerations[found].TolerationSeconds = toleration.TolerationSeconds
	} else {
		glog.Info("toleration not found, add")
		pod.Spec.Tolerations = append(pod.Spec.Tolerations, *toleration)
	}
}

// remove toleration from pod's spec
func removeToleration(toleration *api.Toleration, pod *api.Pod) {
	found := -1
	for i, t := range pod.Spec.Tolerations {
		if t.Key == toleration.Key && t.Operator == toleration.Operator && t.Value == toleration.Value {
			found = i
			break
		}
	}
	// remove
	// swap with last element of slice
	if found != -1 {
		glog.Infof("toleration found <%v>, remove", found)
		lastElemIndex := len(pod.Spec.Tolerations) - 1
		pod.Spec.Tolerations[found] = pod.Spec.Tolerations[lastElemIndex]
		pod.Spec.Tolerations = pod.Spec.Tolerations[:lastElemIndex]
	} else {
		glog.Info("toleration is not found, do nothing")
		// do nothing
	}
}

// Admit will inject toleration for gpu dedicated nodes to pod if it needs gpu resource on creation
func (p *plugin) Admit(a admission.Attributes) (err error) {
	// Ignore all calls to subresources or resources other than pods.
	if len(a.GetSubresource()) != 0 || a.GetResource().GroupResource() != api.Resource("pods") {
		return nil
	}
	if a.GetOperation() != admission.Create {
		return nil
	}
	pod, ok := a.GetObject().(*api.Pod)
	if !ok {
		return apierrors.NewBadRequest("Resource was marked with kind Pod but was unable to be converted")
	}
	if !checkNeedGPU(pod) {
		return nil
	} else {
		addToleration(&toleration, pod)
	}
	return nil
}

func (p *plugin) Handles(operation admission.Operation) bool {
	if operation == admission.Create {
		return true
	}
	return false
}
