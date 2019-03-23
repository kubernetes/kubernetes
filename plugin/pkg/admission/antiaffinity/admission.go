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

package antiaffinity

import (
	"fmt"
	"io"

	"k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apiserver/pkg/admission"
	api "k8s.io/kubernetes/pkg/apis/core"
)

const PluginName = "LimitPodHardAntiAffinityTopology"

// Register registers a plugin
func Register(plugins *admission.Plugins) {
	plugins.Register(PluginName, func(config io.Reader) (admission.Interface, error) {
		return NewInterPodAntiAffinity(), nil
	})
}

// Plugin contains the client used by the admission controller
type Plugin struct {
	*admission.Handler
}

var _ admission.ValidationInterface = &Plugin{}

// NewInterPodAntiAffinity creates a new instance of the LimitPodHardAntiAffinityTopology admission controller
func NewInterPodAntiAffinity() *Plugin {
	return &Plugin{
		Handler: admission.NewHandler(admission.Create, admission.Update),
	}
}

// Validate will deny any pod that defines AntiAffinity topology key other than v1.LabelHostname i.e. "kubernetes.io/hostname"
// in  requiredDuringSchedulingRequiredDuringExecution and requiredDuringSchedulingIgnoredDuringExecution.
func (p *Plugin) Validate(attributes admission.Attributes, o admission.ObjectInterfaces) (err error) {
	// Ignore all calls to subresources or resources other than pods.
	if len(attributes.GetSubresource()) != 0 || attributes.GetResource().GroupResource() != api.Resource("pods") {
		return nil
	}
	pod, ok := attributes.GetObject().(*api.Pod)
	if !ok {
		return apierrors.NewBadRequest("Resource was marked with kind Pod but was unable to be converted")
	}
	affinity := pod.Spec.Affinity
	if affinity != nil && affinity.PodAntiAffinity != nil {
		var podAntiAffinityTerms []api.PodAffinityTerm
		if len(affinity.PodAntiAffinity.RequiredDuringSchedulingIgnoredDuringExecution) != 0 {
			podAntiAffinityTerms = affinity.PodAntiAffinity.RequiredDuringSchedulingIgnoredDuringExecution
		}
		// TODO: Uncomment this block when implement RequiredDuringSchedulingRequiredDuringExecution.
		//if len(affinity.PodAntiAffinity.RequiredDuringSchedulingRequiredDuringExecution) != 0 {
		//        podAntiAffinityTerms = append(podAntiAffinityTerms, affinity.PodAntiAffinity.RequiredDuringSchedulingRequiredDuringExecution...)
		//}
		for _, v := range podAntiAffinityTerms {
			if v.TopologyKey != v1.LabelHostname {
				return apierrors.NewForbidden(attributes.GetResource().GroupResource(), pod.Name, fmt.Errorf("affinity.PodAntiAffinity.RequiredDuringScheduling has TopologyKey %v but only key %v is allowed", v.TopologyKey, v1.LabelHostname))
			}
		}
	}
	return nil
}
