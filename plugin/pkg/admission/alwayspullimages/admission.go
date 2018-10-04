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

// Package alwayspullimages contains an admission controller that modifies every new Pod to force
// the image pull policy to Always. This is useful in a multitenant cluster so that users can be
// assured that their private images can only be used by those who have the credentials to pull
// them. Without this admission controller, once an image has been pulled to a node, any pod from
// any user can use it simply by knowing the image's name (assuming the Pod is scheduled onto the
// right node), without any authorization check against the image. With this admission controller
// enabled, images are always pulled prior to starting containers, which means valid credentials are
// required.
package alwayspullimages

import (
	"io"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/admission"
	api "k8s.io/kubernetes/pkg/apis/core"
	alwayspullimagesapi "k8s.io/kubernetes/plugin/pkg/admission/alwayspullimages/apis/alwayspullimages"
)

const (
	// PluginName indicates name of admission plugin.
	PluginName = "AlwaysPullImages"

	// AlwaysPullImageExcludeAnnotationKey represents the annotation key for a pod to exempt itself from the AlwaysPullImages admission controller
	AlwaysPullImageExcludeAnnotationKey string = "alwayspullimages.admission.kubernetes.io/exclude"
)

// Register registers a plugin
func Register(plugins *admission.Plugins) {
	plugins.Register(PluginName,
		func(config io.Reader) (admission.Interface, error) {
			// load the configuration provided (if any)
			configuration, err := LoadConfiguration(config)
			if err != nil {
				return nil, err
			}

			return NewAlwaysPullImages(configuration), nil
		})
}

// AlwaysPullImages is an implementation of admission.Interface.
// It looks at all new pods and overrides each container's image pull policy to Always.
type AlwaysPullImages struct {
	*admission.Handler
	config *alwayspullimagesapi.Configuration
}

var _ admission.MutationInterface = &AlwaysPullImages{}
var _ admission.ValidationInterface = &AlwaysPullImages{}

// Admit makes an admission decision based on the request attributes
func (a *AlwaysPullImages) Admit(attributes admission.Attributes) (err error) {
	// Ignore all calls to subresources or resources other than pods.
	if shouldIgnore(attributes) {
		return nil
	}
	pod, ok := attributes.GetObject().(*api.Pod)
	if !ok {
		return apierrors.NewBadRequest("Resource was marked with kind Pod but was unable to be converted")
	}

	// Ignore pod if exclusion annotation is present
	if a.config.EnableExcludeAnnotation && shouldExclude(pod) {
		return nil
	}

	for i := range pod.Spec.InitContainers {
		pod.Spec.InitContainers[i].ImagePullPolicy = api.PullAlways
	}

	for i := range pod.Spec.Containers {
		pod.Spec.Containers[i].ImagePullPolicy = api.PullAlways
	}

	return nil
}

// Validate makes sure that all containers are set to always pull images
func (a *AlwaysPullImages) Validate(attributes admission.Attributes) (err error) {
	if shouldIgnore(attributes) {
		return nil
	}

	pod, ok := attributes.GetObject().(*api.Pod)
	if !ok {
		return apierrors.NewBadRequest("Resource was marked with kind Pod but was unable to be converted")
	}

	// Ignore pod if exclusion annotation is present
	if a.config.EnableExcludeAnnotation && shouldExclude(pod) {
		return nil
	}

	for i := range pod.Spec.InitContainers {
		if pod.Spec.InitContainers[i].ImagePullPolicy != api.PullAlways {
			return admission.NewForbidden(attributes,
				field.NotSupported(field.NewPath("spec", "initContainers").Index(i).Child("imagePullPolicy"),
					pod.Spec.InitContainers[i].ImagePullPolicy, []string{string(api.PullAlways)},
				),
			)
		}
	}
	for i := range pod.Spec.Containers {
		if pod.Spec.Containers[i].ImagePullPolicy != api.PullAlways {
			return admission.NewForbidden(attributes,
				field.NotSupported(field.NewPath("spec", "containers").Index(i).Child("imagePullPolicy"),
					pod.Spec.Containers[i].ImagePullPolicy, []string{string(api.PullAlways)},
				),
			)
		}
	}

	return nil
}

func shouldIgnore(attributes admission.Attributes) bool {
	// Ignore all calls to subresources or resources other than pods.
	if len(attributes.GetSubresource()) != 0 || attributes.GetResource().GroupResource() != api.Resource("pods") {
		return true
	}

	return false
}

func shouldExclude(pod *api.Pod) bool {
	// Exclude (opt-out) pods that have the AlwaysPullImageExcludeAnnotationKey set to true
	if podAnnotations := pod.GetAnnotations(); podAnnotations != nil {
		if podAnnotations[AlwaysPullImageExcludeAnnotationKey] == "true" {
			return true
		}
	}

	return false
}

// NewAlwaysPullImages creates a new always pull images admission control handler
func NewAlwaysPullImages(config *alwayspullimagesapi.Configuration) *AlwaysPullImages {
	return &AlwaysPullImages{
		Handler: admission.NewHandler(admission.Create, admission.Update),
		config:  config,
	}
}
