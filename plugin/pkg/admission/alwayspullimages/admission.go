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
	"context"
	"io"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/klog/v2"
	podutil "k8s.io/kubernetes/pkg/api/pod"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/core/pods"
)

// PluginName indicates name of admission plugin.
const PluginName = "AlwaysPullImages"

// Register registers a plugin
func Register(plugins *admission.Plugins) {
	plugins.Register(PluginName, func(config io.Reader) (admission.Interface, error) {
		return NewAlwaysPullImages(), nil
	})
}

// AlwaysPullImages is an implementation of admission.Interface.
// It looks at all new pods and overrides each container's image pull policy to Always.
type AlwaysPullImages struct {
	*admission.Handler
}

var _ admission.MutationInterface = &AlwaysPullImages{}
var _ admission.ValidationInterface = &AlwaysPullImages{}

// Admit makes an admission decision based on the request attributes
func (a *AlwaysPullImages) Admit(ctx context.Context, attributes admission.Attributes, o admission.ObjectInterfaces) (err error) {
	// Ignore all calls to subresources or resources other than pods.
	if shouldIgnore(attributes) {
		return nil
	}
	pod, ok := attributes.GetObject().(*api.Pod)
	if !ok {
		return apierrors.NewBadRequest("Resource was marked with kind Pod but was unable to be converted")
	}

	visitPulledContainers(attributes, pod, func(c *api.Container, _ *field.Path) bool {
		c.ImagePullPolicy = api.PullAlways
		return true
	})

	// See: https://kep.k8s.io/4639
	visitPulledImageVolumes(attributes, pod, func(img *api.ImageVolumeSource, _ *field.Path) {
		img.PullPolicy = api.PullAlways
	})

	return nil
}

// Validate makes sure that all containers are set to always pull images
func (*AlwaysPullImages) Validate(ctx context.Context, attributes admission.Attributes, o admission.ObjectInterfaces) (err error) {
	if shouldIgnore(attributes) {
		return nil
	}

	pod, ok := attributes.GetObject().(*api.Pod)
	if !ok {
		return apierrors.NewBadRequest("Resource was marked with kind Pod but was unable to be converted")
	}

	var allErrs []error
	visitPulledContainers(attributes, pod, func(c *api.Container, p *field.Path) bool {
		if c.ImagePullPolicy != api.PullAlways {
			allErrs = append(allErrs, admission.NewForbidden(attributes,
				field.NotSupported(p.Child("imagePullPolicy"), c.ImagePullPolicy, []string{string(api.PullAlways)}),
			))
		}
		return true
	})

	// See: https://kep.k8s.io/4639
	visitPulledImageVolumes(attributes, pod, func(img *api.ImageVolumeSource, p *field.Path) {
		if img.PullPolicy != api.PullAlways {
			allErrs = append(allErrs, admission.NewForbidden(attributes,
				field.NotSupported(p.Child("pullPolicy"), img.PullPolicy, []string{string(api.PullAlways)}),
			))
		}
	})

	if len(allErrs) > 0 {
		return utilerrors.NewAggregate(allErrs)
	}

	return nil
}

// visitPulledContainers visits the containers whose images must always be pulled, with their
// paths. On a pods/dynamic update, these are only the containers it adds or changes the image of,
// so that other existing containers, whose pull policy is immutable, aren't mutated.
func visitPulledContainers(attributes admission.Attributes, pod *api.Pod, visitor pods.ContainerVisitorWithPath) {
	oldPod, ok := attributes.GetOldObject().(*api.Pod)
	if attributes.GetSubresource() != "dynamic" || !ok {
		pods.VisitContainersWithPath(&pod.Spec, field.NewPath("spec"), visitor)
		return
	}
	initIndex, index := 0, 0
	for pair, containerType := range pods.ContainerPairs(pod, oldPod, podutil.InitContainers|podutil.Containers) {
		if pair.New == nil {
			continue
		}
		var path *field.Path
		if containerType == podutil.InitContainers {
			path = field.NewPath("spec", "initContainers").Index(initIndex)
			initIndex++
		} else {
			path = field.NewPath("spec", "containers").Index(index)
			index++
		}
		if pair.Old != nil && pair.Old.Image == pair.New.Image {
			continue
		}
		if !visitor(pair.New, path) {
			return
		}
	}
}

// visitPulledImageVolumes visits the image volumes whose images must always be pulled, with their
// paths. On a pods/dynamic update, these are only the image volumes added or whose reference
// changed, so that existing volumes, whose pull policy is immutable, aren't mutated.
func visitPulledImageVolumes(attributes admission.Attributes, pod *api.Pod, visitor func(*api.ImageVolumeSource, *field.Path)) {
	oldPod, ok := attributes.GetOldObject().(*api.Pod)
	volsPath := field.NewPath("spec", "volumes")
	if attributes.GetSubresource() != "dynamic" || !ok {
		for i := range pod.Spec.Volumes {
			if pod.Spec.Volumes[i].Image != nil {
				visitor(pod.Spec.Volumes[i].Image, volsPath.Index(i).Child("image"))
			}
		}
		return
	}
	index := 0
	for pair := range pods.VolumePairs(pod, oldPod) {
		if pair.New == nil {
			continue
		}
		path := volsPath.Index(index).Child("image")
		index++
		if pair.New.Image == nil {
			continue
		}
		if pair.Old != nil && pair.Old.Image != nil && pair.Old.Image.Reference == pair.New.Image.Reference {
			continue
		}
		visitor(pair.New.Image, path)
	}
}

// check if it's update and it doesn't add containers or image volumes or change the images referenced by the pod spec
func isUpdateWithNoNewImages(attributes admission.Attributes) bool {
	if attributes.GetOperation() != admission.Update {
		return false
	}

	pod, ok := attributes.GetObject().(*api.Pod)
	if !ok {
		klog.Warningf("Resource was marked with kind Pod but pod was unable to be converted.")
		return false
	}

	oldPod, ok := attributes.GetOldObject().(*api.Pod)
	if !ok {
		klog.Warningf("Resource was marked with kind Pod but old pod was unable to be converted.")
		return false
	}

	// A container or image volume added through the dynamic subresource must always pull, even if
	// it reuses an image that is already in the pod.
	for pair := range pods.ContainerPairs(pod, oldPod, podutil.Containers) {
		if pair.Old == nil {
			return false
		}
	}
	for pair := range pods.VolumePairs(pod, oldPod) {
		if pair.New != nil && pair.New.Image != nil && (pair.Old == nil || pair.Old.Image == nil || pair.Old.Image.Reference != pair.New.Image.Reference) {
			return false
		}
	}

	oldImages := sets.NewString()
	pods.VisitContainersWithPath(&oldPod.Spec, field.NewPath("spec"), func(c *api.Container, _ *field.Path) bool {
		oldImages.Insert(c.Image)
		return true
	})

	hasNewImage := false
	pods.VisitContainersWithPath(&pod.Spec, field.NewPath("spec"), func(c *api.Container, _ *field.Path) bool {
		if !oldImages.Has(c.Image) {
			hasNewImage = true
		}
		return !hasNewImage
	})
	return !hasNewImage
}

func shouldIgnore(attributes admission.Attributes) bool {
	// Ignore calls to resources other than pods, and to subresources that can't modify images.
	subresource := attributes.GetSubresource()
	if (subresource != "" && subresource != "dynamic") || attributes.GetResource().GroupResource() != api.Resource("pods") {
		return true
	}

	if isUpdateWithNoNewImages(attributes) {
		return true
	}
	return false
}

// NewAlwaysPullImages creates a new always pull images admission control handler
func NewAlwaysPullImages() *AlwaysPullImages {
	return &AlwaysPullImages{
		Handler: admission.NewHandler(admission.Create, admission.Update),
	}
}
