/*
Copyright 2020 The Kubernetes Authors.

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

// Package podtcpresolution contains an admission controller that modifies every new Pod,
// that uses the GNU libc library, to force the DNS resolution to use TCP.
// This is useful in busy clusters that may have issues due to UDP packet drops.
// With this admission controller enabled, an environment variable, RES_OPTIONS=use-vc,
// is added to each POD. This variable only works for applications and systems that use
// the libc library resolver, other applications will be working as always.
// ref: https://man7.org/linux/man-pages/man5/resolv.conf.5.html
package podtcpresolution

import (
	"context"
	"io"
	"strings"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/kubernetes/pkg/apis/core"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/core/pods"
)

// PluginName indicates name of admission plugin.
const PluginName = "PodTCPResolution"

// Register registers a plugin
func Register(plugins *admission.Plugins) {
	plugins.Register(PluginName, func(config io.Reader) (admission.Interface, error) {
		return NewPodTCPResolution(), nil
	})
}

// PodTCPResolution is an implementation of admission.Interface.
// It looks at all new pods and sets the environment variable RES_OPTIONS in all containers .
type PodTCPResolution struct {
	*admission.Handler
}

var _ admission.MutationInterface = &PodTCPResolution{}
var _ admission.ValidationInterface = &PodTCPResolution{}

// Admit makes an admission decision based on the request attributes
func (a *PodTCPResolution) Admit(ctx context.Context, attributes admission.Attributes, o admission.ObjectInterfaces) (err error) {
	// Ignore all calls to subresources or resources other than pods.
	if shouldIgnore(attributes) {
		return nil
	}
	pod, ok := attributes.GetObject().(*api.Pod)
	if !ok {
		return apierrors.NewBadRequest("Resource was marked with kind Pod but was unable to be converted")
	}

	pods.VisitContainersWithPath(&pod.Spec, field.NewPath("spec"), func(c *api.Container, _ *field.Path) bool {
		idx, value := getEnvValueByName("RES_OPTIONS", c.Env)
		// append the environment variable if it doesn't exist
		if value == "" {
			c.Env = append(c.Env, api.EnvVar{Name: "RES_OPTIONS", Value: "use-vc"})
		} else if !strings.Contains(value, "use-vc") {
			c.Env[idx].Value = value + " use-vc"
		}
		return true
	})
	return nil
}

// Validate makes sure that all containers are set to always pull images
func (*PodTCPResolution) Validate(ctx context.Context, attributes admission.Attributes, o admission.ObjectInterfaces) (err error) {
	if shouldIgnore(attributes) {
		return nil
	}

	pod, ok := attributes.GetObject().(*api.Pod)
	if !ok {
		return apierrors.NewBadRequest("Resource was marked with kind Pod but was unable to be converted")
	}

	var allErrs []error
	pods.VisitContainersWithPath(&pod.Spec, field.NewPath("spec"), func(c *api.Container, p *field.Path) bool {
		_, value := getEnvValueByName("RES_OPTIONS", c.Env)
		if value == "" || !strings.Contains(value, "use-vc") {
			allErrs = append(allErrs, field.NotFound(p.Child("env"), "RES_OPTIONS=\"use-vc\" to force tcp dns resolution"))
		}
		return true
	})
	if len(allErrs) > 0 {
		return utilerrors.NewAggregate(allErrs)
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

// getEnvValueByName returns the value and position in the slice of the environment variable specified
// or empty if not found
func getEnvValueByName(name string, envVars []core.EnvVar) (int, string) {
	for idx, env := range envVars {
		if env.Name == name {
			return idx, env.Value
		}
	}
	return 0, ""
}

// NewPodTCPResolution creates a new force DNS TCP admission control handler
func NewPodTCPResolution() *PodTCPResolution {
	return &PodTCPResolution{
		Handler: admission.NewHandler(admission.Create, admission.Update),
	}
}
