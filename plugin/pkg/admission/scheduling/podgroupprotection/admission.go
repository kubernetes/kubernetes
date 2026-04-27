/*
Copyright The Kubernetes Authors.

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

package podgroupprotection

import (
	"context"
	"fmt"
	"io"
	"slices"

	"k8s.io/apiserver/pkg/admission"
	apiserveradmission "k8s.io/apiserver/pkg/admission/initializer"
	"k8s.io/component-base/featuregate"
	"k8s.io/klog/v2"
	schedulingapi "k8s.io/kubernetes/pkg/apis/scheduling"
	"k8s.io/kubernetes/pkg/features"
)

const (
	PluginName = "PodGroupProtection"
)

// Register registers the plugin.
func Register(plugins *admission.Plugins) {
	plugins.Register(PluginName, func(config io.Reader) (admission.Interface, error) {
		return newPlugin(), nil
	})
}

type podGroupProtectionPlugin struct {
	*admission.Handler
	enabled               bool
	inspectedFeatureGates bool
}

var _ admission.MutationInterface = &podGroupProtectionPlugin{}
var _ apiserveradmission.WantsFeatures = &podGroupProtectionPlugin{}

func newPlugin() *podGroupProtectionPlugin {
	return &podGroupProtectionPlugin{
		Handler: admission.NewHandler(admission.Create),
	}
}

func (p *podGroupProtectionPlugin) InspectFeatureGates(featureGates featuregate.FeatureGate) {
	p.enabled = featureGates.Enabled(features.GenericWorkload)
	p.inspectedFeatureGates = true
}

func (p *podGroupProtectionPlugin) ValidateInitialization() error {
	if !p.inspectedFeatureGates {
		return fmt.Errorf("feature gates not inspected")
	}
	return nil
}

var podGroupResource = schedulingapi.Resource("podgroups")

// Admit stamps the PodGroupProtectionFinalizer on every newly created PodGroup
// so that it cannot be deleted while pods still reference it.
// The finalizer is removed by the PodGroupProtection controller when the
// PodGroup is no longer in use.
func (p *podGroupProtectionPlugin) Admit(ctx context.Context, a admission.Attributes, o admission.ObjectInterfaces) error {
	if !p.enabled {
		return nil
	}

	if a.GetResource().GroupResource() != podGroupResource {
		return nil
	}
	if len(a.GetSubresource()) != 0 {
		return nil
	}

	pg, ok := a.GetObject().(*schedulingapi.PodGroup)
	if !ok {
		return nil
	}

	if slices.Contains(pg.Finalizers, schedulingapi.PodGroupProtectionFinalizer) {
		return nil
	}

	klog.V(4).InfoS("Adding protection finalizer to PodGroup", "podGroup", klog.KObj(pg))
	pg.Finalizers = append(pg.Finalizers, schedulingapi.PodGroupProtectionFinalizer)
	return nil
}
