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
	genericadmissioninitializer "k8s.io/apiserver/pkg/admission/initializer"
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
	genericWorkloadEnabled   bool
	compositePodGroupEnabled bool
	inspectedFeatureGates    bool
}

var _ admission.MutationInterface = &podGroupProtectionPlugin{}
var _ genericadmissioninitializer.WantsFeatures = &podGroupProtectionPlugin{}

func newPlugin() *podGroupProtectionPlugin {
	return &podGroupProtectionPlugin{
		Handler: admission.NewHandler(admission.Create),
	}
}

func (p *podGroupProtectionPlugin) InspectFeatureGates(featureGates featuregate.FeatureGate) {
	p.genericWorkloadEnabled = featureGates.Enabled(features.GenericWorkload)
	p.compositePodGroupEnabled = featureGates.Enabled(features.CompositePodGroup)
	p.inspectedFeatureGates = true
}

func (p *podGroupProtectionPlugin) ValidateInitialization() error {
	if !p.inspectedFeatureGates {
		return fmt.Errorf("feature gates not inspected")
	}
	return nil
}

var (
	podGroupResource          = schedulingapi.Resource("podgroups")
	compositePodGroupResource = schedulingapi.Resource("compositepodgroups")
)

// Admit stamps the PodGroupProtectionFinalizer on newly created PodGroups
// and CompositePodGroupProtectionFinalizer on newly created CompositePodGroups
// so that they cannot be deleted while child resources still reference them.
// The finalizers are removed by the PodGroupProtection controller when the
// resource is no longer in use.
func (p *podGroupProtectionPlugin) Admit(ctx context.Context, a admission.Attributes, o admission.ObjectInterfaces) error {
	if !p.genericWorkloadEnabled {
		return nil
	}
	if a.GetOperation() != admission.Create {
		return nil
	}

	gr := a.GetResource().GroupResource()
	if gr != podGroupResource && gr != compositePodGroupResource {
		return nil
	}
	if len(a.GetSubresource()) != 0 {
		return nil
	}

	logger := klog.FromContext(ctx)

	if gr == podGroupResource {
		pg, ok := a.GetObject().(*schedulingapi.PodGroup)
		if !ok {
			return nil
		}
		if slices.Contains(pg.Finalizers, schedulingapi.PodGroupProtectionFinalizer) {
			return nil
		}
		logger.V(4).Info("Adding protection finalizer to PodGroup", "podGroup", klog.KObj(pg))
		pg.Finalizers = append(pg.Finalizers, schedulingapi.PodGroupProtectionFinalizer)
		return nil
	}

	if !p.compositePodGroupEnabled {
		return nil
	}

	cpg, ok := a.GetObject().(*schedulingapi.CompositePodGroup)
	if !ok {
		return nil
	}
	if slices.Contains(cpg.Finalizers, schedulingapi.CompositePodGroupProtectionFinalizer) {
		return nil
	}
	logger.V(4).Info("Adding protection finalizer to CompositePodGroup", "compositePodGroup", klog.KObj(cpg))
	cpg.Finalizers = append(cpg.Finalizers, schedulingapi.CompositePodGroupProtectionFinalizer)
	return nil
}
