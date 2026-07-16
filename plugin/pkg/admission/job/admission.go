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

package job

import (
	"context"
	"fmt"
	"io"

	"k8s.io/apiserver/pkg/admission"
	genericadmissioninitializer "k8s.io/apiserver/pkg/admission/initializer"
	"k8s.io/client-go/informers"
	schedulingv1alpha3listers "k8s.io/client-go/listers/scheduling/v1alpha3"
	"k8s.io/component-base/featuregate"
	"k8s.io/kubernetes/pkg/apis/batch"
	"k8s.io/kubernetes/pkg/features"
)

const (
	// PluginName is the name of the admission plugin.
	PluginName = "JobValidation"
)

// Register registers the plugin.
func Register(plugins *admission.Plugins) {
	plugins.Register(PluginName, func(config io.Reader) (admission.Interface, error) {
		return NewPlugin(), nil
	})
}

var _ admission.Interface = &Plugin{}
var _ admission.ValidationInterface = &Plugin{}
var _ = genericadmissioninitializer.WantsExternalKubeInformerFactory(&Plugin{})
var _ genericadmissioninitializer.WantsFeatures = &Plugin{}

// Plugin is an admission controller that validates Job updates
// against gang-scheduled PodGroups.
type Plugin struct {
	*admission.Handler
	genericWorkloadEnabled bool
	workloadWithJobEnabled bool
	inspectedFeatureGates  bool
	pgLister               schedulingv1alpha3listers.PodGroupLister
}

// NewPlugin creates a new JobValidation admission plugin.
func NewPlugin() *Plugin {
	return &Plugin{
		Handler: admission.NewHandler(admission.Update),
	}
}

func (p *Plugin) InspectFeatureGates(featureGates featuregate.FeatureGate) {
	p.genericWorkloadEnabled = featureGates.Enabled(features.GenericWorkload)
	p.workloadWithJobEnabled = featureGates.Enabled(features.WorkloadWithJob)
	p.inspectedFeatureGates = true
}

func (p *Plugin) SetExternalKubeInformerFactory(f informers.SharedInformerFactory) {
	if !p.genericWorkloadEnabled {
		return
	}
	pgInformer := f.Scheduling().V1alpha3().PodGroups()
	p.pgLister = pgInformer.Lister()
	p.SetReadyFunc(pgInformer.Informer().HasSynced)
}

// ValidateInitialization ensures the lister is set when the feature gate is enabled.
func (p *Plugin) ValidateInitialization() error {
	if !p.inspectedFeatureGates {
		return fmt.Errorf("%s has not inspected feature gates", PluginName)
	}
	if p.genericWorkloadEnabled && p.pgLister == nil {
		return fmt.Errorf("missing PodGroup lister")
	}
	return nil
}

// Validate performs admission checks on Job updates that require
// cross-referencing other API objects. It is only active when
// the workload scheduling feature gates are enabled.
func (p *Plugin) Validate(ctx context.Context, a admission.Attributes, o admission.ObjectInterfaces) error {
	if a.GetResource().GroupResource() != batch.Resource("jobs") {
		return nil
	}
	if a.GetSubresource() != "" {
		return nil
	}

	return nil
}
