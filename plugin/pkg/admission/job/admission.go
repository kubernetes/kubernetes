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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apiserver/pkg/admission"
	genericadmissioninitializer "k8s.io/apiserver/pkg/admission/initializer"
	"k8s.io/client-go/informers"
	schedulingv1alpha2listers "k8s.io/client-go/listers/scheduling/v1alpha2"
	"k8s.io/component-base/featuregate"
	"k8s.io/kubernetes/pkg/apis/batch"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/utils/ptr"
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
	genericWorkloadEnabled       bool
	enableWorkloadWithJobEnabled bool
	inspectedFeatureGates        bool
	pgLister                     schedulingv1alpha2listers.PodGroupLister
}

// NewPlugin creates a new JobValidation admission plugin.
func NewPlugin() *Plugin {
	return &Plugin{
		Handler: admission.NewHandler(admission.Update),
	}
}

func (p *Plugin) InspectFeatureGates(featureGates featuregate.FeatureGate) {
	p.genericWorkloadEnabled = featureGates.Enabled(features.GenericWorkload)
	p.enableWorkloadWithJobEnabled = featureGates.Enabled(features.EnableWorkloadWithJob)
	p.inspectedFeatureGates = true
}

func (p *Plugin) SetExternalKubeInformerFactory(f informers.SharedInformerFactory) {
	if !p.genericWorkloadEnabled {
		return
	}
	pgInformer := f.Scheduling().V1alpha2().PodGroups()
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
// cross-referencing other API objects.
func (p *Plugin) Validate(ctx context.Context, a admission.Attributes, o admission.ObjectInterfaces) error {
	if a.GetResource().GroupResource() != batch.Resource("jobs") {
		return nil
	}
	if a.GetSubresource() != "" {
		return nil
	}

	job, ok := a.GetObject().(*batch.Job)
	if !ok {
		return nil
	}
	oldJob, ok := a.GetOldObject().(*batch.Job)
	if !ok {
		return nil
	}

	if err := p.validateParallelismChange(a, job, oldJob); err != nil {
		return err
	}

	return nil
}

// validateParallelismChange rejects parallelism changes on Jobs whose
// PodGroup uses gang scheduling.
func (p *Plugin) validateParallelismChange(a admission.Attributes, job, oldJob *batch.Job) error {
	if !p.genericWorkloadEnabled && !p.enableWorkloadWithJobEnabled {
		return nil
	}
	if ptr.Equal(job.Spec.Parallelism, oldJob.Spec.Parallelism) {
		return nil
	}

	// When SchedulingGroup is set in the template, look up that PodGroup directly.
	sg := oldJob.Spec.Template.Spec.SchedulingGroup
	if sg != nil && sg.PodGroupName != nil {
		pg, err := p.pgLister.PodGroups(oldJob.Namespace).Get(*sg.PodGroupName)
		if err != nil {
			return nil
		}
		if pg.Spec.SchedulingPolicy.Gang != nil {
			return admission.NewForbidden(a, fmt.Errorf(
				"cannot change parallelism for a Job referencing gang-scheduled PodGroup %q", pg.Name))
		}
		return nil
	}

	// When SchedulingGroup is not in the template, scan PodGroups in the namespace owned by this Job.
	pgs, err := p.pgLister.PodGroups(oldJob.Namespace).List(labels.Everything())
	if err != nil {
		return nil
	}
	for _, pg := range pgs {
		if !metav1.IsControlledBy(pg, oldJob) {
			continue
		}
		if pg.Spec.SchedulingPolicy.Gang != nil {
			return admission.NewForbidden(a, fmt.Errorf(
				"cannot change parallelism for a Job referencing gang-scheduled PodGroup %q", pg.Name))
		}
	}

	return nil
}
