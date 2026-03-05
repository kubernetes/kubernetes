/*
Copyright 2026 The Kubernetes Authors.

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

package podgroup

import (
	"context"
	"fmt"
	"io"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/admission"
	genericadmissioninitializer "k8s.io/apiserver/pkg/admission/initializer"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	schedulingv1alpha2listers "k8s.io/client-go/listers/scheduling/v1alpha2"
	"k8s.io/component-base/featuregate"
	scheduling "k8s.io/kubernetes/pkg/apis/scheduling"
	"k8s.io/kubernetes/pkg/features"
)

// PluginName indicates name of admission plugin.
const PluginName = "PodGroupWorkloadExists"

// Register registers a plugin.
func Register(plugins *admission.Plugins) {
	plugins.Register(PluginName, func(config io.Reader) (admission.Interface, error) {
		return NewPodGroupWorkloadExists(), nil
	})
}

// PodGroupWorkloadExists is an admission plugin that validates a PodGroup's
// referenced Workload exists and contains the referenced PodGroupTemplate.
type PodGroupWorkloadExists struct {
	*admission.Handler
	workloadLister schedulingv1alpha2listers.WorkloadLister
	client         kubernetes.Interface
	enabled        bool
}

var _ admission.ValidationInterface = &PodGroupWorkloadExists{}
var _ genericadmissioninitializer.WantsExternalKubeInformerFactory = &PodGroupWorkloadExists{}
var _ genericadmissioninitializer.WantsExternalKubeClientSet = &PodGroupWorkloadExists{}
var _ genericadmissioninitializer.WantsFeatures = &PodGroupWorkloadExists{}

// NewPodGroupWorkloadExists creates a new PodGroupWorkloadExists admission plugin.
func NewPodGroupWorkloadExists() *PodGroupWorkloadExists {
	return &PodGroupWorkloadExists{
		Handler: admission.NewHandler(admission.Create),
	}
}

// InspectFeatureGates captures the GenericWorkload feature gate state.
func (p *PodGroupWorkloadExists) InspectFeatureGates(featureGates featuregate.FeatureGate) {
	p.enabled = featureGates.Enabled(features.GenericWorkload)
}

// SetExternalKubeClientSet sets the client for the plugin.
func (p *PodGroupWorkloadExists) SetExternalKubeClientSet(client kubernetes.Interface) {
	p.client = client
}

// SetExternalKubeInformerFactory sets the informer factory for the plugin.
func (p *PodGroupWorkloadExists) SetExternalKubeInformerFactory(f informers.SharedInformerFactory) {
	workloadInformer := f.Scheduling().V1alpha2().Workloads()
	p.SetReadyFunc(workloadInformer.Informer().HasSynced)
	p.workloadLister = workloadInformer.Lister()
}

// ValidateInitialization ensures the plugin is properly initialized.
func (p *PodGroupWorkloadExists) ValidateInitialization() error {
	if p.workloadLister == nil {
		return fmt.Errorf("missing Workload lister")
	}
	if p.client == nil {
		return fmt.Errorf("missing client")
	}
	return nil
}

// Validate rejects PodGroup creation if the referenced Workload does not exist
// or does not contain the referenced PodGroupTemplate.
func (p *PodGroupWorkloadExists) Validate(ctx context.Context, a admission.Attributes, o admission.ObjectInterfaces) error {
	if !p.enabled {
		return nil
	}

	if a.GetResource().GroupResource() != scheduling.Resource("podgroups") {
		return nil
	}
	if len(a.GetSubresource()) != 0 {
		return nil
	}

	podGroup, ok := a.GetObject().(*scheduling.PodGroup)
	if !ok {
		return apierrors.NewBadRequest("resource was marked with kind PodGroup but could not be converted")
	}

	if podGroup.Spec.PodGroupTemplateRef == nil || podGroup.Spec.PodGroupTemplateRef.Workload == nil {
		return nil
	}

	if !p.WaitForReady() {
		return admission.NewForbidden(a, fmt.Errorf("not yet ready to handle request"))
	}

	workloadRef := podGroup.Spec.PodGroupTemplateRef.Workload
	namespace := a.GetNamespace()

	workload, err := p.workloadLister.Workloads(namespace).Get(workloadRef.WorkloadName)
	if apierrors.IsNotFound(err) {
		workload, err = p.client.SchedulingV1alpha2().Workloads(namespace).Get(ctx, workloadRef.WorkloadName, metav1.GetOptions{})
		if apierrors.IsNotFound(err) {
			return admission.NewForbidden(a, fmt.Errorf("PodGroup rejected: Workload %q not found", workloadRef.WorkloadName))
		}
	}
	if err != nil {
		return apierrors.NewInternalError(err)
	}

	for _, tmpl := range workload.Spec.PodGroupTemplates {
		if tmpl.Name == workloadRef.PodGroupTemplateName {
			return nil
		}
	}

	return admission.NewForbidden(a, fmt.Errorf(
		"PodGroup rejected: PodGroupTemplate %q not found in Workload %q",
		workloadRef.PodGroupTemplateName, workloadRef.WorkloadName,
	))
}
