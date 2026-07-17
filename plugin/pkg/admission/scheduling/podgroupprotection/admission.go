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

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/admission"
	genericadmissioninitializer "k8s.io/apiserver/pkg/admission/initializer"
	"k8s.io/client-go/informers"
	schedulinglistersv1alpha3 "k8s.io/client-go/listers/scheduling/v1alpha3"
	schedulinglistersv1beta1 "k8s.io/client-go/listers/scheduling/v1beta1"
	"k8s.io/component-base/featuregate"
	"k8s.io/klog/v2"
	api "k8s.io/kubernetes/pkg/apis/core"
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

	podGroupLister          schedulinglistersv1beta1.PodGroupLister
	compositePodGroupLister schedulinglistersv1alpha3.CompositePodGroupLister
}

var _ admission.MutationInterface = &podGroupProtectionPlugin{}
var _ admission.ValidationInterface = &podGroupProtectionPlugin{}
var _ genericadmissioninitializer.WantsExternalKubeInformerFactory = &podGroupProtectionPlugin{}
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

func (p *podGroupProtectionPlugin) SetExternalKubeInformerFactory(f informers.SharedInformerFactory) {
	if !p.genericWorkloadEnabled {
		return
	}
	pgInformer := f.Scheduling().V1beta1().PodGroups()
	p.podGroupLister = pgInformer.Lister()

	if p.compositePodGroupEnabled {
		cpgInformer := f.Scheduling().V1alpha3().CompositePodGroups()
		p.compositePodGroupLister = cpgInformer.Lister()

		p.SetReadyFunc(func() bool {
			return pgInformer.Informer().HasSynced() && cpgInformer.Informer().HasSynced()
		})
	} else {
		p.SetReadyFunc(pgInformer.Informer().HasSynced)
	}
}

func (p *podGroupProtectionPlugin) ValidateInitialization() error {
	if !p.inspectedFeatureGates {
		return fmt.Errorf("feature gates not inspected")
	}
	if !p.genericWorkloadEnabled {
		return nil
	}
	if p.podGroupLister == nil {
		return fmt.Errorf("missing PodGroup lister")
	}
	if p.compositePodGroupEnabled && p.compositePodGroupLister == nil {
		return fmt.Errorf("missing CompositePodGroup lister")
	}
	return nil
}

var (
	podResource               = api.Resource("pods")
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

// Validate prevents creation of child objects (Pods, PodGroups, CompositePodGroups)
// when the referenced parent object is already being deleted (has a non-nil DeletionTimestamp).
// This eliminates the race condition where a child could attach to a terminating parent after the
// controller has already verified zero children and is in the process of releasing the parent's finalizer.
func (p *podGroupProtectionPlugin) Validate(ctx context.Context, a admission.Attributes, o admission.ObjectInterfaces) error {
	if !p.genericWorkloadEnabled {
		return nil
	}
	if a.GetOperation() != admission.Create {
		return nil
	}
	if len(a.GetSubresource()) != 0 {
		return nil
	}

	gr := a.GetResource().GroupResource()
	if gr != podResource && gr != podGroupResource && gr != compositePodGroupResource {
		return nil
	}
	if gr == compositePodGroupResource && !p.compositePodGroupEnabled {
		return nil
	}

	if !p.WaitForReady() {
		return admission.NewForbidden(a, fmt.Errorf("not yet ready to handle request"))
	}

	switch gr {
	case podResource:
		return p.validatePod(a)
	case podGroupResource:
		return p.validatePodGroup(a)
	case compositePodGroupResource:
		return p.validateCompositePodGroup(a)
	}

	return nil
}

func (p *podGroupProtectionPlugin) validatePod(a admission.Attributes) error {
	return validateParentReference(
		a,
		"Pod",
		"PodGroup",
		func(pod *api.Pod) *string {
			if pod.Spec.SchedulingGroup != nil {
				return pod.Spec.SchedulingGroup.PodGroupName
			}
			return nil
		},
		func(namespace, name string) (metav1.Object, error) {
			return p.podGroupLister.PodGroups(namespace).Get(name)
		},
	)
}

func (p *podGroupProtectionPlugin) validatePodGroup(a admission.Attributes) error {
	if !p.compositePodGroupEnabled {
		return nil
	}
	return validateParentReference(
		a,
		"PodGroup",
		"CompositePodGroup",
		func(pg *schedulingapi.PodGroup) *string {
			return pg.Spec.ParentCompositePodGroupName
		},
		func(namespace, name string) (metav1.Object, error) {
			return p.compositePodGroupLister.CompositePodGroups(namespace).Get(name)
		},
	)
}

func (p *podGroupProtectionPlugin) validateCompositePodGroup(a admission.Attributes) error {
	if !p.compositePodGroupEnabled {
		return nil
	}
	return validateParentReference(
		a,
		"CompositePodGroup",
		"CompositePodGroup",
		func(cpg *schedulingapi.CompositePodGroup) *string {
			return cpg.Spec.ParentCompositePodGroupName
		},
		func(namespace, name string) (metav1.Object, error) {
			return p.compositePodGroupLister.CompositePodGroups(namespace).Get(name)
		},
	)
}

// validateParentReference checks if the parent object referenced by child is marked for deletion.
func validateParentReference[T any](
	a admission.Attributes,
	childKind, parentKind string,
	getParentName func(T) *string,
	getParent func(namespace, name string) (metav1.Object, error),
) error {
	child, ok := a.GetObject().(T)
	if !ok {
		return nil
	}
	parentName := getParentName(child)
	if parentName == nil || len(*parentName) == 0 {
		return nil
	}

	parent, err := getParent(a.GetNamespace(), *parentName)
	if err != nil {
		if apierrors.IsNotFound(err) {
			return nil
		}
		return admission.NewForbidden(a, err)
	}
	if parent.GetDeletionTimestamp() != nil {
		return admission.NewForbidden(a, fmt.Errorf("cannot create %s referencing %s %q because it is being deleted", childKind, parentKind, *parentName))
	}
	return nil
}
