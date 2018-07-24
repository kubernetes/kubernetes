/*
Copyright 2017 The Kubernetes Authors.

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

package priority

import (
	"fmt"
	"io"

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apiserver/pkg/admission"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/scheduling"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	informers "k8s.io/kubernetes/pkg/client/informers/informers_generated/internalversion"
	schedulinglisters "k8s.io/kubernetes/pkg/client/listers/scheduling/internalversion"
	"k8s.io/kubernetes/pkg/features"
	kubeapiserveradmission "k8s.io/kubernetes/pkg/kubeapiserver/admission"
	kubelettypes "k8s.io/kubernetes/pkg/kubelet/types"
)

const (
	// PluginName indicates name of admission plugin.
	PluginName = "Priority"
)

// Register registers a plugin
func Register(plugins *admission.Plugins) {
	plugins.Register(PluginName, func(config io.Reader) (admission.Interface, error) {
		return newPlugin(), nil
	})
}

// priorityPlugin is an implementation of admission.Interface.
type priorityPlugin struct {
	*admission.Handler
	client internalclientset.Interface
	lister schedulinglisters.PriorityClassLister
}

var _ admission.MutationInterface = &priorityPlugin{}
var _ admission.ValidationInterface = &priorityPlugin{}
var _ = kubeapiserveradmission.WantsInternalKubeInformerFactory(&priorityPlugin{})
var _ = kubeapiserveradmission.WantsInternalKubeClientSet(&priorityPlugin{})

// NewPlugin creates a new priority admission plugin.
func newPlugin() *priorityPlugin {
	return &priorityPlugin{
		Handler: admission.NewHandler(admission.Create, admission.Update, admission.Delete),
	}
}

// ValidateInitialization implements the InitializationValidator interface.
func (p *priorityPlugin) ValidateInitialization() error {
	if p.client == nil {
		return fmt.Errorf("%s requires a client", PluginName)
	}
	if p.lister == nil {
		return fmt.Errorf("%s requires a lister", PluginName)
	}
	return nil
}

// SetInternalKubeClientSet implements the WantsInternalKubeClientSet interface.
func (p *priorityPlugin) SetInternalKubeClientSet(client internalclientset.Interface) {
	p.client = client
}

// SetInternalKubeInformerFactory implements the WantsInternalKubeInformerFactory interface.
func (p *priorityPlugin) SetInternalKubeInformerFactory(f informers.SharedInformerFactory) {
	priorityInformer := f.Scheduling().InternalVersion().PriorityClasses()
	p.lister = priorityInformer.Lister()
	p.SetReadyFunc(priorityInformer.Informer().HasSynced)
}

var (
	podResource           = api.Resource("pods")
	priorityClassResource = scheduling.Resource("priorityclasses")
)

// Admit checks Pods and admits or rejects them. It also resolves the priority of pods based on their PriorityClass.
// Note that pod validation mechanism prevents update of a pod priority.
func (p *priorityPlugin) Admit(a admission.Attributes) error {
	if !utilfeature.DefaultFeatureGate.Enabled(features.PodPriority) {
		return nil
	}

	operation := a.GetOperation()
	// Ignore all calls to subresources
	if len(a.GetSubresource()) != 0 {
		return nil
	}

	switch a.GetResource().GroupResource() {
	case podResource:
		if operation == admission.Create || operation == admission.Update {
			return p.admitPod(a)
		}
		return nil

	default:
		return nil
	}
}

// Validate checks PriorityClasses and admits or rejects them.
func (p *priorityPlugin) Validate(a admission.Attributes) error {
	operation := a.GetOperation()
	// Ignore all calls to subresources
	if len(a.GetSubresource()) != 0 {
		return nil
	}

	switch a.GetResource().GroupResource() {
	case priorityClassResource:
		if operation == admission.Create || operation == admission.Update {
			return p.validatePriorityClass(a)
		}
		return nil

	default:
		return nil
	}
}

// priorityClassPermittedInNamespace returns true if we allow the given priority class name in the
// given namespace. It currently checks that system priorities are created only in the system namespace.
func priorityClassPermittedInNamespace(priorityClassName string, namespace string) bool {
	// Only allow system priorities in the system namespace. This is to prevent abuse or incorrect
	// usage of these priorities. Pods created at these priorities could preempt system critical
	// components.
	for _, spc := range scheduling.SystemPriorityClasses() {
		if spc.Name == priorityClassName && namespace != metav1.NamespaceSystem {
			return false
		}
	}
	return true
}

// admitPod makes sure a new pod does not set spec.Priority field. It also makes sure that the PriorityClassName exists if it is provided and resolves the pod priority from the PriorityClassName.
func (p *priorityPlugin) admitPod(a admission.Attributes) error {
	operation := a.GetOperation()
	pod, ok := a.GetObject().(*api.Pod)
	if !ok {
		return errors.NewBadRequest("resource was marked with kind Pod but was unable to be converted")
	}

	if operation == admission.Update {
		oldPod, ok := a.GetOldObject().(*api.Pod)
		if !ok {
			return errors.NewBadRequest("resource was marked with kind Pod but was unable to be converted")
		}

		// This admission plugin set pod.Spec.Priority on create.
		// Ensure the existing priority is preserved on update.
		// API validation prevents mutations to Priority and PriorityClassName, so any other changes will fail update validation and not be persisted.
		if pod.Spec.Priority == nil && oldPod.Spec.Priority != nil {
			pod.Spec.Priority = oldPod.Spec.Priority
		}
		return nil
	}

	if operation == admission.Create {
		var priority int32
		// TODO: @ravig - This is for backwards compatibility to ensure that critical pods with annotations just work fine.
		// Remove when no longer needed.
		if len(pod.Spec.PriorityClassName) == 0 &&
			utilfeature.DefaultFeatureGate.Enabled(features.ExperimentalCriticalPodAnnotation) &&
			kubelettypes.IsCritical(a.GetNamespace(), pod.Annotations) {
			pod.Spec.PriorityClassName = scheduling.SystemClusterCritical
		}
		if len(pod.Spec.PriorityClassName) == 0 {
			var err error
			priority, err = p.getDefaultPriority()
			if err != nil {
				return fmt.Errorf("failed to get default priority class: %v", err)
			}
		} else {
			pcName := pod.Spec.PriorityClassName
			if !priorityClassPermittedInNamespace(pcName, a.GetNamespace()) {
				return admission.NewForbidden(a, fmt.Errorf("pods with %v priorityClass is not permitted in %v namespace", pcName, a.GetNamespace()))
			}

			// Try resolving the priority class name.
			pc, err := p.lister.Get(pod.Spec.PriorityClassName)
			if err != nil {
				if errors.IsNotFound(err) {
					return admission.NewForbidden(a, fmt.Errorf("no PriorityClass with name %v was found", pod.Spec.PriorityClassName))
				}

				return fmt.Errorf("failed to get PriorityClass with name %s: %v", pod.Spec.PriorityClassName, err)
			}

			priority = pc.Value
		}
		// if the pod contained a priority that differs from the one computed from the priority class, error
		if pod.Spec.Priority != nil && *pod.Spec.Priority != priority {
			return admission.NewForbidden(a, fmt.Errorf("the integer value of priority (%d) must not be provided in pod spec; priority admission controller computed %d from the given PriorityClass name", *pod.Spec.Priority, priority))
		}
		pod.Spec.Priority = &priority
	}
	return nil
}

// validatePriorityClass ensures that the value field is not larger than the highest user definable priority. If the GlobalDefault is set, it ensures that there is no other PriorityClass whose GlobalDefault is set.
func (p *priorityPlugin) validatePriorityClass(a admission.Attributes) error {
	operation := a.GetOperation()
	pc, ok := a.GetObject().(*scheduling.PriorityClass)
	if !ok {
		return errors.NewBadRequest("resource was marked with kind PriorityClass but was unable to be converted")
	}
	// If the new PriorityClass tries to be the default priority, make sure that no other priority class is marked as default.
	if pc.GlobalDefault {
		dpc, err := p.getDefaultPriorityClass()
		if err != nil {
			return fmt.Errorf("failed to get default priority class: %v", err)
		}
		if dpc != nil {
			// Throw an error if a second default priority class is being created, or an existing priority class is being marked as default, while another default already exists.
			if operation == admission.Create || (operation == admission.Update && dpc.GetName() != pc.GetName()) {
				return admission.NewForbidden(a, fmt.Errorf("PriorityClass %v is already marked as default. Only one default can exist", dpc.GetName()))
			}
		}
	}
	return nil
}

func (p *priorityPlugin) getDefaultPriorityClass() (*scheduling.PriorityClass, error) {
	list, err := p.lister.List(labels.Everything())
	if err != nil {
		return nil, err
	}
	// In case more than one global default priority class is added as a result of a race condition,
	// we pick the one with the lowest priority value.
	var defaultPC *scheduling.PriorityClass
	for _, pci := range list {
		if pci.GlobalDefault {
			if defaultPC == nil || defaultPC.Value > pci.Value {
				defaultPC = pci
			}
		}
	}
	return defaultPC, nil
}

func (p *priorityPlugin) getDefaultPriority() (int32, error) {
	dpc, err := p.getDefaultPriorityClass()
	if err != nil {
		return 0, err
	}
	if dpc != nil {
		return dpc.Value, nil
	}
	return int32(scheduling.DefaultPriorityWhenNoDefaultClassExists), nil
}
