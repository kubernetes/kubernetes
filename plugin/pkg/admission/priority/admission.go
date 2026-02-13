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
	"context"
	"fmt"
	"io"

	apiv1 "k8s.io/api/core/v1"
	schedulingv1 "k8s.io/api/scheduling/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apiserver/pkg/admission"
	genericadmissioninitializers "k8s.io/apiserver/pkg/admission/initializer"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	schedulingv1listers "k8s.io/client-go/listers/scheduling/v1"
	"k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/scheduling"
	"k8s.io/kubernetes/pkg/features"
)

const (
	// PluginName indicates name of admission plugin.
	PluginName = "Priority"
)

// Register registers a plugin
func Register(plugins *admission.Plugins) {
	plugins.Register(PluginName, func(config io.Reader) (admission.Interface, error) {
		return NewPlugin(), nil
	})
}

// Plugin is an implementation of admission.Interface.
type Plugin struct {
	*admission.Handler
	client kubernetes.Interface
	lister schedulingv1listers.PriorityClassLister
}

var _ admission.MutationInterface = &Plugin{}
var _ admission.ValidationInterface = &Plugin{}
var _ = genericadmissioninitializers.WantsExternalKubeInformerFactory(&Plugin{})
var _ = genericadmissioninitializers.WantsExternalKubeClientSet(&Plugin{})

// NewPlugin creates a new priority admission plugin.
func NewPlugin() *Plugin {
	return &Plugin{
		Handler: admission.NewHandler(admission.Create, admission.Update, admission.Delete),
	}
}

// ValidateInitialization implements the InitializationValidator interface.
func (p *Plugin) ValidateInitialization() error {
	if p.client == nil {
		return fmt.Errorf("%s requires a client", PluginName)
	}
	if p.lister == nil {
		return fmt.Errorf("%s requires a lister", PluginName)
	}
	return nil
}

// SetExternalKubeClientSet implements the WantsInternalKubeClientSet interface.
func (p *Plugin) SetExternalKubeClientSet(client kubernetes.Interface) {
	p.client = client
}

// SetExternalKubeInformerFactory implements the WantsInternalKubeInformerFactory interface.
func (p *Plugin) SetExternalKubeInformerFactory(f informers.SharedInformerFactory) {
	priorityInformer := f.Scheduling().V1().PriorityClasses()
	p.lister = priorityInformer.Lister()
	p.SetReadyFunc(priorityInformer.Informer().HasSynced)
}

var (
	podResource           = core.Resource("pods")
	workloadResource      = scheduling.Resource("workloads")
	priorityClassResource = scheduling.Resource("priorityclasses")
)

// Admit checks Pods and Workloads and admits or rejects them. It also resolves the priority of pods/workloads based on their PriorityClass.
// Note that pod/workload validation mechanism prevents update of a pod/workload priority.
func (p *Plugin) Admit(ctx context.Context, a admission.Attributes, o admission.ObjectInterfaces) error {
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
	case workloadResource:
		if operation == admission.Create || operation == admission.Update {
			return p.admitWorkload(a)
		}
		return nil
	default:
		return nil
	}
}

// admitWorkload validates PriorityClassName and resolves Priority value for a workload.
// This is only performed when workload-aware-preemption feature gate is enabled.
//
// PriorityClassName validation rules:
// 1. If PriorityClassName is provided, it must refer to an existing PriorityClass.
// 2. Otherwise, the PriorityClassName is set to the global default PriorityClass if it exists.
//
// Priority value resolution:
// 1. If a valid PriorityClassName is provided, use the priority value of that PriorityClass.
// 2. If a global default PriorityClass exists, use its priority value.
// 3. Otherwise, use the default priority (0).
func (p *Plugin) admitWorkload(a admission.Attributes) error {
	if !utilfeature.DefaultFeatureGate.Enabled(features.WorkloadAwarePreemption) {
		return nil
	}

	operation := a.GetOperation()
	workload, ok := a.GetObject().(*scheduling.Workload)
	if !ok {
		return errors.NewBadRequest("resource was marked with kind Workload but was unable to be converted")
	}

	if operation == admission.Create {
		var priority int32 = 0
		if workload.Spec.PriorityClassName == nil || len(*workload.Spec.PriorityClassName) == 0 {
			defaultPriorityClassName, defaultPriority, _, err := p.getDefaultPriority()
			if err != nil {
				return fmt.Errorf("failed to get default priority class: %w", err)
			}
			workload.Spec.PriorityClassName = &defaultPriorityClassName
			priority = defaultPriority
		} else {
			priorityClass, err := p.lister.Get(*workload.Spec.PriorityClassName)
			if err != nil {
				if errors.IsNotFound(err) {
					return admission.NewForbidden(a, fmt.Errorf("no PriorityClass with name %v was found", *workload.Spec.PriorityClassName))
				}
				return fmt.Errorf("failed to resolve PriorityClass with name %s: %w", *workload.Spec.PriorityClassName, err)
			}
			priority = priorityClass.Value
		}
		workload.Spec.Priority = &priority
	}
	return nil
}

// Validate checks PriorityClasses and admits or rejects them.
func (p *Plugin) Validate(ctx context.Context, a admission.Attributes, o admission.ObjectInterfaces) error {
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

// admitPod makes sure a new pod does not set spec.Priority field. It also makes sure that the PriorityClassName exists if it is provided and resolves the pod priority from the PriorityClassName.
func (p *Plugin) admitPod(a admission.Attributes) error {
	operation := a.GetOperation()
	pod, ok := a.GetObject().(*core.Pod)
	if !ok {
		return errors.NewBadRequest("resource was marked with kind Pod but was unable to be converted")
	}

	if operation == admission.Update {
		oldPod, ok := a.GetOldObject().(*core.Pod)
		if !ok {
			return errors.NewBadRequest("resource was marked with kind Pod but was unable to be converted")
		}

		// This admission plugin set pod.Spec.Priority on create.
		// Ensure the existing priority is preserved on update.
		// API validation prevents mutations to Priority and PriorityClassName, so any other changes will fail update validation and not be persisted.
		if pod.Spec.Priority == nil && oldPod.Spec.Priority != nil {
			pod.Spec.Priority = oldPod.Spec.Priority
		}
		if pod.Spec.PreemptionPolicy == nil && oldPod.Spec.PreemptionPolicy != nil {
			pod.Spec.PreemptionPolicy = oldPod.Spec.PreemptionPolicy
		}
		return nil
	}

	if operation == admission.Create {
		var priority int32
		var preemptionPolicy *apiv1.PreemptionPolicy
		if len(pod.Spec.PriorityClassName) == 0 {
			var err error
			var pcName string
			pcName, priority, preemptionPolicy, err = p.getDefaultPriority()
			if err != nil {
				return fmt.Errorf("failed to get default priority class: %v", err)
			}
			pod.Spec.PriorityClassName = pcName
		} else {
			// Try resolving the priority class name.
			pc, err := p.lister.Get(pod.Spec.PriorityClassName)
			if err != nil {
				if errors.IsNotFound(err) {
					return admission.NewForbidden(a, fmt.Errorf("no PriorityClass with name %v was found", pod.Spec.PriorityClassName))
				}

				return fmt.Errorf("failed to get PriorityClass with name %s: %v", pod.Spec.PriorityClassName, err)
			}

			priority = pc.Value
			preemptionPolicy = pc.PreemptionPolicy
		}
		// if the pod contained a priority that differs from the one computed from the priority class, error
		if pod.Spec.Priority != nil && *pod.Spec.Priority != priority {
			return admission.NewForbidden(a, fmt.Errorf("the integer value of priority (%d) must not be provided in pod spec; priority admission controller computed %d from the given PriorityClass name", *pod.Spec.Priority, priority))
		}
		pod.Spec.Priority = &priority

		var corePolicy core.PreemptionPolicy
		if preemptionPolicy != nil {
			corePolicy = core.PreemptionPolicy(*preemptionPolicy)
			if pod.Spec.PreemptionPolicy != nil && *pod.Spec.PreemptionPolicy != corePolicy {
				return admission.NewForbidden(a, fmt.Errorf("the string value of PreemptionPolicy (%s) must not be provided in pod spec; priority admission controller computed %s from the given PriorityClass name", *pod.Spec.PreemptionPolicy, corePolicy))
			}
			pod.Spec.PreemptionPolicy = &corePolicy
		}
	}
	return nil
}

// validatePriorityClass ensures that the value field is not larger than the highest user definable priority. If the GlobalDefault is set, it ensures that there is no other PriorityClass whose GlobalDefault is set.
func (p *Plugin) validatePriorityClass(a admission.Attributes) error {
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

func (p *Plugin) getDefaultPriorityClass() (*schedulingv1.PriorityClass, error) {
	list, err := p.lister.List(labels.Everything())
	if err != nil {
		return nil, err
	}
	// In case more than one global default priority class is added as a result of a race condition,
	// we pick the one with the lowest priority value.
	var defaultPC *schedulingv1.PriorityClass
	for _, pci := range list {
		if pci.GlobalDefault {
			if defaultPC == nil || defaultPC.Value > pci.Value {
				defaultPC = pci
			}
		}
	}
	return defaultPC, nil
}

func (p *Plugin) getDefaultPriority() (string, int32, *apiv1.PreemptionPolicy, error) {
	dpc, err := p.getDefaultPriorityClass()
	if err != nil {
		return "", 0, nil, err
	}
	if dpc != nil {
		return dpc.Name, dpc.Value, dpc.PreemptionPolicy, nil
	}
	preemptLowerPriority := apiv1.PreemptLowerPriority
	return "", int32(scheduling.DefaultPriorityWhenNoDefaultClassExists), &preemptLowerPriority, nil
}
