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

package admission

import (
	"fmt"
	"io"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apiserver/pkg/admission"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/scheduling"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	informers "k8s.io/kubernetes/pkg/client/informers/informers_generated/internalversion"
	schedulinglisters "k8s.io/kubernetes/pkg/client/listers/scheduling/internalversion"
	"k8s.io/kubernetes/pkg/features"
	kubeapiserveradmission "k8s.io/kubernetes/pkg/kubeapiserver/admission"
)

const (
	pluginName = "Priority"

	// HighestUserDefinablePriority is the highest priority for user defined priority classes. Priority values larger than 1 billion are reserved for Kubernetes system use.
	HighestUserDefinablePriority = 1000000000
	// SystemCriticalPriority is the beginning of the range of priority values for critical system components.
	SystemCriticalPriority = 2 * HighestUserDefinablePriority
)

// SystemPriorityClasses defines special priority classes which are used by system critical pods that should not be preempted by workload pods.
var SystemPriorityClasses = map[string]int32{
	"system-cluster-critical": SystemCriticalPriority,
	"system-node-critical":    SystemCriticalPriority + 1000,
}

// Register registers a plugin
func Register(plugins *admission.Plugins) {
	plugins.Register(pluginName, func(config io.Reader) (admission.Interface, error) {
		return NewPlugin(), nil
	})
}

// priorityPlugin is an implementation of admission.Interface.
type priorityPlugin struct {
	*admission.Handler
	client internalclientset.Interface
	lister schedulinglisters.PriorityClassLister
	// globalDefaultPriority caches the value of global default priority class.
	globalDefaultPriority *int32
}

var _ = kubeapiserveradmission.WantsInternalKubeInformerFactory(&priorityPlugin{})
var _ = kubeapiserveradmission.WantsInternalKubeClientSet(&priorityPlugin{})

// NewPlugin creates a new priority admission plugin.
func NewPlugin() admission.Interface {
	return &priorityPlugin{
		Handler: admission.NewHandler(admission.Create, admission.Update, admission.Delete),
	}
}

func (p *priorityPlugin) Validate() error {
	if p.client == nil {
		return fmt.Errorf("%s requires a client", pluginName)
	}
	if p.lister == nil {
		return fmt.Errorf("%s requires a lister", pluginName)
	}
	return nil
}

func (p *priorityPlugin) SetInternalKubeClientSet(client internalclientset.Interface) {
	p.client = client
}

func (p *priorityPlugin) SetInternalKubeInformerFactory(f informers.SharedInformerFactory) {
	priorityInformer := f.Scheduling().InternalVersion().PriorityClasses()
	p.lister = priorityInformer.Lister()
	p.SetReadyFunc(priorityInformer.Informer().HasSynced)
}

var (
	podResource           = api.Resource("pods")
	priorityClassResource = api.Resource("priorityclasses")
)

// Admit checks Pods and PriorityClasses and admits or rejects them. It also resolves the priority of pods based on their PriorityClass.
func (p *priorityPlugin) Admit(a admission.Attributes) error {
	operation := a.GetOperation()
	// Ignore all calls to subresources or resources other than pods.
	// Ignore all operations other than Create and Update.
	if len(a.GetSubresource()) != 0 {
		return nil
	}

	switch a.GetResource().GroupResource() {
	case podResource:
		if operation == admission.Create || operation == admission.Update {
			return p.admitPod(a)
		}
		return nil

	case priorityClassResource:
		if operation == admission.Create || operation == admission.Update {
			return p.admitPriorityClass(a)
		}
		if operation == admission.Delete {
			p.invalidateCachedDefaultPriority()
			return nil
		}
		return nil

	default:
		return nil
	}
}

// admitPod makes sure a new pod does not set spec.Priority field. It also makes sure that the PriorityClassName exists if it is provided and resolves the pod priority from the PriorityClassName.
// Note that pod validation mechanism prevents update of a pod priority.
func (p *priorityPlugin) admitPod(a admission.Attributes) error {
	operation := a.GetOperation()
	pod, ok := a.GetObject().(*api.Pod)
	if !ok {
		return errors.NewBadRequest("resource was marked with kind Pod but was unable to be converted")
	}
	if _, isMirrorPod := pod.Annotations[api.MirrorPodAnnotationKey]; isMirrorPod {
		return nil
	}
	// Make sure that the client has not set `priority` at the time of pod creation.
	if operation == admission.Create && pod.Spec.Priority != nil {
		return admission.NewForbidden(a, fmt.Errorf("the integer value of priority must not be provided in pod spec. Priority admission controller populates the value from the given PriorityClass name"))
	}
	if utilfeature.DefaultFeatureGate.Enabled(features.PodPriority) {
		var priority int32
		if len(pod.Spec.PriorityClassName) == 0 {
			var err error
			priority, err = p.getDefaultPriority()
			if err != nil {
				return fmt.Errorf("failed to get default priority class: %v", err)
			}
		} else {
			// First try to resolve by system priority classes.
			priority, ok = SystemPriorityClasses[pod.Spec.PriorityClassName]
			if !ok {
				// Now that we didn't find any system priority, try resolving by user defined priority classes.
				pc, err := p.lister.Get(pod.Spec.PriorityClassName)
				if err != nil {
					return fmt.Errorf("failed to get default priority class %s: %v", pod.Spec.PriorityClassName, err)
				}
				if pc == nil {
					return admission.NewForbidden(a, fmt.Errorf("no PriorityClass with name %v was found", pod.Spec.PriorityClassName))
				}
				priority = pc.Value
			}
		}
		pod.Spec.Priority = &priority
	}
	return nil
}

// admitPriorityClass ensures that the value field is not larger than the highest user definable priority. If the GlobalDefault is set, it ensures that there is no other PriorityClass whose GlobalDefault is set.
func (p *priorityPlugin) admitPriorityClass(a admission.Attributes) error {
	operation := a.GetOperation()
	pc, ok := a.GetObject().(*scheduling.PriorityClass)
	if !ok {
		return errors.NewBadRequest("resource was marked with kind PriorityClass but was unable to be converted")
	}
	if pc.Value > HighestUserDefinablePriority {
		return admission.NewForbidden(a, fmt.Errorf("maximum allowed value of a user defined priority is %v", HighestUserDefinablePriority))
	}
	if _, ok := SystemPriorityClasses[pc.Name]; ok {
		return admission.NewForbidden(a, fmt.Errorf("the name of the priority class is a reserved name for system use only: %v", pc.Name))
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
	// We conservatively invalidate our cache of global default priority upon any changes to any of the existing classes or creation of a new class.
	p.invalidateCachedDefaultPriority()
	return nil
}

func (p *priorityPlugin) getDefaultPriorityClass() (*scheduling.PriorityClass, error) {
	list, err := p.lister.List(labels.Everything())
	if err != nil {
		return nil, err
	}
	for _, pci := range list {
		if pci.GlobalDefault {
			return pci, nil
		}
	}
	return nil, nil
}

func (p *priorityPlugin) getDefaultPriority() (int32, error) {
	// If global default priority is cached, return it.
	if p.globalDefaultPriority != nil {
		return *p.globalDefaultPriority, nil
	}
	dpc, err := p.getDefaultPriorityClass()
	if err != nil {
		return 0, err
	}
	priority := int32(scheduling.DefaultPriorityWhenNoDefaultClassExists)
	if dpc != nil {
		priority = dpc.Value
	}
	// Cache the value.
	p.globalDefaultPriority = &priority
	return priority, nil
}

// invalidateCachedDefaultPriority sets global default priority to nil to indicate that it should be looked up again.
func (p *priorityPlugin) invalidateCachedDefaultPriority() {
	p.globalDefaultPriority = nil
}
