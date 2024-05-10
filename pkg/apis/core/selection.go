/*
Copyright 2024 The Kubernetes Authors.

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

package core

import (
	"fmt"
	"strconv"

	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
)

func init() {
	SchemeBuilder.Register(addSelectorFuncs)
}

// addSelectorFuncs adds versioned selector funcs for resources to the scheme.
func addSelectorFuncs(scheme *runtime.Scheme) error {
	scheme.AddSelectorFunc(SchemeGroupVersion.WithKind("Event"), EventSelectorFunc)
	scheme.AddSelectorFunc(SchemeGroupVersion.WithKind("Namespace"), NamespaceSelectorFunc)
	scheme.AddSelectorFunc(SchemeGroupVersion.WithKind("Node"), NodeSelectorFunc)
	scheme.AddSelectorFunc(SchemeGroupVersion.WithKind("PersistentVolume"), PersistentVolumeSelectorFunc)
	scheme.AddSelectorFunc(SchemeGroupVersion.WithKind("PersistentVolumeClaim"), PersistentVolumeClaimSelectorFunc)
	scheme.AddSelectorFunc(SchemeGroupVersion.WithKind("Pod"), PodSelectorFunc)
	scheme.AddSelectorFunc(SchemeGroupVersion.WithKind("ReplicationController"), ReplicationControllerSelectorFunc)
	scheme.AddSelectorFunc(SchemeGroupVersion.WithKind("Secret"), SecretSelectorFunc)
	return nil
}

// EventSelectorFunc returns true if the object is an event that matches the label and field selectors.
func EventSelectorFunc(obj runtime.Object, selector runtime.Selectors) (bool, error) {
	return EventMatcher(selector).Matches(obj)
}

// EventMatcher returns a selection predicate for a given label and field selector.
func EventMatcher(selector runtime.Selectors) runtime.SelectionPredicate {
	return runtime.SelectionPredicate{
		Selectors: selector,
		GetAttrs:  EventGetAttrs,
	}
}

// EventGetAttrs returns labels and fields of a given object for filtering purposes.
func EventGetAttrs(obj runtime.Object) (labels.Set, fields.Set, error) {
	event, ok := obj.(*Event)
	if !ok {
		return nil, nil, fmt.Errorf("not an event")
	}
	return labels.Set(event.Labels), EventToSelectableFields(event), nil
}

// EventToSelectableFields returns a field set that represents the object.
func EventToSelectableFields(event *Event) fields.Set {
	defaultFields := runtime.DefaultSelectableFields(event)
	// Use exact size to reduce memory allocations.
	// If you change the fields, adjust the size.
	specificFieldsSet := make(fields.Set, len(defaultFields)+11)
	source := event.Source.Component
	if source == "" {
		source = event.ReportingController
	}
	specificFieldsSet["involvedObject.kind"] = event.InvolvedObject.Kind
	specificFieldsSet["involvedObject.namespace"] = event.InvolvedObject.Namespace
	specificFieldsSet["involvedObject.name"] = event.InvolvedObject.Name
	specificFieldsSet["involvedObject.uid"] = string(event.InvolvedObject.UID)
	specificFieldsSet["involvedObject.apiVersion"] = event.InvolvedObject.APIVersion
	specificFieldsSet["involvedObject.resourceVersion"] = event.InvolvedObject.ResourceVersion
	specificFieldsSet["involvedObject.fieldPath"] = event.InvolvedObject.FieldPath
	specificFieldsSet["reason"] = event.Reason
	specificFieldsSet["reportingComponent"] = event.ReportingController // use the core/v1 field name
	specificFieldsSet["source"] = source
	specificFieldsSet["type"] = event.Type
	return runtime.MergeFieldsSets(specificFieldsSet, defaultFields)
}

// NamespaceSelectorFunc returns true if the object matches the label and field selectors.
func NamespaceSelectorFunc(obj runtime.Object, selector runtime.Selectors) (bool, error) {
	return NamespaceMatcher(selector).Matches(obj)
}

// NamespaceMatchNamespace returns a generic matcher for a given label and field selector.
func NamespaceMatcher(selector runtime.Selectors) runtime.SelectionPredicate {
	return runtime.SelectionPredicate{
		Selectors: selector,
		GetAttrs:  NamespaceGetAttrs,
	}
}

// NamespaceGetAttrs returns labels and fields of a given object for filtering purposes.
func NamespaceGetAttrs(obj runtime.Object) (labels.Set, fields.Set, error) {
	namespaceObj, ok := obj.(*Namespace)
	if !ok {
		return nil, nil, fmt.Errorf("not a namespace")
	}
	return labels.Set(namespaceObj.Labels), NamespaceToSelectableFields(namespaceObj), nil
}

// NamespaceToSelectableFields returns a field set that represents the object
func NamespaceToSelectableFields(namespace *Namespace) fields.Set {
	defaultFields := runtime.DefaultSelectableFields(namespace)
	// Use exact size to reduce memory allocations.
	// If you change the fields, adjust the size.
	specificFieldsSet := make(fields.Set, len(defaultFields)+2)
	specificFieldsSet["status.phase"] = string(namespace.Status.Phase)
	// This is a bug, but we need to support it for backward compatibility.
	specificFieldsSet["name"] = namespace.Name
	return runtime.MergeFieldsSets(specificFieldsSet, defaultFields)
}

// NodeSelectorFunc returns true if the object matches the label and field selectors.
func NodeSelectorFunc(obj runtime.Object, selector runtime.Selectors) (bool, error) {
	return NodeMatcher(selector).Matches(obj)
}

// NodeMatcher returns a generic matcher for a given label and field selector.
func NodeMatcher(selector runtime.Selectors) runtime.SelectionPredicate {
	return runtime.SelectionPredicate{
		Selectors: selector,
		GetAttrs:  NodeGetAttrs,
	}
}

// NodeGetAttrs returns labels and fields of a given object for filtering purposes.
func NodeGetAttrs(obj runtime.Object) (labels.Set, fields.Set, error) {
	nodeObj, ok := obj.(*Node)
	if !ok {
		return nil, nil, fmt.Errorf("not a node")
	}
	return labels.Set(nodeObj.ObjectMeta.Labels), NodeToSelectableFields(nodeObj), nil
}

// NodeToSelectableFields returns a field set that represents the object.
func NodeToSelectableFields(node *Node) fields.Set {
	defaultFields := runtime.DefaultSelectableFields(node)
	// Use exact size to reduce memory allocations.
	// If you change the fields, adjust the size.
	specificFieldsSet := make(fields.Set, len(defaultFields)+1)
	specificFieldsSet["spec.unschedulable"] = fmt.Sprint(node.Spec.Unschedulable)
	return runtime.MergeFieldsSets(specificFieldsSet, defaultFields)
}

// PersistentVolumeClaimSelectorFunc returns true if the object matches the label and field selectors.
func PersistentVolumeClaimSelectorFunc(obj runtime.Object, selector runtime.Selectors) (bool, error) {
	return PersistentVolumeClaimMatcher(selector).Matches(obj)
}

// PersistentVolumeClaimMatcher returns a generic matcher for a given label and field selector.
func PersistentVolumeClaimMatcher(selector runtime.Selectors) runtime.SelectionPredicate {
	return runtime.SelectionPredicate{
		Selectors: selector,
		GetAttrs:  PersistentVolumeClaimGetAttrs,
	}
}

// PersistentVolumeClaimGetAttrs returns labels and fields of a given object for filtering purposes.
func PersistentVolumeClaimGetAttrs(obj runtime.Object) (labels.Set, fields.Set, error) {
	persistentvolumeclaimObj, ok := obj.(*PersistentVolumeClaim)
	if !ok {
		return nil, nil, fmt.Errorf("not a persistentvolumeclaim")
	}
	return labels.Set(persistentvolumeclaimObj.Labels), PersistentVolumeClaimToSelectableFields(persistentvolumeclaimObj), nil
}

// PersistentVolumeClaimToSelectableFields returns a field set that represents the object
func PersistentVolumeClaimToSelectableFields(persistentvolumeclaim *PersistentVolumeClaim) fields.Set {
	defaultFields := runtime.DefaultSelectableFields(persistentvolumeclaim)
	// Use exact size to reduce memory allocations.
	// If you change the fields, adjust the size.
	specificFieldsSet := make(fields.Set, len(defaultFields)+1)
	// This is a bug, but we need to support it for backward compatibility.
	specificFieldsSet["name"] = persistentvolumeclaim.Name
	return runtime.MergeFieldsSets(specificFieldsSet, defaultFields)
}

// PersistentVolumeSelectorFunc returns true if the object matches the label and field selectors.
func PersistentVolumeSelectorFunc(obj runtime.Object, selector runtime.Selectors) (bool, error) {
	return PersistentVolumeMatcher(selector).Matches(obj)
}

// PersistentVolumeMatcher returns a generic matcher for a given label and field selector.
func PersistentVolumeMatcher(selector runtime.Selectors) runtime.SelectionPredicate {
	return runtime.SelectionPredicate{
		Selectors: selector,
		GetAttrs:  PersistentVolumeGetAttrs,
	}
}

// PersistentVolumeGetAttrs returns labels and fields of a given object for filtering purposes.
func PersistentVolumeGetAttrs(obj runtime.Object) (labels.Set, fields.Set, error) {
	persistentvolumeObj, ok := obj.(*PersistentVolume)
	if !ok {
		return nil, nil, fmt.Errorf("not a persistentvolume")
	}
	return labels.Set(persistentvolumeObj.Labels), PersistentVolumeToSelectableFields(persistentvolumeObj), nil
}

// PersistentVolumeToSelectableFields returns a field set that represents the object
func PersistentVolumeToSelectableFields(persistentvolume *PersistentVolume) fields.Set {
	defaultFields := runtime.DefaultSelectableFields(persistentvolume)
	// Use exact size to reduce memory allocations.
	// If you change the fields, adjust the size.
	specificFieldsSet := make(fields.Set, len(defaultFields)+1)
	// This is a bug, but we need to support it for backward compatibility.
	specificFieldsSet["name"] = persistentvolume.Name
	return runtime.MergeFieldsSets(specificFieldsSet, defaultFields)
}

// PodSelectorFunc returns true if the object matches the label and field selectors.
func PodSelectorFunc(obj runtime.Object, selector runtime.Selectors) (bool, error) {
	return PodMatcher(selector).Matches(obj)
}

// PodMatcher returns a pod matcher for a given label and field selector.
func PodMatcher(selector runtime.Selectors) runtime.SelectionPredicate {
	return runtime.SelectionPredicate{
		Selectors: selector,
		GetAttrs:  PodGetAttrs,
	}
}

// PodGetAttrs returns labels and fields of a given pod for filtering purposes.
func PodGetAttrs(obj runtime.Object) (labels.Set, fields.Set, error) {
	pod, ok := obj.(*Pod)
	if !ok {
		return nil, nil, fmt.Errorf("not a pod")
	}
	return labels.Set(pod.ObjectMeta.Labels), PodToSelectableFields(pod), nil
}

// PodToSelectableFields returns a field set that represents the pod.
// TODO: fields are not labels, and the validation rules for them do not apply.
func PodToSelectableFields(pod *Pod) fields.Set {
	defaultFields := runtime.DefaultSelectableFields(pod)
	// Use exact size to reduce memory allocations.
	// If you change the fields, adjust the size.
	podSpecificFieldsSet := make(fields.Set, len(defaultFields)+8)
	podSpecificFieldsSet["spec.nodeName"] = pod.Spec.NodeName
	podSpecificFieldsSet["spec.restartPolicy"] = string(pod.Spec.RestartPolicy)
	podSpecificFieldsSet["spec.schedulerName"] = string(pod.Spec.SchedulerName)
	podSpecificFieldsSet["spec.serviceAccountName"] = string(pod.Spec.ServiceAccountName)
	if pod.Spec.SecurityContext != nil {
		podSpecificFieldsSet["spec.hostNetwork"] = strconv.FormatBool(pod.Spec.SecurityContext.HostNetwork)
	} else {
		// default to false
		podSpecificFieldsSet["spec.hostNetwork"] = strconv.FormatBool(false)
	}
	podSpecificFieldsSet["status.phase"] = string(pod.Status.Phase)
	// TODO: add podIPs as a downward API value(s) with proper format
	podIP := ""
	if len(pod.Status.PodIPs) > 0 {
		podIP = string(pod.Status.PodIPs[0].IP)
	}
	podSpecificFieldsSet["status.podIP"] = podIP
	podSpecificFieldsSet["status.nominatedNodeName"] = string(pod.Status.NominatedNodeName)
	return runtime.MergeFieldsSets(podSpecificFieldsSet, defaultFields)
}

// ReplicationControllerSelectorFunc returns true if the object matches the label and field selectors.
func ReplicationControllerSelectorFunc(obj runtime.Object, selector runtime.Selectors) (bool, error) {
	return ReplicationControllerMatcher(selector).Matches(obj)
}

// ReplicationControllerMatcher returns a generic matcher for a given label and field selector.
func ReplicationControllerMatcher(selector runtime.Selectors) runtime.SelectionPredicate {
	return runtime.SelectionPredicate{
		Selectors: selector,
		GetAttrs:  ReplicationControllerGetAttrs,
	}
}

// ReplicationControllerGetAttrs returns labels and fields of a given object for filtering purposes.
func ReplicationControllerGetAttrs(obj runtime.Object) (labels.Set, fields.Set, error) {
	rc, ok := obj.(*ReplicationController)
	if !ok {
		return nil, nil, fmt.Errorf("given object is not a replication controller")
	}
	return labels.Set(rc.ObjectMeta.Labels), ReplicationControllerToSelectableFields(rc), nil
}

// ReplicationControllerToSelectableFields returns a field set that represents the object.
func ReplicationControllerToSelectableFields(controller *ReplicationController) fields.Set {
	defaultFields := runtime.DefaultSelectableFields(controller)
	// Use exact size to reduce memory allocations.
	// If you change the fields, adjust the size.
	specificFieldsSet := make(fields.Set, len(defaultFields)+1)
	specificFieldsSet["status.replicas"] = strconv.Itoa(int(controller.Status.Replicas))
	return runtime.MergeFieldsSets(specificFieldsSet, defaultFields)
}

// SecretSelectorFunc returns true if the object matches the label and field selectors.
func SecretSelectorFunc(obj runtime.Object, selector runtime.Selectors) (bool, error) {
	return SecretMatcher(selector).Matches(obj)
}

// SecretMatcher returns a selection predicate for a given label and field selector.
func SecretMatcher(selector runtime.Selectors) runtime.SelectionPredicate {
	return runtime.SelectionPredicate{
		Selectors: selector,
		GetAttrs:  SecretGetAttrs,
	}
}

// SecretGetAttrs returns labels and fields of a given object for filtering purposes.
func SecretGetAttrs(obj runtime.Object) (labels.Set, fields.Set, error) {
	secret, ok := obj.(*Secret)
	if !ok {
		return nil, nil, fmt.Errorf("not a secret")
	}
	return labels.Set(secret.Labels), SecretToSelectableFields(secret), nil
}

// SecretSelectableFields returns a field set that can be used for filter selection
func SecretToSelectableFields(secret *Secret) fields.Set {
	defaultFields := runtime.DefaultSelectableFields(secret)
	// Use exact size to reduce memory allocations.
	// If you change the fields, adjust the size.
	specificFieldsSet := make(fields.Set, len(defaultFields)+1)
	specificFieldsSet["type"] = string(secret.Type)
	return runtime.MergeFieldsSets(specificFieldsSet, defaultFields)
}

// ServiceSelectorFunc returns true if the object matches the label and field selectors.
func ServiceSelectorFunc(obj runtime.Object, selector runtime.Selectors) (bool, error) {
	return ServiceMatcher(selector).Matches(obj)
}

// ServiceMatcher returns a selection predicate for a given label and field selector.
func ServiceMatcher(selector runtime.Selectors) runtime.SelectionPredicate {
	return runtime.SelectionPredicate{
		Selectors: selector,
		GetAttrs:  ServiceGetAttrs,
	}
}

// GetAttrs returns labels and fields of a given object for filtering purposes.
func ServiceGetAttrs(obj runtime.Object) (labels.Set, fields.Set, error) {
	service, ok := obj.(*Service)
	if !ok {
		return nil, nil, fmt.Errorf("not a service")
	}
	return service.Labels, ServiceToSelectableFields(service), nil
}

// ServiceSelectableFields returns a field set that can be used for filter selection
func ServiceToSelectableFields(service *Service) fields.Set {
	defaultFields := runtime.DefaultSelectableFields(service)
	// Use exact size to reduce memory allocations.
	// If you change the fields, adjust the size.
	specificFieldsSet := make(fields.Set, len(defaultFields)+2)
	specificFieldsSet["spec.clusterIP"] = service.Spec.ClusterIP
	specificFieldsSet["spec.type"] = string(service.Spec.Type)
	return runtime.MergeFieldsSets(specificFieldsSet, defaultFields)
}
