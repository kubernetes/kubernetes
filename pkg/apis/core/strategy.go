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

package core

import (
	"fmt"
	"strconv"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
)

// PodToSelectableFields returns a field set that represents the object
// TODO: fields are not labels, and the validation rules for them do not apply.
func PodToSelectableFields(pod *Pod) fields.Set {
	// The purpose of allocation with a given number of elements is to reduce
	// amount of allocations needed to create the fields.Set. If you add any
	// field here or the number of object-meta related fields changes, this should
	// be adjusted.
	specificFieldsSet := fields.Set{
		"spec.nodeName":      string(pod.Spec.NodeName),
		"spec.restartPolicy": string(pod.Spec.RestartPolicy),
		"spec.schedulerName": string(pod.Spec.SchedulerName),
		"status.phase":       string(pod.Status.Phase),
		"status.podIP":       string(pod.Status.PodIP),
	}
	return AddObjectMetaFieldsSet(specificFieldsSet, &pod.ObjectMeta, true)
}

// NodeToSelectableFields returns a field set that represents the object.
func NodeToSelectableFields(node *Node) fields.Set {
	objectMetaFieldsSet := ObjectMetaFieldsSet(&node.ObjectMeta, false)
	specificFieldsSet := fields.Set{
		"spec.unschedulable": fmt.Sprint(node.Spec.Unschedulable),
	}
	return MergeFieldsSets(objectMetaFieldsSet, specificFieldsSet)
}

// ControllerToSelectableFields returns a field set that represents the object.
func ControllerToSelectableFields(controller *ReplicationController) fields.Set {
	objectMetaFieldsSet := ObjectMetaFieldsSet(&controller.ObjectMeta, true)
	controllerSpecificFieldsSet := fields.Set{
		"status.replicas": strconv.Itoa(int(controller.Status.Replicas)),
	}
	return MergeFieldsSets(objectMetaFieldsSet, controllerSpecificFieldsSet)
}

// PersistentVolumeToSelectableFields returns a field set that represents the object
func PersistentVolumeToSelectableFields(persistentvolume *PersistentVolume) fields.Set {
	objectMetaFieldsSet := ObjectMetaFieldsSet(&persistentvolume.ObjectMeta, false)
	specificFieldsSet := fields.Set{
		// This is a bug, but we need to support it for backward compatibility.
		"name": persistentvolume.Name,
	}
	return MergeFieldsSets(objectMetaFieldsSet, specificFieldsSet)
}

// EventToSelectableFields returns a field set that represents the object
func EventToSelectableFields(event *Event) fields.Set {
	specificFieldsSet := fields.Set{
		"involvedObject.kind":            event.InvolvedObject.Kind,
		"involvedObject.namespace":       event.InvolvedObject.Namespace,
		"involvedObject.name":            event.InvolvedObject.Name,
		"involvedObject.uid":             string(event.InvolvedObject.UID),
		"involvedObject.apiVersion":      event.InvolvedObject.APIVersion,
		"involvedObject.resourceVersion": event.InvolvedObject.ResourceVersion,
		"involvedObject.fieldPath":       event.InvolvedObject.FieldPath,
		"reason":                         event.Reason,
		"source":                         event.Source.Component,
		"type":                           event.Type,
	}
	return AddObjectMetaFieldsSet(specificFieldsSet, &event.ObjectMeta, true)
}

// NamespaceToSelectableFields returns a field set that represents the object
func NamespaceToSelectableFields(namespace *Namespace) fields.Set {
	specificFieldsSet := fields.Set{
		"status.phase": string(namespace.Status.Phase),
		// This is a bug, but we need to support it for backward compatibility.
		"name": namespace.Name,
	}
	return AddObjectMetaFieldsSet(specificFieldsSet, &namespace.ObjectMeta, false)
}

// SecretToSelectableFields returns a field set that represents the object
func SecretToSelectableFields(secret *Secret) fields.Set {
	specificFieldsSet := fields.Set{
		"type": string(secret.Type),
	}
	return AddObjectMetaFieldsSet(specificFieldsSet, &secret.ObjectMeta, false)
}

// PersistentVolumeClaimToSelectableFields returns a field set that represents the object
func PersistentVolumeClaimToSelectableFields(pvClaim *PersistentVolumeClaim) fields.Set {
	specificFieldsSet := fields.Set{
		// This is a bug, but we need to support it for backward compatibility.
		"name": pvClaim.Name,
	}
	return AddObjectMetaFieldsSet(specificFieldsSet, &pvClaim.ObjectMeta, false)
}

// AdObjectMetaField add fields that represent the ObjectMeta to source.
func AddObjectMetaFieldsSet(source fields.Set, objectMeta *metav1.ObjectMeta, hasNamespaceField bool) fields.Set {
	source["metadata.name"] = objectMeta.Name
	if hasNamespaceField {
		source["metadata.namespace"] = objectMeta.Namespace
	}
	return source
}

// MergeFieldsSets merges a fields'set from fragment into the source.
func MergeFieldsSets(source fields.Set, fragment fields.Set) fields.Set {
	for k, value := range fragment {
		source[k] = value
	}
	return source
}

// ObjectMetaFieldsSet returns a fields that represent the ObjectMeta.
func ObjectMetaFieldsSet(objectMeta *metav1.ObjectMeta, hasNamespaceField bool) fields.Set {
	if !hasNamespaceField {
		return fields.Set{
			"metadata.name": objectMeta.Name,
		}
	}
	return fields.Set{
		"metadata.name":      objectMeta.Name,
		"metadata.namespace": objectMeta.Namespace,
	}
}
