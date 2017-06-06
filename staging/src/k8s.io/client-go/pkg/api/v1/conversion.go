/*
Copyright 2015 The Kubernetes Authors.

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

package v1

import (
	"encoding/json"
	"fmt"
	"reflect"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/conversion"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/client-go/pkg/api"
	"k8s.io/client-go/pkg/apis/extensions"
)

// This is a "fast-path" that avoids reflection for common types. It focuses on the objects that are
// converted the most in the cluster.
// TODO: generate one of these for every external API group - this is to prove the impact
func addFastPathConversionFuncs(scheme *runtime.Scheme) error {
	scheme.AddGenericConversionFunc(func(objA, objB interface{}, s conversion.Scope) (bool, error) {
		switch a := objA.(type) {
		case *Pod:
			switch b := objB.(type) {
			case *api.Pod:
				return true, Convert_v1_Pod_To_api_Pod(a, b, s)
			}
		case *api.Pod:
			switch b := objB.(type) {
			case *Pod:
				return true, Convert_api_Pod_To_v1_Pod(a, b, s)
			}

		case *Event:
			switch b := objB.(type) {
			case *api.Event:
				return true, Convert_v1_Event_To_api_Event(a, b, s)
			}
		case *api.Event:
			switch b := objB.(type) {
			case *Event:
				return true, Convert_api_Event_To_v1_Event(a, b, s)
			}

		case *ReplicationController:
			switch b := objB.(type) {
			case *api.ReplicationController:
				return true, Convert_v1_ReplicationController_To_api_ReplicationController(a, b, s)
			}
		case *api.ReplicationController:
			switch b := objB.(type) {
			case *ReplicationController:
				return true, Convert_api_ReplicationController_To_v1_ReplicationController(a, b, s)
			}

		case *Node:
			switch b := objB.(type) {
			case *api.Node:
				return true, Convert_v1_Node_To_api_Node(a, b, s)
			}
		case *api.Node:
			switch b := objB.(type) {
			case *Node:
				return true, Convert_api_Node_To_v1_Node(a, b, s)
			}

		case *Namespace:
			switch b := objB.(type) {
			case *api.Namespace:
				return true, Convert_v1_Namespace_To_api_Namespace(a, b, s)
			}
		case *api.Namespace:
			switch b := objB.(type) {
			case *Namespace:
				return true, Convert_api_Namespace_To_v1_Namespace(a, b, s)
			}

		case *Service:
			switch b := objB.(type) {
			case *api.Service:
				return true, Convert_v1_Service_To_api_Service(a, b, s)
			}
		case *api.Service:
			switch b := objB.(type) {
			case *Service:
				return true, Convert_api_Service_To_v1_Service(a, b, s)
			}

		case *Endpoints:
			switch b := objB.(type) {
			case *api.Endpoints:
				return true, Convert_v1_Endpoints_To_api_Endpoints(a, b, s)
			}
		case *api.Endpoints:
			switch b := objB.(type) {
			case *Endpoints:
				return true, Convert_api_Endpoints_To_v1_Endpoints(a, b, s)
			}

		case *metav1.WatchEvent:
			switch b := objB.(type) {
			case *metav1.InternalEvent:
				return true, metav1.Convert_versioned_Event_to_versioned_InternalEvent(a, b, s)
			}
		case *metav1.InternalEvent:
			switch b := objB.(type) {
			case *metav1.WatchEvent:
				return true, metav1.Convert_versioned_InternalEvent_to_versioned_Event(a, b, s)
			}
		}
		return false, nil
	})
	return nil
}

func addConversionFuncs(scheme *runtime.Scheme) error {
	// Add non-generated conversion functions
	err := scheme.AddConversionFuncs(
		Convert_api_Pod_To_v1_Pod,
		Convert_api_PodSpec_To_v1_PodSpec,
		Convert_api_ReplicationControllerSpec_To_v1_ReplicationControllerSpec,
		Convert_api_ServiceSpec_To_v1_ServiceSpec,
		Convert_v1_Pod_To_api_Pod,
		Convert_v1_PodSpec_To_api_PodSpec,
		Convert_v1_ReplicationControllerSpec_To_api_ReplicationControllerSpec,
		Convert_v1_Secret_To_api_Secret,
		Convert_v1_ServiceSpec_To_api_ServiceSpec,
		Convert_v1_ResourceList_To_api_ResourceList,
		Convert_v1_ReplicationController_to_extensions_ReplicaSet,
		Convert_v1_ReplicationControllerSpec_to_extensions_ReplicaSetSpec,
		Convert_v1_ReplicationControllerStatus_to_extensions_ReplicaSetStatus,
		Convert_extensions_ReplicaSet_to_v1_ReplicationController,
		Convert_extensions_ReplicaSetSpec_to_v1_ReplicationControllerSpec,
		Convert_extensions_ReplicaSetStatus_to_v1_ReplicationControllerStatus,
	)
	if err != nil {
		return err
	}

	// Add field label conversions for kinds having selectable nothing but ObjectMeta fields.
	for _, k := range []string{
		"Endpoints",
		"ResourceQuota",
		"PersistentVolumeClaim",
		"Service",
		"ServiceAccount",
		"ConfigMap",
	} {
		kind := k // don't close over range variables
		err = scheme.AddFieldLabelConversionFunc("v1", kind,
			func(label, value string) (string, string, error) {
				switch label {
				case "metadata.namespace",
					"metadata.name":
					return label, value, nil
				default:
					return "", "", fmt.Errorf("field label %q not supported for %q", label, kind)
				}
			},
		)
		if err != nil {
			return err
		}
	}

	// Add field conversion funcs.
	err = scheme.AddFieldLabelConversionFunc("v1", "Pod",
		func(label, value string) (string, string, error) {
			switch label {
			case "metadata.annotations",
				"metadata.labels",
				"metadata.name",
				"metadata.namespace",
				"spec.nodeName",
				"spec.restartPolicy",
				"spec.serviceAccountName",
				"status.phase",
				"status.hostIP",
				"status.podIP":
				return label, value, nil
				// This is for backwards compatibility with old v1 clients which send spec.host
			case "spec.host":
				return "spec.nodeName", value, nil
			default:
				return "", "", fmt.Errorf("field label not supported: %s", label)
			}
		},
	)
	if err != nil {
		return err
	}
	err = scheme.AddFieldLabelConversionFunc("v1", "Node",
		func(label, value string) (string, string, error) {
			switch label {
			case "metadata.name":
				return label, value, nil
			case "spec.unschedulable":
				return label, value, nil
			default:
				return "", "", fmt.Errorf("field label not supported: %s", label)
			}
		},
	)
	if err != nil {
		return err
	}
	err = scheme.AddFieldLabelConversionFunc("v1", "ReplicationController",
		func(label, value string) (string, string, error) {
			switch label {
			case "metadata.name",
				"metadata.namespace",
				"status.replicas":
				return label, value, nil
			default:
				return "", "", fmt.Errorf("field label not supported: %s", label)
			}
		})
	if err != nil {
		return err
	}
	err = scheme.AddFieldLabelConversionFunc("v1", "PersistentVolume",
		func(label, value string) (string, string, error) {
			switch label {
			case "metadata.name":
				return label, value, nil
			default:
				return "", "", fmt.Errorf("field label not supported: %s", label)
			}
		},
	)
	if err != nil {
		return err
	}
	if err := AddFieldLabelConversionsForEvent(scheme); err != nil {
		return err
	}
	if err := AddFieldLabelConversionsForNamespace(scheme); err != nil {
		return err
	}
	if err := AddFieldLabelConversionsForSecret(scheme); err != nil {
		return err
	}
	return nil
}

func Convert_v1_ReplicationController_to_extensions_ReplicaSet(in *ReplicationController, out *extensions.ReplicaSet, s conversion.Scope) error {
	out.ObjectMeta = in.ObjectMeta
	if err := Convert_v1_ReplicationControllerSpec_to_extensions_ReplicaSetSpec(&in.Spec, &out.Spec, s); err != nil {
		return err
	}
	if err := Convert_v1_ReplicationControllerStatus_to_extensions_ReplicaSetStatus(&in.Status, &out.Status, s); err != nil {
		return err
	}
	return nil
}

func Convert_v1_ReplicationControllerSpec_to_extensions_ReplicaSetSpec(in *ReplicationControllerSpec, out *extensions.ReplicaSetSpec, s conversion.Scope) error {
	out.Replicas = *in.Replicas
	if in.Selector != nil {
		metav1.Convert_map_to_unversioned_LabelSelector(&in.Selector, out.Selector, s)
	}
	if in.Template != nil {
		if err := Convert_v1_PodTemplateSpec_To_api_PodTemplateSpec(in.Template, &out.Template, s); err != nil {
			return err
		}
	}
	return nil
}

func Convert_v1_ReplicationControllerStatus_to_extensions_ReplicaSetStatus(in *ReplicationControllerStatus, out *extensions.ReplicaSetStatus, s conversion.Scope) error {
	out.Replicas = in.Replicas
	out.FullyLabeledReplicas = in.FullyLabeledReplicas
	out.ReadyReplicas = in.ReadyReplicas
	out.AvailableReplicas = in.AvailableReplicas
	out.ObservedGeneration = in.ObservedGeneration
	return nil
}

func Convert_extensions_ReplicaSet_to_v1_ReplicationController(in *extensions.ReplicaSet, out *ReplicationController, s conversion.Scope) error {
	out.ObjectMeta = in.ObjectMeta
	if err := Convert_extensions_ReplicaSetSpec_to_v1_ReplicationControllerSpec(&in.Spec, &out.Spec, s); err != nil {
		fieldErr, ok := err.(*field.Error)
		if !ok {
			return err
		}
		if out.Annotations == nil {
			out.Annotations = make(map[string]string)
		}
		out.Annotations[NonConvertibleAnnotationPrefix+"/"+fieldErr.Field] = reflect.ValueOf(fieldErr.BadValue).String()
	}
	if err := Convert_extensions_ReplicaSetStatus_to_v1_ReplicationControllerStatus(&in.Status, &out.Status, s); err != nil {
		return err
	}
	return nil
}

func Convert_extensions_ReplicaSetSpec_to_v1_ReplicationControllerSpec(in *extensions.ReplicaSetSpec, out *ReplicationControllerSpec, s conversion.Scope) error {
	out.Replicas = new(int32)
	*out.Replicas = in.Replicas
	out.MinReadySeconds = in.MinReadySeconds
	var invalidErr error
	if in.Selector != nil {
		invalidErr = metav1.Convert_unversioned_LabelSelector_to_map(in.Selector, &out.Selector, s)
	}
	out.Template = new(PodTemplateSpec)
	if err := Convert_api_PodTemplateSpec_To_v1_PodTemplateSpec(&in.Template, out.Template, s); err != nil {
		return err
	}
	return invalidErr
}

func Convert_extensions_ReplicaSetStatus_to_v1_ReplicationControllerStatus(in *extensions.ReplicaSetStatus, out *ReplicationControllerStatus, s conversion.Scope) error {
	out.Replicas = in.Replicas
	out.FullyLabeledReplicas = in.FullyLabeledReplicas
	out.ReadyReplicas = in.ReadyReplicas
	out.AvailableReplicas = in.AvailableReplicas
	out.ObservedGeneration = in.ObservedGeneration
	return nil
}

func Convert_api_ReplicationControllerSpec_To_v1_ReplicationControllerSpec(in *api.ReplicationControllerSpec, out *ReplicationControllerSpec, s conversion.Scope) error {
	out.Replicas = &in.Replicas
	out.MinReadySeconds = in.MinReadySeconds
	out.Selector = in.Selector
	if in.Template != nil {
		out.Template = new(PodTemplateSpec)
		if err := Convert_api_PodTemplateSpec_To_v1_PodTemplateSpec(in.Template, out.Template, s); err != nil {
			return err
		}
	} else {
		out.Template = nil
	}
	return nil
}

func Convert_v1_ReplicationControllerSpec_To_api_ReplicationControllerSpec(in *ReplicationControllerSpec, out *api.ReplicationControllerSpec, s conversion.Scope) error {
	if in.Replicas != nil {
		out.Replicas = *in.Replicas
	}
	out.MinReadySeconds = in.MinReadySeconds
	out.Selector = in.Selector
	if in.Template != nil {
		out.Template = new(api.PodTemplateSpec)
		if err := Convert_v1_PodTemplateSpec_To_api_PodTemplateSpec(in.Template, out.Template, s); err != nil {
			return err
		}
	} else {
		out.Template = nil
	}
	return nil
}

func Convert_api_PodStatusResult_To_v1_PodStatusResult(in *api.PodStatusResult, out *PodStatusResult, s conversion.Scope) error {
	if err := autoConvert_api_PodStatusResult_To_v1_PodStatusResult(in, out, s); err != nil {
		return err
	}

	if old := out.Annotations; old != nil {
		out.Annotations = make(map[string]string, len(old))
		for k, v := range old {
			out.Annotations[k] = v
		}
	}
	if len(out.Status.InitContainerStatuses) > 0 {
		if out.Annotations == nil {
			out.Annotations = make(map[string]string)
		}
		value, err := json.Marshal(out.Status.InitContainerStatuses)
		if err != nil {
			return err
		}
		out.Annotations[PodInitContainerStatusesAnnotationKey] = string(value)
		out.Annotations[PodInitContainerStatusesBetaAnnotationKey] = string(value)
	} else {
		delete(out.Annotations, PodInitContainerStatusesAnnotationKey)
		delete(out.Annotations, PodInitContainerStatusesBetaAnnotationKey)
	}
	return nil
}

func Convert_v1_PodStatusResult_To_api_PodStatusResult(in *PodStatusResult, out *api.PodStatusResult, s conversion.Scope) error {
	// TODO: sometime after we move init container to stable, remove these conversions
	// If there is a beta annotation, copy to alpha key.
	// See commit log for PR #31026 for why we do this.
	if valueBeta, okBeta := in.Annotations[PodInitContainerStatusesBetaAnnotationKey]; okBeta {
		in.Annotations[PodInitContainerStatusesAnnotationKey] = valueBeta
	}
	// Move the annotation to the internal repr. field
	if value, ok := in.Annotations[PodInitContainerStatusesAnnotationKey]; ok {
		var values []ContainerStatus
		if err := json.Unmarshal([]byte(value), &values); err != nil {
			return err
		}
		// Conversion from external to internal version exists more to
		// satisfy the needs of the decoder than it does to be a general
		// purpose tool. And Decode always creates an intermediate object
		// to decode to. Thus the caller of UnsafeConvertToVersion is
		// taking responsibility to ensure mutation of in is not exposed
		// back to the caller.
		in.Status.InitContainerStatuses = values
	}

	if err := autoConvert_v1_PodStatusResult_To_api_PodStatusResult(in, out, s); err != nil {
		return err
	}
	if len(out.Annotations) > 0 {
		old := out.Annotations
		out.Annotations = make(map[string]string, len(old))
		for k, v := range old {
			out.Annotations[k] = v
		}
		delete(out.Annotations, PodInitContainerStatusesAnnotationKey)
		delete(out.Annotations, PodInitContainerStatusesBetaAnnotationKey)
	}
	return nil
}

func Convert_api_PodTemplateSpec_To_v1_PodTemplateSpec(in *api.PodTemplateSpec, out *PodTemplateSpec, s conversion.Scope) error {
	if err := autoConvert_api_PodTemplateSpec_To_v1_PodTemplateSpec(in, out, s); err != nil {
		return err
	}

	// TODO: sometime after we move init container to stable, remove these conversions.
	if old := out.Annotations; old != nil {
		out.Annotations = make(map[string]string, len(old))
		for k, v := range old {
			out.Annotations[k] = v
		}
	}
	if len(out.Spec.InitContainers) > 0 {
		if out.Annotations == nil {
			out.Annotations = make(map[string]string)
		}
		value, err := json.Marshal(out.Spec.InitContainers)
		if err != nil {
			return err
		}
		out.Annotations[PodInitContainersAnnotationKey] = string(value)
		out.Annotations[PodInitContainersBetaAnnotationKey] = string(value)
	} else {
		delete(out.Annotations, PodInitContainersAnnotationKey)
		delete(out.Annotations, PodInitContainersBetaAnnotationKey)
	}
	return nil
}

func Convert_v1_PodTemplateSpec_To_api_PodTemplateSpec(in *PodTemplateSpec, out *api.PodTemplateSpec, s conversion.Scope) error {
	// TODO: sometime after we move init container to stable, remove these conversions
	// If there is a beta annotation, copy to alpha key.
	// See commit log for PR #31026 for why we do this.
	if valueBeta, okBeta := in.Annotations[PodInitContainersBetaAnnotationKey]; okBeta {
		in.Annotations[PodInitContainersAnnotationKey] = valueBeta
	}
	// Move the annotation to the internal repr. field
	if value, ok := in.Annotations[PodInitContainersAnnotationKey]; ok {
		var values []Container
		if err := json.Unmarshal([]byte(value), &values); err != nil {
			return err
		}
		// Conversion from external to internal version exists more to
		// satisfy the needs of the decoder than it does to be a general
		// purpose tool. And Decode always creates an intermediate object
		// to decode to. Thus the caller of UnsafeConvertToVersion is
		// taking responsibility to ensure mutation of in is not exposed
		// back to the caller.
		in.Spec.InitContainers = values

		// Call defaulters explicitly until annotations are removed
		tmpPodTemp := &PodTemplate{
			Template: PodTemplateSpec{
				Spec: PodSpec{
					HostNetwork:    in.Spec.HostNetwork,
					InitContainers: values,
				},
			},
		}
		SetObjectDefaults_PodTemplate(tmpPodTemp)
		in.Spec.InitContainers = tmpPodTemp.Template.Spec.InitContainers
	}

	if err := autoConvert_v1_PodTemplateSpec_To_api_PodTemplateSpec(in, out, s); err != nil {
		return err
	}
	if len(out.Annotations) > 0 {
		old := out.Annotations
		out.Annotations = make(map[string]string, len(old))
		for k, v := range old {
			out.Annotations[k] = v
		}
		delete(out.Annotations, PodInitContainersAnnotationKey)
		delete(out.Annotations, PodInitContainersBetaAnnotationKey)
	}
	return nil
}

// The following two PodSpec conversions are done here to support ServiceAccount
// as an alias for ServiceAccountName.
func Convert_api_PodSpec_To_v1_PodSpec(in *api.PodSpec, out *PodSpec, s conversion.Scope) error {
	if err := autoConvert_api_PodSpec_To_v1_PodSpec(in, out, s); err != nil {
		return err
	}

	// DeprecatedServiceAccount is an alias for ServiceAccountName.
	out.DeprecatedServiceAccount = in.ServiceAccountName

	if in.SecurityContext != nil {
		// the host namespace fields have to be handled here for backward compatibility
		// with v1.0.0
		out.HostPID = in.SecurityContext.HostPID
		out.HostNetwork = in.SecurityContext.HostNetwork
		out.HostIPC = in.SecurityContext.HostIPC
	}

	return nil
}

func Convert_v1_PodSpec_To_api_PodSpec(in *PodSpec, out *api.PodSpec, s conversion.Scope) error {
	if err := autoConvert_v1_PodSpec_To_api_PodSpec(in, out, s); err != nil {
		return err
	}

	// We support DeprecatedServiceAccount as an alias for ServiceAccountName.
	// If both are specified, ServiceAccountName (the new field) wins.
	if in.ServiceAccountName == "" {
		out.ServiceAccountName = in.DeprecatedServiceAccount
	}

	// the host namespace fields have to be handled specially for backward compatibility
	// with v1.0.0
	if out.SecurityContext == nil {
		out.SecurityContext = new(api.PodSecurityContext)
	}
	out.SecurityContext.HostNetwork = in.HostNetwork
	out.SecurityContext.HostPID = in.HostPID
	out.SecurityContext.HostIPC = in.HostIPC

	return nil
}

func Convert_api_Pod_To_v1_Pod(in *api.Pod, out *Pod, s conversion.Scope) error {
	if err := autoConvert_api_Pod_To_v1_Pod(in, out, s); err != nil {
		return err
	}

	// TODO: sometime after we move init container to stable, remove these conversions
	if len(out.Spec.InitContainers) > 0 || len(out.Status.InitContainerStatuses) > 0 {
		old := out.Annotations
		out.Annotations = make(map[string]string, len(old))
		for k, v := range old {
			out.Annotations[k] = v
		}
		delete(out.Annotations, PodInitContainersAnnotationKey)
		delete(out.Annotations, PodInitContainersBetaAnnotationKey)
		delete(out.Annotations, PodInitContainerStatusesAnnotationKey)
		delete(out.Annotations, PodInitContainerStatusesBetaAnnotationKey)
	}
	if len(out.Spec.InitContainers) > 0 {
		value, err := json.Marshal(out.Spec.InitContainers)
		if err != nil {
			return err
		}
		out.Annotations[PodInitContainersAnnotationKey] = string(value)
		out.Annotations[PodInitContainersBetaAnnotationKey] = string(value)
	}
	if len(out.Status.InitContainerStatuses) > 0 {
		value, err := json.Marshal(out.Status.InitContainerStatuses)
		if err != nil {
			return err
		}
		out.Annotations[PodInitContainerStatusesAnnotationKey] = string(value)
		out.Annotations[PodInitContainerStatusesBetaAnnotationKey] = string(value)
	}

	return nil
}

func Convert_v1_Pod_To_api_Pod(in *Pod, out *api.Pod, s conversion.Scope) error {
	// If there is a beta annotation, copy to alpha key.
	// See commit log for PR #31026 for why we do this.
	if valueBeta, okBeta := in.Annotations[PodInitContainersBetaAnnotationKey]; okBeta {
		in.Annotations[PodInitContainersAnnotationKey] = valueBeta
	}
	// TODO: sometime after we move init container to stable, remove these conversions
	// Move the annotation to the internal repr. field
	if value, ok := in.Annotations[PodInitContainersAnnotationKey]; ok {
		var values []Container
		if err := json.Unmarshal([]byte(value), &values); err != nil {
			return err
		}
		// Conversion from external to internal version exists more to
		// satisfy the needs of the decoder than it does to be a general
		// purpose tool. And Decode always creates an intermediate object
		// to decode to. Thus the caller of UnsafeConvertToVersion is
		// taking responsibility to ensure mutation of in is not exposed
		// back to the caller.
		in.Spec.InitContainers = values
		// Call defaulters explicitly until annotations are removed
		tmpPod := &Pod{
			Spec: PodSpec{
				HostNetwork:    in.Spec.HostNetwork,
				InitContainers: values,
			},
		}
		SetObjectDefaults_Pod(tmpPod)
		in.Spec.InitContainers = tmpPod.Spec.InitContainers
	}
	// If there is a beta annotation, copy to alpha key.
	// See commit log for PR #31026 for why we do this.
	if valueBeta, okBeta := in.Annotations[PodInitContainerStatusesBetaAnnotationKey]; okBeta {
		in.Annotations[PodInitContainerStatusesAnnotationKey] = valueBeta
	}
	if value, ok := in.Annotations[PodInitContainerStatusesAnnotationKey]; ok {
		var values []ContainerStatus
		if err := json.Unmarshal([]byte(value), &values); err != nil {
			return err
		}
		// Conversion from external to internal version exists more to
		// satisfy the needs of the decoder than it does to be a general
		// purpose tool. And Decode always creates an intermediate object
		// to decode to. Thus the caller of UnsafeConvertToVersion is
		// taking responsibility to ensure mutation of in is not exposed
		// back to the caller.
		in.Status.InitContainerStatuses = values
	}

	if err := autoConvert_v1_Pod_To_api_Pod(in, out, s); err != nil {
		return err
	}
	if len(out.Annotations) > 0 {
		old := out.Annotations
		out.Annotations = make(map[string]string, len(old))
		for k, v := range old {
			out.Annotations[k] = v
		}
		delete(out.Annotations, PodInitContainersAnnotationKey)
		delete(out.Annotations, PodInitContainersBetaAnnotationKey)
		delete(out.Annotations, PodInitContainerStatusesAnnotationKey)
		delete(out.Annotations, PodInitContainerStatusesBetaAnnotationKey)
	}
	return nil
}

func Convert_v1_Secret_To_api_Secret(in *Secret, out *api.Secret, s conversion.Scope) error {
	if err := autoConvert_v1_Secret_To_api_Secret(in, out, s); err != nil {
		return err
	}

	// StringData overwrites Data
	if len(in.StringData) > 0 {
		if out.Data == nil {
			out.Data = map[string][]byte{}
		}
		for k, v := range in.StringData {
			out.Data[k] = []byte(v)
		}
	}

	return nil
}

func Convert_api_PodSecurityContext_To_v1_PodSecurityContext(in *api.PodSecurityContext, out *PodSecurityContext, s conversion.Scope) error {
	out.SupplementalGroups = in.SupplementalGroups
	if in.SELinuxOptions != nil {
		out.SELinuxOptions = new(SELinuxOptions)
		if err := Convert_api_SELinuxOptions_To_v1_SELinuxOptions(in.SELinuxOptions, out.SELinuxOptions, s); err != nil {
			return err
		}
	} else {
		out.SELinuxOptions = nil
	}
	out.RunAsUser = in.RunAsUser
	out.RunAsNonRoot = in.RunAsNonRoot
	out.FSGroup = in.FSGroup
	return nil
}

func Convert_v1_PodSecurityContext_To_api_PodSecurityContext(in *PodSecurityContext, out *api.PodSecurityContext, s conversion.Scope) error {
	out.SupplementalGroups = in.SupplementalGroups
	if in.SELinuxOptions != nil {
		out.SELinuxOptions = new(api.SELinuxOptions)
		if err := Convert_v1_SELinuxOptions_To_api_SELinuxOptions(in.SELinuxOptions, out.SELinuxOptions, s); err != nil {
			return err
		}
	} else {
		out.SELinuxOptions = nil
	}
	out.RunAsUser = in.RunAsUser
	out.RunAsNonRoot = in.RunAsNonRoot
	out.FSGroup = in.FSGroup
	return nil
}

// +k8s:conversion-fn=copy-only
func Convert_v1_ResourceList_To_api_ResourceList(in *ResourceList, out *api.ResourceList, s conversion.Scope) error {
	if *in == nil {
		return nil
	}
	if *out == nil {
		*out = make(api.ResourceList, len(*in))
	}
	for key, val := range *in {
		// Moved to defaults
		// TODO(#18538): We round up resource values to milli scale to maintain API compatibility.
		// In the future, we should instead reject values that need rounding.
		// const milliScale = -3
		// val.RoundUp(milliScale)

		(*out)[api.ResourceName(key)] = val
	}
	return nil
}

func AddFieldLabelConversionsForEvent(scheme *runtime.Scheme) error {
	return scheme.AddFieldLabelConversionFunc("v1", "Event",
		func(label, value string) (string, string, error) {
			switch label {
			case "involvedObject.kind",
				"involvedObject.namespace",
				"involvedObject.name",
				"involvedObject.uid",
				"involvedObject.apiVersion",
				"involvedObject.resourceVersion",
				"involvedObject.fieldPath",
				"reason",
				"source",
				"type",
				"metadata.namespace",
				"metadata.name":
				return label, value, nil
			default:
				return "", "", fmt.Errorf("field label not supported: %s", label)
			}
		})
}

func AddFieldLabelConversionsForNamespace(scheme *runtime.Scheme) error {
	return scheme.AddFieldLabelConversionFunc("v1", "Namespace",
		func(label, value string) (string, string, error) {
			switch label {
			case "status.phase",
				"metadata.name":
				return label, value, nil
			default:
				return "", "", fmt.Errorf("field label not supported: %s", label)
			}
		})
}

func AddFieldLabelConversionsForSecret(scheme *runtime.Scheme) error {
	return scheme.AddFieldLabelConversionFunc("v1", "Secret",
		func(label, value string) (string, string, error) {
			switch label {
			case "type",
				"metadata.namespace",
				"metadata.name":
				return label, value, nil
			default:
				return "", "", fmt.Errorf("field label not supported: %s", label)
			}
		})
}
