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
	"fmt"
	"reflect"

	"k8s.io/utils/ptr"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/conversion"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kubernetes/pkg/apis/apps"
	"k8s.io/kubernetes/pkg/apis/core"
)

func addConversionFuncs(scheme *runtime.Scheme) error {
	// Add field conversion funcs.
	err := scheme.AddFieldLabelConversionFunc(SchemeGroupVersion.WithKind("Pod"),
		func(label, value string) (string, string, error) {
			switch label {
			case "metadata.name",
				"metadata.namespace",
				"spec.nodeName",
				"spec.restartPolicy",
				"spec.schedulerName",
				"spec.serviceAccountName",
				"spec.hostNetwork",
				"status.phase",
				"status.podIP",
				"status.podIPs",
				"status.nominatedNodeName":
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
	err = scheme.AddFieldLabelConversionFunc(SchemeGroupVersion.WithKind("Node"),
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
	err = scheme.AddFieldLabelConversionFunc(SchemeGroupVersion.WithKind("ReplicationController"),
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
	if err := AddFieldLabelConversionsForEvent(scheme); err != nil {
		return err
	}
	if err := AddFieldLabelConversionsForNamespace(scheme); err != nil {
		return err
	}
	if err := AddFieldLabelConversionsForSecret(scheme); err != nil {
		return err
	}
	if err := AddFieldLabelConversionsForService(scheme); err != nil {
		return err
	}
	return nil
}

func Convert_v1_ReplicationController_To_apps_ReplicaSet(in *v1.ReplicationController, out *apps.ReplicaSet, s conversion.Scope) error {
	out.ObjectMeta = in.ObjectMeta
	if err := Convert_v1_ReplicationControllerSpec_To_apps_ReplicaSetSpec(&in.Spec, &out.Spec, s); err != nil {
		return err
	}
	if err := Convert_v1_ReplicationControllerStatus_To_apps_ReplicaSetStatus(&in.Status, &out.Status, s); err != nil {
		return err
	}
	return nil
}

func Convert_v1_ReplicationControllerSpec_To_apps_ReplicaSetSpec(in *v1.ReplicationControllerSpec, out *apps.ReplicaSetSpec, s conversion.Scope) error {
	out.Replicas = *in.Replicas
	out.MinReadySeconds = in.MinReadySeconds
	if in.Selector != nil {
		out.Selector = new(metav1.LabelSelector)
		metav1.Convert_Map_string_To_string_To_v1_LabelSelector(&in.Selector, out.Selector, s)
	}
	if in.Template != nil {
		if err := Convert_v1_PodTemplateSpec_To_core_PodTemplateSpec(in.Template, &out.Template, s); err != nil {
			return err
		}
	}
	return nil
}

func Convert_v1_ReplicationControllerStatus_To_apps_ReplicaSetStatus(in *v1.ReplicationControllerStatus, out *apps.ReplicaSetStatus, s conversion.Scope) error {
	out.Replicas = in.Replicas
	out.FullyLabeledReplicas = in.FullyLabeledReplicas
	out.ReadyReplicas = in.ReadyReplicas
	out.AvailableReplicas = in.AvailableReplicas
	out.ObservedGeneration = in.ObservedGeneration
	for _, cond := range in.Conditions {
		out.Conditions = append(out.Conditions, apps.ReplicaSetCondition{
			Type:               apps.ReplicaSetConditionType(cond.Type),
			Status:             core.ConditionStatus(cond.Status),
			LastTransitionTime: cond.LastTransitionTime,
			Reason:             cond.Reason,
			Message:            cond.Message,
		})
	}
	return nil
}

func Convert_apps_ReplicaSet_To_v1_ReplicationController(in *apps.ReplicaSet, out *v1.ReplicationController, s conversion.Scope) error {
	out.ObjectMeta = in.ObjectMeta
	if err := Convert_apps_ReplicaSetSpec_To_v1_ReplicationControllerSpec(&in.Spec, &out.Spec, s); err != nil {
		fieldErr, ok := err.(*field.Error)
		if !ok {
			return err
		}
		if out.Annotations == nil {
			out.Annotations = make(map[string]string)
		}
		out.Annotations[v1.NonConvertibleAnnotationPrefix+"/"+fieldErr.Field] = reflect.ValueOf(fieldErr.BadValue).String()
	}
	if err := Convert_apps_ReplicaSetStatus_To_v1_ReplicationControllerStatus(&in.Status, &out.Status, s); err != nil {
		return err
	}
	return nil
}

func Convert_apps_ReplicaSetSpec_To_v1_ReplicationControllerSpec(in *apps.ReplicaSetSpec, out *v1.ReplicationControllerSpec, s conversion.Scope) error {
	out.Replicas = new(int32)
	*out.Replicas = in.Replicas
	out.MinReadySeconds = in.MinReadySeconds
	var invalidErr error
	if in.Selector != nil {
		invalidErr = metav1.Convert_v1_LabelSelector_To_Map_string_To_string(in.Selector, &out.Selector, s)
	}
	out.Template = new(v1.PodTemplateSpec)
	if err := Convert_core_PodTemplateSpec_To_v1_PodTemplateSpec(&in.Template, out.Template, s); err != nil {
		return err
	}
	return invalidErr
}

func Convert_apps_ReplicaSetStatus_To_v1_ReplicationControllerStatus(in *apps.ReplicaSetStatus, out *v1.ReplicationControllerStatus, s conversion.Scope) error {
	out.Replicas = in.Replicas
	out.FullyLabeledReplicas = in.FullyLabeledReplicas
	out.ReadyReplicas = in.ReadyReplicas
	out.AvailableReplicas = in.AvailableReplicas
	out.ObservedGeneration = in.ObservedGeneration
	for _, cond := range in.Conditions {
		out.Conditions = append(out.Conditions, v1.ReplicationControllerCondition{
			Type:               v1.ReplicationControllerConditionType(cond.Type),
			Status:             v1.ConditionStatus(cond.Status),
			LastTransitionTime: cond.LastTransitionTime,
			Reason:             cond.Reason,
			Message:            cond.Message,
		})
	}
	return nil
}

func Convert_core_ReplicationControllerSpec_To_v1_ReplicationControllerSpec(in *core.ReplicationControllerSpec, out *v1.ReplicationControllerSpec, s conversion.Scope) error {
	if err := autoConvert_core_ReplicationControllerSpec_To_v1_ReplicationControllerSpec(in, out, s); err != nil {
		return err
	}
	out.MinReadySeconds = in.MinReadySeconds
	out.Selector = in.Selector
	if in.Template != nil {
		out.Template = new(v1.PodTemplateSpec)
		if err := Convert_core_PodTemplateSpec_To_v1_PodTemplateSpec(in.Template, out.Template, s); err != nil {
			return err
		}
	} else {
		out.Template = nil
	}
	return nil
}

func Convert_v1_ReplicationControllerSpec_To_core_ReplicationControllerSpec(in *v1.ReplicationControllerSpec, out *core.ReplicationControllerSpec, s conversion.Scope) error {
	if err := autoConvert_v1_ReplicationControllerSpec_To_core_ReplicationControllerSpec(in, out, s); err != nil {
		return err
	}
	out.MinReadySeconds = in.MinReadySeconds
	out.Selector = in.Selector
	if in.Template != nil {
		out.Template = new(core.PodTemplateSpec)
		if err := Convert_v1_PodTemplateSpec_To_core_PodTemplateSpec(in.Template, out.Template, s); err != nil {
			return err
		}
	} else {
		out.Template = nil
	}
	return nil
}

func Convert_core_PodTemplateSpec_To_v1_PodTemplateSpec(in *core.PodTemplateSpec, out *v1.PodTemplateSpec, s conversion.Scope) error {
	if err := autoConvert_core_PodTemplateSpec_To_v1_PodTemplateSpec(in, out, s); err != nil {
		return err
	}

	// drop init container annotations so they don't take effect on legacy kubelets.
	// remove this once the oldest supported kubelet no longer honors the annotations over the field.
	out.Annotations = dropInitContainerAnnotations(out.Annotations)

	return nil
}

func Convert_v1_PodTemplateSpec_To_core_PodTemplateSpec(in *v1.PodTemplateSpec, out *core.PodTemplateSpec, s conversion.Scope) error {
	if err := autoConvert_v1_PodTemplateSpec_To_core_PodTemplateSpec(in, out, s); err != nil {
		return err
	}

	// drop init container annotations so they don't show up as differences when receiving requests from old clients
	out.Annotations = dropInitContainerAnnotations(out.Annotations)

	return nil
}

func Convert_v1_PodStatus_To_core_PodStatus(in *v1.PodStatus, out *core.PodStatus, s conversion.Scope) error {
	if err := autoConvert_v1_PodStatus_To_core_PodStatus(in, out, s); err != nil {
		return err
	}

	// If both fields (v1.PodIPs and v1.PodIP) are provided and differ, then PodIP is authoritative for compatibility with older kubelets
	if (len(in.PodIP) > 0 && len(in.PodIPs) > 0) && (in.PodIP != in.PodIPs[0].IP) {
		out.PodIPs = []core.PodIP{
			{
				IP: in.PodIP,
			},
		}
	}
	// at the this point, autoConvert copied v1.PodIPs -> core.PodIPs
	// if v1.PodIPs was empty but v1.PodIP is not, then set core.PodIPs[0] with v1.PodIP
	if len(in.PodIP) > 0 && len(in.PodIPs) == 0 {
		out.PodIPs = []core.PodIP{
			{
				IP: in.PodIP,
			},
		}
	}
	return nil
}

func Convert_core_PodStatus_To_v1_PodStatus(in *core.PodStatus, out *v1.PodStatus, s conversion.Scope) error {
	if err := autoConvert_core_PodStatus_To_v1_PodStatus(in, out, s); err != nil {
		return err
	}
	// at the this point autoConvert copied core.PodIPs -> v1.PodIPs
	//  v1.PodIP (singular value field, which does not exist in core) needs to
	// be set with core.PodIPs[0]
	if len(in.PodIPs) > 0 {
		out.PodIP = in.PodIPs[0].IP
	}
	return nil
}

// The following two v1.PodSpec conversions are done here to support v1.ServiceAccount
// as an alias for ServiceAccountName.
func Convert_core_PodSpec_To_v1_PodSpec(in *core.PodSpec, out *v1.PodSpec, s conversion.Scope) error {
	if err := autoConvert_core_PodSpec_To_v1_PodSpec(in, out, s); err != nil {
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
		out.ShareProcessNamespace = in.SecurityContext.ShareProcessNamespace
		out.HostUsers = in.SecurityContext.HostUsers
	}

	return nil
}

func Convert_core_NodeSpec_To_v1_NodeSpec(in *core.NodeSpec, out *v1.NodeSpec, s conversion.Scope) error {
	if err := autoConvert_core_NodeSpec_To_v1_NodeSpec(in, out, s); err != nil {
		return err
	}
	// at the this point autoConvert copied core.PodCIDRs -> v1.PodCIDRs
	// v1.PodCIDR (singular value field, which does not exist in core) needs to
	// be set with core.PodCIDRs[0]
	if len(in.PodCIDRs) > 0 {
		out.PodCIDR = in.PodCIDRs[0]
	}
	return nil
}

func Convert_v1_NodeSpec_To_core_NodeSpec(in *v1.NodeSpec, out *core.NodeSpec, s conversion.Scope) error {
	if err := autoConvert_v1_NodeSpec_To_core_NodeSpec(in, out, s); err != nil {
		return err
	}
	// If both fields (v1.PodCIDRs and v1.PodCIDR) are provided and differ, then PodCIDR is authoritative for compatibility with older clients
	if (len(in.PodCIDR) > 0 && len(in.PodCIDRs) > 0) && (in.PodCIDR != in.PodCIDRs[0]) {
		out.PodCIDRs = []string{in.PodCIDR}
	}

	// at the this point, autoConvert copied v1.PodCIDRs -> core.PodCIDRs
	// if v1.PodCIDRs was empty but v1.PodCIDR is not, then set core.PodCIDRs[0] with v1.PodCIDR
	if len(in.PodCIDR) > 0 && len(in.PodCIDRs) == 0 {
		out.PodCIDRs = []string{in.PodCIDR}
	}
	return nil
}

func Convert_v1_PodSpec_To_core_PodSpec(in *v1.PodSpec, out *core.PodSpec, s conversion.Scope) error {
	if err := autoConvert_v1_PodSpec_To_core_PodSpec(in, out, s); err != nil {
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
		out.SecurityContext = new(core.PodSecurityContext)
	}
	out.SecurityContext.HostNetwork = in.HostNetwork
	out.SecurityContext.HostPID = in.HostPID
	out.SecurityContext.HostIPC = in.HostIPC
	out.SecurityContext.ShareProcessNamespace = in.ShareProcessNamespace
	out.SecurityContext.HostUsers = in.HostUsers

	return nil
}

func Convert_v1_Pod_To_core_Pod(in *v1.Pod, out *core.Pod, s conversion.Scope) error {
	if err := autoConvert_v1_Pod_To_core_Pod(in, out, s); err != nil {
		return err
	}

	// drop init container annotations so they don't show up as differences when receiving requests from old clients
	out.Annotations = dropInitContainerAnnotations(out.Annotations)

	// Forcing the value of TerminationGracePeriodSeconds to 1 if it is negative.
	// Just for Pod, not for PodSpec, because we don't want to change the behavior of the PodTemplate.
	if in.Spec.TerminationGracePeriodSeconds != nil && *in.Spec.TerminationGracePeriodSeconds < 0 {
		out.Spec.TerminationGracePeriodSeconds = ptr.To[int64](1)
	}
	return nil
}

func Convert_core_Pod_To_v1_Pod(in *core.Pod, out *v1.Pod, s conversion.Scope) error {
	if err := autoConvert_core_Pod_To_v1_Pod(in, out, s); err != nil {
		return err
	}

	// drop init container annotations so they don't take effect on legacy kubelets.
	// remove this once the oldest supported kubelet no longer honors the annotations over the field.
	out.Annotations = dropInitContainerAnnotations(out.Annotations)

	// Forcing the value of TerminationGracePeriodSeconds to 1 if it is negative.
	// Just for Pod, not for PodSpec, because we don't want to change the behavior of the PodTemplate.
	if in.Spec.TerminationGracePeriodSeconds != nil && *in.Spec.TerminationGracePeriodSeconds < 0 {
		out.Spec.TerminationGracePeriodSeconds = ptr.To[int64](1)
	}
	return nil
}

func Convert_v1_Secret_To_core_Secret(in *v1.Secret, out *core.Secret, s conversion.Scope) error {
	if err := autoConvert_v1_Secret_To_core_Secret(in, out, s); err != nil {
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

// +k8s:conversion-fn=copy-only
func Convert_v1_ResourceList_To_core_ResourceList(in *v1.ResourceList, out *core.ResourceList, s conversion.Scope) error {
	if *in == nil {
		return nil
	}
	if *out == nil {
		*out = make(core.ResourceList, len(*in))
	}
	for key, val := range *in {
		// Moved to defaults
		// TODO(#18538): We round up resource values to milli scale to maintain API compatibility.
		// In the future, we should instead reject values that need rounding.
		// const milliScale = -3
		// val.RoundUp(milliScale)

		(*out)[core.ResourceName(key)] = val
	}
	return nil
}

func AddFieldLabelConversionsForEvent(scheme *runtime.Scheme) error {
	return scheme.AddFieldLabelConversionFunc(SchemeGroupVersion.WithKind("Event"),
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
				"reportingComponent",
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
	return scheme.AddFieldLabelConversionFunc(SchemeGroupVersion.WithKind("Namespace"),
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
	return scheme.AddFieldLabelConversionFunc(SchemeGroupVersion.WithKind("Secret"),
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

func AddFieldLabelConversionsForService(scheme *runtime.Scheme) error {
	return scheme.AddFieldLabelConversionFunc(SchemeGroupVersion.WithKind("Service"),
		func(label, value string) (string, string, error) {
			switch label {
			case "metadata.namespace",
				"metadata.name",
				"spec.clusterIP",
				"spec.type":
				return label, value, nil
			default:
				return "", "", fmt.Errorf("field label not supported: %s", label)
			}
		})
}

var initContainerAnnotations = map[string]bool{
	"pod.beta.kubernetes.io/init-containers":          true,
	"pod.alpha.kubernetes.io/init-containers":         true,
	"pod.beta.kubernetes.io/init-container-statuses":  true,
	"pod.alpha.kubernetes.io/init-container-statuses": true,
}

// dropInitContainerAnnotations returns a copy of the annotations with init container annotations removed,
// or the original annotations if no init container annotations were present.
//
// this can be removed once no clients prior to 1.8 are supported, and no kubelets prior to 1.8 can be run
// (we don't support kubelets older than 2 versions skewed from the apiserver, but we don't prevent them, either)
func dropInitContainerAnnotations(oldAnnotations map[string]string) map[string]string {
	if len(oldAnnotations) == 0 {
		return oldAnnotations
	}

	found := false
	for k := range initContainerAnnotations {
		if _, ok := oldAnnotations[k]; ok {
			found = true
			break
		}
	}
	if !found {
		return oldAnnotations
	}

	newAnnotations := make(map[string]string, len(oldAnnotations))
	for k, v := range oldAnnotations {
		if !initContainerAnnotations[k] {
			newAnnotations[k] = v
		}
	}
	return newAnnotations
}

// Convert_core_PersistentVolumeSpec_To_v1_PersistentVolumeSpec is defined outside the autogenerated file for use by other API packages
// This is needed because it is referenced from other APIs, but is invisible at code-generation time because of the build tags.
func Convert_core_PersistentVolumeSpec_To_v1_PersistentVolumeSpec(in *core.PersistentVolumeSpec, out *v1.PersistentVolumeSpec, s conversion.Scope) error {
	return autoConvert_core_PersistentVolumeSpec_To_v1_PersistentVolumeSpec(in, out, s)
}

// Convert_v1_PersistentVolumeSpec_To_core_PersistentVolumeSpec is defined outside the autogenerated file for use by other API packages
// This is needed because it is referenced from other APIs, but is invisible at code-generation time because of the build tags.
func Convert_v1_PersistentVolumeSpec_To_core_PersistentVolumeSpec(in *v1.PersistentVolumeSpec, out *core.PersistentVolumeSpec, s conversion.Scope) error {
	return autoConvert_v1_PersistentVolumeSpec_To_core_PersistentVolumeSpec(in, out, s)
}

// Convert_Slice_string_To_Pointer_string is needed because decoding URL parameters requires manual assistance.
func Convert_Slice_string_To_Pointer_string(in *[]string, out **string, s conversion.Scope) error {
	if len(*in) == 0 {
		return nil
	}
	temp := (*in)[0]
	*out = &temp
	return nil
}
