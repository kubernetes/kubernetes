/*
Copyright 2016 The Kubernetes Authors.

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

package v1beta2

import (
	"fmt"
	"strconv"

	appsv1beta2 "k8s.io/api/apps/v1beta2"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/conversion"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/pkg/apis/apps"
	autoscaling "k8s.io/kubernetes/pkg/apis/autoscaling"
	k8s_api_v1 "k8s.io/kubernetes/pkg/apis/core/v1"
)

func addConversionFuncs(scheme *runtime.Scheme) error {
	// Add field label conversions for kinds having selectable nothing but ObjectMeta fields.
	if err := scheme.AddFieldLabelConversionFunc(SchemeGroupVersion.WithKind("StatefulSet"),
		func(label, value string) (string, string, error) {
			switch label {
			case "metadata.name", "metadata.namespace", "status.successful":
				return label, value, nil
			default:
				return "", "", fmt.Errorf("field label not supported for appsv1beta2.StatefulSet: %s", label)
			}
		}); err != nil {
		return err
	}

	return nil
}

func Convert_autoscaling_ScaleStatus_To_v1beta2_ScaleStatus(in *autoscaling.ScaleStatus, out *appsv1beta2.ScaleStatus, s conversion.Scope) error {
	out.Replicas = int32(in.Replicas)
	out.TargetSelector = in.Selector

	out.Selector = nil
	selector, err := metav1.ParseToLabelSelector(in.Selector)
	if err != nil {
		return fmt.Errorf("failed to parse selector: %v", err)
	}
	if len(selector.MatchExpressions) == 0 {
		out.Selector = selector.MatchLabels
	}

	return nil
}

func Convert_v1beta2_ScaleStatus_To_autoscaling_ScaleStatus(in *appsv1beta2.ScaleStatus, out *autoscaling.ScaleStatus, s conversion.Scope) error {
	out.Replicas = in.Replicas

	// Normally when 2 fields map to the same internal value we favor the old field, since
	// old clients can't be expected to know about new fields but clients that know about the
	// new field can be expected to know about the old field (though that's not quite true, due
	// to kubectl apply). However, these fields are readonly, so any non-nil value should work.
	if in.TargetSelector != "" {
		out.Selector = in.TargetSelector
	} else if in.Selector != nil {
		set := labels.Set{}
		for key, val := range in.Selector {
			set[key] = val
		}
		out.Selector = labels.SelectorFromSet(set).String()
	} else {
		out.Selector = ""
	}
	return nil
}

func Convert_v1beta2_DeploymentSpec_To_apps_DeploymentSpec(in *appsv1beta2.DeploymentSpec, out *apps.DeploymentSpec, s conversion.Scope) error {
	if in.Replicas != nil {
		out.Replicas = *in.Replicas
	}
	out.Selector = in.Selector
	if err := k8s_api_v1.Convert_v1_PodTemplateSpec_To_core_PodTemplateSpec(&in.Template, &out.Template, s); err != nil {
		return err
	}
	if err := Convert_v1beta2_DeploymentStrategy_To_apps_DeploymentStrategy(&in.Strategy, &out.Strategy, s); err != nil {
		return err
	}
	out.RevisionHistoryLimit = in.RevisionHistoryLimit
	out.MinReadySeconds = in.MinReadySeconds
	out.Paused = in.Paused
	if in.ProgressDeadlineSeconds != nil {
		out.ProgressDeadlineSeconds = new(int32)
		*out.ProgressDeadlineSeconds = *in.ProgressDeadlineSeconds
	}
	return nil
}

func Convert_apps_DeploymentSpec_To_v1beta2_DeploymentSpec(in *apps.DeploymentSpec, out *appsv1beta2.DeploymentSpec, s conversion.Scope) error {
	out.Replicas = &in.Replicas
	out.Selector = in.Selector
	if err := k8s_api_v1.Convert_core_PodTemplateSpec_To_v1_PodTemplateSpec(&in.Template, &out.Template, s); err != nil {
		return err
	}
	if err := Convert_apps_DeploymentStrategy_To_v1beta2_DeploymentStrategy(&in.Strategy, &out.Strategy, s); err != nil {
		return err
	}
	if in.RevisionHistoryLimit != nil {
		out.RevisionHistoryLimit = new(int32)
		*out.RevisionHistoryLimit = int32(*in.RevisionHistoryLimit)
	}
	out.MinReadySeconds = int32(in.MinReadySeconds)
	out.Paused = in.Paused
	if in.ProgressDeadlineSeconds != nil {
		out.ProgressDeadlineSeconds = new(int32)
		*out.ProgressDeadlineSeconds = *in.ProgressDeadlineSeconds
	}
	return nil
}

func Convert_v1beta2_Deployment_To_apps_Deployment(in *appsv1beta2.Deployment, out *apps.Deployment, s conversion.Scope) error {
	out.ObjectMeta = in.ObjectMeta
	if err := Convert_v1beta2_DeploymentSpec_To_apps_DeploymentSpec(&in.Spec, &out.Spec, s); err != nil {
		return err
	}

	// Copy annotation to deprecated rollbackTo field for roundtrip
	// TODO: remove this conversion after we delete extensions/v1beta1 and apps/v1beta1 Deployment
	if revision := in.Annotations[appsv1beta2.DeprecatedRollbackTo]; revision != "" {
		if revision64, err := strconv.ParseInt(revision, 10, 64); err != nil {
			return fmt.Errorf("failed to parse annotation[%s]=%s as int64: %v", appsv1beta2.DeprecatedRollbackTo, revision, err)
		} else {
			out.Spec.RollbackTo = new(apps.RollbackConfig)
			out.Spec.RollbackTo.Revision = revision64
		}
		out.Annotations = deepCopyStringMap(out.Annotations)
		delete(out.Annotations, appsv1beta2.DeprecatedRollbackTo)
	} else {
		out.Spec.RollbackTo = nil
	}

	if err := Convert_v1beta2_DeploymentStatus_To_apps_DeploymentStatus(&in.Status, &out.Status, s); err != nil {
		return err
	}
	return nil
}

func Convert_apps_Deployment_To_v1beta2_Deployment(in *apps.Deployment, out *appsv1beta2.Deployment, s conversion.Scope) error {
	out.ObjectMeta = in.ObjectMeta
	out.Annotations = deepCopyStringMap(out.Annotations) // deep copy because we modify annotations below

	if err := Convert_apps_DeploymentSpec_To_v1beta2_DeploymentSpec(&in.Spec, &out.Spec, s); err != nil {
		return err
	}

	// Copy deprecated rollbackTo field to annotation for roundtrip
	// TODO: remove this conversion after we delete extensions/v1beta1 and apps/v1beta1 Deployment
	if in.Spec.RollbackTo != nil {
		if out.Annotations == nil {
			out.Annotations = make(map[string]string)
		}
		out.Annotations[appsv1beta2.DeprecatedRollbackTo] = strconv.FormatInt(in.Spec.RollbackTo.Revision, 10)
	} else {
		delete(out.Annotations, appsv1beta2.DeprecatedRollbackTo)
	}

	if err := Convert_apps_DeploymentStatus_To_v1beta2_DeploymentStatus(&in.Status, &out.Status, s); err != nil {
		return err
	}
	return nil
}

func Convert_apps_DaemonSet_To_v1beta2_DaemonSet(in *apps.DaemonSet, out *appsv1beta2.DaemonSet, s conversion.Scope) error {
	out.ObjectMeta = in.ObjectMeta
	out.Annotations = deepCopyStringMap(out.Annotations)
	out.Annotations[appsv1beta2.DeprecatedTemplateGeneration] = strconv.FormatInt(in.Spec.TemplateGeneration, 10)
	if err := Convert_apps_DaemonSetSpec_To_v1beta2_DaemonSetSpec(&in.Spec, &out.Spec, s); err != nil {
		return err
	}
	if err := s.Convert(&in.Status, &out.Status, 0); err != nil {
		return err
	}
	return nil
}

func Convert_apps_DaemonSetSpec_To_v1beta2_DaemonSetSpec(in *apps.DaemonSetSpec, out *appsv1beta2.DaemonSetSpec, s conversion.Scope) error {
	out.Selector = in.Selector
	if err := k8s_api_v1.Convert_core_PodTemplateSpec_To_v1_PodTemplateSpec(&in.Template, &out.Template, s); err != nil {
		return err
	}
	if err := Convert_apps_DaemonSetUpdateStrategy_To_v1beta2_DaemonSetUpdateStrategy(&in.UpdateStrategy, &out.UpdateStrategy, s); err != nil {
		return err
	}
	out.MinReadySeconds = int32(in.MinReadySeconds)
	if in.RevisionHistoryLimit != nil {
		out.RevisionHistoryLimit = new(int32)
		*out.RevisionHistoryLimit = *in.RevisionHistoryLimit
	} else {
		out.RevisionHistoryLimit = nil
	}
	return nil
}

func Convert_v1beta2_DaemonSet_To_apps_DaemonSet(in *appsv1beta2.DaemonSet, out *apps.DaemonSet, s conversion.Scope) error {
	out.ObjectMeta = in.ObjectMeta
	if err := Convert_v1beta2_DaemonSetSpec_To_apps_DaemonSetSpec(&in.Spec, &out.Spec, s); err != nil {
		return err
	}
	if value, ok := in.Annotations[appsv1beta2.DeprecatedTemplateGeneration]; ok {
		if value64, err := strconv.ParseInt(value, 10, 64); err != nil {
			return err
		} else {
			out.Spec.TemplateGeneration = value64
			out.Annotations = deepCopyStringMap(out.Annotations)
			delete(out.Annotations, appsv1beta2.DeprecatedTemplateGeneration)
		}
	}
	if err := s.Convert(&in.Status, &out.Status, 0); err != nil {
		return err
	}
	return nil
}

func Convert_v1beta2_DaemonSetSpec_To_apps_DaemonSetSpec(in *appsv1beta2.DaemonSetSpec, out *apps.DaemonSetSpec, s conversion.Scope) error {
	out.Selector = in.Selector
	if err := k8s_api_v1.Convert_v1_PodTemplateSpec_To_core_PodTemplateSpec(&in.Template, &out.Template, s); err != nil {
		return err
	}
	if err := Convert_v1beta2_DaemonSetUpdateStrategy_To_apps_DaemonSetUpdateStrategy(&in.UpdateStrategy, &out.UpdateStrategy, s); err != nil {
		return err
	}
	if in.RevisionHistoryLimit != nil {
		out.RevisionHistoryLimit = new(int32)
		*out.RevisionHistoryLimit = *in.RevisionHistoryLimit
	} else {
		out.RevisionHistoryLimit = nil
	}
	out.MinReadySeconds = in.MinReadySeconds
	return nil
}

func deepCopyStringMap(m map[string]string) map[string]string {
	ret := make(map[string]string, len(m))
	for k, v := range m {
		ret[k] = v
	}
	return ret
}
