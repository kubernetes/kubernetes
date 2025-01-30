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

package v1

import (
	"fmt"
	"strconv"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/conversion"
	"k8s.io/kubernetes/pkg/apis/apps"
	"k8s.io/kubernetes/pkg/apis/core"
)

// Convert_apps_DeploymentSpec_To_v1_DeploymentSpec is defined here, because public
// conversion is not auto-generated due to existing warnings.
func Convert_apps_DeploymentSpec_To_v1_DeploymentSpec(in *apps.DeploymentSpec, out *appsv1.DeploymentSpec, s conversion.Scope) error {
	if err := autoConvert_apps_DeploymentSpec_To_v1_DeploymentSpec(in, out, s); err != nil {
		return err
	}
	return nil
}

func Convert_v1_Deployment_To_apps_Deployment(in *appsv1.Deployment, out *apps.Deployment, s conversion.Scope) error {
	if err := autoConvert_v1_Deployment_To_apps_Deployment(in, out, s); err != nil {
		return err
	}

	// Copy annotation to deprecated rollbackTo field for roundtrip
	// TODO: remove this conversion after we delete extensions/v1beta1 and apps/v1beta1 Deployment
	if revision := in.Annotations[appsv1.DeprecatedRollbackTo]; revision != "" {
		if revision64, err := strconv.ParseInt(revision, 10, 64); err != nil {
			return fmt.Errorf("failed to parse annotation[%s]=%s as int64: %v", appsv1.DeprecatedRollbackTo, revision, err)
		} else {
			out.Spec.RollbackTo = new(apps.RollbackConfig)
			out.Spec.RollbackTo.Revision = revision64
		}
		out.Annotations = deepCopyStringMap(out.Annotations)
		delete(out.Annotations, appsv1.DeprecatedRollbackTo)
	} else {
		out.Spec.RollbackTo = nil
	}

	return nil
}

func Convert_apps_Deployment_To_v1_Deployment(in *apps.Deployment, out *appsv1.Deployment, s conversion.Scope) error {
	if err := autoConvert_apps_Deployment_To_v1_Deployment(in, out, s); err != nil {
		return err
	}

	out.Annotations = deepCopyStringMap(out.Annotations) // deep copy because we modify it below

	// Copy deprecated rollbackTo field to annotation for roundtrip
	// TODO: remove this conversion after we delete extensions/v1beta1 and apps/v1beta1 Deployment
	if in.Spec.RollbackTo != nil {
		if out.Annotations == nil {
			out.Annotations = make(map[string]string)
		}
		out.Annotations[appsv1.DeprecatedRollbackTo] = strconv.FormatInt(in.Spec.RollbackTo.Revision, 10)
	} else {
		delete(out.Annotations, appsv1.DeprecatedRollbackTo)
	}

	return nil
}

func Convert_apps_DaemonSet_To_v1_DaemonSet(in *apps.DaemonSet, out *appsv1.DaemonSet, s conversion.Scope) error {
	if err := autoConvert_apps_DaemonSet_To_v1_DaemonSet(in, out, s); err != nil {
		return err
	}

	out.Annotations = deepCopyStringMap(out.Annotations) // deep copy annotations because we change them below
	out.Annotations[appsv1.DeprecatedTemplateGeneration] = strconv.FormatInt(in.Spec.TemplateGeneration, 10)
	return nil
}

// Convert_apps_DaemonSetSpec_To_v1_DaemonSetSpec is defined here, because public
// conversion is not auto-generated due to existing warnings.
func Convert_apps_DaemonSetSpec_To_v1_DaemonSetSpec(in *apps.DaemonSetSpec, out *appsv1.DaemonSetSpec, s conversion.Scope) error {
	if err := autoConvert_apps_DaemonSetSpec_To_v1_DaemonSetSpec(in, out, s); err != nil {
		return err
	}
	return nil
}

func Convert_v1_DaemonSet_To_apps_DaemonSet(in *appsv1.DaemonSet, out *apps.DaemonSet, s conversion.Scope) error {
	if err := autoConvert_v1_DaemonSet_To_apps_DaemonSet(in, out, s); err != nil {
		return err
	}
	if value, ok := in.Annotations[appsv1.DeprecatedTemplateGeneration]; ok {
		if value64, err := strconv.ParseInt(value, 10, 64); err != nil {
			return err
		} else {
			out.Spec.TemplateGeneration = value64
			out.Annotations = deepCopyStringMap(out.Annotations)
			delete(out.Annotations, appsv1.DeprecatedTemplateGeneration)
		}
	}
	return nil
}

func deepCopyStringMap(m map[string]string) map[string]string {
	ret := make(map[string]string, len(m))
	for k, v := range m {
		ret[k] = v
	}
	return ret
}

// Convert_apps_StatefulSetSpec_To_v1_StatefulSetSpec augments auto-conversion to preserve < 1.17 behavior
// setting apiVersion/kind in nested persistent volume claim objects.
func Convert_v1_StatefulSetSpec_To_apps_StatefulSetSpec(in *appsv1.StatefulSetSpec, out *apps.StatefulSetSpec, s conversion.Scope) error {
	if err := autoConvert_v1_StatefulSetSpec_To_apps_StatefulSetSpec(in, out, s); err != nil {
		return err
	}
	// set APIVersion/Kind to behave the same as reflective conversion < 1.17.
	// see https://issue.k8s.io/87583
	if out.VolumeClaimTemplates != nil {
		// copy so we don't modify the input
		templatesCopy := make([]core.PersistentVolumeClaim, len(out.VolumeClaimTemplates))
		copy(templatesCopy, out.VolumeClaimTemplates)
		out.VolumeClaimTemplates = templatesCopy
		for i := range out.VolumeClaimTemplates {
			out.VolumeClaimTemplates[i].APIVersion = ""
			out.VolumeClaimTemplates[i].Kind = ""
		}
	}
	return nil
}

// Convert_apps_StatefulSetSpec_To_v1_StatefulSetSpec augments auto-conversion to preserve < 1.17 behavior
// setting apiVersion/kind in nested persistent volume claim objects.
func Convert_apps_StatefulSetSpec_To_v1_StatefulSetSpec(in *apps.StatefulSetSpec, out *appsv1.StatefulSetSpec, s conversion.Scope) error {
	if err := autoConvert_apps_StatefulSetSpec_To_v1_StatefulSetSpec(in, out, s); err != nil {
		return err
	}
	// set APIVersion/Kind to behave the same as reflective conversion < 1.17.
	// see https://issue.k8s.io/87583
	if out.VolumeClaimTemplates != nil {
		// copy so we don't modify the input
		templatesCopy := make([]corev1.PersistentVolumeClaim, len(out.VolumeClaimTemplates))
		copy(templatesCopy, out.VolumeClaimTemplates)
		out.VolumeClaimTemplates = templatesCopy
		for i := range out.VolumeClaimTemplates {
			out.VolumeClaimTemplates[i].APIVersion = "v1"
			out.VolumeClaimTemplates[i].Kind = "PersistentVolumeClaim"
		}
	}
	return nil
}
