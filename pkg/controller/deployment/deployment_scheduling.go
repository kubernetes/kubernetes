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

package deployment

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	apps "k8s.io/api/apps/v1"
	schedulingv1beta1 "k8s.io/api/scheduling/v1beta1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/strategicpatch"
	"k8s.io/apimachinery/pkg/util/validation"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	workloadbuilder "k8s.io/component-helpers/scheduling/schedulingv1/workloadbuilder"
	"k8s.io/klog/v2"
	deploymentutil "k8s.io/kubernetes/pkg/api/deployment"
	"k8s.io/kubernetes/pkg/features"
)

// resolveMinCount returns the effective gang minCount for the Workload
// template. If the user set minCount explicitly, that value is always used.
// Otherwise minCount follows positive replicas and defaults to 1 when replicas
// is unset or zero.
func resolveMinCount(d *apps.Deployment) int32 {
	if s := d.Spec.Scheduling; s != nil &&
		s.SchedulingPolicy != nil &&
		s.SchedulingPolicy.Gang != nil &&
		s.SchedulingPolicy.Gang.MinCount != nil {
		return *s.SchedulingPolicy.Gang.MinCount
	}
	if d.Spec.Replicas != nil && *d.Spec.Replicas > 0 {
		return *d.Spec.Replicas
	}
	return 1
}

func shouldManageWorkloadForDeployment(d *apps.Deployment) bool {
	return utilfeature.DefaultFeatureGate.Enabled(features.WorkloadWithDeployment) && d.Spec.Scheduling != nil
}

// podGroupTemplateName is a DNS label (63 bytes), tighter than the Workload name.
func podGroupTemplateName(d *apps.Deployment) string {
	const suffix = "-pgt-0"
	prefix := d.Name
	if max := validation.DNS1123LabelMaxLength - len(suffix); len(prefix) > max {
		prefix = prefix[:max]
	}
	prefix = strings.TrimRight(prefix, ".-")
	return prefix + suffix
}

// buildWorkloadForDeployment compiles the Deployment's scheduling configuration into a Workload.
// For Gang scheduling, it stores the resolved minCount in the template so each ReplicaSet's PodGroup inherits it.
func buildWorkloadForDeployment(d *apps.Deployment) (*schedulingv1beta1.Workload, error) {
	minCount := resolveMinCount(d)
	input := deploymentutil.WorkloadInputForDeploymentV1(d.Spec.Scheduling)

	wlName := workloadbuilder.GenerateWorkloadName(d.Name, d.UID)
	item := deploymentutil.WorkloadItemForDeployment(
		podGroupTemplateName(d), d.Spec.Template.Spec.PriorityClassName, input, nil,
	)
	item.Callbacks = []workloadbuilder.SchedulingConfigFunc{
		func(cfg *workloadbuilder.SchedulingConfig) {
			if cfg != nil && cfg.Policy != nil && cfg.Policy.Gang != nil {
				cfg.Policy.Gang.MinCount = &minCount
			}
		},
	}

	builder := workloadbuilder.NewBuilder(item, workloadbuilder.BuildOptions{
		Name:      wlName,
		Namespace: d.Namespace,
		Owner:     metav1.NewControllerRef(d, controllerKind),
	})

	return builder.BuildWorkload()
}

// ensureWorkloadForDeployment reconciles the Workload owned by a Deployment.
// Its template minCount is kept current so new PodGroups inherit the right value.
func (dc *DeploymentController) ensureWorkloadForDeployment(ctx context.Context, d *apps.Deployment) (*schedulingv1beta1.Workload, error) {
	logger := klog.FromContext(ctx)
	wlName := workloadbuilder.GenerateWorkloadName(d.Name, d.UID)
	minCount := resolveMinCount(d)

	if dc.workloadLister == nil {
		return nil, fmt.Errorf("workload informer is not configured")
	}
	existing, err := dc.workloadLister.Workloads(d.Namespace).Get(wlName)
	if err == nil {
		if !isWorkloadControlledByDeployment(existing, d) {
			return nil, fmt.Errorf("workload %s/%s is not controlled by Deployment %s/%s with UID %q",
				existing.Namespace, existing.Name, d.Namespace, d.Name, d.UID)
		}
		if len(existing.Spec.PodGroupTemplates) != 1 {
			return nil, fmt.Errorf("workload %s/%s for Deployment %s/%s has %d PodGroupTemplates, want exactly 1",
				existing.Namespace, existing.Name, d.Namespace, d.Name, len(existing.Spec.PodGroupTemplates))
		}
		if existing.Spec.PodGroupTemplates[0].SchedulingPolicy.Gang != nil &&
			existing.Spec.PodGroupTemplates[0].SchedulingPolicy.Gang.MinCount != minCount {
			logger.V(2).Info("Workload template minCount differs, patching",
				"workload", klog.KObj(existing),
				"currentMinCount", existing.Spec.PodGroupTemplates[0].SchedulingPolicy.Gang.MinCount,
				"desiredMinCount", minCount)
			return dc.patchWorkloadTemplateMinCount(ctx, existing, minCount)
		}
		return existing, nil
	}
	if !errors.IsNotFound(err) {
		return nil, fmt.Errorf("failed to get Workload %s/%s: %w",
			d.Namespace, wlName, err)
	}

	desired, err := buildWorkloadForDeployment(d)
	if err != nil {
		return nil, err
	}

	created, err := dc.client.SchedulingV1beta1().Workloads(d.Namespace).Create(ctx, desired, metav1.CreateOptions{})
	if err == nil {
		logger.V(2).Info("Created Workload for Deployment",
			"workload", klog.KObj(created),
			"deployment", klog.KObj(d))
		return created, nil
	}
	if errors.IsAlreadyExists(err) {
		return nil, err
	}
	return nil, fmt.Errorf("failed to create Workload %s/%s: %w",
		d.Namespace, wlName, err)
}

func isWorkloadControlledByDeployment(workload *schedulingv1beta1.Workload, d *apps.Deployment) bool {
	owner := metav1.GetControllerOf(workload)
	if owner == nil ||
		owner.APIVersion != apps.SchemeGroupVersion.String() ||
		owner.Kind != "Deployment" ||
		owner.Name != d.Name ||
		owner.UID != d.UID {
		return false
	}
	controllerRef := workload.Spec.ControllerRef
	return controllerRef != nil &&
		controllerRef.APIGroup == apps.GroupName &&
		controllerRef.Kind == "Deployment" &&
		controllerRef.Name == d.Name
}

// patchWorkloadTemplateMinCount updates the gang minCount on the Workload's
// PodGroupTemplate so new PodGroups stamped from it inherit the correct value.
func (dc *DeploymentController) patchWorkloadTemplateMinCount(ctx context.Context, workload *schedulingv1beta1.Workload, minCount int32) (*schedulingv1beta1.Workload, error) {
	updated := workload.DeepCopy()
	updated.Spec.PodGroupTemplates[0].SchedulingPolicy.Gang.MinCount = minCount

	oldData, err := json.Marshal(workload)
	if err != nil {
		return nil, err
	}
	newData, err := json.Marshal(updated)
	if err != nil {
		return nil, err
	}
	patch, err := strategicpatch.CreateTwoWayMergePatch(oldData, newData, &schedulingv1beta1.Workload{})
	if err != nil {
		return nil, err
	}

	return dc.client.SchedulingV1beta1().Workloads(workload.Namespace).
		Patch(ctx, workload.Name, types.StrategicMergePatchType, patch, metav1.PatchOptions{})
}
