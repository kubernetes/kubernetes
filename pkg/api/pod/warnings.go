/*
Copyright 2021 The Kubernetes Authors.

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

package pod

import (
	"context"
	"fmt"
	"strings"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation/field"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/core/pods"
)

func GetWarningsForPod(ctx context.Context, pod, oldPod *api.Pod) []string {
	if pod == nil {
		return nil
	}

	var (
		oldSpec *api.PodSpec
		oldMeta *metav1.ObjectMeta
	)
	if oldPod != nil {
		oldSpec = &oldPod.Spec
		oldMeta = &oldPod.ObjectMeta
	}
	return warningsForPodSpecAndMeta(nil, &pod.Spec, &pod.ObjectMeta, oldSpec, oldMeta)
}

func GetWarningsForPodTemplate(ctx context.Context, fieldPath *field.Path, podTemplate, oldPodTemplate *api.PodTemplateSpec) []string {
	if podTemplate == nil {
		return nil
	}

	var (
		oldSpec *api.PodSpec
		oldMeta *metav1.ObjectMeta
	)
	if oldPodTemplate != nil {
		oldSpec = &oldPodTemplate.Spec
		oldMeta = &oldPodTemplate.ObjectMeta
	}
	return warningsForPodSpecAndMeta(fieldPath, &podTemplate.Spec, &podTemplate.ObjectMeta, oldSpec, oldMeta)
}

var deprecatedNodeLabels = map[string]string{
	`beta.kubernetes.io/arch`:                  `deprecated since v1.14; use "kubernetes.io/arch" instead`,
	`beta.kubernetes.io/os`:                    `deprecated since v1.14; use "kubernetes.io/os" instead`,
	`failure-domain.beta.kubernetes.io/region`: `deprecated since v1.17; use "topology.kubernetes.io/region" instead`,
	`failure-domain.beta.kubernetes.io/zone`:   `deprecated since v1.17; use "topology.kubernetes.io/zone" instead`,
	`beta.kubernetes.io/instance-type`:         `deprecated since v1.17; use "node.kubernetes.io/instance-type" instead`,
}

var deprecatedAnnotations = []struct {
	key     string
	prefix  string
	message string
}{
	{
		key:     `scheduler.alpha.kubernetes.io/critical-pod`,
		message: `non-functional in v1.16+; use the "priorityClassName" field instead`,
	},
	{
		key:     `security.alpha.kubernetes.io/sysctls`,
		message: `non-functional in v1.11+; use the "sysctls" field instead`,
	},
	{
		key:     `security.alpha.kubernetes.io/unsafe-sysctls`,
		message: `non-functional in v1.11+; use the "sysctls" field instead`,
	},
}

func warningsForPodSpecAndMeta(fieldPath *field.Path, podSpec *api.PodSpec, meta *metav1.ObjectMeta, oldPodSpec *api.PodSpec, oldMeta *metav1.ObjectMeta) []string {
	var warnings []string

	// use of deprecated node labels in selectors/affinity/topology
	for k := range podSpec.NodeSelector {
		if msg, deprecated := deprecatedNodeLabels[k]; deprecated {
			warnings = append(warnings, fmt.Sprintf("%s: %s", fieldPath.Child("spec", "nodeSelector").Key(k), msg))
		}
	}
	if podSpec.Affinity != nil && podSpec.Affinity.NodeAffinity != nil {
		n := podSpec.Affinity.NodeAffinity
		if n.RequiredDuringSchedulingIgnoredDuringExecution != nil {
			for i, t := range n.RequiredDuringSchedulingIgnoredDuringExecution.NodeSelectorTerms {
				for j, e := range t.MatchExpressions {
					if msg, deprecated := deprecatedNodeLabels[e.Key]; deprecated {
						warnings = append(
							warnings,
							fmt.Sprintf(
								"%s: %s is %s",
								fieldPath.Child("spec", "affinity", "nodeAffinity", "requiredDuringSchedulingIgnoredDuringExecution", "nodeSelectorTerms").Index(i).
									Child("matchExpressions").Index(j).
									Child("key"),
								e.Key,
								msg,
							),
						)
					}
				}
			}
		}
		for i, t := range n.PreferredDuringSchedulingIgnoredDuringExecution {
			for j, e := range t.Preference.MatchExpressions {
				if msg, deprecated := deprecatedNodeLabels[e.Key]; deprecated {
					warnings = append(
						warnings,
						fmt.Sprintf(
							"%s: %s is %s",
							fieldPath.Child("spec", "affinity", "nodeAffinity", "preferredDuringSchedulingIgnoredDuringExecution").Index(i).
								Child("preference").
								Child("matchExpressions").Index(j).
								Child("key"),
							e.Key,
							msg,
						),
					)
				}
			}
		}
	}
	for i, t := range podSpec.TopologySpreadConstraints {
		if msg, deprecated := deprecatedNodeLabels[t.TopologyKey]; deprecated {
			warnings = append(warnings, fmt.Sprintf(
				"%s: %s is %s",
				fieldPath.Child("spec", "topologySpreadConstraints").Index(i).Child("topologyKey"),
				t.TopologyKey,
				msg,
			))
		}
	}

	// use of deprecated annotations
	for _, deprecated := range deprecatedAnnotations {
		if _, exists := meta.Annotations[deprecated.key]; exists {
			warnings = append(warnings, fmt.Sprintf("%s: %s", fieldPath.Child("metadata", "annotations").Key(deprecated.key), deprecated.message))
		}
		if len(deprecated.prefix) > 0 {
			for k := range meta.Annotations {
				if strings.HasPrefix(k, deprecated.prefix) {
					warnings = append(warnings, fmt.Sprintf("%s: %s", fieldPath.Child("metadata", "annotations").Key(k), deprecated.message))
					break
				}
			}
		}
	}

	// deprecated and removed volume plugins
	for i, v := range podSpec.Volumes {
		if v.PhotonPersistentDisk != nil {
			warnings = append(warnings, fmt.Sprintf("%s: deprecated in v1.11, non-functional in v1.16+", fieldPath.Child("spec", "volumes").Index(i).Child("photonPersistentDisk")))
		}
		if v.GitRepo != nil {
			warnings = append(warnings, fmt.Sprintf("%s: deprecated in v1.11", fieldPath.Child("spec", "volumes").Index(i).Child("gitRepo")))
		}
		if v.ScaleIO != nil {
			warnings = append(warnings, fmt.Sprintf("%s: deprecated in v1.16, non-functional in v1.22+", fieldPath.Child("spec", "volumes").Index(i).Child("scaleIO")))
		}
		if v.Flocker != nil {
			warnings = append(warnings, fmt.Sprintf("%s: deprecated in v1.22, support removal is planned in v1.26", fieldPath.Child("spec", "volumes").Index(i).Child("flocker")))
		}
		if v.StorageOS != nil {
			warnings = append(warnings, fmt.Sprintf("%s: deprecated in v1.22, support removal is planned in v1.26", fieldPath.Child("spec", "volumes").Index(i).Child("storageOS")))
		}
		if v.Quobyte != nil {
			warnings = append(warnings, fmt.Sprintf("%s: deprecated in v1.22, support removal is planned in v1.26", fieldPath.Child("spec", "volumes").Index(i).Child("quobyte")))
		}
	}

	// duplicate hostAliases (#91670, #58477)
	if len(podSpec.HostAliases) > 1 {
		items := sets.NewString()
		for i, item := range podSpec.HostAliases {
			if items.Has(item.IP) {
				warnings = append(warnings, fmt.Sprintf("%s: duplicate ip %q", fieldPath.Child("spec", "hostAliases").Index(i).Child("ip"), item.IP))
			} else {
				items.Insert(item.IP)
			}
		}
	}

	// duplicate imagePullSecrets (#91629, #58477)
	if len(podSpec.ImagePullSecrets) > 1 {
		items := sets.NewString()
		for i, item := range podSpec.ImagePullSecrets {
			if items.Has(item.Name) {
				warnings = append(warnings, fmt.Sprintf("%s: duplicate name %q", fieldPath.Child("spec", "imagePullSecrets").Index(i).Child("name"), item.Name))
			} else {
				items.Insert(item.Name)
			}
		}
	}
	// imagePullSecrets with empty name (#99454#issuecomment-787838112)
	for i, item := range podSpec.ImagePullSecrets {
		if len(item.Name) == 0 {
			warnings = append(warnings, fmt.Sprintf("%s: invalid empty name %q", fieldPath.Child("spec", "imagePullSecrets").Index(i).Child("name"), item.Name))
		}
	}

	// duplicate volume names (#78266, #58477)
	if len(podSpec.Volumes) > 1 {
		items := sets.NewString()
		for i, item := range podSpec.Volumes {
			if items.Has(item.Name) {
				warnings = append(warnings, fmt.Sprintf("%s: duplicate name %q", fieldPath.Child("spec", "volumes").Index(i).Child("name"), item.Name))
			} else {
				items.Insert(item.Name)
			}
		}
	}

	// fractional memory/ephemeral-storage requests/limits (#79950, #49442, #18538)
	if value, ok := podSpec.Overhead[api.ResourceMemory]; ok && value.MilliValue()%int64(1000) != int64(0) {
		warnings = append(warnings, fmt.Sprintf("%s: fractional byte value %q is invalid, must be an integer", fieldPath.Child("spec", "overhead").Key(string(api.ResourceMemory)), value.String()))
	}
	if value, ok := podSpec.Overhead[api.ResourceEphemeralStorage]; ok && value.MilliValue()%int64(1000) != int64(0) {
		warnings = append(warnings, fmt.Sprintf("%s: fractional byte value %q is invalid, must be an integer", fieldPath.Child("spec", "overhead").Key(string(api.ResourceEphemeralStorage)), value.String()))
	}

	// use of pod seccomp annotation without accompanying field
	if podSpec.SecurityContext == nil || podSpec.SecurityContext.SeccompProfile == nil {
		if _, exists := meta.Annotations[api.SeccompPodAnnotationKey]; exists {
			warnings = append(warnings, fmt.Sprintf(`%s: deprecated since v1.19; use the "seccompProfile" field instead`, fieldPath.Child("metadata", "annotations").Key(api.SeccompPodAnnotationKey)))
		}
	}

	pods.VisitContainersWithPath(podSpec, fieldPath.Child("spec"), func(c *api.Container, p *field.Path) bool {
		// use of container seccomp annotation without accompanying field
		if c.SecurityContext == nil || c.SecurityContext.SeccompProfile == nil {
			if _, exists := meta.Annotations[api.SeccompContainerAnnotationKeyPrefix+c.Name]; exists {
				warnings = append(warnings, fmt.Sprintf(`%s: deprecated since v1.19; use the "seccompProfile" field instead`, fieldPath.Child("metadata", "annotations").Key(api.SeccompContainerAnnotationKeyPrefix+c.Name)))
			}
		}

		// fractional memory/ephemeral-storage requests/limits (#79950, #49442, #18538)
		if value, ok := c.Resources.Limits[api.ResourceMemory]; ok && value.MilliValue()%int64(1000) != int64(0) {
			warnings = append(warnings, fmt.Sprintf("%s: fractional byte value %q is invalid, must be an integer", p.Child("resources", "limits").Key(string(api.ResourceMemory)), value.String()))
		}
		if value, ok := c.Resources.Requests[api.ResourceMemory]; ok && value.MilliValue()%int64(1000) != int64(0) {
			warnings = append(warnings, fmt.Sprintf("%s: fractional byte value %q is invalid, must be an integer", p.Child("resources", "requests").Key(string(api.ResourceMemory)), value.String()))
		}
		if value, ok := c.Resources.Limits[api.ResourceEphemeralStorage]; ok && value.MilliValue()%int64(1000) != int64(0) {
			warnings = append(warnings, fmt.Sprintf("%s: fractional byte value %q is invalid, must be an integer", p.Child("resources", "limits").Key(string(api.ResourceEphemeralStorage)), value.String()))
		}
		if value, ok := c.Resources.Requests[api.ResourceEphemeralStorage]; ok && value.MilliValue()%int64(1000) != int64(0) {
			warnings = append(warnings, fmt.Sprintf("%s: fractional byte value %q is invalid, must be an integer", p.Child("resources", "requests").Key(string(api.ResourceEphemeralStorage)), value.String()))
		}

		// duplicate containers[*].env (#86163, #93266, #58477)
		if len(c.Env) > 1 {
			items := sets.NewString()
			for i, item := range c.Env {
				if items.Has(item.Name) {
					warnings = append(warnings, fmt.Sprintf("%s: duplicate name %q", p.Child("env").Index(i).Child("name"), item.Name))
				} else {
					items.Insert(item.Name)
				}
			}
		}
		return true
	})

	return warnings
}
