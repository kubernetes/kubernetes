package v1alpha1

import (
	"k8s.io/apimachinery/pkg/conversion"
	"k8s.io/kubernetes/pkg/kubelet/log/api"
)

func Convert_v1alpha1_PodLogPolicy_To_api_PodPolicy(in *PodLogPolicy, out *api.PodLogPolicy, s conversion.Scope) error {
	out.PluginName = in.LogPlugin
	out.SafeDeletionEnabled = in.SafeDeletionEnabled
	for containerName, containerLogPolicies := range in.ContainerLogPolicies {
		for _, containerLogPolicy := range containerLogPolicies {
			policy := api.ContainerLogPolicy{
				ContainerName:   containerName,
				Name:            containerLogPolicy.Category,
				Path:            containerLogPolicy.Path,
				VolumeName:      containerLogPolicy.VolumeName,
				PluginConfigMap: containerLogPolicy.PluginConfigMap,
			}
			out.ContainerLogPolicies = append(out.ContainerLogPolicies, policy)
		}
	}
	return nil
}
