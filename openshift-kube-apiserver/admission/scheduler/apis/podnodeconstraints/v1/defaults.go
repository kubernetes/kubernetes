package v1

import (
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
)

func SetDefaults_PodNodeConstraintsConfig(obj *PodNodeConstraintsConfig) {
	if obj.NodeSelectorLabelBlacklist == nil {
		obj.NodeSelectorLabelBlacklist = []string{
			corev1.LabelHostname,
		}
	}
}

func addDefaultingFuncs(scheme *runtime.Scheme) error {
	scheme.AddTypeDefaultingFunc(&PodNodeConstraintsConfig{}, func(obj interface{}) { SetDefaults_PodNodeConstraintsConfig(obj.(*PodNodeConstraintsConfig)) })
	return nil
}
