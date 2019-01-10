package util

import (
	"encoding/json"
	"fmt"

	"github.com/golang/glog"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/kubernetes/pkg/kubelet/log/api/v1alpha1"
	"k8s.io/kubernetes/pkg/kubelet/util/format"
)

func IsPodLogPolicyExists(pod *v1.Pod) bool {
	_, exists := pod.Annotations[v1alpha1.PodLogPolicyLabelKey]
	if exists {
		return true
	}
	return false
}

func GetPodLogPolicy(pod *v1.Pod) (*v1alpha1.PodLogPolicy, error) {
	// get log policy from pod annotations
	podLogPolicyLabelValue, exists := pod.Annotations[v1alpha1.PodLogPolicyLabelKey]
	if !exists {
		return nil, fmt.Errorf("key %q is not exists", podLogPolicyLabelValue)
	}

	podLogPolicy := &v1alpha1.PodLogPolicy{}
	err := json.Unmarshal([]byte(podLogPolicyLabelValue), podLogPolicy)
	if err != nil {
		glog.Errorf("json unmarshal error, %v, podLogPolicyLabelValue: %s", err, podLogPolicyLabelValue)
		return nil, err
	}

	return podLogPolicy, nil
}

func GetPodLogConfigMapNames(pod *v1.Pod) sets.String {
	// configMap name set
	configMapNames := sets.NewString()
	podLogPolicy, err := GetPodLogPolicy(pod)
	if err != nil {
		glog.Errorf("get pod log policy error, %v, pod: %q", err, format.Pod(pod))
		return configMapNames
	}
	for _, containerLogPolicies := range podLogPolicy.ContainerLogPolicies {
		for _, containerLogPolicy := range containerLogPolicies {
			configMapNames.Insert(containerLogPolicy.PluginConfigMap)
		}
	}
	return configMapNames
}
