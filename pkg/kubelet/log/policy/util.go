package policy

import (
	"encoding/json"

	"github.com/golang/glog"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/kubernetes/pkg/kubelet/util/format"
)

func IsPodLogPolicyExists(pod *v1.Pod) bool {
	_, exists := pod.Annotations[PodLogPolicyLabelKey]
	if !exists {
		return false
	}
	return true
}

func GetPodLogPolicy(pod *v1.Pod) (*PodLogPolicy, error) {
	// get log policy from pod annotations
	podLogPolicyLabelValue := pod.Annotations[PodLogPolicyLabelKey]
	podLogPolicy := &PodLogPolicy{}
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
	if !IsPodLogPolicyExists(pod) {
		return configMapNames
	}
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
