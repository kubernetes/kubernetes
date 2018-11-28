package util

import (
	"fmt"

	"github.com/golang/glog"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/kubernetes/pkg/kubelet/log/api"
	"k8s.io/kubernetes/pkg/kubelet/log/api/v1alpha1"
	utilv1alpha1 "k8s.io/kubernetes/pkg/kubelet/log/api/v1alpha1/util"
	"k8s.io/kubernetes/pkg/kubelet/log/api/v1beta1"
	utilv1beta1 "k8s.io/kubernetes/pkg/kubelet/log/api/v1beta1/util"
	"k8s.io/kubernetes/pkg/kubelet/util/format"
)

func IsPodLogPolicyExists(pod *v1.Pod) bool {
	if utilv1beta1.IsPodLogPolicyExists(pod) {
		return true
	}
	if utilv1alpha1.IsPodLogPolicyExists(pod) {
		return true
	}
	return false
}

func GetPodLogPolicy(pod *v1.Pod) (*api.PodLogPolicy, error) {
	exists := utilv1beta1.IsPodLogPolicyExists(pod)
	if exists {
		v1beta1PodLogPolicy, err := utilv1beta1.GetPodLogPolicy(pod)
		if err != nil {
			glog.Errorf("get pog log policy error, %v, pod: %q", err, format.Pod(pod))
			return nil, err
		}
		podLogPolicy := &api.PodLogPolicy{}
		err = v1beta1.Convert_v1beta1_PodLogPolicy_To_api_PodPolicy(v1beta1PodLogPolicy, podLogPolicy, nil)
		if err != nil {
			glog.Errorf("convert pod log policy from v1beta to api error, %v, pod: %q", err, format.Pod(pod))
			return nil, err
		}
		return podLogPolicy, nil
	}

	exists = utilv1alpha1.IsPodLogPolicyExists(pod)
	if exists {
		v1alpha1PodLogPolicy, err := utilv1alpha1.GetPodLogPolicy(pod)
		if err != nil {
			glog.Errorf("get pog log policy error, %v, pod: %q", err, format.Pod(pod))
			return nil, err
		}
		podLogPolicy := &api.PodLogPolicy{}
		err = v1alpha1.Convert_v1alpha1_PodLogPolicy_To_api_PodPolicy(v1alpha1PodLogPolicy, podLogPolicy, nil)
		if err != nil {
			glog.Errorf("convert pod log policy from v1alpha1 to api error, %v, pod: %q", err, format.Pod(pod))
			return nil, err
		}
		return podLogPolicy, nil
	}

	return nil, fmt.Errorf("pod log policy is not exists, pod: %q", format.Pod(pod))
}

func GetPodLogConfigMapNames(pod *v1.Pod) sets.String {
	exists := utilv1beta1.IsPodLogPolicyExists(pod)
	if exists {
		return utilv1beta1.GetPodLogConfigMapNames(pod)
	}

	exists = utilv1alpha1.IsPodLogPolicyExists(pod)
	if exists {
		return utilv1alpha1.GetPodLogConfigMapNames(pod)
	}

	return sets.NewString()
}
