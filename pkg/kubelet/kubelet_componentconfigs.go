package kubelet

import (
	"k8s.io/kubernetes/pkg/apis/componentconfig"
)

func (k *Kubelet) GetComponentConfigs() componentconfig.KubeletConfiguration {
	return k.GetConfiguration()
}
