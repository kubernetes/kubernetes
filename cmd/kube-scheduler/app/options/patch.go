package options

import (
	"k8s.io/klog/v2"

	kubeschedulerconfig "k8s.io/kubernetes/pkg/scheduler/apis/config"
)

func LoadKubeSchedulerConfiguration(logger klog.Logger, file string) (*kubeschedulerconfig.KubeSchedulerConfiguration, error) {
	return LoadConfigFromFile(logger, file)
}
