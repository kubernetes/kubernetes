package options

import kubeschedulerconfig "k8s.io/kubernetes/pkg/scheduler/apis/config"

func LoadKubeSchedulerConfiguration(file string) (*kubeschedulerconfig.KubeSchedulerConfiguration, error) {
	return loadConfigFromFile(file)
}
