/*
Copyright 2018 The Kubernetes Authors.

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

package phases

import (
	"errors"

	clientset "k8s.io/client-go/kubernetes"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/phases/workflow"
	kubeletphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/kubelet"
	"k8s.io/kubernetes/cmd/kubeadm/app/preflight"
	"k8s.io/kubernetes/pkg/util/normalizer"
	utilsexec "k8s.io/utils/exec"
)

var (
	dynamicKubeletConfigLongDesc = normalizer.LongDesc(`
		Enables or updates dynamic kubelet configuration for a Node, against the kubelet-config-1.X ConfigMap in the cluster,
		where X is the minor version of the desired kubelet version.

		WARNING: This feature is still experimental, and disabled by default. Enable only if you know what you are doing, as it
		may have surprising side-effects at this stage.
		`)

	dynamicKubeletConfigExample = normalizer.Examples(`
		# Enables dynamic kubelet configuration for a Node.
		kubeadm init phase enable-dynamic-kubelet-config --node-name node-1

		WARNING: This feature is still experimental, and disabled by default. Enable only if you know what you are doing, as it
		may have surprising side-effects at this stage.
		`)
)

type enableDynamicKubeletConfigData interface {
	Cfg() *kubeadmapi.InitConfiguration
	Client() (clientset.Interface, error)
}

// NewEnableDynamicKubeletConfigPhase returns the phase to EnableDynamicKubeletConfig
func NewEnableDynamicKubeletConfigPhase() workflow.Phase {
	return workflow.Phase{
		Name:    "enable-dynamic-kubelet-config",
		Short:   "EXPERIMENTAL: Enables or updates dynamic kubelet configuration for a Node",
		Long:    dynamicKubeletConfigLongDesc,
		Example: dynamicKubeletConfigExample,
		Hidden:  true,
		Run:     EnableDynamicKubeletConfig,
	}
}

// EnableDynamicKubeletConfig enables dynamic kubelet configuration on node
// This feature is still in experimental state
func EnableDynamicKubeletConfig(c workflow.RunData) error {
	data, ok := c.(enableDynamicKubeletConfigData)
	if !ok {
		return errors.New("enable-dynamic-kubelet-config phase invoked with an invalid data struct")
	}

	cfg := data.Cfg()
	client, err := data.Client()
	if err != nil {
		return err
	}
	nodeName := cfg.NodeRegistration.Name
	if len(nodeName) == 0 {
		return errors.New("NodeRegistration.Name is required for the enable-dynamic-kubelet-config phase")
	}

	kubeletVersion, err := preflight.GetKubeletVersion(utilsexec.New())
	if err != nil {
		return err
	}

	return kubeletphase.EnableDynamicConfigForNode(client, nodeName, kubeletVersion)
}
