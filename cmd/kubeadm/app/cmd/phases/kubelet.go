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
	"github.com/pkg/errors"
	"k8s.io/klog"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/options"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/phases/workflow"
	kubeletphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/kubelet"
	"k8s.io/kubernetes/pkg/util/normalizer"
)

var (
	kubeletStartPhaseExample = normalizer.Examples(`
		# Writes a dynamic environment file with kubelet flags from a InitConfiguration file.
		kubeadm init phase kubelet-start --config masterconfig.yaml
		`)
)

// kubeletStartData defines the behavior that a runtime data struct passed to the kubelet start phase
// should have. Please note that we are using an interface in order to make this phase reusable in different workflows
// (and thus with different runtime data struct, all of them requested to be compliant to this interface)
type kubeletStartData interface {
	Cfg() *kubeadmapi.InitConfiguration
	DryRun() bool
	KubeletDir() string
}

// NewKubeletStartPhase creates a kubeadm workflow phase that start kubelet on a node.
func NewKubeletStartPhase() workflow.Phase {
	return workflow.Phase{
		Name:    "kubelet-start",
		Short:   "Writes kubelet settings and (re)starts the kubelet",
		Long:    "Writes a file with KubeletConfiguration and an environment file with node specific kubelet settings, and then (re)starts kubelet.",
		Example: kubeletStartPhaseExample,
		Run:     runKubeletStart,
		InheritFlags: []string{
			options.CfgPath,
			options.NodeCRISocket,
			options.NodeName,
		},
	}
}

// runKubeletStart executes kubelet start logic.
func runKubeletStart(c workflow.RunData) error {
	data, ok := c.(kubeletStartData)
	if !ok {
		return errors.New("kubelet-start phase invoked with an invalid data struct")
	}

	// First off, configure the kubelet. In this short timeframe, kubeadm is trying to stop/restart the kubelet
	// Try to stop the kubelet service so no race conditions occur when configuring it
	if !data.DryRun() {
		klog.V(1).Infof("Stopping the kubelet")
		kubeletphase.TryStopKubelet()
	}

	// Write env file with flags for the kubelet to use. We do not need to write the --register-with-taints for the master,
	// as we handle that ourselves in the markmaster phase
	// TODO: Maybe we want to do that some time in the future, in order to remove some logic from the markmaster phase?
	if err := kubeletphase.WriteKubeletDynamicEnvFile(data.Cfg(), false, data.KubeletDir()); err != nil {
		return errors.Wrap(err, "error writing a dynamic environment file for the kubelet")
	}

	// Write the kubelet configuration file to disk.
	if err := kubeletphase.WriteConfigToDisk(data.Cfg().ComponentConfigs.Kubelet, data.KubeletDir()); err != nil {
		return errors.Wrap(err, "error writing kubelet configuration to disk")
	}

	// Try to start the kubelet service in case it's inactive
	if !data.DryRun() {
		klog.V(1).Infof("Starting the kubelet")
		kubeletphase.TryStartKubelet()
	}

	return nil
}
