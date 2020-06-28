/*
Copyright 2017 The Kubernetes Authors.

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
	"fmt"

	"github.com/pkg/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/klog/v2"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/options"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/phases/workflow"
	cmdutil "k8s.io/kubernetes/cmd/kubeadm/app/cmd/util"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	kubeletphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/kubelet"
	patchnodephase "k8s.io/kubernetes/cmd/kubeadm/app/phases/patchnode"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/uploadconfig"
)

var (
	uploadKubeadmConfigLongDesc = fmt.Sprintf(cmdutil.LongDesc(`
		Upload the kubeadm ClusterConfiguration to a ConfigMap called %s in the %s namespace.
		This enables correct configuration of system components and a seamless user experience when upgrading.

		Alternatively, you can use kubeadm config.
		`), kubeadmconstants.KubeadmConfigConfigMap, metav1.NamespaceSystem)

	uploadKubeadmConfigExample = cmdutil.Examples(`
		# upload the configuration of your cluster
		kubeadm init phase upload-config --config=myConfig.yaml
		`)

	uploadKubeletConfigLongDesc = cmdutil.LongDesc(`
		Upload kubelet configuration extracted from the kubeadm InitConfiguration object to a ConfigMap
		of the form kubelet-config-1.X in the cluster, where X is the minor version of the current (API Server) Kubernetes version.
		`)

	uploadKubeletConfigExample = cmdutil.Examples(`
		# Upload the kubelet configuration from the kubeadm Config file to a ConfigMap in the cluster.
		kubeadm init phase upload-config kubelet --config kubeadm.yaml
		`)
)

// NewUploadConfigPhase returns the phase to uploadConfig
func NewUploadConfigPhase() workflow.Phase {
	return workflow.Phase{
		Name:    "upload-config",
		Aliases: []string{"uploadconfig"},
		Short:   "Upload the kubeadm and kubelet configuration to a ConfigMap",
		Long:    cmdutil.MacroCommandLongDescription,
		Phases: []workflow.Phase{
			{
				Name:           "all",
				Short:          "Upload all configuration to a config map",
				RunAllSiblings: true,
				InheritFlags:   getUploadConfigPhaseFlags(),
			},
			{
				Name:         "kubeadm",
				Short:        "Upload the kubeadm ClusterConfiguration to a ConfigMap",
				Long:         uploadKubeadmConfigLongDesc,
				Example:      uploadKubeadmConfigExample,
				Run:          runUploadKubeadmConfig,
				InheritFlags: getUploadConfigPhaseFlags(),
			},
			{
				Name:         "kubelet",
				Short:        "Upload the kubelet component config to a ConfigMap",
				Long:         uploadKubeletConfigLongDesc,
				Example:      uploadKubeletConfigExample,
				Run:          runUploadKubeletConfig,
				InheritFlags: getUploadConfigPhaseFlags(),
			},
		},
	}
}

func getUploadConfigPhaseFlags() []string {
	return []string{
		options.CfgPath,
		options.KubeconfigPath,
	}
}

// runUploadKubeadmConfig uploads the kubeadm configuration to a ConfigMap
func runUploadKubeadmConfig(c workflow.RunData) error {
	cfg, client, err := getUploadConfigData(c)
	if err != nil {
		return err
	}

	klog.V(1).Infoln("[upload-config] Uploading the kubeadm ClusterConfiguration to a ConfigMap")
	if err := uploadconfig.UploadConfiguration(cfg, client); err != nil {
		return errors.Wrap(err, "error uploading the kubeadm ClusterConfiguration")
	}
	return nil
}

// runUploadKubeletConfig uploads the kubelet configuration to a ConfigMap
func runUploadKubeletConfig(c workflow.RunData) error {
	cfg, client, err := getUploadConfigData(c)
	if err != nil {
		return err
	}

	klog.V(1).Infoln("[upload-config] Uploading the kubelet component config to a ConfigMap")
	if err = kubeletphase.CreateConfigMap(&cfg.ClusterConfiguration, client); err != nil {
		return errors.Wrap(err, "error creating kubelet configuration ConfigMap")
	}

	klog.V(1).Infoln("[upload-config] Preserving the CRISocket information for the control-plane node")
	if err := patchnodephase.AnnotateCRISocket(client, cfg.NodeRegistration.Name, cfg.NodeRegistration.CRISocket); err != nil {
		return errors.Wrap(err, "Error writing Crisocket information for the control-plane node")
	}
	return nil
}

func getUploadConfigData(c workflow.RunData) (*kubeadmapi.InitConfiguration, clientset.Interface, error) {
	data, ok := c.(InitData)
	if !ok {
		return nil, nil, errors.New("upload-config phase invoked with an invalid data struct")
	}
	cfg := data.Cfg()
	client, err := data.Client()
	if err != nil {
		return nil, nil, err
	}
	return cfg, client, err
}
