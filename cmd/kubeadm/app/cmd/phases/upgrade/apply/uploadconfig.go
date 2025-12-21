/*
Copyright 2024 The Kubernetes Authors.

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

// Package apply implements phases of 'kubeadm upgrade apply'.
package apply

import (
	"fmt"

	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/klog/v2"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/options"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/phases/workflow"
	kubeletphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/kubelet"
	patchnodephase "k8s.io/kubernetes/cmd/kubeadm/app/phases/patchnode"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/uploadconfig"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/errors"
)

// NewUploadConfigPhase returns a new upload-config phase.
func NewUploadConfigPhase() workflow.Phase {
	return workflow.Phase{
		Name:    "upload-config",
		Aliases: []string{"uploadconfig"},
		Short:   "Upload the kubeadm and kubelet configurations to ConfigMaps",
		Phases: []workflow.Phase{
			{
				Name:           "all",
				Short:          "Upload all the configurations to ConfigMaps",
				RunAllSiblings: true,
				InheritFlags:   getUploadConfigPhaseFlags(),
			},
			{
				Name:         "kubeadm",
				Short:        "Upload the kubeadm ClusterConfiguration to a ConfigMap",
				Run:          runUploadKubeadmConfig,
				InheritFlags: getUploadConfigPhaseFlags(),
			},
			{
				Name:         "kubelet",
				Short:        "Upload the kubelet configuration to a ConfigMap",
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
		options.DryRun,
	}
}

// runUploadKubeadmConfig uploads the kubeadm configuration to a ConfigMap.
func runUploadKubeadmConfig(c workflow.RunData) error {
	cfg, client, dryRun, err := getUploadConfigData(c)
	if err != nil {
		return err
	}

	if dryRun {
		fmt.Println("[dryrun] Would upload the kubeadm ClusterConfiguration to a ConfigMap")
		return nil
	}

	klog.V(1).Infoln("[upgrade/upload-config] Uploading the kubeadm ClusterConfiguration to a ConfigMap")
	if err := uploadconfig.UploadConfiguration(cfg, client); err != nil {
		return errors.Wrap(err, "error uploading the kubeadm ClusterConfiguration")
	}
	return nil
}

// runUploadKubeletConfig uploads the kubelet configuration to a ConfigMap.
func runUploadKubeletConfig(c workflow.RunData) error {
	cfg, client, dryRun, err := getUploadConfigData(c)
	if err != nil {
		return err
	}

	if dryRun {
		fmt.Println("[dryrun] Would upload the kubelet configuration to a ConfigMap")
		fmt.Println("[dryrun] Would write the CRISocket annotation for the control-plane node")
		return nil
	}

	klog.V(1).Infoln("[upgrade/upload-config] Uploading the kubelet configuration to a ConfigMap")
	if err = kubeletphase.CreateConfigMap(&cfg.ClusterConfiguration, client); err != nil {
		return errors.Wrap(err, "error creating kubelet configuration ConfigMap")
	}

	// TODO Remove once NodeLocalCRISocket is removed in 1.37.
	if err := patchnodephase.RemoveCRISocketAnnotation(client, cfg.NodeRegistration.Name); err != nil {
		return err
	}

	return nil
}

func getUploadConfigData(c workflow.RunData) (*kubeadmapi.InitConfiguration, clientset.Interface, bool, error) {
	data, ok := c.(Data)
	if !ok {
		return nil, nil, false, errors.New("upload-config phase invoked with an invalid data struct")
	}

	return data.InitCfg(), data.Client(), data.DryRun(), nil
}
