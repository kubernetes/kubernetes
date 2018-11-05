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
	"github.com/golang/glog"
	"github.com/pkg/errors"
	"github.com/spf13/cobra"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapiv1beta1 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta1"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/options"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/phases/workflow"
	cmdutil "k8s.io/kubernetes/cmd/kubeadm/app/cmd/util"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	kubeletphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/kubelet"
	patchnodephase "k8s.io/kubernetes/cmd/kubeadm/app/phases/patchnode"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	configutil "k8s.io/kubernetes/cmd/kubeadm/app/util/config"
	kubeconfigutil "k8s.io/kubernetes/cmd/kubeadm/app/util/kubeconfig"
	"k8s.io/kubernetes/pkg/util/normalizer"
)

var (
	kubeletStartPhaseExample = normalizer.Examples(`
		# Writes a dynamic environment file with kubelet flags from a InitConfiguration file.
		kubeadm init phase kubelet-start --config masterconfig.yaml
		`)

	kubeletConfigUploadLongDesc = normalizer.LongDesc(`
		Uploads kubelet configuration extracted from the kubeadm InitConfiguration object to a ConfigMap
		of the form kubelet-config-1.X in the cluster, where X is the minor version of the current (API Server) Kubernetes version.
		` + cmdutil.AlphaDisclaimer)

	kubeletConfigUploadExample = normalizer.Examples(`
		# Uploads the kubelet configuration from the kubeadm Config file to a ConfigMap in the cluster.
		kubeadm alpha phase kubelet config upload --config kubeadm.yaml
		`)

	kubeletConfigAnnotateCRILongDesc = normalizer.LongDesc(`
		Adds an annotation to the current node with the CRI socket specified in the kubeadm InitConfiguration object.
		` + cmdutil.AlphaDisclaimer)

	kubeletConfigAnnotateCRIExample = normalizer.Examples(`
		kubeadm alpha phase kubelet config annotate-cri --config kubeadm.yaml
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
		CmdFlags: []string{
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
		glog.V(1).Infof("Stopping the kubelet")
		kubeletphase.TryStopKubelet()
	}

	// Write env file with flags for the kubelet to use. We do not need to write the --register-with-taints for the master,
	// as we handle that ourselves in the markmaster phase
	// TODO: Maybe we want to do that some time in the future, in order to remove some logic from the markmaster phase?
	if err := kubeletphase.WriteKubeletDynamicEnvFile(&data.Cfg().NodeRegistration, data.Cfg().FeatureGates, false, data.KubeletDir()); err != nil {
		return errors.Wrap(err, "error writing a dynamic environment file for the kubelet")
	}

	// Write the kubelet configuration file to disk.
	if err := kubeletphase.WriteConfigToDisk(data.Cfg().ComponentConfigs.Kubelet, data.KubeletDir()); err != nil {
		return errors.Wrap(err, "error writing kubelet configuration to disk")
	}

	// Try to start the kubelet service in case it's inactive
	if !data.DryRun() {
		glog.V(1).Infof("Starting the kubelet")
		kubeletphase.TryStartKubelet()
	}

	return nil
}

// NewCmdKubelet returns command for `kubeadm phase kubelet`
func NewCmdKubelet() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "kubelet",
		Short: "Commands related to handling the kubelet.",
		Long:  cmdutil.MacroCommandLongDescription,
	}

	cmd.AddCommand(NewCmdKubeletConfig())
	return cmd
}

// NewCmdKubeletConfig returns command for `kubeadm phase kubelet config`
func NewCmdKubeletConfig() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "config",
		Short: "Handles kubelet configuration.",
		Long:  cmdutil.MacroCommandLongDescription,
	}

	cmd.AddCommand(NewCmdKubeletConfigUpload())
	cmd.AddCommand(NewCmdKubeletAnnotateCRI())
	return cmd
}

// NewCmdKubeletConfigUpload calls cobra.Command for uploading dynamic kubelet configuration
func NewCmdKubeletConfigUpload() *cobra.Command {
	cfg := &kubeadmapiv1beta1.InitConfiguration{}
	var cfgPath string
	kubeConfigFile := constants.GetAdminKubeConfigPath()

	cmd := &cobra.Command{
		Use:     "upload",
		Short:   "Uploads kubelet configuration to a ConfigMap based on a kubeadm InitConfiguration file.",
		Long:    kubeletConfigUploadLongDesc,
		Example: kubeletConfigUploadExample,
		Run: func(cmd *cobra.Command, args []string) {
			if len(cfgPath) == 0 {
				kubeadmutil.CheckErr(errors.New("The --config argument is required"))
			}

			// KubernetesVersion is not used, but we set it explicitly to avoid the lookup
			// of the version from the internet when executing ConfigFileAndDefaultsToInternalConfig
			SetKubernetesVersion(cfg)

			// This call returns the ready-to-use configuration based on the configuration file
			internalcfg, err := configutil.ConfigFileAndDefaultsToInternalConfig(cfgPath, cfg)
			kubeadmutil.CheckErr(err)

			kubeConfigFile = cmdutil.FindExistingKubeConfig(kubeConfigFile)
			client, err := kubeconfigutil.ClientSetFromFile(kubeConfigFile)
			kubeadmutil.CheckErr(err)

			err = kubeletphase.CreateConfigMap(internalcfg, client)
			kubeadmutil.CheckErr(err)
		},
	}

	options.AddKubeConfigFlag(cmd.Flags(), &kubeConfigFile)
	options.AddConfigFlag(cmd.Flags(), &cfgPath)
	return cmd
}

// NewCmdKubeletAnnotateCRI calls cobra.Command for annotating the node with the given crisocket
func NewCmdKubeletAnnotateCRI() *cobra.Command {
	cfg := &kubeadmapiv1beta1.InitConfiguration{}
	var cfgPath string
	kubeConfigFile := constants.GetAdminKubeConfigPath()

	cmd := &cobra.Command{
		Use:     "annotate-cri",
		Short:   "annotates the node with the given crisocket",
		Long:    kubeletConfigAnnotateCRILongDesc,
		Example: kubeletConfigAnnotateCRIExample,
		Run: func(cmd *cobra.Command, args []string) {
			if len(cfgPath) == 0 {
				kubeadmutil.CheckErr(errors.New("The --config argument is required"))
			}

			// KubernetesVersion is not used, but we set it explicitly to avoid the lookup
			// of the version from the internet when executing ConfigFileAndDefaultsToInternalConfig
			SetKubernetesVersion(cfg)

			// This call returns the ready-to-use configuration based on the configuration file
			internalcfg, err := configutil.ConfigFileAndDefaultsToInternalConfig(cfgPath, cfg)
			kubeadmutil.CheckErr(err)

			kubeConfigFile = cmdutil.FindExistingKubeConfig(kubeConfigFile)
			client, err := kubeconfigutil.ClientSetFromFile(kubeConfigFile)
			kubeadmutil.CheckErr(err)

			err = patchnodephase.AnnotateCRISocket(client, internalcfg.NodeRegistration.Name, internalcfg.NodeRegistration.CRISocket)
			kubeadmutil.CheckErr(err)
		},
	}

	options.AddKubeConfigFlag(cmd.Flags(), &kubeConfigFile)
	options.AddConfigFlag(cmd.Flags(), &cfgPath)
	return cmd
}
