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

package upgrade

import (
	"os"

	"github.com/pkg/errors"
	"github.com/spf13/cobra"
	flag "github.com/spf13/pflag"

	clientset "k8s.io/client-go/kubernetes"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/options"
	phases "k8s.io/kubernetes/cmd/kubeadm/app/cmd/phases/upgrade/node"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/phases/workflow"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	configutil "k8s.io/kubernetes/cmd/kubeadm/app/util/config"
)

// nodeOptions defines all the options exposed via flags by kubeadm upgrade node.
// Please note that this structure includes the public kubeadm config API, but only a subset of the options
// supported by this api will be exposed as a flag.
type nodeOptions struct {
	kubeConfigPath   string
	kubeletVersion   string
	advertiseAddress string
	nodeName         string
	etcdUpgrade      bool
	renewCerts       bool
	dryRun           bool
}

// compile-time assert that the local data object satisfies the phases data interface.
var _ phases.Data = &nodeData{}

// nodeData defines all the runtime information used when running the kubeadm upgrade node worklow;
// this data is shared across all the phases that are included in the workflow.
type nodeData struct {
	etcdUpgrade        bool
	renewCerts         bool
	dryRun             bool
	kubeletVersion     string
	cfg                *kubeadmapi.InitConfiguration
	isControlPlaneNode bool
	client             clientset.Interface
}

// NewCmdNode returns the cobra command for `kubeadm upgrade node`
func NewCmdNode() *cobra.Command {
	nodeOptions := newNodeOptions()
	nodeRunner := workflow.NewRunner()

	cmd := &cobra.Command{
		Use:   "node",
		Short: "Upgrade commands for a node in the cluster",
		Run: func(cmd *cobra.Command, args []string) {
			err := nodeRunner.Run(args)
			kubeadmutil.CheckErr(err)
		},
		Args: cobra.NoArgs,
	}

	// adds flags to the node command
	// flags could be eventually inherited by the sub-commands automatically generated for phases
	addUpgradeNodeFlags(cmd.Flags(), nodeOptions)

	// initialize the workflow runner with the list of phases
	nodeRunner.AppendPhase(phases.NewControlPlane())
	nodeRunner.AppendPhase(phases.NewKubeletConfigPhase())

	// sets the data builder function, that will be used by the runner
	// both when running the entire workflow or single phases
	nodeRunner.SetDataInitializer(func(cmd *cobra.Command, args []string) (workflow.RunData, error) {
		return newNodeData(cmd, args, nodeOptions)
	})

	// binds the Runner to kubeadm upgrade node command by altering
	// command help, adding --skip-phases flag and by adding phases subcommands
	nodeRunner.BindToCommand(cmd)

	// upgrade node config command is subject to GA deprecation policy, so we should deprecate it
	// and keep it here for one year or three releases - the longer of the two - starting from v1.15 included
	cmd.AddCommand(NewCmdUpgradeNodeConfig())

	return cmd
}

// newNodeOptions returns a struct ready for being used for creating cmd kubeadm upgrade node flags.
func newNodeOptions() *nodeOptions {
	return &nodeOptions{
		kubeConfigPath: constants.GetKubeletKubeConfigPath(),
		dryRun:         false,
	}
}

func addUpgradeNodeFlags(flagSet *flag.FlagSet, nodeOptions *nodeOptions) {
	options.AddKubeConfigFlag(flagSet, &nodeOptions.kubeConfigPath)
	flagSet.BoolVar(&nodeOptions.dryRun, options.DryRun, nodeOptions.dryRun, "Do not change any state, just output the actions that would be performed.")
	flagSet.StringVar(&nodeOptions.kubeletVersion, options.KubeletVersion, nodeOptions.kubeletVersion, "The *desired* version for the kubelet config after the upgrade. If not specified, the KubernetesVersion from the kubeadm-config ConfigMap will be used")
	flagSet.BoolVar(&nodeOptions.renewCerts, options.CertificateRenewal, nodeOptions.renewCerts, "Perform the renewal of certificates used by component changed during upgrades.")
	flagSet.BoolVar(&nodeOptions.etcdUpgrade, options.EtcdUpgrade, nodeOptions.etcdUpgrade, "Perform the upgrade of etcd.")
}

// newNodeData returns a new nodeData struct to be used for the execution of the kubeadm upgrade node workflow.
// This func takes care of validating nodeOptions passed to the command, and then it converts
// options into the internal InitConfiguration type that is used as input all the phases in the kubeadm upgrade node workflow
func newNodeData(cmd *cobra.Command, args []string, options *nodeOptions) (*nodeData, error) {
	client, err := getClient(options.kubeConfigPath, options.dryRun)
	if err != nil {
		return nil, errors.Wrapf(err, "couldn't create a Kubernetes client from file %q", options.kubeConfigPath)
	}

	// isControlPlane checks if a node is a control-plane node by looking up
	// the kube-apiserver manifest file
	isControlPlaneNode := true
	filepath := kubeadmconstants.GetStaticPodFilepath(kubeadmconstants.KubeAPIServer, kubeadmconstants.GetStaticPodDirectory())
	if _, err := os.Stat(filepath); os.IsNotExist(err) {
		isControlPlaneNode = false
	}

	// Fetches the cluster configuration
	// NB in case of control-plane node, we are reading all the info for the node; in case of NOT control-plane node
	//    (worker node), we are not reading local API address and the CRI socket from the node object
	cfg, err := configutil.FetchInitConfigurationFromCluster(client, os.Stdout, "upgrade", !isControlPlaneNode)
	if err != nil {
		return nil, errors.Wrap(err, "unable to fetch the kubeadm-config ConfigMap")
	}

	return &nodeData{
		etcdUpgrade:        options.etcdUpgrade,
		renewCerts:         options.renewCerts,
		dryRun:             options.dryRun,
		kubeletVersion:     options.kubeletVersion,
		cfg:                cfg,
		client:             client,
		isControlPlaneNode: isControlPlaneNode,
	}, nil
}

// DryRun returns the dryRun flag.
func (d *nodeData) DryRun() bool {
	return d.dryRun
}

// EtcdUpgrade returns the etcdUpgrade flag.
func (d *nodeData) EtcdUpgrade() bool {
	return d.etcdUpgrade
}

// RenewCerts returns the renewCerts flag.
func (d *nodeData) RenewCerts() bool {
	return d.renewCerts
}

// KubeletVersion returns the kubeletVersion flag.
func (d *nodeData) KubeletVersion() string {
	return d.kubeletVersion
}

// Cfg returns initConfiguration.
func (d *nodeData) Cfg() *kubeadmapi.InitConfiguration {
	return d.cfg
}

// IsControlPlaneNode returns the isControlPlaneNode flag.
func (d *nodeData) IsControlPlaneNode() bool {
	return d.isControlPlaneNode
}

// Client returns a Kubernetes client to be used by kubeadm.
func (d *nodeData) Client() clientset.Interface {
	return d.client
}

// NewCmdUpgradeNodeConfig returns the cobra.Command for downloading the new/upgrading the kubelet configuration from the kubelet-config-1.X
// ConfigMap in the cluster
// TODO: to remove when 1.18 is released
func NewCmdUpgradeNodeConfig() *cobra.Command {
	nodeOptions := newNodeOptions()
	nodeRunner := workflow.NewRunner()

	cmd := &cobra.Command{
		Use:        "config",
		Short:      "Download the kubelet configuration from the cluster ConfigMap kubelet-config-1.X, where X is the minor version of the kubelet",
		Deprecated: "use \"kubeadm upgrade node\" instead",
		Run: func(cmd *cobra.Command, args []string) {
			// This is required for preserving the old behavior of `kubeadm upgrade node config`.
			// The new implementation exposed as a phase under `kubeadm upgrade node` infers the target
			// kubelet config version from the kubeadm-config ConfigMap
			if len(nodeOptions.kubeletVersion) == 0 {
				kubeadmutil.CheckErr(errors.New("the --kubelet-version argument is required"))
			}

			err := nodeRunner.Run(args)
			kubeadmutil.CheckErr(err)
		},
	}

	// adds flags to the node command
	addUpgradeNodeFlags(cmd.Flags(), nodeOptions)

	// initialize the workflow runner with the list of phases
	nodeRunner.AppendPhase(phases.NewKubeletConfigPhase())

	// sets the data builder function, that will be used by the runner
	// both when running the entire workflow or single phases
	nodeRunner.SetDataInitializer(func(cmd *cobra.Command, args []string) (workflow.RunData, error) {
		return newNodeData(cmd, args, nodeOptions)
	})

	return cmd
}
