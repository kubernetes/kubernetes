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
	"io"
	"os"

	"github.com/pkg/errors"
	"github.com/spf13/cobra"
	flag "github.com/spf13/pflag"

	"k8s.io/apimachinery/pkg/util/sets"
	clientset "k8s.io/client-go/kubernetes"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/validation"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/options"
	phases "k8s.io/kubernetes/cmd/kubeadm/app/cmd/phases/upgrade/node"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/phases/workflow"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	configutil "k8s.io/kubernetes/cmd/kubeadm/app/util/config"
)

// nodeOptions defines all the options exposed via flags by kubeadm upgrade node.
// Please note that this structure includes the public kubeadm config API, but only a subset of the options
// supported by this api will be exposed as a flag.
type nodeOptions struct {
	kubeConfigPath        string
	etcdUpgrade           bool
	renewCerts            bool
	dryRun                bool
	patchesDir            string
	ignorePreflightErrors []string
}

// compile-time assert that the local data object satisfies the phases data interface.
var _ phases.Data = &nodeData{}

// nodeData defines all the runtime information used when running the kubeadm upgrade node workflow;
// this data is shared across all the phases that are included in the workflow.
type nodeData struct {
	etcdUpgrade           bool
	renewCerts            bool
	dryRun                bool
	cfg                   *kubeadmapi.InitConfiguration
	isControlPlaneNode    bool
	client                clientset.Interface
	patchesDir            string
	ignorePreflightErrors sets.Set[string]
	kubeConfigPath        string
	outputWriter          io.Writer
}

// newCmdNode returns the cobra command for `kubeadm upgrade node`
func newCmdNode(out io.Writer) *cobra.Command {
	nodeOptions := newNodeOptions()
	nodeRunner := workflow.NewRunner()

	cmd := &cobra.Command{
		Use:   "node",
		Short: "Upgrade commands for a node in the cluster",
		RunE: func(cmd *cobra.Command, args []string) error {
			return nodeRunner.Run(args)
		},
		Args: cobra.NoArgs,
	}

	// adds flags to the node command
	// flags could be eventually inherited by the sub-commands automatically generated for phases
	addUpgradeNodeFlags(cmd.Flags(), nodeOptions)
	options.AddPatchesFlag(cmd.Flags(), &nodeOptions.patchesDir)

	// initialize the workflow runner with the list of phases
	nodeRunner.AppendPhase(phases.NewPreflightPhase())
	nodeRunner.AppendPhase(phases.NewControlPlane())
	nodeRunner.AppendPhase(phases.NewKubeletConfigPhase())

	// sets the data builder function, that will be used by the runner
	// both when running the entire workflow or single phases
	nodeRunner.SetDataInitializer(func(cmd *cobra.Command, args []string) (workflow.RunData, error) {
		return newNodeData(cmd, args, nodeOptions, out)
	})

	// binds the Runner to kubeadm upgrade node command by altering
	// command help, adding --skip-phases flag and by adding phases subcommands
	nodeRunner.BindToCommand(cmd)

	return cmd
}

// newNodeOptions returns a struct ready for being used for creating cmd kubeadm upgrade node flags.
func newNodeOptions() *nodeOptions {
	return &nodeOptions{
		kubeConfigPath: "", // This is populated in newNodeData() on runtime
		dryRun:         false,
		renewCerts:     true,
		etcdUpgrade:    true,
	}
}

func addUpgradeNodeFlags(flagSet *flag.FlagSet, nodeOptions *nodeOptions) {
	options.AddKubeConfigFlag(flagSet, &nodeOptions.kubeConfigPath)
	flagSet.BoolVar(&nodeOptions.dryRun, options.DryRun, nodeOptions.dryRun, "Do not change any state, just output the actions that would be performed.")
	flagSet.BoolVar(&nodeOptions.renewCerts, options.CertificateRenewal, nodeOptions.renewCerts, "Perform the renewal of certificates used by component changed during upgrades.")
	flagSet.BoolVar(&nodeOptions.etcdUpgrade, options.EtcdUpgrade, nodeOptions.etcdUpgrade, "Perform the upgrade of etcd.")
	flagSet.StringSliceVar(&nodeOptions.ignorePreflightErrors, options.IgnorePreflightErrors, nodeOptions.ignorePreflightErrors, "A list of checks whose errors will be shown as warnings. Example: 'IsPrivilegedUser,Swap'. Value 'all' ignores errors from all checks.")
}

// newNodeData returns a new nodeData struct to be used for the execution of the kubeadm upgrade node workflow.
// This func takes care of validating nodeOptions passed to the command, and then it converts
// options into the internal InitConfiguration type that is used as input all the phases in the kubeadm upgrade node workflow
func newNodeData(cmd *cobra.Command, args []string, options *nodeOptions, out io.Writer) (*nodeData, error) {
	// Checks if a node is a control-plane node by looking up the kube-apiserver manifest file
	isControlPlaneNode := true
	filepath := constants.GetStaticPodFilepath(constants.KubeAPIServer, constants.GetStaticPodDirectory())
	if _, err := os.Stat(filepath); os.IsNotExist(err) {
		isControlPlaneNode = false
	}
	if len(options.kubeConfigPath) == 0 {
		// Update the kubeconfig path depending on whether this is a control plane node or not.
		options.kubeConfigPath = constants.GetKubeletKubeConfigPath()
		if isControlPlaneNode {
			options.kubeConfigPath = constants.GetAdminKubeConfigPath()
		}
	}

	client, err := getClient(options.kubeConfigPath, options.dryRun)
	if err != nil {
		return nil, errors.Wrapf(err, "couldn't create a Kubernetes client from file %q", options.kubeConfigPath)
	}
	// Fetches the cluster configuration
	// NB in case of control-plane node, we are reading all the info for the node; in case of NOT control-plane node
	//    (worker node), we are not reading local API address and the CRI socket from the node object
	cfg, err := configutil.FetchInitConfigurationFromCluster(client, nil, "upgrade", !isControlPlaneNode, false)
	if err != nil {
		return nil, errors.Wrap(err, "unable to fetch the kubeadm-config ConfigMap")
	}

	ignorePreflightErrorsSet, err := validation.ValidateIgnorePreflightErrors(options.ignorePreflightErrors, cfg.NodeRegistration.IgnorePreflightErrors)
	if err != nil {
		return nil, err
	}
	// Also set the union of pre-flight errors to JoinConfiguration, to provide a consistent view of the runtime configuration:
	cfg.NodeRegistration.IgnorePreflightErrors = sets.List(ignorePreflightErrorsSet)
	return &nodeData{
		etcdUpgrade:           options.etcdUpgrade,
		renewCerts:            options.renewCerts,
		dryRun:                options.dryRun,
		cfg:                   cfg,
		client:                client,
		isControlPlaneNode:    isControlPlaneNode,
		patchesDir:            options.patchesDir,
		ignorePreflightErrors: ignorePreflightErrorsSet,
		kubeConfigPath:        options.kubeConfigPath,
		outputWriter:          out,
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

// PatchesDir returns the folder where patches for components are stored
func (d *nodeData) PatchesDir() string {
	return d.patchesDir
}

// IgnorePreflightErrors returns the list of preflight errors to ignore.
func (d *nodeData) IgnorePreflightErrors() sets.Set[string] {
	return d.ignorePreflightErrors
}

// KubeConfigPath returns the path to the user kubeconfig file.
func (d *nodeData) KubeConfigPath() string {
	return d.kubeConfigPath
}

func (d *nodeData) OutputWriter() io.Writer {
	return d.outputWriter
}
