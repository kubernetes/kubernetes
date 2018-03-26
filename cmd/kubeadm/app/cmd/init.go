/*
Copyright 2016 The Kubernetes Authors.

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

package cmd

import (
	"fmt"
	"html/template"
	"io"
	"strings"

	"github.com/renstrom/dedent"
	"github.com/spf13/cobra"
	flag "github.com/spf13/pflag"

	"k8s.io/apimachinery/pkg/util/sets"
	kubeadmapiext "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1alpha1"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/validation"
	cmdutil "k8s.io/kubernetes/cmd/kubeadm/app/cmd/util"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/util/factory"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/util/phases"
	"k8s.io/kubernetes/cmd/kubeadm/app/features"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
)

var (
	initDoneTemplate = template.Must(template.New("init").Parse(dedent.Dedent(`
		Your Kubernetes master has initialized successfully!

		To start using your cluster, you need to run the following as a regular user:

		  mkdir -p $HOME/.kube
		  sudo cp -i {{.KubeConfigPath}} $HOME/.kube/config
		  sudo chown $(id -u):$(id -g) $HOME/.kube/config

		You should now deploy a pod network to the cluster.
		Run "kubectl apply -f [podnetwork].yaml" with one of the options listed at:
		  https://kubernetes.io/docs/concepts/cluster-administration/addons/

		You can now join any number of machines by running the following on each node
		as root:

		  {{.joinCommand}}

		`)))

	kubeletFailTemplate = template.Must(template.New("init").Parse(dedent.Dedent(`
		Unfortunately, an error has occurred:
			{{ .Error }}

		This error is likely caused by:
			- The kubelet is not running
			- The kubelet is unhealthy due to a misconfiguration of the node in some way (required cgroups disabled)
			- Either there is no internet connection, or imagePullPolicy is set to "Never",
			  so the kubelet cannot pull or find the following control plane images:
				- {{ .APIServerImage }}
				- {{ .ControllerManagerImage }}
				- {{ .SchedulerImage }}
				- {{ .EtcdImage }} (only if no external etcd endpoints are configured)

		If you are on a systemd-powered system, you can try to troubleshoot the error with the following commands:
			- 'systemctl status kubelet'
			- 'journalctl -xeu kubelet'
		`)))
)

// initContext define the execution context for the init command.
// All the methods implementing the phases use this object as a receiver, so all the attributes
// of this object will be passed through the init phases.
type initContext struct {
	// init input parameters
	out io.Writer

	// init flags not overlapping with the masterConfiguration object
	dryRun                bool
	skipPreFlight         bool
	skipTokenPrint        bool
	ignorePreflightErrors sets.String

	// factories with the responsibility to create/and store accessory components used during the phases execution
	// such components will are accessible to any phase as a methods added via composition to the initContext
	factory.MasterConfigurationFactory
	factory.ClientFactory
	factory.WaiterFactory

	// worklow flags to be shared across init phases
	usingExternalCAEvaluated bool
	usingExternalCA          bool
}

// NewCmdInit returns "kubeadm init" command.
func NewCmdInit(out io.Writer) *cobra.Command {
	context := &initContext{out: out}

	var cfgPath string
	var featureGatesString string
	var ignorePreflightErrors []string
	var cfg = &kubeadmapiext.MasterConfiguration{}
	legacyscheme.Scheme.Default(cfg)

	// creates a PhasedCommandBuilder, that allows to build customized cobra.Command configured
	// for supporting execution of the entire logic or only specific phases/atomic parts of the init logic.
	var cmdBuilder = &phases.PhasedCommandBuilder{
		Use:   "init",
		Short: "Run this command in order to set up the Kubernetes master.",
		// Phases provides the definition of the command workflow as a sequence of ordered phases
		Phases: context.initWorkflow(),
		// RunPhases provide the function the executes the entire command logic (all the phases) or
		// or only specific phases selected by the user.
		// For init, we are overriding the default RunPhases method in order to provide a set of
		// actions (e.g. flag validation) that will be always executed before and after the execution
		// of entire init workflow or the execution of the selected phases
		RunPhases: func(cmd *cobra.Command, phasesToRun phases.PhaseWorkflow, args []string) error {
			var err error

			// If --config file is passed, errors if any overlapping flag is used too (mixedArguments)
			err = validation.ValidateMixedArguments(cmd.Flags())
			kubeadmutil.CheckErr(err)

			// parses the IgnorePreflightErrors strings
			context.ignorePreflightErrors, err = validation.ValidateIgnorePreflightErrors(ignorePreflightErrors, context.skipPreFlight)
			kubeadmutil.CheckErr(err)

			// parses the featureGatesString into a map[string]bool as expected by the cfg object
			cfg.FeatureGates, err = features.NewFeatureGate(&features.InitFeatureGates, featureGatesString)
			kubeadmutil.CheckErr(err)

			// If the kubernetes version is not used in the phasesToRun, inhibits kubernetes version lookup
			// (an unnecessary access to internet in those cases) - by setting it to a default value -
			context.inhibitVersionLookupWhenPossible(phasesToRun, cfg)

			// inits the masterConfiguration starting from the v1alpha1 configuration passed to the command;
			// it includes also a validation of all the configuration entry and defaulting for missing values
			err = context.InitMasterConfiguration(cfg, cfgPath, context.dryRun)
			kubeadmutil.CheckErr(err)

			// Executes the entire init workflow or the selected phases
			err = phases.DefaultPhaseExecutor(cmd, phasesToRun, args)
			kubeadmutil.CheckErr(err)

			// In case a token was created, but not join message printed, prints it
			if phasesToRun.HasByArgOrAlias("bootstrap-token/token") && !phasesToRun.HasByArgOrAlias("init-completed") {
				err = context.printJoinMessage()
				kubeadmutil.CheckErr(err)
			}

			return nil
		},
	}

	// Builds the customized cobra.Command configured above
	cmd := cmdBuilder.MustBuild()

	// Add flags to the cobra.Command
	AddInitConfigFlags(cmd.PersistentFlags(), cfg, &featureGatesString)
	AddInitOtherFlags(cmd.PersistentFlags(), &cfgPath, &context.skipPreFlight, &context.skipTokenPrint, &context.dryRun, &ignorePreflightErrors)

	return cmd
}

// AddInitConfigFlags adds init flags bound to the config to the specified flagset
func AddInitConfigFlags(flagSet *flag.FlagSet, cfg *kubeadmapiext.MasterConfiguration, featureGatesString *string) {
	flagSet.StringVar(
		&cfg.API.AdvertiseAddress, "apiserver-advertise-address", cfg.API.AdvertiseAddress,
		"The IP address the API Server will advertise it's listening on. Specify '0.0.0.0' to use the address of the default network interface.",
	)
	flagSet.Int32Var(
		&cfg.API.BindPort, "apiserver-bind-port", cfg.API.BindPort,
		"Port for the API Server to bind to.",
	)
	flagSet.StringVar(
		&cfg.Networking.ServiceSubnet, "service-cidr", cfg.Networking.ServiceSubnet,
		"Use alternative range of IP address for service VIPs.",
	)
	flagSet.StringVar(
		&cfg.Networking.PodSubnet, "pod-network-cidr", cfg.Networking.PodSubnet,
		"Specify range of IP addresses for the pod network. If set, the control plane will automatically allocate CIDRs for every node.",
	)
	flagSet.StringVar(
		&cfg.Networking.DNSDomain, "service-dns-domain", cfg.Networking.DNSDomain,
		`Use alternative domain for services, e.g. "myorg.internal".`,
	)
	flagSet.StringVar(
		&cfg.KubernetesVersion, "kubernetes-version", cfg.KubernetesVersion,
		`Choose a specific Kubernetes version for the control plane.`,
	)
	flagSet.StringVar(
		&cfg.CertificatesDir, "cert-dir", cfg.CertificatesDir,
		`The path where to save and store the certificates.`,
	)
	flagSet.StringSliceVar(
		&cfg.APIServerCertSANs, "apiserver-cert-extra-sans", cfg.APIServerCertSANs,
		`Optional extra Subject Alternative Names (SANs) to use for the API Server serving certificate. Can be both IP addresses and DNS names.`,
	)
	flagSet.StringVar(
		&cfg.NodeName, "node-name", cfg.NodeName,
		`Specify the node name.`,
	)
	flagSet.StringVar(
		&cfg.Token, "token", cfg.Token,
		"The token to use for establishing bidirectional trust between nodes and masters.",
	)
	flagSet.DurationVar(
		&cfg.TokenTTL.Duration, "token-ttl", cfg.TokenTTL.Duration,
		"The duration before the bootstrap token is automatically deleted. If set to '0', the token will never expire.",
	)
	flagSet.StringVar(
		&cfg.CRISocket, "cri-socket", cfg.CRISocket,
		`Specify the CRI socket to connect to.`,
	)
	flagSet.StringVar(featureGatesString, "feature-gates", *featureGatesString, "A set of key=value pairs that describe feature gates for various features. "+
		"Options are:\n"+strings.Join(features.KnownFeatures(&features.InitFeatureGates), "\n"))

}

// AddInitOtherFlags adds init flags that are not bound to a configuration file to the given flagset
func AddInitOtherFlags(flagSet *flag.FlagSet, cfgPath *string, skipPreFlight, skipTokenPrint, dryRun *bool, ignorePreflightErrors *[]string) {
	flagSet.StringVar(
		cfgPath, "config", *cfgPath,
		"Path to kubeadm config file. WARNING: Usage of a configuration file is experimental.",
	)
	flagSet.StringSliceVar(
		ignorePreflightErrors, "ignore-preflight-errors", *ignorePreflightErrors,
		"A list of checks whose errors will be shown as warnings. Example: 'IsPrivilegedUser,Swap'. Value 'all' ignores errors from all checks.",
	)
	// Note: All flags that are not bound to the cfg object should be whitelisted in cmd/kubeadm/app/apis/kubeadm/validation/validation.go
	flagSet.BoolVar(
		skipPreFlight, "skip-preflight-checks", *skipPreFlight,
		"Skip preflight checks which normally run before modifying the system.",
	)
	flagSet.MarkDeprecated("skip-preflight-checks", "it is now equivalent to --ignore-preflight-errors=all")
	// Note: All flags that are not bound to the cfg object should be whitelisted in cmd/kubeadm/app/apis/kubeadm/validation/validation.go
	flagSet.BoolVar(
		skipTokenPrint, "skip-token-print", *skipTokenPrint,
		"Skip printing of the default bootstrap token generated by 'kubeadm init'.",
	)
	// Note: All flags that are not bound to the cfg object should be whitelisted in cmd/kubeadm/app/apis/kubeadm/validation/validation.go
	flagSet.BoolVar(
		dryRun, "dry-run", *dryRun,
		"Don't apply any changes; just output what would be done.",
	)
}

// inhibitVersionLookupWhenPossible checks if the kubernetesVersion will be used by the requested phaseToRun;
// if not, it sets the kubernetesVersion explicitly to avoid the lookup of the version from the internet
func (c *initContext) inhibitVersionLookupWhenPossible(phasesToRun phases.PhaseWorkflow, v1alpha1Cfg *kubeadmapiext.MasterConfiguration) {
	// If at least a featureGate feature is requested, returns because the kubernetesVersion will be required for checking the compatibility of
	// features resquested with the chosen kubernetesVersion
	if len(v1alpha1Cfg.FeatureGates) > 0 {
		return
	}

	// Returns if at least one of the phasesToRun requires the kubernetesVersion
	for _, p := range phasesUsingKubernetesVersion {
		if phasesToRun.HasByArgOrAlias(p) {
			return
		}
	}

	// Otherwise, sets the kubernetesVersion explicitly to avoid the lookup of the version from the internet
	v1alpha1Cfg.KubernetesVersion = "v1.11.0"
}

// printJoinMessage prints a simplified join message in case a token was create but the init complete message was not printed
func (c *initContext) printJoinMessage() error {

	// Gets the join command
	fmt.Printf("[bootstraptoken] You can now join any number of machines by running the following on each node	as root:\n\n")
	joinCommand, err := cmdutil.GetJoinCommand(c.AdminKubeConfigPath(), c.MasterConfiguration().Token, c.skipTokenPrint)
	if err != nil {
		return fmt.Errorf("failed to get join command: %v", err)
	}
	fmt.Printf("  %s\n\n", joinCommand)

	return nil
}
