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
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"
	"text/template"

	"github.com/renstrom/dedent"
	"github.com/spf13/cobra"

	"k8s.io/apimachinery/pkg/runtime"
	apivalidation "k8s.io/apimachinery/pkg/util/validation"
	certutil "k8s.io/client-go/util/cert"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapiext "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1alpha1"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/validation"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/discovery"
	kubenode "k8s.io/kubernetes/cmd/kubeadm/app/node"
	"k8s.io/kubernetes/cmd/kubeadm/app/preflight"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	kubeconfigutil "k8s.io/kubernetes/cmd/kubeadm/app/util/kubeconfig"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/util/initsystem"
	utiltaints "k8s.io/kubernetes/pkg/util/taints"
)

var (
	joinDoneMsgf = dedent.Dedent(`
		Node join complete:
		* Certificate signing request sent to master and response
		  received.
		* Kubelet informed of new secure connection details.

		Run 'kubectl get nodes' on the master to see this machine join.
		`)
	kubeletDropIn = template.Must(template.New("dropIn").Parse(dedent.Dedent(`
		[Service]
		Environment="KUBELET_EXTRA_ARGS={{if .Labels}}--node-labels={{.Labels}} {{end}}{{if .Taints}}--register-with-taints={{.Taints}}{{end}}"
		`)))
	kubeletManualSettings = template.Must(template.New("dropIn").Parse(dedent.Dedent(`
		Kubelet should be started manually with following options:
		{{if .Labels}} --node-labels={{.Labels}}{{end}}
		{{if .Taints}} --register-with-taints={{.Taints}}{{end}}"
		`)))
)

// NewCmdJoin returns "kubeadm join" command.
func NewCmdJoin(out io.Writer) *cobra.Command {
	cfg := &kubeadmapiext.NodeConfiguration{}
	api.Scheme.Default(cfg)

	var skipPreFlight bool
	var cfgPath string
	var taints []string
	var labels []string

	cmd := &cobra.Command{
		Use:   "join <flags> [DiscoveryTokenAPIServers]",
		Short: "Run this on any machine you wish to join an existing cluster",
		Long: dedent.Dedent(`
		When joining a kubeadm initialized cluster, we need to establish 
		bidirectional trust. This is split into discovery (having the Node 
		trust the Kubernetes Master) and TLS bootstrap (having the Kubernetes 
		Master trust the Node).

		There are 2 main schemes for discovery. The first is to use a shared 
		token along with the IP address of the API server. The second is to 
		provide a file (a subset of the standard kubeconfig file). This file 
		can be a local file or downloaded via an HTTPS URL. The forms are 
		kubeadm join --discovery-token abcdef.1234567890abcdef 1.2.3.4:6443, 
		kubeadm join --discovery-file path/to/file.conf, or kubeadm join
		--discovery-file https://url/file.conf. Only one form can be used. If 
		the discovery information is loaded from a URL, HTTPS must be used and 
		the host installed CA bundle is used to verify the connection.

		The TLS bootstrap mechanism is also driven via a shared token. This is 
		used to temporarily authenticate with the Kubernetes Master to submit a
		certificate signing request (CSR) for a locally created key pair. By 
		default kubeadm will set up the Kubernetes Master to automatically 
		approve these signing requests. This token is passed in with the 
		--tls-bootstrap-token abcdef.1234567890abcdef flag.

		Often times the same token is used for both parts. In this case, the
		--token flag can be used instead of specifying each token individually.
		`),
		Run: func(cmd *cobra.Command, args []string) {
			cfg.DiscoveryTokenAPIServers = args

			api.Scheme.Default(cfg)
			internalcfg := &kubeadmapi.NodeConfiguration{}
			api.Scheme.Convert(cfg, internalcfg, nil)

			j, err := NewJoin(cfgPath, args, internalcfg, skipPreFlight, labels, taints)
			kubeadmutil.CheckErr(err)
			kubeadmutil.CheckErr(j.Validate())
			kubeadmutil.CheckErr(j.Run(out))
		},
	}

	cmd.PersistentFlags().StringVar(
		&cfgPath, "config", cfgPath,
		"Path to kubeadm config file")

	cmd.PersistentFlags().StringVar(
		&cfg.DiscoveryFile, "discovery-file", "",
		"A file or url from which to load cluster information")
	cmd.PersistentFlags().StringVar(
		&cfg.DiscoveryToken, "discovery-token", "",
		"A token used to validate cluster information fetched from the master")
	cmd.PersistentFlags().StringVar(
		&cfg.TLSBootstrapToken, "tls-bootstrap-token", "",
		"A token used for TLS bootstrapping")
	cmd.PersistentFlags().StringVar(
		&cfg.Token, "token", "",
		"Use this token for both discovery-token and tls-bootstrap-token")
	cmd.PersistentFlags().StringSliceVar(
		&taints, "register-with-taints", taints,
		`Optional taints to add to the node when the node registers itself.`,
	)
	cmd.PersistentFlags().StringSliceVar(
		&labels, "node-labels", labels,
		`Optional labels to add to the node when the node registers itself.`,
	)
	cmd.PersistentFlags().BoolVar(
		&skipPreFlight, "skip-preflight-checks", false,
		"Skip preflight checks normally run before modifying the system",
	)

	return cmd
}

type Join struct {
	cfg *kubeadmapi.NodeConfiguration
}

func NewJoin(cfgPath string, args []string, cfg *kubeadmapi.NodeConfiguration, skipPreFlight bool, labels []string, taints []string) (*Join, error) {
	fmt.Println("[kubeadm] WARNING: kubeadm is in beta, please do not use it for production clusters.")

	if cfgPath != "" {
		b, err := ioutil.ReadFile(cfgPath)
		if err != nil {
			return nil, fmt.Errorf("unable to read config from %q [%v]", cfgPath, err)
		}
		if err := runtime.DecodeInto(api.Codecs.UniversalDecoder(), b, cfg); err != nil {
			return nil, fmt.Errorf("unable to decode config from %q [%v]", cfgPath, err)
		}
	}

	labelObj, err := parseLabels(labels)
	if err != nil {
		return nil, fmt.Errorf("unable to parse labels %s [%v]", cfgPath, err)
	}
	cfg.Labels = labelObj

	taintsObj, err := parseTaints(taints)
	if err != nil {
		return nil, fmt.Errorf("unable to parse taints %s [%v]", cfgPath, err)
	}
	cfg.Taints = taintsObj

	if !skipPreFlight {
		fmt.Println("[preflight] Running pre-flight checks")

		// First, check if we're root separately from the other preflight checks and fail fast
		if err := preflight.RunRootCheckOnly(); err != nil {
			return nil, err
		}

		// Then continue with the others...
		if err := preflight.RunJoinNodeChecks(cfg); err != nil {
			return nil, err
		}

		// Try to start the kubelet service in case it's inactive
		preflight.TryStartKubelet()
	} else {
		fmt.Println("[preflight] Skipping pre-flight checks")
	}

	return &Join{cfg: cfg}, nil
}

func (j *Join) Validate() error {
	return validation.ValidateNodeConfiguration(j.cfg).ToAggregate()
}

// Run executes worker node provisioning and tries to join an existing cluster.
func (j *Join) Run(out io.Writer) error {

	cfg, err := discovery.For(j.cfg)
	if err != nil {
		return err
	}

	hostname, err := os.Hostname()
	if err != nil {
		return err
	}
	client, err := kubeconfigutil.KubeConfigToClientSet(cfg)
	if err != nil {
		return err
	}
	if err := kubenode.ValidateAPIServer(client); err != nil {
		return err
	}

	if err := updateKubeletConfig(out, j.cfg); err != nil {
		return err
	}

	if err := kubenode.PerformTLSBootstrap(cfg, hostname); err != nil {
		return err
	}

	kubeconfigFile := filepath.Join(kubeadmapi.GlobalEnvParams.KubernetesDir, kubeadmconstants.KubeletKubeConfigFileName)
	if err := kubeconfigutil.WriteToDisk(kubeconfigFile, cfg); err != nil {
		return err
	}

	// Write the ca certificate to disk so kubelet can use it for authentication
	cluster := cfg.Contexts[cfg.CurrentContext].Cluster
	err = certutil.WriteCert(j.cfg.CACertPath, cfg.Clusters[cluster].CertificateAuthorityData)
	if err != nil {
		return fmt.Errorf("couldn't save the CA certificate to disk [%v]", err)
	}

	fmt.Fprintf(out, joinDoneMsgf)
	return nil
}

func parseLabels(spec []string) (map[string]string, error) {
	labels := map[string]string{}
	for _, labelSpec := range spec {
		if strings.Index(labelSpec, "=") != -1 {
			parts := strings.Split(labelSpec, "=")
			if len(parts) != 2 {
				return nil, fmt.Errorf("invalid label spec: %v", labelSpec)
			}
			if errs := apivalidation.IsValidLabelValue(parts[1]); len(errs) != 0 {
				return nil, fmt.Errorf("invalid label value: %q: %s", labelSpec, strings.Join(errs, ";"))
			}
			labels[parts[0]] = parts[1]
		} else {
			return nil, fmt.Errorf("unknown label spec: %v", labelSpec)
		}
	}
	return labels, nil
}

func parseTaints(spec []string) ([]v1.Taint, error) {
	var taints []v1.Taint

	for _, taintSpec := range spec {
		if strings.Index(taintSpec, "=") != -1 && strings.Index(taintSpec, ":") != -1 {
			newTaint, err := utiltaints.ParseTaint(taintSpec)
			if err != nil {
				return nil, err
			}
			taints = append(taints, newTaint)
		} else {
			return nil, fmt.Errorf("unknown taint spec: %v", taintSpec)
		}
	}
	return taints, nil
}

// TODO: move into a separate file e.g. app/node/kubelet.go
func updateKubeletConfig(out io.Writer, cfg *kubeadmapi.NodeConfiguration) error {

	// it si required to updateKubeletConfig only if labels or taints should be applied
	if len(cfg.Labels) > 0 || len(cfg.Taints) > 0 {

		//converts label and taints back to strings
		var labels, taints []string

		for label, value := range cfg.Labels {
			labels = append(labels, fmt.Sprintf("%s=%s", label, value))
		}

		for _, taint := range cfg.Taints {
			taints = append(taints, fmt.Sprintf("%s=%s:%s", taint.Key, taint.Value, taint.Effect))
		}

		ctx := map[string]string{
			"Labels": strings.Join(labels, ","),
			"Taints": strings.Join(taints, ","),
		}

		// check if there is a supported init system
		initSystem, err := initsystem.GetInitSystem()
		if err != nil {
			// if not, prints a message with instructions for manual configuration
			fmt.Println("[kubelet] WARNING: No supported init system detected, won't ensure kubelet is set with taints and labels.")
			err = kubeletManualSettings.Execute(out, ctx)
			if err != nil {
				return fmt.Errorf("failed to printing kubelet manual settings instructions [%v]", err)
			}
		} else {
			// if there is a managed init system,
			// checks if kubelet service is in place
			if !initSystem.ServiceExists("kubelet") {
				return fmt.Errorf("failure to set kubelet labels and taints, kubelet service is not installed")
			}

			// creates a new dop-in configuration file into the kubelet service configuration
			filename := "/etc/systemd/system/kubelet.service.d/20-labels-taints.conf"
			fmt.Printf("[kubelet] Creating %s\n", filename)
			err := createKubeletDropIn(filename, ctx)
			if err != nil {
				return fmt.Errorf("failure saving kubelet configuration in %s [%v]", filename, err)
			}

			// forces reloading of the init system configuration
			fmt.Println("[kubelet] Reloading init system configuration")
			err = initSystem.DaemonReload()
			if err != nil {
				return fmt.Errorf("failure reloading init system configuration [%v]", err)
			}

			// if the kubelet service is active, force restarts in order to make new settings effective
			// NB. this is considered safe, because this command is issued before kubelet providing
			// the kubeconfig file (and thus before the node joining the cluster)
			if initSystem.ServiceIsActive("kubelet") {
				fmt.Println("[kubelet] Restarting the kubelet service")
				if err := initSystem.ServiceRestart("kubelet"); err != nil {
					fmt.Printf("[kubelet] WARNING: Unable to restart start the kubelet service: [%v]\n", err)
				}
			}
		}
	}

	return nil
}

func createKubeletDropIn(filename string, ctx map[string]string) error {
	if err := os.MkdirAll(filepath.Dir(filename), os.FileMode(0755)); err != nil {
		return err
	}

	f, err := os.OpenFile(filename, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, os.FileMode(0600))
	if err != nil {
		return err
	}

	err = kubeletDropIn.Execute(f, ctx)
	if err != nil {
		return err
	}

	if err = f.Close(); err == nil {
		return err
	}

	return nil
}
