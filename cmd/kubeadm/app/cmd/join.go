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
	"path/filepath"

	"github.com/renstrom/dedent"
	"github.com/spf13/cobra"

	"k8s.io/apimachinery/pkg/runtime"
	certutil "k8s.io/client-go/util/cert"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapiext "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1alpha1"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/validation"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/discovery"
	"k8s.io/kubernetes/cmd/kubeadm/app/preflight"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	kubeconfigutil "k8s.io/kubernetes/cmd/kubeadm/app/util/kubeconfig"
	"k8s.io/kubernetes/pkg/api"
	nodeutil "k8s.io/kubernetes/pkg/util/node"
)

var (
	joinDoneMsgf = dedent.Dedent(`
		Node join complete:
		* Certificate signing request sent to master and response
		  received.
		* Kubelet informed of new secure connection details.

		Run 'kubectl get nodes' on the master to see this machine join.
		`)
)

// NewCmdJoin returns "kubeadm join" command.
func NewCmdJoin(out io.Writer) *cobra.Command {
	cfg := &kubeadmapiext.NodeConfiguration{}
	api.Scheme.Default(cfg)

	var skipPreFlight bool
	var cfgPath string

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

		If you use a shared token for discovery, you should also pass the
		--discovery-token-ca-cert-hash flag to validate the public key of the
		root certificate authority (CA) presented by the Kubernetes Master. The
		value of this flag is specified as "<hash-type>:<hex-encoded-value>",
		where the supported hash type is "sha256". The hash is calculated over
		the bytes of the Subject Public Key Info (SPKI) object (as in RFC7469).
		This value is available in the output of "kubeadm init" or can be
		calcuated using standard tools. The --discovery-token-ca-cert-hash flag
		may be repeated multiple times to allow more than one public key.

		If you cannot know the CA public key hash ahead of time, you can pass
		the --discovery-token-unsafe-skip-ca-verification flag to disable this
		verification. This weakens the kubeadm security model since other nodes
		can potentially impersonate the Kubernetes Master.

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

			j, err := NewJoin(cfgPath, args, internalcfg, skipPreFlight)
			kubeadmutil.CheckErr(err)
			kubeadmutil.CheckErr(j.Validate(cmd))
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
		&cfg.NodeName, "node-name", "",
		"Specify the node name")
	cmd.PersistentFlags().StringVar(
		&cfg.TLSBootstrapToken, "tls-bootstrap-token", "",
		"A token used for TLS bootstrapping")
	cmd.PersistentFlags().StringSliceVar(
		&cfg.DiscoveryTokenCACertHashes, "discovery-token-ca-cert-hash", []string{},
		"For token-based discovery, validate that the root CA public key matches this hash (format: \"<type>:<value>\").")
	cmd.PersistentFlags().BoolVar(
		&cfg.DiscoveryTokenUnsafeSkipCAVerification, "discovery-token-unsafe-skip-ca-verification", false,
		"For token-based discovery, allow joining without --discovery-token-ca-cert-hash pinning.")

	cmd.PersistentFlags().StringVar(
		&cfg.Token, "token", "",
		"Use this token for both discovery-token and tls-bootstrap-token")

	cmd.PersistentFlags().BoolVar(
		&skipPreFlight, "skip-preflight-checks", false,
		"Skip preflight checks normally run before modifying the system",
	)

	return cmd
}

type Join struct {
	cfg *kubeadmapi.NodeConfiguration
}

func NewJoin(cfgPath string, args []string, cfg *kubeadmapi.NodeConfiguration, skipPreFlight bool) (*Join, error) {
	fmt.Println("[kubeadm] WARNING: kubeadm is in beta, please do not use it for production clusters.")

	if cfg.NodeName == "" {
		cfg.NodeName = nodeutil.GetHostname("")
	}

	if cfgPath != "" {
		b, err := ioutil.ReadFile(cfgPath)
		if err != nil {
			return nil, fmt.Errorf("unable to read config from %q [%v]", cfgPath, err)
		}
		if err := runtime.DecodeInto(api.Codecs.UniversalDecoder(), b, cfg); err != nil {
			return nil, fmt.Errorf("unable to decode config from %q [%v]", cfgPath, err)
		}
	}

	if !skipPreFlight {
		fmt.Println("[preflight] Running pre-flight checks")

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

func (j *Join) Validate(cmd *cobra.Command) error {
	if err := validation.ValidateMixedArguments(cmd.PersistentFlags()); err != nil {
		return err
	}
	return validation.ValidateNodeConfiguration(j.cfg).ToAggregate()
}

// Run executes worker node provisioning and tries to join an existing cluster.
func (j *Join) Run(out io.Writer) error {
	cfg, err := discovery.For(j.cfg)
	if err != nil {
		return err
	}

	kubeconfigFile := filepath.Join(kubeadmconstants.KubernetesDir, kubeadmconstants.KubeletBootstrapKubeConfigFileName)
	if err := kubeconfigutil.WriteToDisk(kubeconfigFile, cfg); err != nil {
		return err
	}

	// Write the ca certificate to disk so kubelet can use it for authentication
	cluster := cfg.Contexts[cfg.CurrentContext].Cluster
	err = certutil.WriteCert(j.cfg.CACertPath, cfg.Clusters[cluster].CertificateAuthorityData)
	if err != nil {
		return fmt.Errorf("couldn't save the CA certificate to disk: %v", err)
	}

	fmt.Fprintf(out, joinDoneMsgf)
	return nil
}
