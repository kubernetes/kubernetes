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
	"bytes"
	"fmt"
	"text/template"

	"github.com/lithammer/dedent"
	"github.com/pkg/errors"
	"k8s.io/klog"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/options"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/phases/workflow"
	cmdutil "k8s.io/kubernetes/cmd/kubeadm/app/cmd/util"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/certs"
	"k8s.io/kubernetes/cmd/kubeadm/app/preflight"
	utilsexec "k8s.io/utils/exec"
)

var (
	preflightExample = cmdutil.Examples(`
		# Run join pre-flight checks using a config file.
		kubeadm join phase preflight --config kubeadm-config.yml
		`)

	notReadyToJoinControlPlaneTemp = template.Must(template.New("join").Parse(dedent.Dedent(`
		One or more conditions for hosting a new control plane instance is not satisfied.

		{{.Error}}

		Please ensure that:
		* The cluster has a stable controlPlaneEndpoint address.
		* The certificates that must be shared among control plane instances are provided.

		`)))
)

// NewPreflightPhase creates a kubeadm workflow phase that implements preflight checks for a new node join
func NewPreflightPhase() workflow.Phase {
	return workflow.Phase{
		Name:    "preflight [api-server-endpoint]",
		Short:   "Run join pre-flight checks",
		Long:    "Run pre-flight checks for kubeadm join.",
		Example: preflightExample,
		Run:     runPreflight,
		InheritFlags: []string{
			options.CfgPath,
			options.IgnorePreflightErrors,
			options.TLSBootstrapToken,
			options.TokenStr,
			options.ControlPlane,
			options.APIServerAdvertiseAddress,
			options.APIServerBindPort,
			options.NodeCRISocket,
			options.NodeName,
			options.FileDiscovery,
			options.TokenDiscovery,
			options.TokenDiscoveryCAHash,
			options.TokenDiscoverySkipCAHash,
			options.CertificateKey,
		},
	}
}

// runPreflight executes preflight checks logic.
func runPreflight(c workflow.RunData) error {
	j, ok := c.(JoinData)
	if !ok {
		return errors.New("preflight phase invoked with an invalid data struct")
	}
	fmt.Println("[preflight] Running pre-flight checks")

	// Start with general checks
	klog.V(1).Infoln("[preflight] Running general checks")
	if err := preflight.RunJoinNodeChecks(utilsexec.New(), j.Cfg(), j.IgnorePreflightErrors()); err != nil {
		return err
	}

	initCfg, err := j.InitCfg()
	if err != nil {
		return err
	}

	// Continue with more specific checks based on the init configuration
	klog.V(1).Infoln("[preflight] Running configuration dependant checks")
	if j.Cfg().ControlPlane != nil {
		// Checks if the cluster configuration supports
		// joining a new control plane instance and if all the necessary certificates are provided
		hasCertificateKey := len(j.CertificateKey()) > 0
		if err := checkIfReadyForAdditionalControlPlane(&initCfg.ClusterConfiguration, hasCertificateKey); err != nil {
			// outputs the not ready for hosting a new control plane instance message
			ctx := map[string]string{
				"Error": err.Error(),
			}

			var msg bytes.Buffer
			notReadyToJoinControlPlaneTemp.Execute(&msg, ctx)
			return errors.New(msg.String())
		}

		// run kubeadm init preflight checks for checking all the prerequisites
		fmt.Println("[preflight] Running pre-flight checks before initializing the new control plane instance")

		if err := preflight.RunInitNodeChecks(utilsexec.New(), initCfg, j.IgnorePreflightErrors(), true, hasCertificateKey); err != nil {
			return err
		}

		fmt.Println("[preflight] Pulling images required for setting up a Kubernetes cluster")
		fmt.Println("[preflight] This might take a minute or two, depending on the speed of your internet connection")
		fmt.Println("[preflight] You can also perform this action in beforehand using 'kubeadm config images pull'")
		if err := preflight.RunPullImagesCheck(utilsexec.New(), initCfg, j.IgnorePreflightErrors()); err != nil {
			return err
		}
	}
	return nil
}

// checkIfReadyForAdditionalControlPlane ensures that the cluster is in a state that supports
// joining an additional control plane instance and if the node is ready to preflight
func checkIfReadyForAdditionalControlPlane(initConfiguration *kubeadmapi.ClusterConfiguration, hasCertificateKey bool) error {
	// blocks if the cluster was created without a stable control plane endpoint
	if initConfiguration.ControlPlaneEndpoint == "" {
		return errors.New("unable to add a new control plane instance a cluster that doesn't have a stable controlPlaneEndpoint address")
	}

	if !hasCertificateKey {
		// checks if the certificates that must be equal across controlplane instances are provided
		if ret, err := certs.SharedCertificateExists(initConfiguration); !ret {
			return err
		}
	}

	return nil
}
