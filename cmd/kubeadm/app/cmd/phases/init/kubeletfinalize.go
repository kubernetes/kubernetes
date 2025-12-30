/*
Copyright 2019 The Kubernetes Authors.

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
	"os"
	"path/filepath"

	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/klog/v2"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/options"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/phases/workflow"
	cmdutil "k8s.io/kubernetes/cmd/kubeadm/app/cmd/util"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	kubeletphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/kubelet"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/errors"
)

var (
	kubeletFinalizePhaseExample = cmdutil.Examples(`
		# Updates settings relevant to the kubelet after TLS bootstrap
		kubeadm init phase kubelet-finalize all --config
		`)
)

// NewKubeletFinalizePhase creates a kubeadm workflow phase that updates settings
// relevant to the kubelet after TLS bootstrap.
func NewKubeletFinalizePhase() workflow.Phase {
	return workflow.Phase{
		Name:    "kubelet-finalize",
		Short:   "Updates settings relevant to the kubelet after TLS bootstrap",
		Example: kubeletFinalizePhaseExample,
		Phases: []workflow.Phase{
			{
				Name:           "all",
				Short:          "Run all kubelet-finalize phases",
				InheritFlags:   []string{options.CfgPath, options.CertificatesDir, options.DryRun},
				Example:        kubeletFinalizePhaseExample,
				RunAllSiblings: true,
			},
			{
				Name:         "enable-client-cert-rotation",
				Short:        "Enable kubelet client certificate rotation",
				InheritFlags: []string{options.CfgPath, options.CertificatesDir, options.DryRun},
				Run:          runKubeletFinalizeEnableClientCertRotation,
			},
		},
	}
}

// runKubeletFinalizeEnableClientCertRotation detects if the kubelet certificate rotation is enabled
// and updates the kubelet.conf file to point to a rotatable certificate and key for the
// Node user.
func runKubeletFinalizeEnableClientCertRotation(c workflow.RunData) error {
	data, ok := c.(InitData)
	if !ok {
		return errors.New("kubelet-finalize phase invoked with an invalid data struct")
	}

	// Check if the user has added the kubelet --cert-dir flag.
	// If yes, use that path, else use the kubeadm provided value.
	cfg := data.Cfg()
	pkiPath := filepath.Join(data.KubeletDir(), "pki")
	val, idx := kubeadmapi.GetArgValue(cfg.NodeRegistration.KubeletExtraArgs, "cert-dir", -1)
	if idx > -1 {
		pkiPath = val
	}

	// Check for the existence of the kubelet-client-current.pem file in the kubelet certificate directory.
	rotate := false
	pemPath := filepath.Join(pkiPath, "kubelet-client-current.pem")
	if _, err := os.Stat(pemPath); err == nil {
		klog.V(1).Infof("[kubelet-finalize] Assuming that kubelet client certificate rotation is enabled: found %q", pemPath)
		rotate = true
	} else {
		klog.V(1).Infof("[kubelet-finalize] Assuming that kubelet client certificate rotation is disabled: %v", err)
	}

	// Exit early if rotation is disabled.
	if !rotate {
		return nil
	}

	kubeconfigPath := filepath.Join(kubeadmconstants.KubernetesDir, kubeadmconstants.KubeletKubeConfigFileName)
	fmt.Printf("[kubelet-finalize] Updating %q to point to a rotatable kubelet client certificate and key\n", kubeconfigPath)

	// Exit early if dry-running is enabled.
	if data.DryRun() {
		return nil
	}

	// Load the kubeconfig from disk.
	kubeconfig, err := clientcmd.LoadFromFile(kubeconfigPath)
	if err != nil {
		return errors.Wrapf(err, "could not load %q", kubeconfigPath)
	}

	// Perform basic validation. The errors here can only happen if the kubelet.conf was corrupted.
	if len(kubeconfig.CurrentContext) == 0 {
		return errors.Errorf("the file %q does not have current context set", kubeconfigPath)
	}
	currentContext, ok := kubeconfig.Contexts[kubeconfig.CurrentContext]
	if !ok {
		return errors.Errorf("the file %q is not a valid kubeconfig: %q set as current-context, but not found in context list", kubeconfigPath, kubeconfig.CurrentContext)
	}
	userName := currentContext.AuthInfo
	if len(userName) == 0 {
		return errors.Errorf("the file %q is not a valid kubeconfig: empty username for current context", kubeconfigPath)
	}
	info, ok := kubeconfig.AuthInfos[userName]
	if !ok {
		return errors.Errorf("the file %q does not contain authentication for user %q", kubeconfigPath, cfg.NodeRegistration.Name)
	}

	// Update the client certificate and key of the node authorizer to point to the PEM symbolic link.
	info.ClientKeyData = []byte{}
	info.ClientCertificateData = []byte{}
	info.ClientKey = pemPath
	info.ClientCertificate = pemPath

	// Writes the kubeconfig back to disk.
	if err = clientcmd.WriteToFile(*kubeconfig, kubeconfigPath); err != nil {
		return errors.Wrapf(err, "failed to serialize %q", kubeconfigPath)
	}

	// Restart the kubelet.
	klog.V(1).Info("[kubelet-finalize] Restarting the kubelet to enable client certificate rotation")
	kubeletphase.TryRestartKubelet()

	return nil
}
