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

	"github.com/lithammer/dedent"
	"github.com/pkg/errors"
	"k8s.io/apimachinery/pkg/util/wait"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	certutil "k8s.io/client-go/util/cert"
	"k8s.io/klog"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/options"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/phases/workflow"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	kubeletphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/kubelet"
	patchnodephase "k8s.io/kubernetes/cmd/kubeadm/app/phases/patchnode"
	"k8s.io/kubernetes/cmd/kubeadm/app/preflight"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/apiclient"
	kubeconfigutil "k8s.io/kubernetes/cmd/kubeadm/app/util/kubeconfig"
	utilsexec "k8s.io/utils/exec"
)

var (
	kubeadmJoinFailMsg = dedent.Dedent(`
		Unfortunately, an error has occurred:
			%v

		This error is likely caused by:
			- The kubelet is not running
			- The kubelet is unhealthy due to a misconfiguration of the node in some way (required cgroups disabled)

		If you are on a systemd-powered system, you can try to troubleshoot the error with the following commands:
			- 'systemctl status kubelet'
			- 'journalctl -xeu kubelet'
		`)
)

// NewKubeletStartPhase creates a kubeadm workflow phase that start kubelet on a node.
func NewKubeletStartPhase() workflow.Phase {
	return workflow.Phase{
		Name:  "kubelet-start [api-server-endpoint]",
		Short: "Writes kubelet settings, certificates and (re)starts the kubelet",
		Long:  "Writes a file with KubeletConfiguration and an environment file with node specific kubelet settings, and then (re)starts kubelet.",
		Run:   runKubeletStartJoinPhase,
		InheritFlags: []string{
			options.CfgPath,
			options.NodeCRISocket,
			options.NodeName,
			options.FileDiscovery,
			options.TokenDiscovery,
			options.TokenDiscoveryCAHash,
			options.TokenDiscoverySkipCAHash,
			options.TLSBootstrapToken,
			options.TokenStr,
		},
	}
}

func getKubeletStartJoinData(c workflow.RunData) (*kubeadmapi.JoinConfiguration, *kubeadmapi.InitConfiguration, *clientcmdapi.Config, error) {
	data, ok := c.(JoinData)
	if !ok {
		return nil, nil, nil, errors.New("kubelet-start phase invoked with an invalid data struct")
	}
	cfg := data.Cfg()
	initCfg, err := data.InitCfg()
	if err != nil {
		return nil, nil, nil, err
	}
	tlsBootstrapCfg, err := data.TLSBootstrapCfg()
	if err != nil {
		return nil, nil, nil, err
	}
	return cfg, initCfg, tlsBootstrapCfg, nil
}

// runKubeletStartJoinPhase executes the kubelet TLS bootstrap process.
// This process is executed by the kubelet and completes with the node joining the cluster
// with a dedicates set of credentials as required by the node authorizer
func runKubeletStartJoinPhase(c workflow.RunData) error {
	cfg, initCfg, tlsBootstrapCfg, err := getKubeletStartJoinData(c)
	if err != nil {
		return err
	}
	bootstrapKubeConfigFile := kubeadmconstants.GetBootstrapKubeletKubeConfigPath()

	// Write the bootstrap kubelet config file or the TLS-Boostrapped kubelet config file down to disk
	klog.V(1).Infoln("[kubelet-start] writing bootstrap kubelet config file at", bootstrapKubeConfigFile)
	if err := kubeconfigutil.WriteToDisk(bootstrapKubeConfigFile, tlsBootstrapCfg); err != nil {
		return errors.Wrap(err, "couldn't save bootstrap-kubelet.conf to disk")
	}

	// Write the ca certificate to disk so kubelet can use it for authentication
	cluster := tlsBootstrapCfg.Contexts[tlsBootstrapCfg.CurrentContext].Cluster
	if _, err := os.Stat(cfg.CACertPath); os.IsNotExist(err) {
		if err := certutil.WriteCert(cfg.CACertPath, tlsBootstrapCfg.Clusters[cluster].CertificateAuthorityData); err != nil {
			return errors.Wrap(err, "couldn't save the CA certificate to disk")
		}
	}

	kubeletVersion, err := preflight.GetKubeletVersion(utilsexec.New())
	if err != nil {
		return err
	}

	bootstrapClient, err := kubeconfigutil.ClientSetFromFile(bootstrapKubeConfigFile)
	if err != nil {
		return errors.Errorf("couldn't create client from kubeconfig file %q", bootstrapKubeConfigFile)
	}

	// Configure the kubelet. In this short timeframe, kubeadm is trying to stop/restart the kubelet
	// Try to stop the kubelet service so no race conditions occur when configuring it
	klog.V(1).Infoln("[kubelet-start] Stopping the kubelet")
	kubeletphase.TryStopKubelet()

	// Write the configuration for the kubelet (using the bootstrap token credentials) to disk so the kubelet can start
	if err := kubeletphase.DownloadConfig(bootstrapClient, kubeletVersion, kubeadmconstants.KubeletRunDirectory); err != nil {
		return err
	}

	// Write env file with flags for the kubelet to use. We only want to
	// register the joining node with the specified taints if the node
	// is not a control-plane. The mark-control-plane phase will register the taints otherwise.
	registerTaintsUsingFlags := cfg.ControlPlane == nil
	if err := kubeletphase.WriteKubeletDynamicEnvFile(&initCfg.ClusterConfiguration, &initCfg.NodeRegistration, registerTaintsUsingFlags, kubeadmconstants.KubeletRunDirectory); err != nil {
		return err
	}

	// Try to start the kubelet service in case it's inactive
	klog.V(1).Infoln("[kubelet-start] Starting the kubelet")
	kubeletphase.TryStartKubelet()

	// Now the kubelet will perform the TLS Bootstrap, transforming /etc/kubernetes/bootstrap-kubelet.conf to /etc/kubernetes/kubelet.conf
	// Wait for the kubelet to create the /etc/kubernetes/kubelet.conf kubeconfig file. If this process
	// times out, display a somewhat user-friendly message.
	waiter := apiclient.NewKubeWaiter(nil, kubeadmconstants.TLSBootstrapTimeout, os.Stdout)
	if err := waiter.WaitForKubeletAndFunc(waitForTLSBootstrappedClient); err != nil {
		fmt.Printf(kubeadmJoinFailMsg, err)
		return err
	}

	// When we know the /etc/kubernetes/kubelet.conf file is available, get the client
	client, err := kubeconfigutil.ClientSetFromFile(kubeadmconstants.GetKubeletKubeConfigPath())
	if err != nil {
		return err
	}

	klog.V(1).Infoln("[kubelet-start] preserving the crisocket information for the node")
	if err := patchnodephase.AnnotateCRISocket(client, cfg.NodeRegistration.Name, cfg.NodeRegistration.CRISocket); err != nil {
		return errors.Wrap(err, "error uploading crisocket")
	}

	return nil
}

// waitForTLSBootstrappedClient waits for the /etc/kubernetes/kubelet.conf file to be available
func waitForTLSBootstrappedClient() error {
	fmt.Println("[kubelet-start] Waiting for the kubelet to perform the TLS Bootstrap...")

	// Loop on every falsy return. Return with an error if raised. Exit successfully if true is returned.
	return wait.PollImmediate(kubeadmconstants.APICallRetryInterval, kubeadmconstants.TLSBootstrapTimeout, func() (bool, error) {
		// Check that we can create a client set out of the kubelet kubeconfig. This ensures not
		// only that the kubeconfig file exists, but that other files required by it also exist (like
		// client certificate and key)
		_, err := kubeconfigutil.ClientSetFromFile(kubeadmconstants.GetKubeletKubeConfigPath())
		return (err == nil), nil
	})
}
