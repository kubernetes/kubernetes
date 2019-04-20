package phases

import (
	"fmt"
	"github.com/pkg/errors"

	kubeadmapiv1beta1 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta1"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/options"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/phases/workflow"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	kubeletphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/kubelet"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/kubeletserver"
	kubeconfigutil "k8s.io/kubernetes/cmd/kubeadm/app/util/kubeconfig"
	"k8s.io/kubernetes/pkg/util/normalizer"
)

var (
	kubeletServerCertExample = normalizer.Examples(`
		# Generates key and certificate signing request and approves it with kube-apiserver
		kubeadm init phase kubelet-server
		`)
)

// NewKubeletStartPhase creates a kubeadm workflow phase that start kubelet on a node.
func NewKubeletServerCertAndKey() workflow.Phase {
	return workflow.Phase{
		Name:    "kubelet-server",
		Short:   "Generates, approves and writes kubelet server cert and key to file",
		Long:    "Generates, approves and writes kubelet server cert and key by submitting certificate signing request to kube-apiserver.",
		Example: kubeletServerCertExample,
		Run:     kubeletServerCerts,
		InheritFlags: []string{
			options.CfgPath,
			options.NodeName,
			options.KubeconfigDir,
			options.KubeconfigPath,
			options.DryRun,
		},
	}
}

// runKubeletStart executes kubelet start logic.
func kubeletServerCerts(c workflow.RunData) error {
	fmt.Println("[kubelet-server] Generate kubelet server certificates")
	data, ok := c.(JoinData)
	if !ok {
		return errors.New("kubelet-server phase invoked with an invalid data struct")
	}

	initCfg, err := data.InitCfg()

	initCfg.ComponentConfigs.Kubelet.TLSCertFile = kubeadmapiv1beta1.DefaultKubeletServerCertFileName
	initCfg.ComponentConfigs.Kubelet.TLSPrivateKeyFile = kubeadmapiv1beta1.DefaultKubeletServerKeyFileName

	if !ok {
		return errors.New("kubelet-server join phase get InitCfg")
	}

	kubeClient, err := kubeconfigutil.ClientSetFromFile(kubeadmconstants.GetBootstrapKubeletKubeConfigPath())
	if err != nil {
		return errors.Errorf("couldn't create client from kubeconfig file %q", kubeadmconstants.GetBootstrapKubeletKubeConfigPath())
	}

	certDataPem, keyDataPem, err := kubeletserver.KubeletServerCert(kubeClient, initCfg.LocalAPIEndpoint, data.Cfg().NodeRegistration)

	if err != nil {
		return errors.Wrap(err, "join phase kubelet server cert")
	}

	err = kubeletserver.WriteKeyToFile(initCfg.CertificatesDir, initCfg.ComponentConfigs.Kubelet.TLSPrivateKeyFile, keyDataPem)

	if err != nil {
		return errors.Wrap(err, "join phase kubelet server cert")
	}

	err = kubeletserver.WriteCertToFile(initCfg.CertificatesDir, initCfg.ComponentConfigs.Kubelet.TLSCertFile, certDataPem)

	if err != nil {
		return errors.Wrap(err, "join phase kubelet server cert")
	}

	// Configure the kubelet. In this short timeframe, kubeadm is trying to stop/restart the kubelet
	// Try to stop the kubelet service so no race conditions occur when configuring it
	fmt.Println("[kubelet-server] Stopping the kubelet")
	kubeletphase.TryStopKubelet()

	registerTaintsUsingFlags := data.Cfg().ControlPlane == nil
	if err := kubeletphase.WriteKubeletDynamicEnvFile(&initCfg.ClusterConfiguration, &initCfg.NodeRegistration, registerTaintsUsingFlags, kubeadmconstants.KubeletRunDirectory); err != nil {
		return err
	}

	// Try to start the kubelet service in case it's inactive
	fmt.Println("[kubelet-server] Starting the kubelet")
	kubeletphase.TryStartKubelet()

	return nil
}
