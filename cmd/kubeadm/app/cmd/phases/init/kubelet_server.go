package phases

import (
	"fmt"
	"github.com/pkg/errors"

	kubeadmapiv1beta1 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta1"

	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/options"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/phases/workflow"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/kubeletserver"
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
	data, ok := c.(InitData)
	if !ok {
		return errors.New("kubelet-start phase invoked with an invalid data struct")
	}

	data.Cfg().ComponentConfigs.Kubelet.TLSCertFile = kubeadmapiv1beta1.DefaultKubeletServerCertFileName
	data.Cfg().ComponentConfigs.Kubelet.TLSPrivateKeyFile = kubeadmapiv1beta1.DefaultKubeletServerKeyFileName

	kubeClient, err := data.Client()

	if err != nil {
		return errors.Wrap(err, "error getting kubernetes client")
	}

	certDataPem, keyDataPem, err := kubeletserver.KubeletServerCert(kubeClient, data.Cfg().LocalAPIEndpoint, data.Cfg().NodeRegistration)

	if err != nil {
		return errors.Wrap(err, "init phase kubelet server cert")
	}

	err = kubeletserver.WriteKeyToFile(data.Cfg().CertificatesDir, data.Cfg().ComponentConfigs.Kubelet.TLSPrivateKeyFile, keyDataPem)

	if err != nil {
		return errors.Wrap(err, "init phase kubelet server cert")
	}

	err = kubeletserver.WriteCertToFile(data.Cfg().CertificatesDir, data.Cfg().ComponentConfigs.Kubelet.TLSCertFile, certDataPem)

	if err != nil {
		return errors.Wrap(err, "init phase kubelet server cert")
	}

	return nil
}
