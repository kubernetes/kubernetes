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

// Package app implements a server that runs a set of active
// components.  This includes replication controllers, service endpoints and
// nodes.
package app

import (
	"context"
	"fmt"

	"k8s.io/controller-manager/controller"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/cmd/kube-controller-manager/names"
	"k8s.io/kubernetes/pkg/controller/certificates/approver"
	"k8s.io/kubernetes/pkg/controller/certificates/cleaner"
	"k8s.io/kubernetes/pkg/controller/certificates/rootcacertpublisher"
	"k8s.io/kubernetes/pkg/controller/certificates/signer"
	csrsigningconfig "k8s.io/kubernetes/pkg/controller/certificates/signer/config"
)

func newCertificateSigningRequestSigningControllerDescriptor() *ControllerDescriptor {
	return &ControllerDescriptor{
		name:     names.CertificateSigningRequestSigningController,
		aliases:  []string{"csrsigning"},
		initFunc: startCertificateSigningRequestSigningController,
	}
}

func startCertificateSigningRequestSigningController(ctx context.Context, controllerContext ControllerContext, controllerName string) (controller.Interface, bool, error) {
	logger := klog.FromContext(ctx)
	missingSingleSigningFile := controllerContext.ComponentConfig.CSRSigningController.ClusterSigningCertFile == "" || controllerContext.ComponentConfig.CSRSigningController.ClusterSigningKeyFile == ""
	if missingSingleSigningFile && !anySpecificFilesSet(controllerContext.ComponentConfig.CSRSigningController) {
		logger.Info("Skipping CSR signer controller because no csr cert/key was specified")
		return nil, false, nil
	}
	if !missingSingleSigningFile && anySpecificFilesSet(controllerContext.ComponentConfig.CSRSigningController) {
		return nil, false, fmt.Errorf("cannot specify default and per controller certs at the same time")
	}

	c := controllerContext.ClientBuilder.ClientOrDie(logger, "certificate-controller")
	csrInformer := controllerContext.InformerFactory.Certificates().V1().CertificateSigningRequests()
	certTTL := controllerContext.ComponentConfig.CSRSigningController.ClusterSigningDuration.Duration

	if kubeletServingSignerCertFile, kubeletServingSignerKeyFile := getKubeletServingSignerFiles(controllerContext.ComponentConfig.CSRSigningController); len(kubeletServingSignerCertFile) > 0 || len(kubeletServingSignerKeyFile) > 0 {
		kubeletServingSigner, err := signer.NewKubeletServingCSRSigningController(ctx, c, csrInformer, kubeletServingSignerCertFile, kubeletServingSignerKeyFile, certTTL)
		if err != nil {
			return nil, false, fmt.Errorf("failed to start kubernetes.io/kubelet-serving certificate controller: %v", err)
		}
		go kubeletServingSigner.Run(ctx, 5)
	} else {
		logger.Info("Skipping CSR signer controller because specific files were specified for other signers and not this one", "controller", "kubernetes.io/kubelet-serving")
	}

	if kubeletClientSignerCertFile, kubeletClientSignerKeyFile := getKubeletClientSignerFiles(controllerContext.ComponentConfig.CSRSigningController); len(kubeletClientSignerCertFile) > 0 || len(kubeletClientSignerKeyFile) > 0 {
		kubeletClientSigner, err := signer.NewKubeletClientCSRSigningController(ctx, c, csrInformer, kubeletClientSignerCertFile, kubeletClientSignerKeyFile, certTTL)
		if err != nil {
			return nil, false, fmt.Errorf("failed to start kubernetes.io/kube-apiserver-client-kubelet certificate controller: %v", err)
		}
		go kubeletClientSigner.Run(ctx, 5)
	} else {
		logger.Info("Skipping CSR signer controller because specific files were specified for other signers and not this one", "controller", "kubernetes.io/kube-apiserver-client-kubelet")
	}

	if kubeAPIServerSignerCertFile, kubeAPIServerSignerKeyFile := getKubeAPIServerClientSignerFiles(controllerContext.ComponentConfig.CSRSigningController); len(kubeAPIServerSignerCertFile) > 0 || len(kubeAPIServerSignerKeyFile) > 0 {
		kubeAPIServerClientSigner, err := signer.NewKubeAPIServerClientCSRSigningController(ctx, c, csrInformer, kubeAPIServerSignerCertFile, kubeAPIServerSignerKeyFile, certTTL)
		if err != nil {
			return nil, false, fmt.Errorf("failed to start kubernetes.io/kube-apiserver-client certificate controller: %v", err)
		}
		go kubeAPIServerClientSigner.Run(ctx, 5)
	} else {
		logger.Info("Skipping CSR signer controller because specific files were specified for other signers and not this one", "controller", "kubernetes.io/kube-apiserver-client")
	}

	if legacyUnknownSignerCertFile, legacyUnknownSignerKeyFile := getLegacyUnknownSignerFiles(controllerContext.ComponentConfig.CSRSigningController); len(legacyUnknownSignerCertFile) > 0 || len(legacyUnknownSignerKeyFile) > 0 {
		legacyUnknownSigner, err := signer.NewLegacyUnknownCSRSigningController(ctx, c, csrInformer, legacyUnknownSignerCertFile, legacyUnknownSignerKeyFile, certTTL)
		if err != nil {
			return nil, false, fmt.Errorf("failed to start kubernetes.io/legacy-unknown certificate controller: %v", err)
		}
		go legacyUnknownSigner.Run(ctx, 5)
	} else {
		logger.Info("Skipping CSR signer controller because specific files were specified for other signers and not this one", "controller", "kubernetes.io/legacy-unknown")
	}

	return nil, true, nil
}

func areKubeletServingSignerFilesSpecified(config csrsigningconfig.CSRSigningControllerConfiguration) bool {
	// if only one is specified, it will error later during construction
	return len(config.KubeletServingSignerConfiguration.CertFile) > 0 || len(config.KubeletServingSignerConfiguration.KeyFile) > 0
}
func areKubeletClientSignerFilesSpecified(config csrsigningconfig.CSRSigningControllerConfiguration) bool {
	// if only one is specified, it will error later during construction
	return len(config.KubeletClientSignerConfiguration.CertFile) > 0 || len(config.KubeletClientSignerConfiguration.KeyFile) > 0
}

func areKubeAPIServerClientSignerFilesSpecified(config csrsigningconfig.CSRSigningControllerConfiguration) bool {
	// if only one is specified, it will error later during construction
	return len(config.KubeAPIServerClientSignerConfiguration.CertFile) > 0 || len(config.KubeAPIServerClientSignerConfiguration.KeyFile) > 0
}

func areLegacyUnknownSignerFilesSpecified(config csrsigningconfig.CSRSigningControllerConfiguration) bool {
	// if only one is specified, it will error later during construction
	return len(config.LegacyUnknownSignerConfiguration.CertFile) > 0 || len(config.LegacyUnknownSignerConfiguration.KeyFile) > 0
}

func anySpecificFilesSet(config csrsigningconfig.CSRSigningControllerConfiguration) bool {
	return areKubeletServingSignerFilesSpecified(config) ||
		areKubeletClientSignerFilesSpecified(config) ||
		areKubeAPIServerClientSignerFilesSpecified(config) ||
		areLegacyUnknownSignerFilesSpecified(config)
}

func getKubeletServingSignerFiles(config csrsigningconfig.CSRSigningControllerConfiguration) (string, string) {
	// if any cert/key is set for specific CSR signing loops, then the --cluster-signing-{cert,key}-file are not used for any CSR signing loop.
	if anySpecificFilesSet(config) {
		return config.KubeletServingSignerConfiguration.CertFile, config.KubeletServingSignerConfiguration.KeyFile
	}
	return config.ClusterSigningCertFile, config.ClusterSigningKeyFile
}

func getKubeletClientSignerFiles(config csrsigningconfig.CSRSigningControllerConfiguration) (string, string) {
	// if any cert/key is set for specific CSR signing loops, then the --cluster-signing-{cert,key}-file are not used for any CSR signing loop.
	if anySpecificFilesSet(config) {
		return config.KubeletClientSignerConfiguration.CertFile, config.KubeletClientSignerConfiguration.KeyFile
	}
	return config.ClusterSigningCertFile, config.ClusterSigningKeyFile
}

func getKubeAPIServerClientSignerFiles(config csrsigningconfig.CSRSigningControllerConfiguration) (string, string) {
	// if any cert/key is set for specific CSR signing loops, then the --cluster-signing-{cert,key}-file are not used for any CSR signing loop.
	if anySpecificFilesSet(config) {
		return config.KubeAPIServerClientSignerConfiguration.CertFile, config.KubeAPIServerClientSignerConfiguration.KeyFile
	}
	return config.ClusterSigningCertFile, config.ClusterSigningKeyFile
}

func getLegacyUnknownSignerFiles(config csrsigningconfig.CSRSigningControllerConfiguration) (string, string) {
	// if any cert/key is set for specific CSR signing loops, then the --cluster-signing-{cert,key}-file are not used for any CSR signing loop.
	if anySpecificFilesSet(config) {
		return config.LegacyUnknownSignerConfiguration.CertFile, config.LegacyUnknownSignerConfiguration.KeyFile
	}
	return config.ClusterSigningCertFile, config.ClusterSigningKeyFile
}

func newCertificateSigningRequestApprovingControllerDescriptor() *ControllerDescriptor {
	return &ControllerDescriptor{
		name:     names.CertificateSigningRequestApprovingController,
		aliases:  []string{"csrapproving"},
		initFunc: startCertificateSigningRequestApprovingController,
	}
}
func startCertificateSigningRequestApprovingController(ctx context.Context, controllerContext ControllerContext, controllerName string) (controller.Interface, bool, error) {
	logger := klog.FromContext(ctx)
	approver := approver.NewCSRApprovingController(
		ctx,
		controllerContext.ClientBuilder.ClientOrDie(logger, "certificate-controller"),
		controllerContext.InformerFactory.Certificates().V1().CertificateSigningRequests(),
	)
	go approver.Run(ctx, 5)

	return nil, true, nil
}

func newCertificateSigningRequestCleanerControllerDescriptor() *ControllerDescriptor {
	return &ControllerDescriptor{
		name:     names.CertificateSigningRequestCleanerController,
		aliases:  []string{"csrcleaner"},
		initFunc: startCertificateSigningRequestCleanerController,
	}
}
func startCertificateSigningRequestCleanerController(ctx context.Context, controllerContext ControllerContext, controllerName string) (controller.Interface, bool, error) {
	logger := klog.FromContext(ctx)
	cleaner := cleaner.NewCSRCleanerController(
		controllerContext.ClientBuilder.ClientOrDie(logger, "certificate-controller").CertificatesV1().CertificateSigningRequests(),
		controllerContext.InformerFactory.Certificates().V1().CertificateSigningRequests(),
	)
	go cleaner.Run(ctx, 1)
	return nil, true, nil
}

func newRootCACertificatePublisherControllerDescriptor() *ControllerDescriptor {
	return &ControllerDescriptor{
		name:     names.RootCACertificatePublisherController,
		aliases:  []string{"root-ca-cert-publisher"},
		initFunc: startRootCACertificatePublisherController,
	}
}

func startRootCACertificatePublisherController(ctx context.Context, controllerContext ControllerContext, controllerName string) (controller.Interface, bool, error) {
	logger := klog.FromContext(ctx)
	var (
		rootCA []byte
		err    error
	)
	if controllerContext.ComponentConfig.SAController.RootCAFile != "" {
		if rootCA, err = readCA(controllerContext.ComponentConfig.SAController.RootCAFile); err != nil {
			return nil, true, fmt.Errorf("error parsing root-ca-file at %s: %v", controllerContext.ComponentConfig.SAController.RootCAFile, err)
		}
	} else {
		rootCA = controllerContext.ClientBuilder.ConfigOrDie(logger, "root-ca-cert-publisher").CAData
	}

	sac, err := rootcacertpublisher.NewPublisher(
		controllerContext.InformerFactory.Core().V1().ConfigMaps(),
		controllerContext.InformerFactory.Core().V1().Namespaces(),
		controllerContext.ClientBuilder.ClientOrDie(logger, "root-ca-cert-publisher"),
		rootCA,
	)
	if err != nil {
		return nil, true, fmt.Errorf("error creating root CA certificate publisher: %v", err)
	}
	go sac.Run(ctx, 1)
	return nil, true, nil
}
