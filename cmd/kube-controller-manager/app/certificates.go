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
//
package app

import (
	"fmt"
	"net/http"

	"k8s.io/apimachinery/pkg/runtime/schema"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/controller/certificates/approver"
	"k8s.io/kubernetes/pkg/controller/certificates/cleaner"
	"k8s.io/kubernetes/pkg/controller/certificates/rootcacertpublisher"
	"k8s.io/kubernetes/pkg/controller/certificates/signer"
	csrsigningconfig "k8s.io/kubernetes/pkg/controller/certificates/signer/config"
	"k8s.io/kubernetes/pkg/features"
)

func startCSRSigningController(ctx ControllerContext) (http.Handler, bool, error) {
	gvr := schema.GroupVersionResource{Group: "certificates.k8s.io", Version: "v1", Resource: "certificatesigningrequests"}
	if !ctx.AvailableResources[gvr] {
		klog.Warningf("Resource %s is not available now", gvr.String())
		return nil, false, nil
	}
	missingSingleSigningFile := ctx.ComponentConfig.CSRSigningController.ClusterSigningCertFile == "" || ctx.ComponentConfig.CSRSigningController.ClusterSigningKeyFile == ""
	if missingSingleSigningFile && !anySpecificFilesSet(ctx.ComponentConfig.CSRSigningController) {
		klog.V(2).Info("skipping CSR signer controller because no csr cert/key was specified")
		return nil, false, nil
	}
	if !missingSingleSigningFile && anySpecificFilesSet(ctx.ComponentConfig.CSRSigningController) {
		return nil, false, fmt.Errorf("cannot specify default and per controller certs at the same time")
	}

	c := ctx.ClientBuilder.ClientOrDie("certificate-controller")
	csrInformer := ctx.InformerFactory.Certificates().V1().CertificateSigningRequests()
	certTTL := ctx.ComponentConfig.CSRSigningController.ClusterSigningDuration.Duration

	if kubeletServingSignerCertFile, kubeletServingSignerKeyFile := getKubeletServingSignerFiles(ctx.ComponentConfig.CSRSigningController); len(kubeletServingSignerCertFile) > 0 || len(kubeletServingSignerKeyFile) > 0 {
		kubeletServingSigner, err := signer.NewKubeletServingCSRSigningController(c, csrInformer, kubeletServingSignerCertFile, kubeletServingSignerKeyFile, certTTL)
		if err != nil {
			return nil, false, fmt.Errorf("failed to start kubernetes.io/kubelet-serving certificate controller: %v", err)
		}
		go kubeletServingSigner.Run(1, ctx.Stop)
	} else {
		klog.V(2).Infof("skipping CSR signer controller %q because specific files were specified for other signers and not this one.", "kubernetes.io/kubelet-serving")
	}

	if kubeletClientSignerCertFile, kubeletClientSignerKeyFile := getKubeletClientSignerFiles(ctx.ComponentConfig.CSRSigningController); len(kubeletClientSignerCertFile) > 0 || len(kubeletClientSignerKeyFile) > 0 {
		kubeletClientSigner, err := signer.NewKubeletClientCSRSigningController(c, csrInformer, kubeletClientSignerCertFile, kubeletClientSignerKeyFile, certTTL)
		if err != nil {
			return nil, false, fmt.Errorf("failed to start kubernetes.io/kube-apiserver-client-kubelet certificate controller: %v", err)
		}
		go kubeletClientSigner.Run(1, ctx.Stop)
	} else {
		klog.V(2).Infof("skipping CSR signer controller %q because specific files were specified for other signers and not this one.", "kubernetes.io/kube-apiserver-client-kubelet")
	}

	if kubeAPIServerSignerCertFile, kubeAPIServerSignerKeyFile := getKubeAPIServerClientSignerFiles(ctx.ComponentConfig.CSRSigningController); len(kubeAPIServerSignerCertFile) > 0 || len(kubeAPIServerSignerKeyFile) > 0 {
		kubeAPIServerClientSigner, err := signer.NewKubeAPIServerClientCSRSigningController(c, csrInformer, kubeAPIServerSignerCertFile, kubeAPIServerSignerKeyFile, certTTL)
		if err != nil {
			return nil, false, fmt.Errorf("failed to start kubernetes.io/kube-apiserver-client certificate controller: %v", err)
		}
		go kubeAPIServerClientSigner.Run(1, ctx.Stop)
	} else {
		klog.V(2).Infof("skipping CSR signer controller %q because specific files were specified for other signers and not this one.", "kubernetes.io/kube-apiserver-client")
	}

	if legacyUnknownSignerCertFile, legacyUnknownSignerKeyFile := getLegacyUnknownSignerFiles(ctx.ComponentConfig.CSRSigningController); len(legacyUnknownSignerCertFile) > 0 || len(legacyUnknownSignerKeyFile) > 0 {
		legacyUnknownSigner, err := signer.NewLegacyUnknownCSRSigningController(c, csrInformer, legacyUnknownSignerCertFile, legacyUnknownSignerKeyFile, certTTL)
		if err != nil {
			return nil, false, fmt.Errorf("failed to start kubernetes.io/legacy-unknown certificate controller: %v", err)
		}
		go legacyUnknownSigner.Run(1, ctx.Stop)
	} else {
		klog.V(2).Infof("skipping CSR signer controller %q because specific files were specified for other signers and not this one.", "kubernetes.io/legacy-unknown")
	}

	return nil, true, nil
}

func areKubeletServingSignerFilesSpecified(config csrsigningconfig.CSRSigningControllerConfiguration) bool {
	if len(config.KubeletServingSignerConfiguration.CertFile) > 0 || len(config.KubeletServingSignerConfiguration.KeyFile) > 0 {
		// if only one is specified, it will error later during construction
		return true
	}
	return false
}
func areKubeletClientSignerFilesSpecified(config csrsigningconfig.CSRSigningControllerConfiguration) bool {
	if len(config.KubeletClientSignerConfiguration.CertFile) > 0 || len(config.KubeletClientSignerConfiguration.KeyFile) > 0 {
		// if only one is specified, it will error later during construction
		return true
	}
	return false
}

func areKubeAPIServerClientSignerFilesSpecified(config csrsigningconfig.CSRSigningControllerConfiguration) bool {
	if len(config.KubeAPIServerClientSignerConfiguration.CertFile) > 0 || len(config.KubeAPIServerClientSignerConfiguration.KeyFile) > 0 {
		// if only one is specified, it will error later during construction
		return true
	}
	return false
}

func areLegacyUnknownSignerFilesSpecified(config csrsigningconfig.CSRSigningControllerConfiguration) bool {
	if len(config.LegacyUnknownSignerConfiguration.CertFile) > 0 || len(config.LegacyUnknownSignerConfiguration.KeyFile) > 0 {
		// if only one is specified, it will error later during construction
		return true
	}
	return false
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

func startCSRApprovingController(ctx ControllerContext) (http.Handler, bool, error) {
	gvr := schema.GroupVersionResource{Group: "certificates.k8s.io", Version: "v1", Resource: "certificatesigningrequests"}
	if !ctx.AvailableResources[gvr] {
		klog.Warningf("Resource %s is not available now", gvr.String())
		return nil, false, nil
	}

	approver := approver.NewCSRApprovingController(
		ctx.ClientBuilder.ClientOrDie("certificate-controller"),
		ctx.InformerFactory.Certificates().V1().CertificateSigningRequests(),
	)
	go approver.Run(1, ctx.Stop)

	return nil, true, nil
}

func startCSRCleanerController(ctx ControllerContext) (http.Handler, bool, error) {
	cleaner := cleaner.NewCSRCleanerController(
		ctx.ClientBuilder.ClientOrDie("certificate-controller").CertificatesV1().CertificateSigningRequests(),
		ctx.InformerFactory.Certificates().V1().CertificateSigningRequests(),
	)
	go cleaner.Run(1, ctx.Stop)
	return nil, true, nil
}

func startRootCACertPublisher(ctx ControllerContext) (http.Handler, bool, error) {
	if !utilfeature.DefaultFeatureGate.Enabled(features.RootCAConfigMap) {
		return nil, false, nil
	}

	var (
		rootCA []byte
		err    error
	)
	if ctx.ComponentConfig.SAController.RootCAFile != "" {
		if rootCA, err = readCA(ctx.ComponentConfig.SAController.RootCAFile); err != nil {
			return nil, true, fmt.Errorf("error parsing root-ca-file at %s: %v", ctx.ComponentConfig.SAController.RootCAFile, err)
		}
	} else {
		rootCA = ctx.ClientBuilder.ConfigOrDie("root-ca-cert-publisher").CAData
	}

	sac, err := rootcacertpublisher.NewPublisher(
		ctx.InformerFactory.Core().V1().ConfigMaps(),
		ctx.InformerFactory.Core().V1().Namespaces(),
		ctx.ClientBuilder.ClientOrDie("root-ca-cert-publisher"),
		rootCA,
	)
	if err != nil {
		return nil, true, fmt.Errorf("error creating root CA certificate publisher: %v", err)
	}
	go sac.Run(1, ctx.Stop)
	return nil, true, nil
}
