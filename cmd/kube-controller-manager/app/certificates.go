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
	"os"

	"net/http"

	"k8s.io/apimachinery/pkg/runtime/schema"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/klog/v2"
	kubeoptions "k8s.io/kubernetes/cmd/kube-controller-manager/app/options"
	"k8s.io/kubernetes/pkg/controller/certificates/approver"
	"k8s.io/kubernetes/pkg/controller/certificates/cleaner"
	"k8s.io/kubernetes/pkg/controller/certificates/rootcacertpublisher"
	"k8s.io/kubernetes/pkg/controller/certificates/signer"
	csrsigningconfig "k8s.io/kubernetes/pkg/controller/certificates/signer/config"
	"k8s.io/kubernetes/pkg/features"
)

func startCSRSigningController(ctx ControllerContext) (http.Handler, bool, error) {
	gvr := schema.GroupVersionResource{Group: "certificates.k8s.io", Version: "v1beta1", Resource: "certificatesigningrequests"}
	if !ctx.AvailableResources[gvr] {
		klog.Warningf("Resource %s is not available now", gvr.String())
		return nil, false, nil
	}
	if ctx.ComponentConfig.CSRSigningController.ClusterSigningCertFile == "" || ctx.ComponentConfig.CSRSigningController.ClusterSigningKeyFile == "" {
		klog.V(2).Info("skipping CSR signer controller because no csr cert/key was specified")
		return nil, false, nil
	}

	// Deprecation warning for old defaults.
	//
	// * If the signing cert and key are the default paths but the files
	// exist, warn that the paths need to be specified explicitly in a
	// later release and the defaults will be removed. We don't expect this
	// to be the case.
	//
	// * If the signing cert and key are default paths but the files don't exist,
	// bail out of startController without logging.
	var keyFileExists, keyUsesDefault, certFileExists, certUsesDefault bool

	_, err := os.Stat(ctx.ComponentConfig.CSRSigningController.ClusterSigningCertFile)
	certFileExists = !os.IsNotExist(err)

	certUsesDefault = (ctx.ComponentConfig.CSRSigningController.ClusterSigningCertFile == kubeoptions.DefaultClusterSigningCertFile)

	_, err = os.Stat(ctx.ComponentConfig.CSRSigningController.ClusterSigningKeyFile)
	keyFileExists = !os.IsNotExist(err)

	keyUsesDefault = (ctx.ComponentConfig.CSRSigningController.ClusterSigningKeyFile == kubeoptions.DefaultClusterSigningKeyFile)

	switch {
	case (keyFileExists && keyUsesDefault) || (certFileExists && certUsesDefault):
		klog.Warningf("You might be using flag defaulting for --cluster-signing-cert-file and" +
			" --cluster-signing-key-file. These defaults are deprecated and will be removed" +
			" in a subsequent release. Please pass these options explicitly.")
	case (!keyFileExists && keyUsesDefault) && (!certFileExists && certUsesDefault):
		// This is what we expect right now if people aren't
		// setting up the signing controller. This isn't
		// actually a problem since the signer is not a
		// required controller.
		klog.V(2).Info("skipping CSR signer controller because no csr cert/key was specified and the default files are missing")
		return nil, false, nil
	default:
		// Note that '!filesExist && !usesDefaults' is obviously
		// operator error. We don't handle this case here and instead
		// allow it to be handled by NewCSR... below.
	}

	c := ctx.ClientBuilder.ClientOrDie("certificate-controller")
	csrInformer := ctx.InformerFactory.Certificates().V1beta1().CertificateSigningRequests()
	certTTL := ctx.ComponentConfig.CSRSigningController.ClusterSigningDuration.Duration
	caFile, caKeyFile := getKubeletServingSignerFiles(ctx.ComponentConfig.CSRSigningController)

	// TODO get different signer cert and key files for each signer when we add flags.

	kubeletServingSigner, err := signer.NewKubeletServingCSRSigningController(c, csrInformer, caFile, caKeyFile, certTTL)
	if err != nil {
		return nil, false, fmt.Errorf("failed to start kubernetes.io/kubelet-serving certificate controller: %v", err)
	}
	go kubeletServingSigner.Run(1, ctx.Stop)

	kubeletClientSigner, err := signer.NewKubeletClientCSRSigningController(c, csrInformer, caFile, caKeyFile, certTTL)
	if err != nil {
		return nil, false, fmt.Errorf("failed to start kubernetes.io/kube-apiserver-client-kubelet certificate controller: %v", err)
	}
	go kubeletClientSigner.Run(1, ctx.Stop)

	kubeAPIServerClientSigner, err := signer.NewKubeAPIServerClientCSRSigningController(c, csrInformer, caFile, caKeyFile, certTTL)
	if err != nil {
		return nil, false, fmt.Errorf("failed to start kubernetes.io/kube-apiserver-client certificate controller: %v", err)
	}
	go kubeAPIServerClientSigner.Run(1, ctx.Stop)

	legacyUnknownSigner, err := signer.NewLegacyUnknownCSRSigningController(c, csrInformer, caFile, caKeyFile, certTTL)
	if err != nil {
		return nil, false, fmt.Errorf("failed to start kubernetes.io/legacy-unknown certificate controller: %v", err)
	}
	go legacyUnknownSigner.Run(1, ctx.Stop)

	return nil, true, nil
}

// getKubeletServingSignerFiles returns the cert and key for signing.
// TODO we will extended this for each signer so that it prefers the specific flag (to be added) and falls back to the single flag
func getKubeletServingSignerFiles(config csrsigningconfig.CSRSigningControllerConfiguration) (string, string) {
	return config.ClusterSigningCertFile, config.ClusterSigningKeyFile
}

func startCSRApprovingController(ctx ControllerContext) (http.Handler, bool, error) {
	gvr := schema.GroupVersionResource{Group: "certificates.k8s.io", Version: "v1beta1", Resource: "certificatesigningrequests"}
	if !ctx.AvailableResources[gvr] {
		klog.Warningf("Resource %s is not available now", gvr.String())
		return nil, false, nil
	}

	approver := approver.NewCSRApprovingController(
		ctx.ClientBuilder.ClientOrDie("certificate-controller"),
		ctx.InformerFactory.Certificates().V1beta1().CertificateSigningRequests(),
	)
	go approver.Run(1, ctx.Stop)

	return nil, true, nil
}

func startCSRCleanerController(ctx ControllerContext) (http.Handler, bool, error) {
	cleaner := cleaner.NewCSRCleanerController(
		ctx.ClientBuilder.ClientOrDie("certificate-controller").CertificatesV1beta1().CertificateSigningRequests(),
		ctx.InformerFactory.Certificates().V1beta1().CertificateSigningRequests(),
	)
	go cleaner.Run(1, ctx.Stop)
	return nil, true, nil
}

func startRootCACertPublisher(ctx ControllerContext) (http.Handler, bool, error) {
	if !utilfeature.DefaultFeatureGate.Enabled(features.BoundServiceAccountTokenVolume) {
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
