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
	"time"

	certificatesv1alpha1 "k8s.io/api/certificates/v1alpha1"
	certificatesv1beta1 "k8s.io/api/certificates/v1beta1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/server/dynamiccertificates"
	"k8s.io/client-go/kubernetes"
	"k8s.io/component-base/featuregate"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/cmd/kube-controller-manager/names"
	"k8s.io/kubernetes/pkg/controller/certificates/approver"
	"k8s.io/kubernetes/pkg/controller/certificates/cleaner"
	ctbpublisher "k8s.io/kubernetes/pkg/controller/certificates/clustertrustbundlepublisher"
	"k8s.io/kubernetes/pkg/controller/certificates/rootcacertpublisher"
	"k8s.io/kubernetes/pkg/controller/certificates/signer"
	csrsigningconfig "k8s.io/kubernetes/pkg/controller/certificates/signer/config"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/utils/clock"
)

func newCertificateSigningRequestSigningControllerDescriptor() *ControllerDescriptor {
	return &ControllerDescriptor{
		name:        names.CertificateSigningRequestSigningController,
		aliases:     []string{"csrsigning"},
		constructor: newCertificateSigningRequestSigningController,
	}
}

func newCertificateSigningRequestSigningController(ctx context.Context, controllerContext ControllerContext, controllerName string) (Controller, error) {
	logger := klog.FromContext(ctx)
	missingSingleSigningFile := controllerContext.ComponentConfig.CSRSigningController.ClusterSigningCertFile == "" || controllerContext.ComponentConfig.CSRSigningController.ClusterSigningKeyFile == ""
	if missingSingleSigningFile && !anySpecificFilesSet(controllerContext.ComponentConfig.CSRSigningController) {
		logger.Info("Skipping CSR signer controller because no csr cert/key was specified")
		return nil, nil
	}
	if !missingSingleSigningFile && anySpecificFilesSet(controllerContext.ComponentConfig.CSRSigningController) {
		return nil, fmt.Errorf("cannot specify default and per controller certs at the same time")
	}

	c, err := controllerContext.NewClient("certificate-controller")
	if err != nil {
		return nil, err
	}

	csrInformer := controllerContext.InformerFactory.Certificates().V1().CertificateSigningRequests()
	certTTL := controllerContext.ComponentConfig.CSRSigningController.ClusterSigningDuration.Duration

	var rx []runFunc
	if kubeletServingSignerCertFile, kubeletServingSignerKeyFile := getKubeletServingSignerFiles(controllerContext.ComponentConfig.CSRSigningController); len(kubeletServingSignerCertFile) > 0 || len(kubeletServingSignerKeyFile) > 0 {
		kubeletServingSigner, err := signer.NewKubeletServingCSRSigningController(ctx, c, csrInformer, kubeletServingSignerCertFile, kubeletServingSignerKeyFile, certTTL)
		if err != nil {
			return nil, fmt.Errorf("failed to init kubernetes.io/kubelet-serving certificate controller: %w", err)
		}

		rx = append(rx, func(ctx context.Context) {
			kubeletServingSigner.Run(ctx, 5)
		})
	} else {
		logger.Info("Skipping CSR signer controller because specific files were specified for other signers and not this one", "controller", "kubernetes.io/kubelet-serving")
	}

	if kubeletClientSignerCertFile, kubeletClientSignerKeyFile := getKubeletClientSignerFiles(controllerContext.ComponentConfig.CSRSigningController); len(kubeletClientSignerCertFile) > 0 || len(kubeletClientSignerKeyFile) > 0 {
		kubeletClientSigner, err := signer.NewKubeletClientCSRSigningController(ctx, c, csrInformer, kubeletClientSignerCertFile, kubeletClientSignerKeyFile, certTTL)
		if err != nil {
			return nil, fmt.Errorf("failed to init kubernetes.io/kube-apiserver-client-kubelet certificate controller: %w", err)
		}

		rx = append(rx, func(ctx context.Context) {
			kubeletClientSigner.Run(ctx, 5)
		})
	} else {
		logger.Info("Skipping CSR signer controller because specific files were specified for other signers and not this one", "controller", "kubernetes.io/kube-apiserver-client-kubelet")
	}

	if kubeAPIServerSignerCertFile, kubeAPIServerSignerKeyFile := getKubeAPIServerClientSignerFiles(controllerContext.ComponentConfig.CSRSigningController); len(kubeAPIServerSignerCertFile) > 0 || len(kubeAPIServerSignerKeyFile) > 0 {
		kubeAPIServerClientSigner, err := signer.NewKubeAPIServerClientCSRSigningController(ctx, c, csrInformer, kubeAPIServerSignerCertFile, kubeAPIServerSignerKeyFile, certTTL)
		if err != nil {
			return nil, fmt.Errorf("failed to init kubernetes.io/kube-apiserver-client certificate controller: %w", err)
		}

		rx = append(rx, func(ctx context.Context) {
			kubeAPIServerClientSigner.Run(ctx, 5)
		})
	} else {
		logger.Info("Skipping CSR signer controller because specific files were specified for other signers and not this one", "controller", "kubernetes.io/kube-apiserver-client")
	}

	if legacyUnknownSignerCertFile, legacyUnknownSignerKeyFile := getLegacyUnknownSignerFiles(controllerContext.ComponentConfig.CSRSigningController); len(legacyUnknownSignerCertFile) > 0 || len(legacyUnknownSignerKeyFile) > 0 {
		legacyUnknownSigner, err := signer.NewLegacyUnknownCSRSigningController(ctx, c, csrInformer, legacyUnknownSignerCertFile, legacyUnknownSignerKeyFile, certTTL)
		if err != nil {
			return nil, fmt.Errorf("failed to init kubernetes.io/legacy-unknown certificate controller: %w", err)
		}

		rx = append(rx, func(ctx context.Context) {
			legacyUnknownSigner.Run(ctx, 5)
		})
	} else {
		logger.Info("Skipping CSR signer controller because specific files were specified for other signers and not this one", "controller", "kubernetes.io/legacy-unknown")
	}

	return newControllerLoop(concurrentRun(rx...), controllerName), nil
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
		name:        names.CertificateSigningRequestApprovingController,
		aliases:     []string{"csrapproving"},
		constructor: newCertificateSigningRequestApprovingController,
	}
}
func newCertificateSigningRequestApprovingController(ctx context.Context, controllerContext ControllerContext, controllerName string) (Controller, error) {
	client, err := controllerContext.NewClient("certificate-controller")
	if err != nil {
		return nil, err
	}

	ac := approver.NewCSRApprovingController(
		ctx,
		client,
		controllerContext.InformerFactory.Certificates().V1().CertificateSigningRequests(),
	)
	return newControllerLoop(func(ctx context.Context) {
		ac.Run(ctx, 5)
	}, controllerName), nil
}

func newCertificateSigningRequestCleanerControllerDescriptor() *ControllerDescriptor {
	return &ControllerDescriptor{
		name:        names.CertificateSigningRequestCleanerController,
		aliases:     []string{"csrcleaner"},
		constructor: newCertificateSigningRequestCleanerController,
	}
}
func newCertificateSigningRequestCleanerController(ctx context.Context, controllerContext ControllerContext, controllerName string) (Controller, error) {
	client, err := controllerContext.NewClient("certificate-controller")
	if err != nil {
		return nil, err
	}

	cc := cleaner.NewCSRCleanerController(
		client.CertificatesV1().CertificateSigningRequests(),
		controllerContext.InformerFactory.Certificates().V1().CertificateSigningRequests(),
	)
	return newControllerLoop(func(ctx context.Context) {
		cc.Run(ctx, 1)
	}, controllerName), nil
}

func newPodCertificateRequestCleanerControllerDescriptor() *ControllerDescriptor {
	return &ControllerDescriptor{
		name:        names.PodCertificateRequestCleanerController,
		constructor: newPodCertificateRequestCleanerController,
		requiredFeatureGates: []featuregate.Feature{
			features.PodCertificateRequest,
		},
	}
}

func newPodCertificateRequestCleanerController(ctx context.Context, controllerContext ControllerContext, controllerName string) (Controller, error) {
	cleaner := cleaner.NewPCRCleanerController(
		controllerContext.ClientBuilder.ClientOrDie("podcertificaterequestcleaner"),
		controllerContext.InformerFactory.Certificates().V1beta1().PodCertificateRequests(),
		clock.RealClock{},
		15*time.Minute, // We expect all PodCertificateRequest flows to complete faster than this.
		5*time.Minute,
	)
	return newControllerLoop(func(ctx context.Context) {
		cleaner.Run(ctx, 1)
	}, controllerName), nil
}

func newRootCACertificatePublisherControllerDescriptor() *ControllerDescriptor {
	return &ControllerDescriptor{
		name:        names.RootCACertificatePublisherController,
		aliases:     []string{"root-ca-cert-publisher"},
		constructor: newRootCACertificatePublisherController,
	}
}

func newRootCACertificatePublisherController(ctx context.Context, controllerContext ControllerContext, controllerName string) (Controller, error) {
	rootCA, err := getKubeAPIServerCAFileContents(controllerContext)
	if err != nil {
		return nil, err
	}

	client, err := controllerContext.NewClient("root-ca-cert-publisher")
	if err != nil {
		return nil, err
	}

	sac, err := rootcacertpublisher.NewPublisher(
		controllerContext.InformerFactory.Core().V1().ConfigMaps(),
		controllerContext.InformerFactory.Core().V1().Namespaces(),
		client,
		rootCA,
	)
	if err != nil {
		return nil, fmt.Errorf("error creating root CA certificate publisher: %w", err)
	}

	return newControllerLoop(func(ctx context.Context) {
		sac.Run(ctx, 1)
	}, controllerName), nil
}

func newKubeAPIServerSignerClusterTrustBundledPublisherDescriptor() *ControllerDescriptor {
	return &ControllerDescriptor{
		name:                 names.KubeAPIServerClusterTrustBundlePublisherController,
		constructor:          newKubeAPIServerSignerClusterTrustBundledPublisherController,
		requiredFeatureGates: []featuregate.Feature{features.ClusterTrustBundle},
	}
}

type controllerConstructor func(string, dynamiccertificates.CAContentProvider, kubernetes.Interface) (ctbpublisher.PublisherRunner, error)

func newKubeAPIServerSignerClusterTrustBundledPublisherController(
	ctx context.Context, controllerContext ControllerContext, controllerName string,
) (Controller, error) {
	rootCA, err := getKubeAPIServerCAFileContents(controllerContext)
	if err != nil {
		return nil, err
	}
	if len(rootCA) == 0 {
		return nil, nil
	}

	servingSigners, err := dynamiccertificates.NewStaticCAContent("kube-apiserver-serving", rootCA)
	if err != nil {
		return nil, fmt.Errorf("failed to create a static CA content provider for the kube-apiserver-serving signer: %w", err)
	}

	schemaControllerMapping := map[schema.GroupVersion]controllerConstructor{
		certificatesv1alpha1.SchemeGroupVersion: ctbpublisher.NewAlphaClusterTrustBundlePublisher,
		certificatesv1beta1.SchemeGroupVersion:  ctbpublisher.NewBetaClusterTrustBundlePublisher,
	}

	apiserverSignerClient, err := controllerContext.NewClient("kube-apiserver-serving-clustertrustbundle-publisher")
	if err != nil {
		return nil, err
	}

	var runner ctbpublisher.PublisherRunner
	for _, gv := range []schema.GroupVersion{certificatesv1beta1.SchemeGroupVersion, certificatesv1alpha1.SchemeGroupVersion} {
		ctbAvailable, err := clusterTrustBundlesAvailable(apiserverSignerClient, gv)
		if err != nil {
			return nil, fmt.Errorf("discovery failed for ClusterTrustBundle: %w", err)
		}

		if !ctbAvailable {
			continue
		}

		runner, err = schemaControllerMapping[gv](
			"kubernetes.io/kube-apiserver-serving",
			servingSigners,
			apiserverSignerClient,
		)
		if err != nil {
			return nil, fmt.Errorf("error creating kube-apiserver-serving signer certificates publisher: %w", err)
		}
		break
	}

	if runner == nil {
		klog.Info("no known scheme version was found for clustertrustbundles, cannot start kube-apiserver-serving-clustertrustbundle-publisher-controller")
		return nil, nil
	}

	return newControllerLoop(runner.Run, controllerName), nil
}

func clusterTrustBundlesAvailable(client kubernetes.Interface, schemaVersion schema.GroupVersion) (bool, error) {
	resList, err := client.Discovery().ServerResourcesForGroupVersion(schemaVersion.String())
	if errors.IsNotFound(err) {
		return false, nil
	}

	if resList != nil {
		// even in case of an error above there might be a partial list for APIs that
		// were already successfully discovered
		for _, r := range resList.APIResources {
			if r.Name == "clustertrustbundles" {
				return true, nil
			}
		}
	}
	return false, err
}

func getKubeAPIServerCAFileContents(controllerContext ControllerContext) ([]byte, error) {
	if controllerContext.ComponentConfig.SAController.RootCAFile == "" {
		config, err := controllerContext.NewClientConfig("root-ca-cert-publisher")
		if err != nil {
			return nil, err
		}
		return config.CAData, nil
	}

	rootCA, err := readCA(controllerContext.ComponentConfig.SAController.RootCAFile)
	if err != nil {
		return nil, fmt.Errorf("error parsing root-ca-file at %s: %w", controllerContext.ComponentConfig.SAController.RootCAFile, err)
	}
	return rootCA, nil

}
