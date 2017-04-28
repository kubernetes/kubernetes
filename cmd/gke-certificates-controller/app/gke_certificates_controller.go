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

// Package app implements a server that runs a stand-alone version of the
// certificates controller for GKE clusters.
package app

import (
	"time"

	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	clientv1 "k8s.io/client-go/pkg/api/v1"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/client-go/tools/record"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	informers "k8s.io/kubernetes/pkg/client/informers/informers_generated/externalversions"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/certificates"

	// Install all auth plugins
	_ "k8s.io/client-go/plugin/pkg/client/auth"

	"github.com/golang/glog"
	"github.com/spf13/cobra"
)

// NewGKECertificatesControllerCommand creates a new *cobra.Command with default parameters.
func NewGKECertificatesControllerCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use: "gke-certificates-controller",
		Long: `The Kubernetes GKE certificates controller is a daemon that
handles auto-approving and signing certificates for GKE clusters.`,
	}

	return cmd
}

// Run runs the GKECertificatesController. This should never exit.
func Run(s *GKECertificatesController) error {
	kubeconfig, err := clientcmd.BuildConfigFromFlags("", s.Kubeconfig)
	if err != nil {
		return err
	}

	kubeClient, err := clientset.NewForConfig(restclient.AddUserAgent(kubeconfig, "gke-certificates-controller"))
	if err != nil {
		return err
	}

	eventBroadcaster := record.NewBroadcaster()
	eventBroadcaster.StartLogging(glog.Infof)
	eventBroadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: v1core.New(kubeClient.Core().RESTClient()).Events("")})
	recorder := eventBroadcaster.NewRecorder(api.Scheme, clientv1.EventSource{Component: "gke-certificates-controller"})

	clientBuilder := controller.SimpleControllerClientBuilder{ClientConfig: kubeconfig}
	client := clientBuilder.ClientOrDie("certificate-controller")

	sharedInformers := informers.NewSharedInformerFactory(client, time.Duration(12)*time.Hour)

	signer, err := NewGKESigner(s.ClusterSigningGKEKubeconfig, s.ClusterSigningGKERetryBackoff.Duration, recorder)
	if err != nil {
		return err
	}

	controller, err := certificates.NewCertificateController(
		client,
		sharedInformers.Certificates().V1beta1().CertificateSigningRequests(),
		signer,
		certificates.NewGroupApprover(s.ApproveAllKubeletCSRsForGroup),
	)
	if err != nil {
		return err
	}

	sharedInformers.Start(nil)
	controller.Run(1, nil) // runs forever
	return nil
}
