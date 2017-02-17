/*
Copyright 2014 The Kubernetes Authors.

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
// certificates controller.
package app

import (
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"github.com/spf13/pflag"
)

// GKECertificatesController is the main context object for the package.
type GKECertificatesController struct {
	Kubeconfig                    string
	Master                        string
	ClusterSigningCertFile        string
	ClusterSigningKeyFile         string
	ClusterSigningGKEKubeconfig   string
	ClusterSigningGKERetryBackoff metav1.Duration
	ApproveAllKubeletCSRsForGroup string
	UseServiceAccountCredentials  bool
	ServiceAccountKeyFile         string
	RootCAFile                    string
}

// Create a new instance of a GKECertificatesController with default parameters.
func NewGKECertificatesController() *GKECertificatesController {
	s := &GKECertificatesController{
		ClusterSigningCertFile:        "/etc/kubernetes/ca/ca.pem",
		ClusterSigningKeyFile:         "/etc/kubernetes/ca/ca.key",
		ClusterSigningGKERetryBackoff: metav1.Duration{Duration: 500 * time.Millisecond},
	}
	return s
}

// AddFlags adds flags for a specific GKECertificatesController to the
// specified FlagSet.
func (s *GKECertificatesController) AddFlags(fs *pflag.FlagSet) {
	fs.StringVar(&s.Kubeconfig, "kubeconfig", s.Kubeconfig, "Path to kubeconfig file with authorization and master location information.")
	fs.StringVar(&s.Master, "master", s.Master, "The address of the Kubernetes API server (overrides any value in kubeconfig)")

	fs.StringVar(&s.ClusterSigningCertFile, "cluster-signing-cert-file", s.ClusterSigningCertFile, "Filename containing a PEM-encoded X509 CA certificate used to issue cluster-scoped certificates")
	fs.StringVar(&s.ClusterSigningKeyFile, "cluster-signing-key-file", s.ClusterSigningKeyFile, "Filename containing a PEM-encoded RSA or ECDSA private key used to sign cluster-scoped certificates")
	fs.StringVar(&s.ClusterSigningGKEKubeconfig, "cluster-signing-gke-kubeconfig", s.ClusterSigningGKEKubeconfig, "If set, use the kubeconfig file to call GKE to sign cluster-scoped certificates instead of using a local private key.")
	fs.DurationVar(&s.ClusterSigningGKERetryBackoff.Duration, "cluster-signing-gke-retry-backoff", s.ClusterSigningGKERetryBackoff.Duration, "The initial backoff to use when retrying requests to GKE. Additional attempts will use exponential backoff.")

	fs.StringVar(&s.ApproveAllKubeletCSRsForGroup, "insecure-experimental-approve-all-kubelet-csrs-for-group", s.ApproveAllKubeletCSRsForGroup, "The group for which the controller-manager will auto approve all CSRs for kubelet client certificates.")

	fs.BoolVar(&s.UseServiceAccountCredentials, "use-service-account-credentials", s.UseServiceAccountCredentials, "If true, use individual service account credentials for each controller.")
	fs.StringVar(&s.ServiceAccountKeyFile, "service-account-private-key-file", s.ServiceAccountKeyFile, "Filename containing a PEM-encoded private RSA or ECDSA key used to sign service account tokens.")
	fs.StringVar(&s.RootCAFile, "root-ca-file", s.RootCAFile, "If set, this root certificate authority will be included in service account's token secret. This must be a valid PEM-encoded CA bundle.")
}
