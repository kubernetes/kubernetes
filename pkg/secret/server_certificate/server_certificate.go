/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package server_certificate

import "k8s.io/kubernetes/pkg/api"

const (
	// Server certificate secrets hold certificates to secure
	// transport layer.
	//
	// Required fields:
	// - Secret.Annotations["kubernetes.io/server-certificate.hosts"] - comma separated list of valid host names for the certificate
	// - Secret.Annotations["kubernetes.io/server-certificate.certificate"] - the encoded server certificate
	// - Secret.Annotations["kubernetes.io/server-certificate.key"] - the encoded server key
	//
	// Optional fields:
	// - Secret.Annotations["kubernetes.io/server-certificate.bundle"] - the encoded server bundle
	SecretTypeServerCertificate api.SecretType = "kubernetes.io/server-certificate"

	// HostNamesKey is the key for the hosts the generated certificate must be valid for
	HostNamesKey = string(SecretTypeServerCertificate) + ".hosts"
	// CertificateDataKey is the key for the encoded certificate
	CertificateDataKey = "certificate"
	// PrivateKeyDataKey is the key for the encoded private key
	PrivateKeyDataKey = "key"
	// BundleDataKey is the key for the encoded certificate bundle (optional)
	BundleDataKey = "bundle"
	// PrivateKeyAlgoKey is the key for the private key algorithm (optional)
	PrivateKeyAlgoKey = string(SecretTypeServerCertificate) + ".key-algorithm"
	// PrivateKeySizeKey is the key for the private key size (optional)
	PrivateKeySizeKey = string(SecretTypeServerCertificate) + ".key-size"
)
