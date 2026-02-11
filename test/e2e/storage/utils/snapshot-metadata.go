/*
Copyright The Kubernetes Authors.

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

package utils

import (
	"context"
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"fmt"
	"math/big"
	"time"

	"github.com/onsi/ginkgo/v2"
	v1 "k8s.io/api/core/v1"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	apiextensionsclientset "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/test/e2e/framework"
)

// Constants for common values
const (
	CommonName                         = "csi-snapshot-metadata"
	TLSSecretName                      = "csi-snapshot-metadata-server-certs"
	ServicePort                        = 6443
	TargetPort                         = 50051
	CertificateValidityDays            = 365
	ServerCertificateValidity          = 60
	CertificateKeySize                 = 4096
	AppSelectorKey                     = "app.kubernetes.io/name"
	AppSelectorValue                   = "csi-hostpathplugin"
	SnapshotMetadataCRAudience         = "csi-snapshot-metadata-test"
	SnapshotMetadataServiceCRDManifest = "test/e2e/testing-manifests/storage-csi/external-snapshot-metadata/cbt.storage.k8s.io_snapshotmetadataservices.yaml"

	// SnapshotMetadataService API constants
	SnapshotMetadataServiceGroup      = "cbt.storage.k8s.io"
	SnapshotMetadataServiceVersion    = "v1alpha1"
	SnapshotMetadataServiceAPIVersion = SnapshotMetadataServiceGroup + "/" + SnapshotMetadataServiceVersion
	SnapshotMetadataServiceKind       = "SnapshotMetadataService"
	SnapshotMetadataServiceResource   = "snapshotmetadataservices"
)

// generateCA creates a self-signed CA certificate and private key using RSA.
func generateCA() (*x509.Certificate, *rsa.PrivateKey, error) {
	ginkgo.By("Generating CA certificate and private key")

	caPrivateKey, err := rsa.GenerateKey(rand.Reader, CertificateKeySize)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to generate CA private key: %w", err)
	}

	caTemplate := &x509.Certificate{
		SerialNumber: big.NewInt(1),
		Subject: pkix.Name{
			CommonName: CommonName,
		},
		NotBefore:             time.Now(),
		NotAfter:              time.Now().Add(CertificateValidityDays * 24 * time.Hour),
		KeyUsage:              x509.KeyUsageCertSign | x509.KeyUsageDigitalSignature,
		BasicConstraintsValid: true,
		IsCA:                  true,
	}

	caCertBytes, err := x509.CreateCertificate(rand.Reader, caTemplate, caTemplate, &caPrivateKey.PublicKey, caPrivateKey)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to create CA certificate: %w", err)
	}

	caCert, err := x509.ParseCertificate(caCertBytes)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to parse CA certificate: %w", err)
	}
	return caCert, caPrivateKey, nil
}

// generateServerCert creates a server certificate signed by the given CA using RSA.
func generateServerCert(driverNamespace string, caCert *x509.Certificate, caPrivateKey *rsa.PrivateKey) ([]byte, []byte, error) {
	ginkgo.By("Generating server certificate and private key")

	serverPrivateKey, err := rsa.GenerateKey(rand.Reader, CertificateKeySize)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to generate server private key: %w", err)
	}

	serverTemplate := &x509.Certificate{
		SerialNumber: big.NewInt(2),
		Subject: pkix.Name{
			CommonName: CommonName,
		},
		DNSNames: []string{
			fmt.Sprintf(".%s", driverNamespace),
			fmt.Sprintf("%s.%s", CommonName, driverNamespace),
		},
		NotBefore:   time.Now(),
		NotAfter:    time.Now().Add(ServerCertificateValidity * 24 * time.Hour),
		KeyUsage:    x509.KeyUsageDigitalSignature,
		ExtKeyUsage: []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
	}

	serverCertBytes, err := x509.CreateCertificate(rand.Reader, serverTemplate, caCert, &serverPrivateKey.PublicKey, caPrivateKey)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to create server certificate: %w", err)
	}

	serverKeyBytes := x509.MarshalPKCS1PrivateKey(serverPrivateKey)

	return serverCertBytes, serverKeyBytes, nil
}

// createTLSSecret creates or updates a TLS secret in the specified namespace.
func createTLSSecret(ctx context.Context, f *framework.Framework, driverNamespace string, certBytes, keyBytes []byte) error {
	ginkgo.By("Creating TLS secret")

	certPEM := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: certBytes})
	keyPEM := pem.EncodeToMemory(&pem.Block{Type: "RSA PRIVATE KEY", Bytes: keyBytes})

	tlsSecret := &v1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      TLSSecretName,
			Namespace: driverNamespace,
		},
		Type: v1.SecretTypeTLS,
		Data: map[string][]byte{
			"tls.crt": certPEM,
			"tls.key": keyPEM,
		},
	}

	client := f.ClientSet.CoreV1().Secrets(driverNamespace)
	if _, err := client.Create(ctx, tlsSecret, metav1.CreateOptions{}); err != nil {
		return fmt.Errorf("failed to create or update TLS secret: %w", err)
	}

	framework.Logf("TLS secret created successfully: %s", tlsSecret.Name)
	return nil
}

// createSnapshotMetadataSVC creates a Kubernetes service for the snapshot metadata.
func createSnapshotMetadataSVC(ctx context.Context, f *framework.Framework, driverName, driverNamespace string) error {
	ginkgo.By("Creating Kubernetes service for snapshot metadata")
	svc := &v1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      CommonName,
			Namespace: driverNamespace,
		},
		Spec: v1.ServiceSpec{
			Ports: []v1.ServicePort{
				{
					Name:       "snapshot-metadata-port",
					Port:       ServicePort,
					Protocol:   "TCP",
					TargetPort: intstr.FromInt(TargetPort),
				},
			},
			Selector: map[string]string{
				AppSelectorKey: AppSelectorValue,
			},
		},
	}

	client := f.ClientSet.CoreV1().Services(driverNamespace)
	if _, err := client.Create(ctx, svc, metav1.CreateOptions{}); err != nil {
		return fmt.Errorf("failed to create service: %w", err)
	}
	framework.Logf("Service created successfully: %s", svc.Name)

	return nil
}

// crdExistsInDiscovery checks to see if the given CRD exists in discovery at all served versions.
func crdExistsInDiscovery(client apiextensionsclientset.Interface, crd *apiextensionsv1.CustomResourceDefinition) bool {
	var versions []string
	for _, v := range crd.Spec.Versions {
		if v.Served {
			versions = append(versions, v.Name)
		}
	}
	for _, v := range versions {
		if !crdVersionExistsInDiscovery(client, crd, v) {
			return false
		}
	}
	return true
}

func crdVersionExistsInDiscovery(client apiextensionsclientset.Interface, crd *apiextensionsv1.CustomResourceDefinition, version string) bool {
	resourceList, err := client.Discovery().ServerResourcesForGroupVersion(crd.Spec.Group + "/" + version)
	if err != nil {
		return false
	}
	for _, resource := range resourceList.APIResources {
		if resource.Name == crd.Spec.Names.Plural {
			return true
		}
	}
	return false
}

// createSnapshotMetadataServiceCRD creates the SnapshotMetadataService CRD from the manifest if it doesn't already exist.
func createSnapshotMetadataServiceCRD(ctx context.Context, f *framework.Framework) error {
	ginkgo.By("Creating SnapshotMetadataService CRD if not present")

	// Load CRD from manifest
	items, err := LoadFromManifests(SnapshotMetadataServiceCRDManifest)
	if err != nil {
		return fmt.Errorf("failed to load SnapshotMetadataService CRD manifest: %w", err)
	}

	if len(items) == 0 {
		return fmt.Errorf("no items found in SnapshotMetadataService CRD manifest")
	}

	crd, ok := items[0].(*apiextensionsv1.CustomResourceDefinition)
	if !ok {
		return fmt.Errorf("expected CustomResourceDefinition, got %T", items[0])
	}

	// Create apiextensions client
	apiextClient, err := apiextensionsclientset.NewForConfig(f.ClientConfig())
	if err != nil {
		return fmt.Errorf("failed to create apiextensions client: %w", err)
	}

	// Check if CRD already exists in discovery
	if crdExistsInDiscovery(apiextClient, crd) {
		framework.Logf("SnapshotMetadataService CRD already exists, skipping creation")
		return nil
	}

	// Create the CRD
	if _, err := apiextClient.ApiextensionsV1().CustomResourceDefinitions().Create(ctx, crd, metav1.CreateOptions{}); err != nil {
		return fmt.Errorf("failed to create SnapshotMetadataService CRD: %w", err)
	}

	// Wait for CRD to appear in discovery
	if err := wait.PollUntilContextTimeout(ctx, 500*time.Millisecond, wait.ForeverTestTimeout, true, func(ctx context.Context) (bool, error) {
		return crdExistsInDiscovery(apiextClient, crd), nil
	}); err != nil {
		return fmt.Errorf("failed to see SnapshotMetadataService CRD in discovery: %w", err)
	}

	framework.Logf("SnapshotMetadataService CRD created successfully: %s", crd.Name)
	return nil
}

// createSnapshotMetdataServiceCR creates a SnapshotMetadataService custom resource.
func createSnapshotMetdataServiceCR(ctx context.Context, f *framework.Framework, driverName, driverNamespace string, caCert *x509.Certificate) error {
	ginkgo.By("Creating snapshot metadata service CR")

	sms := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"kind":       SnapshotMetadataServiceKind,
			"apiVersion": SnapshotMetadataServiceAPIVersion,
			"metadata": map[string]interface{}{
				"name": fmt.Sprintf("%s-%s", driverName, f.UniqueName),
			},
			"spec": map[string]interface{}{
				"caCert":   pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: caCert.Raw}),
				"audience": SnapshotMetadataCRAudience,
				"address":  fmt.Sprintf("%s.%s:%d", CommonName, driverNamespace, ServicePort),
			},
		},
	}

	gvr := schema.GroupVersionResource{
		Group:    SnapshotMetadataServiceGroup,
		Version:  SnapshotMetadataServiceVersion,
		Resource: SnapshotMetadataServiceResource,
	}

	if _, err := f.DynamicClient.Resource(gvr).Create(ctx, sms, metav1.CreateOptions{}); err != nil {
		return fmt.Errorf("failed to create SnapshotMetadataService: %w", err)
	}

	return nil
}

// CreateSnapshotMetadataResources sets up the snapshot metadata resources.
func CreateSnapshotMetadataResources(ctx context.Context, f *framework.Framework, driverName, driverNamespace string) error {
	caCert, caPrivateKey, err := generateCA()
	if err != nil {
		return fmt.Errorf("failed to generate CA certificate: %w", err)
	}

	serverCertBytes, serverKeyBytes, err := generateServerCert(driverNamespace, caCert, caPrivateKey)
	if err != nil {
		return fmt.Errorf("failed to generate server certificate: %w", err)
	}

	if err := createTLSSecret(ctx, f, driverNamespace, serverCertBytes, serverKeyBytes); err != nil {
		return fmt.Errorf("failed to create TLS secret: %w", err)
	}

	if err := createSnapshotMetadataSVC(ctx, f, driverName, driverNamespace); err != nil {
		return fmt.Errorf("failed to create snapshot metadata svc: %w", err)
	}

	// Create SnapshotMetadataService CRD, if not already present
	if err := createSnapshotMetadataServiceCRD(ctx, f); err != nil {
		return fmt.Errorf("failed to create SnapshotMetadataService CRD: %w", err)
	}

	if err := createSnapshotMetdataServiceCR(ctx, f, driverName, driverNamespace, caCert); err != nil {
		return fmt.Errorf("failed to create SnapshotMetadataService CR: %w", err)
	}

	return nil
}
