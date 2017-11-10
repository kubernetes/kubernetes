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

package etcd

import (
	"crypto/x509"
	"fmt"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	certutil "k8s.io/client-go/util/cert"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/certs/pkiutil"
)

const (
	// Name of the secret used to hold certs for etcd peers
	etcdPeerSecretName = "etcd-peer"

	// Name of the secret used to hold certs for etcd members
	etcdServerSecretName = "etcd-server"
)

type certCreator struct {
	client  clientset.Interface
	certDir string
	caCert  *x509.Certificate
}

type secretPair struct {
	name     string
	certType string
}

func createTLSSecrets(cfg *kubeadmapi.MasterConfiguration, client clientset.Interface) error {
	caCert, err := pkiutil.TryLoadCertFromDisk(cfg.Etcd.SelfHosted.CertificatesDir, kubeadmconstants.CACertAndKeyBaseName)
	if err != nil {
		return err
	}

	creator := certCreator{client, cfg.Etcd.SelfHosted.CertificatesDir, caCert}
	secretPairs := []secretPair{
		secretPair{kubeadmconstants.EtcdOperatorEtcdTLSSecret, kubeadmconstants.EtcdClientCertAndKeyBaseName},
		secretPair{kubeadmconstants.KubeAPIServerEtcdTLSSecret, kubeadmconstants.EtcdClientCertAndKeyBaseName},
	}

	for _, secretPair := range secretPairs {
		if err := creator.createSecret(secretPair.name, secretPair.certType); err != nil {
			return err
		}
	}

	return nil
}

func (c certCreator) createSecret(name, certType string) error {
	cert, key, err := pkiutil.TryLoadCertAndKeyFromDisk(c.certDir, certType)
	if err != nil {
		return err
	}

	secret := &v1.Secret{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "extensions/v1beta1",
			Kind:       "Secret",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: metav1.NamespaceSystem,
		},
		Data: map[string][]byte{
			fmt.Sprintf("%s.crt", certType):    certutil.EncodeCertPEM(cert),
			fmt.Sprintf("%s.key", certType):    certutil.EncodePrivateKeyPEM(key),
			fmt.Sprintf("%s-ca.crt", certType): certutil.EncodeCertPEM(c.caCert),
		},
	}

	if _, err := c.client.CoreV1().Secrets(metav1.NamespaceSystem).Create(secret); err != nil {
		return fmt.Errorf("[self-hosted] Could not create %s secret: %v", name, err)
	}

	return nil
}
