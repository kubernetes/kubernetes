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

package selfhosting

import (
	"fmt"
	"io/ioutil"
	"path/filepath"
	"strings"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/apiclient"
)

type tlsKeyPair struct {
	name string
	cert string
	key  string
}

func apiServerCertificatesVolumeSource() v1.VolumeSource {
	return v1.VolumeSource{
		Projected: &v1.ProjectedVolumeSource{
			Sources: []v1.VolumeProjection{
				{
					Secret: &v1.SecretProjection{
						LocalObjectReference: v1.LocalObjectReference{
							Name: kubeadmconstants.CACertAndKeyBaseName,
						},
						Items: []v1.KeyToPath{
							{
								Key:  v1.TLSCertKey,
								Path: kubeadmconstants.CACertName,
							},
						},
					},
				},
				{
					Secret: &v1.SecretProjection{
						LocalObjectReference: v1.LocalObjectReference{
							Name: kubeadmconstants.APIServerCertAndKeyBaseName,
						},
						Items: []v1.KeyToPath{
							{
								Key:  v1.TLSCertKey,
								Path: kubeadmconstants.APIServerCertName,
							},
							{
								Key:  v1.TLSPrivateKeyKey,
								Path: kubeadmconstants.APIServerKeyName,
							},
						},
					},
				},
				{
					Secret: &v1.SecretProjection{
						LocalObjectReference: v1.LocalObjectReference{
							Name: kubeadmconstants.APIServerKubeletClientCertAndKeyBaseName,
						},
						Items: []v1.KeyToPath{
							{
								Key:  v1.TLSCertKey,
								Path: kubeadmconstants.APIServerKubeletClientCertName,
							},
							{
								Key:  v1.TLSPrivateKeyKey,
								Path: kubeadmconstants.APIServerKubeletClientKeyName,
							},
						},
					},
				},
				{
					Secret: &v1.SecretProjection{
						LocalObjectReference: v1.LocalObjectReference{
							Name: kubeadmconstants.ServiceAccountKeyBaseName,
						},
						Items: []v1.KeyToPath{
							{
								Key:  v1.TLSCertKey,
								Path: kubeadmconstants.ServiceAccountPublicKeyName,
							},
						},
					},
				},
				{
					Secret: &v1.SecretProjection{
						LocalObjectReference: v1.LocalObjectReference{
							Name: kubeadmconstants.FrontProxyCACertAndKeyBaseName,
						},
						Items: []v1.KeyToPath{
							{
								Key:  v1.TLSCertKey,
								Path: kubeadmconstants.FrontProxyCACertName,
							},
						},
					},
				},
				{
					Secret: &v1.SecretProjection{
						LocalObjectReference: v1.LocalObjectReference{
							Name: kubeadmconstants.FrontProxyClientCertAndKeyBaseName,
						},
						Items: []v1.KeyToPath{
							{
								Key:  v1.TLSCertKey,
								Path: kubeadmconstants.FrontProxyClientCertName,
							},
							{
								Key:  v1.TLSPrivateKeyKey,
								Path: kubeadmconstants.FrontProxyClientKeyName,
							},
						},
					},
				},
				{
					Secret: &v1.SecretProjection{
						LocalObjectReference: v1.LocalObjectReference{
							Name: strings.Replace(kubeadmconstants.EtcdCACertAndKeyBaseName, "/", "-", -1),
						},
						Items: []v1.KeyToPath{
							{
								Key:  v1.TLSCertKey,
								Path: kubeadmconstants.EtcdCACertName,
							},
							{
								Key:  v1.TLSPrivateKeyKey,
								Path: kubeadmconstants.EtcdCAKeyName,
							},
						},
					},
				},
				{
					Secret: &v1.SecretProjection{
						LocalObjectReference: v1.LocalObjectReference{
							Name: kubeadmconstants.APIServerEtcdClientCertAndKeyBaseName,
						},
						Items: []v1.KeyToPath{
							{
								Key:  v1.TLSCertKey,
								Path: kubeadmconstants.APIServerEtcdClientCertName,
							},
							{
								Key:  v1.TLSPrivateKeyKey,
								Path: kubeadmconstants.APIServerEtcdClientKeyName,
							},
						},
					},
				},
			},
		},
	}
}

func controllerManagerCertificatesVolumeSource() v1.VolumeSource {
	return v1.VolumeSource{
		Projected: &v1.ProjectedVolumeSource{
			Sources: []v1.VolumeProjection{
				{
					Secret: &v1.SecretProjection{
						LocalObjectReference: v1.LocalObjectReference{
							Name: kubeadmconstants.CACertAndKeyBaseName,
						},
						Items: []v1.KeyToPath{
							{
								Key:  v1.TLSCertKey,
								Path: kubeadmconstants.CACertName,
							},
							{
								Key:  v1.TLSPrivateKeyKey,
								Path: kubeadmconstants.CAKeyName,
							},
						},
					},
				},
				{
					Secret: &v1.SecretProjection{
						LocalObjectReference: v1.LocalObjectReference{
							Name: kubeadmconstants.ServiceAccountKeyBaseName,
						},
						Items: []v1.KeyToPath{
							{
								Key:  v1.TLSPrivateKeyKey,
								Path: kubeadmconstants.ServiceAccountPrivateKeyName,
							},
						},
					},
				},
				{
					Secret: &v1.SecretProjection{
						LocalObjectReference: v1.LocalObjectReference{
							Name: kubeadmconstants.FrontProxyCACertAndKeyBaseName,
						},
						Items: []v1.KeyToPath{
							{
								Key:  v1.TLSCertKey,
								Path: kubeadmconstants.FrontProxyCACertName,
							},
						},
					},
				},
			},
		},
	}
}

func kubeConfigVolumeSource(kubeconfigSecretName string) v1.VolumeSource {
	return v1.VolumeSource{
		Secret: &v1.SecretVolumeSource{
			SecretName: strings.Replace(kubeconfigSecretName, "/", "-", -1),
		},
	}
}

func uploadTLSSecrets(client clientset.Interface, certDir string) error {
	for _, tlsKeyPair := range getTLSKeyPairs() {
		secret, err := createTLSSecretFromFiles(
			tlsKeyPair.name,
			filepath.Join(certDir, tlsKeyPair.cert),
			filepath.Join(certDir, tlsKeyPair.key),
		)
		if err != nil {
			return err
		}

		if err := apiclient.CreateOrUpdateSecret(client, secret); err != nil {
			return err
		}
		fmt.Printf("[self-hosted] Created TLS secret %q from %s and %s\n", tlsKeyPair.name, tlsKeyPair.cert, tlsKeyPair.key)
	}

	return nil
}

func uploadKubeConfigSecrets(client clientset.Interface, kubeConfigDir string) error {
	files := []string{
		kubeadmconstants.SchedulerKubeConfigFileName,
		kubeadmconstants.ControllerManagerKubeConfigFileName,
	}
	for _, file := range files {
		kubeConfigPath := filepath.Join(kubeConfigDir, file)
		secret, err := createOpaqueSecretFromFile(file, kubeConfigPath)
		if err != nil {
			return err
		}

		if err := apiclient.CreateOrUpdateSecret(client, secret); err != nil {
			return err
		}
		fmt.Printf("[self-hosted] Created secret for kubeconfig file %q\n", file)
	}

	return nil
}

func createTLSSecretFromFiles(secretName, crt, key string) (*v1.Secret, error) {
	crtBytes, err := ioutil.ReadFile(crt)
	if err != nil {
		return nil, err
	}
	keyBytes, err := ioutil.ReadFile(key)
	if err != nil {
		return nil, err
	}

	return &v1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      secretName,
			Namespace: metav1.NamespaceSystem,
		},
		Type: v1.SecretTypeTLS,
		Data: map[string][]byte{
			v1.TLSCertKey:       crtBytes,
			v1.TLSPrivateKeyKey: keyBytes,
		},
	}, nil
}

func createOpaqueSecretFromFile(secretName, file string) (*v1.Secret, error) {
	fileBytes, err := ioutil.ReadFile(file)
	if err != nil {
		return nil, err
	}

	return &v1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      secretName,
			Namespace: metav1.NamespaceSystem,
		},
		Type: v1.SecretTypeOpaque,
		Data: map[string][]byte{
			filepath.Base(file): fileBytes,
		},
	}, nil
}

func getTLSKeyPairs() []*tlsKeyPair {
	return []*tlsKeyPair{
		{
			name: kubeadmconstants.CACertAndKeyBaseName,
			cert: kubeadmconstants.CACertName,
			key:  kubeadmconstants.CAKeyName,
		},
		{
			name: kubeadmconstants.APIServerCertAndKeyBaseName,
			cert: kubeadmconstants.APIServerCertName,
			key:  kubeadmconstants.APIServerKeyName,
		},
		{
			name: kubeadmconstants.APIServerKubeletClientCertAndKeyBaseName,
			cert: kubeadmconstants.APIServerKubeletClientCertName,
			key:  kubeadmconstants.APIServerKubeletClientKeyName,
		},
		{
			name: kubeadmconstants.ServiceAccountKeyBaseName,
			cert: kubeadmconstants.ServiceAccountPublicKeyName,
			key:  kubeadmconstants.ServiceAccountPrivateKeyName,
		},
		{
			name: kubeadmconstants.FrontProxyCACertAndKeyBaseName,
			cert: kubeadmconstants.FrontProxyCACertName,
			key:  kubeadmconstants.FrontProxyCAKeyName,
		},
		{
			name: kubeadmconstants.FrontProxyClientCertAndKeyBaseName,
			cert: kubeadmconstants.FrontProxyClientCertName,
			key:  kubeadmconstants.FrontProxyClientKeyName,
		},
		{
			name: strings.Replace(kubeadmconstants.EtcdCACertAndKeyBaseName, "/", "-", -1),
			cert: kubeadmconstants.EtcdCACertName,
			key:  kubeadmconstants.EtcdCAKeyName,
		},
		{
			name: kubeadmconstants.APIServerEtcdClientCertAndKeyBaseName,
			cert: kubeadmconstants.APIServerEtcdClientCertName,
			key:  kubeadmconstants.APIServerEtcdClientKeyName,
		},
	}
}
