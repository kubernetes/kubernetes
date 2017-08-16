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
	"path"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/features"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
)

const (
	volumeName      = "k8s"
	volumeMountName = "k8s"
)

type tlsKeyPair struct {
	name string
	cert string
	key  string
}

func k8sSelfHostedVolumeMount() v1.VolumeMount {
	return v1.VolumeMount{
		Name:      volumeMountName,
		MountPath: kubeadmconstants.KubernetesDir,
		ReadOnly:  true,
	}
}

func apiServerVolume(cfg *kubeadmapi.MasterConfiguration) v1.Volume {
	var volumeSource v1.VolumeSource
	if features.Enabled(cfg.FeatureFlags, features.StoreCertsInSecrets) {
		volumeSource = v1.VolumeSource{
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
									Path: path.Join(path.Base(cfg.CertificatesDir), kubeadmconstants.CACertName),
								},
								{
									Key:  v1.TLSPrivateKeyKey,
									Path: path.Join(path.Base(cfg.CertificatesDir), kubeadmconstants.CAKeyName),
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
									Path: path.Join(path.Base(cfg.CertificatesDir), kubeadmconstants.APIServerCertName),
								},
								{
									Key:  v1.TLSPrivateKeyKey,
									Path: path.Join(path.Base(cfg.CertificatesDir), kubeadmconstants.APIServerKeyName),
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
									Path: path.Join(path.Base(cfg.CertificatesDir), kubeadmconstants.APIServerKubeletClientCertName),
								},
								{
									Key:  v1.TLSPrivateKeyKey,
									Path: path.Join(path.Base(cfg.CertificatesDir), kubeadmconstants.APIServerKubeletClientKeyName),
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
									Path: path.Join(path.Base(cfg.CertificatesDir), kubeadmconstants.ServiceAccountPublicKeyName),
								},
								{
									Key:  v1.TLSPrivateKeyKey,
									Path: path.Join(path.Base(cfg.CertificatesDir), kubeadmconstants.ServiceAccountPrivateKeyName),
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
									Path: path.Join(path.Base(cfg.CertificatesDir), kubeadmconstants.FrontProxyCACertName),
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
									Path: path.Join(path.Base(cfg.CertificatesDir), kubeadmconstants.FrontProxyClientCertName),
								},
								{
									Key:  v1.TLSPrivateKeyKey,
									Path: path.Join(path.Base(cfg.CertificatesDir), kubeadmconstants.FrontProxyClientKeyName),
								},
							},
						},
					},
				},
			},
		}
	} else {
		volumeSource = v1.VolumeSource{
			HostPath: &v1.HostPathVolumeSource{
				Path: kubeadmconstants.KubernetesDir,
			},
		}
	}
	return v1.Volume{
		Name:         volumeName,
		VolumeSource: volumeSource,
	}
}

func schedulerVolume(cfg *kubeadmapi.MasterConfiguration) v1.Volume {
	var volumeSource v1.VolumeSource
	if features.Enabled(cfg.FeatureFlags, features.StoreCertsInSecrets) {
		volumeSource = v1.VolumeSource{
			Projected: &v1.ProjectedVolumeSource{
				Sources: []v1.VolumeProjection{
					{
						Secret: &v1.SecretProjection{
							LocalObjectReference: v1.LocalObjectReference{
								Name: kubeadmconstants.SchedulerKubeConfigFileName,
							},
						},
					},
				},
			},
		}
	} else {
		volumeSource = v1.VolumeSource{
			HostPath: &v1.HostPathVolumeSource{
				Path: kubeadmconstants.KubernetesDir,
			},
		}
	}
	return v1.Volume{
		Name:         volumeName,
		VolumeSource: volumeSource,
	}
}

func controllerManagerVolume(cfg *kubeadmapi.MasterConfiguration) v1.Volume {
	var volumeSource v1.VolumeSource
	if features.Enabled(cfg.FeatureFlags, features.StoreCertsInSecrets) {
		volumeSource = v1.VolumeSource{
			Projected: &v1.ProjectedVolumeSource{
				Sources: []v1.VolumeProjection{
					{
						Secret: &v1.SecretProjection{
							LocalObjectReference: v1.LocalObjectReference{
								Name: kubeadmconstants.ControllerManagerKubeConfigFileName,
							},
						},
					},
					{
						Secret: &v1.SecretProjection{
							LocalObjectReference: v1.LocalObjectReference{
								Name: kubeadmconstants.CACertAndKeyBaseName,
							},
							Items: []v1.KeyToPath{
								{
									Key:  v1.TLSCertKey,
									Path: path.Join(path.Base(cfg.CertificatesDir), kubeadmconstants.CACertName),
								},
								{
									Key:  v1.TLSPrivateKeyKey,
									Path: path.Join(path.Base(cfg.CertificatesDir), kubeadmconstants.CAKeyName),
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
									Path: path.Join(path.Base(cfg.CertificatesDir), kubeadmconstants.ServiceAccountPrivateKeyName),
								},
							},
						},
					},
				},
			},
		}
	} else {
		volumeSource = v1.VolumeSource{
			HostPath: &v1.HostPathVolumeSource{
				Path: kubeadmconstants.KubernetesDir,
			},
		}
	}
	return v1.Volume{
		Name:         volumeName,
		VolumeSource: volumeSource,
	}
}

func createTLSSecrets(cfg *kubeadmapi.MasterConfiguration, client clientset.Interface) error {
	for _, tlsKeyPair := range getTLSKeyPairs() {
		secret, err := createTLSSecretFromFiles(
			tlsKeyPair.name,
			path.Join(cfg.CertificatesDir, tlsKeyPair.cert),
			path.Join(cfg.CertificatesDir, tlsKeyPair.key),
		)
		if err != nil {
			return err
		}

		if _, err := client.CoreV1().Secrets(metav1.NamespaceSystem).Create(secret); err != nil {
			return err
		}
		fmt.Printf("[self-hosted] Created TLS secret %q from %s and %s\n", tlsKeyPair.name, tlsKeyPair.cert, tlsKeyPair.key)
	}

	return nil
}

func createOpaqueSecrets(cfg *kubeadmapi.MasterConfiguration, client clientset.Interface) error {
	files := []string{
		kubeadmconstants.SchedulerKubeConfigFileName,
		kubeadmconstants.ControllerManagerKubeConfigFileName,
	}
	for _, file := range files {
		secret, err := createOpaqueSecretFromFile(
			file,
			path.Join(kubeadmconstants.KubernetesDir, file),
		)
		if err != nil {
			return err
		}

		if _, err := client.CoreV1().Secrets(metav1.NamespaceSystem).Create(secret); err != nil {
			return err
		}
		fmt.Printf("[self-hosted] Created secret %q\n", file)
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
			path.Base(file): fileBytes,
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
	}
}
