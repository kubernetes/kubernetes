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

package master

import (
	"fmt"
	"io/ioutil"
	"path"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/pkg/api/v1"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
)

func apiServerProjectedVolume() *v1.Volume {
	return &v1.Volume{
		Name: "k8s",
		VolumeSource: v1.VolumeSource{
			Projected: &v1.ProjectedVolumeSource{
				Sources: []v1.VolumeProjection{
					{
						Secret: &v1.SecretProjection{
							LocalObjectReference: v1.LocalObjectReference{
								Name: kubeadmconstants.CACertAndKeyBaseName,
							},
							Items: []v1.KeyToPath{
								{
									Key:  "tls.crt",
									Path: path.Join("pki", kubeadmconstants.CACertName),
								},
								{
									Key:  "tls.key",
									Path: path.Join("pki", kubeadmconstants.CAKeyName),
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
									Key:  "tls.crt",
									Path: path.Join("pki", kubeadmconstants.APIServerCertName),
								},
								{
									Key:  "tls.key",
									Path: path.Join("pki", kubeadmconstants.APIServerKeyName),
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
									Key:  "tls.crt",
									Path: path.Join("pki", kubeadmconstants.APIServerKubeletClientCertName),
								},
								{
									Key:  "tls.key",
									Path: path.Join("pki", kubeadmconstants.APIServerKubeletClientKeyName),
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
									Key:  "tls.crt",
									Path: path.Join("pki", kubeadmconstants.ServiceAccountPublicKeyName),
								},
								{
									Key:  "tls.key",
									Path: path.Join("pki", kubeadmconstants.ServiceAccountPrivateKeyName),
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
									Key:  "tls.crt",
									Path: path.Join("pki", kubeadmconstants.FrontProxyCACertName),
								},
							},
						},
					},
				},
			},
		},
	}
}

func schedulerProjectedVolume() *v1.Volume {
	return &v1.Volume{
		Name: "k8s",
		VolumeSource: v1.VolumeSource{
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
		},
	}
}

func controllerManagerProjectedVolume() *v1.Volume {
	return &v1.Volume{
		Name: "k8s",
		VolumeSource: v1.VolumeSource{
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
									Key:  "tls.crt",
									Path: path.Join("pki", kubeadmconstants.CACertName),
								},
								{
									Key:  "tls.key",
									Path: path.Join("pki", kubeadmconstants.CAKeyName),
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
									Key:  "tls.key",
									Path: path.Join("pki", kubeadmconstants.ServiceAccountPrivateKeyName),
								},
							},
						},
					},
				},
			},
		},
	}
}

func createTLSSecrets(client *clientset.Clientset) error {
	tlsPairs := []string{
		kubeadmconstants.CACertAndKeyBaseName,
		kubeadmconstants.CACertName,
		kubeadmconstants.CAKeyName,
		kubeadmconstants.APIServerCertAndKeyBaseName,
		kubeadmconstants.APIServerCertName,
		kubeadmconstants.APIServerKeyName,
		kubeadmconstants.APIServerKubeletClientCertAndKeyBaseName,
		kubeadmconstants.APIServerKubeletClientCertName,
		kubeadmconstants.APIServerKubeletClientKeyName,
		kubeadmconstants.ServiceAccountKeyBaseName,
		kubeadmconstants.ServiceAccountPublicKeyName,
		kubeadmconstants.ServiceAccountPrivateKeyName,
		kubeadmconstants.FrontProxyCACertAndKeyBaseName,
		kubeadmconstants.FrontProxyCACertName,
		kubeadmconstants.FrontProxyCAKeyName,
	}
	for i := 0; i < len(tlsPairs); i = i + 3 {
		fmt.Printf("[self-hosted] Creating TLS secret %q with %s and %s\n", tlsPairs[i], tlsPairs[i+1], tlsPairs[i+2])
		secret, err := createTLSSecretFromFiles(
			tlsPairs[i],
			path.Join(kubeadmapi.GlobalEnvParams.KubernetesDir, "pki", tlsPairs[i+1]),
			path.Join(kubeadmapi.GlobalEnvParams.KubernetesDir, "pki", tlsPairs[i+2]),
		)
		if err != nil {
			return err
		}

		if _, err := client.Secrets(metav1.NamespaceSystem).Create(secret); err != nil {
			return err
		}
	}

	return nil
}

func createOpaqueSecrets(client *clientset.Clientset) error {
	files := []string{
		kubeadmconstants.SchedulerKubeConfigFileName,
		kubeadmconstants.ControllerManagerKubeConfigFileName,
	}
	for i := 0; i < len(files); i++ {
		fmt.Printf("[self-hosted] Creating secret for %q\n", files[i])
		secret, err := createOpaqueSecretFromFile(
			files[i],
			path.Join(kubeadmapi.GlobalEnvParams.KubernetesDir, files[i]),
		)
		if err != nil {
			return err
		}

		if _, err := client.Secrets(metav1.NamespaceSystem).Create(secret); err != nil {
			return err
		}
	}

	return nil
}

func createTLSSecretFromFiles(secretName, crt, key string) (*v1.Secret, error) {
	data := make(map[string][]byte, 0)

	crtBytes, err := ioutil.ReadFile(crt)
	if err != nil {
		return nil, err
	}
	data["tls.crt"] = crtBytes

	keyBytes, err := ioutil.ReadFile(key)
	if err != nil {
		return nil, err
	}
	data["tls.key"] = keyBytes

	return &v1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      secretName,
			Namespace: metav1.NamespaceSystem,
		},
		Type: v1.SecretTypeTLS,
		Data: data,
	}, nil
}

func createOpaqueSecretFromFile(secretName, file string) (*v1.Secret, error) {
	data := make(map[string][]byte, 0)

	fileBytes, err := ioutil.ReadFile(file)
	if err != nil {
		return nil, err
	}
	data[path.Base(file)] = fileBytes

	return &v1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      secretName,
			Namespace: metav1.NamespaceSystem,
		},
		Type: v1.SecretTypeOpaque,
		Data: data,
	}, nil
}
