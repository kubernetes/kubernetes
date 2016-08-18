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
package kubemaster

import (
	"crypto/x509"
	"encoding/hex"
	"encoding/json"
	"fmt"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/extensions"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	kubeadmapi "k8s.io/kubernetes/pkg/kubeadm/api"
	certutil "k8s.io/kubernetes/pkg/util/cert"
)

type kubeDiscovery struct {
	Deployment *extensions.Deployment
	Secret     *api.Secret
}

const (
	kubeDiscoverynName      = "kube-discovery"
	kubeDiscoverySecretName = "clusterinfo"
)

func encodeKubeDiscoverySecretData(params *kubeadmapi.BootstrapParams, caCert *x509.Certificate) map[string][]byte {
	// TODO ListenIP is probably not the right now, although it's best we have right now
	// if user provides a DNS name, or anything else, we should use that, may be it's really
	// the list of all SANs (minus internal DNS names and service IP)?

	var (
		data         = map[string][]byte{}
		endpointList = []string{}
		tokenMap     = map[string]string{}
	)

	endpointList = append(endpointList, fmt.Sprintf("https://%s:443", params.Discovery.ListenIP))
	tokenMap[params.Discovery.TokenID] = hex.EncodeToString(params.Discovery.Token)

	data["endpoint-list.json"], _ = json.Marshal(endpointList)
	data["token-map.json"], _ = json.Marshal(tokenMap)
	data["ca.pem"] = certutil.EncodeCertPEM(caCert)

	return data
}

func newKubeDiscoveryPodSpec(params *kubeadmapi.BootstrapParams) api.PodSpec {
	return api.PodSpec{
		SecurityContext: &api.PodSecurityContext{HostNetwork: true}, // TODO we should just use map it to a host port
		Containers: []api.Container{{
			Name:    kubeDiscoverynName,
			Image:   params.EnvParams["discovery_image"],
			Command: []string{"/usr/bin/kube-discovery"},
			VolumeMounts: []api.VolumeMount{{
				Name:      kubeDiscoverySecretName,
				MountPath: "/tmp/secret", // TODO use a shared constant
				ReadOnly:  true,
			}},
		}},
		Volumes: []api.Volume{{
			Name: kubeDiscoverySecretName,
			VolumeSource: api.VolumeSource{
				Secret: &api.SecretVolumeSource{SecretName: kubeDiscoverySecretName},
			}},
		},
	}
}

func newKubeDiscovery(params *kubeadmapi.BootstrapParams, caCert *x509.Certificate) kubeDiscovery {
	// TODO pin to master
	return kubeDiscovery{
		Deployment: NewDeployment(kubeDiscoverynName, 1, newKubeDiscoveryPodSpec(params)),
		Secret: &api.Secret{
			ObjectMeta: api.ObjectMeta{Name: kubeDiscoverySecretName},
			Type:       api.SecretTypeOpaque,
			Data:       encodeKubeDiscoverySecretData(params, caCert),
		},
	}
}

func CreateDiscoveryDeploymentAndSecret(params *kubeadmapi.BootstrapParams, client *clientset.Clientset, caCert *x509.Certificate) error {
	kd := newKubeDiscovery(params, caCert)

	if _, err := client.Extensions().Deployments(api.NamespaceSystem).Create(kd.Deployment); err != nil {
		return fmt.Errorf("<master/discovery> failed to create %q deployment", kubeDiscoverynName)
	}
	if _, err := client.Secrets(api.NamespaceSystem).Create(kd.Secret); err != nil {
		return fmt.Errorf("<master/discovery> failed to create %q secret", kubeDiscoverySecretName)
	}

	fmt.Println("<master/discovery> created essential addon: kube-discovery")

	// TODO we should probably wait for the pod to become ready

	return nil
}
