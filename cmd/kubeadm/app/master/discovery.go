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

package master

import (
	"crypto/x509"
	"encoding/json"
	"fmt"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapiext "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1alpha1"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	extensions "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	certutil "k8s.io/kubernetes/pkg/util/cert"
)

type kubeDiscovery struct {
	Deployment *extensions.Deployment
	Secret     *v1.Secret
}

const (
	kubeDiscoveryName       = "kube-discovery"
	kubeDiscoverySecretName = "clusterinfo"
)

func encodeKubeDiscoverySecretData(dcfg *kubeadmapi.TokenDiscovery, apicfg kubeadmapi.API, caCert *x509.Certificate) map[string][]byte {
	var (
		data         = map[string][]byte{}
		endpointList = []string{}
		tokenMap     = map[string]string{}
	)

	for _, addr := range apicfg.AdvertiseAddresses {
		endpointList = append(endpointList, fmt.Sprintf("https://%s:%d", addr, apicfg.Port))
	}

	tokenMap[dcfg.ID] = dcfg.Secret

	data["endpoint-list.json"], _ = json.Marshal(endpointList)
	data["token-map.json"], _ = json.Marshal(tokenMap)
	data["ca.pem"] = certutil.EncodeCertPEM(caCert)

	return data
}

func newKubeDiscoveryPodSpec(cfg *kubeadmapi.MasterConfiguration) v1.PodSpec {
	return v1.PodSpec{
		// We have to use host network namespace, as `HostPort`/`HostIP` are Docker's
		// business and CNI support isn't quite there yet (except for kubenet)
		// (see https://github.com/kubernetes/kubernetes/issues/31307)
		// TODO update this when #31307 is resolved
		HostNetwork:     true,
		SecurityContext: &v1.PodSecurityContext{},
		Containers: []v1.Container{{
			Name:    kubeDiscoveryName,
			Image:   kubeadmapi.GlobalEnvParams.DiscoveryImage,
			Command: []string{"/usr/local/bin/kube-discovery"},
			VolumeMounts: []v1.VolumeMount{{
				Name:      kubeDiscoverySecretName,
				MountPath: "/tmp/secret", // TODO use a shared constant
				ReadOnly:  true,
			}},
			Ports: []v1.ContainerPort{
				// TODO when CNI issue (#31307) is resolved, we should consider adding
				// `HostIP: s.API.AdvertiseAddrs[0]`, if there is only one address`
				{Name: "http", ContainerPort: kubeadmapiext.DefaultDiscoveryBindPort, HostPort: kubeadmutil.DiscoveryPort(cfg.Discovery.Token)},
			},
			SecurityContext: &v1.SecurityContext{
				SELinuxOptions: &v1.SELinuxOptions{
					// TODO: This implies our discovery container is not being restricted by
					// SELinux. This is not optimal and would be nice to adjust in future
					// so it can read /tmp/secret, but for now this avoids recommending
					// setenforce 0 system-wide.
					Type: "spc_t",
				},
			},
		}},
		Volumes: []v1.Volume{{
			Name: kubeDiscoverySecretName,
			VolumeSource: v1.VolumeSource{
				Secret: &v1.SecretVolumeSource{SecretName: kubeDiscoverySecretName},
			}},
		},
	}
}

func newKubeDiscovery(cfg *kubeadmapi.MasterConfiguration, caCert *x509.Certificate) kubeDiscovery {
	kd := kubeDiscovery{
		Deployment: NewDeployment(kubeDiscoveryName, 1, newKubeDiscoveryPodSpec(cfg)),
		Secret: &v1.Secret{
			ObjectMeta: v1.ObjectMeta{Name: kubeDiscoverySecretName},
			Type:       v1.SecretTypeOpaque,
			Data:       encodeKubeDiscoverySecretData(cfg.Discovery.Token, cfg.API, caCert),
		},
	}

	SetMasterTaintTolerations(&kd.Deployment.Spec.Template.ObjectMeta)
	SetNodeAffinity(&kd.Deployment.Spec.Template.ObjectMeta, MasterNodeAffinity(), NativeArchitectureNodeAffinity())

	return kd
}

func CreateDiscoveryDeploymentAndSecret(cfg *kubeadmapi.MasterConfiguration, client *clientset.Clientset, caCert *x509.Certificate) error {
	kd := newKubeDiscovery(cfg, caCert)

	if _, err := client.Extensions().Deployments(api.NamespaceSystem).Create(kd.Deployment); err != nil {
		return fmt.Errorf("failed to create %q deployment [%v]", kubeDiscoveryName, err)
	}
	if _, err := client.Secrets(api.NamespaceSystem).Create(kd.Secret); err != nil {
		return fmt.Errorf("failed to create %q secret [%v]", kubeDiscoverySecretName, err)
	}

	fmt.Println("[token-discovery] Created the kube-discovery deployment, waiting for it to become ready")

	start := time.Now()
	wait.PollInfinite(apiCallRetryInterval, func() (bool, error) {
		d, err := client.Extensions().Deployments(api.NamespaceSystem).Get(kubeDiscoveryName, metav1.GetOptions{})
		if err != nil {
			return false, nil
		}
		if d.Status.AvailableReplicas < 1 {
			return false, nil
		}
		return true, nil
	})
	fmt.Printf("[token-discovery] kube-discovery is ready after %f seconds\n", time.Since(start).Seconds())

	return nil
}
