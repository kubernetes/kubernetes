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

package addons

import (
	"fmt"
	"net"

	"runtime"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	kuberuntime "k8s.io/apimachinery/pkg/runtime"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	extensions "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	"k8s.io/kubernetes/pkg/registry/core/service/ipallocator"
)

// CreateEssentialAddons creates the kube-proxy and kube-dns addons
func CreateEssentialAddons(cfg *kubeadmapi.MasterConfiguration, client *clientset.Clientset) error {

	proxyConfigMapBytes, err := kubeadmutil.ParseTemplate(KubeProxyConfigMap, struct{ MasterEndpoint string }{
		// Fetch this value from the kubeconfig file
		MasterEndpoint: fmt.Sprintf("https://%s:%d", cfg.API.AdvertiseAddresses[0], cfg.API.Port),
	})
	if err != nil {
		return fmt.Errorf("error when parsing kube-proxy configmap template: %v", err)
	}

	proxyDaemonSetBytes, err := kubeadmutil.ParseTemplate(KubeProxyDaemonSet, struct{ ImageRepository, Arch, Version string }{
		ImageRepository: kubeadmapi.GlobalEnvParams.RepositoryPrefix,
		Arch:            runtime.GOARCH,
		// TODO: Fetch the version from the {API Server IP}/version
		Version: cfg.KubernetesVersion,
	})
	if err != nil {
		return fmt.Errorf("error when parsing kube-proxy daemonset template: %v", err)
	}

	dnsDeploymentBytes, err := kubeadmutil.ParseTemplate(KubeDNSDeployment, struct {
		ImageRepository, Arch, Version, DNSDomain string
		Replicas                                  int
	}{
		ImageRepository: kubeadmapi.GlobalEnvParams.RepositoryPrefix,
		Arch:            runtime.GOARCH,
		// TODO: Support larger amount of replicas?
		Replicas:  1,
		Version:   KubeDNSVersion,
		DNSDomain: cfg.Networking.DNSDomain,
	})
	if err != nil {
		return fmt.Errorf("error when parsing kube-dns deployment template: %v", err)
	}

	// Get the DNS IP
	dnsip, err := getDNSIP(cfg.Networking.ServiceSubnet)
	if err != nil {
		return err
	}

	dnsServiceBytes, err := kubeadmutil.ParseTemplate(KubeDNSService, struct{ DNSIP string }{
		DNSIP: dnsip.String(),
	})
	if err != nil {
		return fmt.Errorf("error when parsing kube-proxy configmap template: %v", err)
	}

	err = CreateKubeProxyAddon(proxyConfigMapBytes, proxyDaemonSetBytes, client)
	if err != nil {
		return err
	}
	fmt.Println("[addons] Created essential addon: kube-proxy")

	err = CreateKubeDNSAddon(dnsDeploymentBytes, dnsServiceBytes, client)
	if err != nil {
		return err
	}
	fmt.Println("[addons] Created essential addon: kube-dns")
	return nil
}

func CreateKubeProxyAddon(configMapBytes, daemonSetbytes []byte, client *clientset.Clientset) error {
	kubeproxyConfigMap := &v1.ConfigMap{}
	if err := kuberuntime.DecodeInto(api.Codecs.UniversalDecoder(), configMapBytes, kubeproxyConfigMap); err != nil {
		return fmt.Errorf("unable to decode kube-proxy configmap %v", err)
	}

	if _, err := client.CoreV1().ConfigMaps(metav1.NamespaceSystem).Create(kubeproxyConfigMap); err != nil {
		return fmt.Errorf("unable to create a new kube-proxy configmap: %v", err)
	}

	kubeproxyDaemonSet := &extensions.DaemonSet{}
	if err := kuberuntime.DecodeInto(api.Codecs.UniversalDecoder(), daemonSetbytes, kubeproxyDaemonSet); err != nil {
		return fmt.Errorf("unable to decode kube-proxy daemonset %v", err)
	}

	if _, err := client.ExtensionsV1beta1().DaemonSets(metav1.NamespaceSystem).Create(kubeproxyDaemonSet); err != nil {
		return fmt.Errorf("unable to create a new kube-proxy daemonset: %v", err)
	}
	return nil
}

func CreateKubeDNSAddon(deploymentBytes, serviceBytes []byte, client *clientset.Clientset) error {
	kubednsDeployment := &extensions.Deployment{}
	if err := kuberuntime.DecodeInto(api.Codecs.UniversalDecoder(), deploymentBytes, kubednsDeployment); err != nil {
		return fmt.Errorf("unable to decode kube-dns deployment %v", err)
	}

	// TODO: All these .Create(foo) calls should instead be more like "kubectl apply -f" commands; they should not fail if there are existing objects with the same name
	if _, err := client.ExtensionsV1beta1().Deployments(metav1.NamespaceSystem).Create(kubednsDeployment); err != nil {
		return fmt.Errorf("unable to create a new kube-dns deployment: %v", err)
	}

	kubednsService := &v1.Service{}
	if err := kuberuntime.DecodeInto(api.Codecs.UniversalDecoder(), serviceBytes, kubednsService); err != nil {
		return fmt.Errorf("unable to decode kube-dns service %v", err)
	}

	if _, err := client.CoreV1().Services(metav1.NamespaceSystem).Create(kubednsService); err != nil {
		return fmt.Errorf("unable to create a new kube-dns service: %v", err)
	}
	return nil
}

// TODO: Instead of looking at the subnet given to kubeadm, it should be possible to only use /28 or larger subnets and then
// kubeadm should look at the kubernetes service (e.g. 10.96.0.1 or 10.0.0.1) and just append a "0" at the end.
// This way, we don't need the information about the subnet in this phase => good
func getDNSIP(subnet string) (net.IP, error) {
	_, n, err := net.ParseCIDR(subnet)
	if err != nil {
		return nil, fmt.Errorf("could not parse %q: %v", subnet, err)
	}
	ip, err := ipallocator.GetIndexedIP(n, 10)
	if err != nil {
		return nil, fmt.Errorf("unable to allocate IP address for kube-dns addon from the given CIDR %q: [%v]", subnet, err)
	}
	return ip, nil
}
