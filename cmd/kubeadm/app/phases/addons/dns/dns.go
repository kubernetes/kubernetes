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

package dns

import (
	"fmt"
	"net"
	"runtime"

	"k8s.io/api/core/v1"
	extensions "k8s.io/api/extensions/v1beta1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	kuberuntime "k8s.io/apimachinery/pkg/runtime"
	clientset "k8s.io/client-go/kubernetes"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	"k8s.io/kubernetes/pkg/api"
)

// EnsureDNSAddon creates the kube-dns addon
func EnsureDNSAddon(cfg *kubeadmapi.MasterConfiguration, client clientset.Interface) error {
	if err := CreateServiceAccount(client); err != nil {
		return err
	}

	dnsDeploymentBytes, err := kubeadmutil.ParseTemplate(KubeDNSDeployment, struct{ ImageRepository, Arch, Version, DNSDomain, MasterTaintKey string }{
		ImageRepository: cfg.ImageRepository,
		Arch:            runtime.GOARCH,
		Version:         KubeDNSVersion,
		DNSDomain:       cfg.Networking.DNSDomain,
		MasterTaintKey:  kubeadmconstants.LabelNodeRoleMaster,
	})
	if err != nil {
		return fmt.Errorf("error when parsing kube-dns deployment template: %v", err)
	}

	dnsip, err := getDNSIP(client)
	if err != nil {
		return err
	}

	dnsServiceBytes, err := kubeadmutil.ParseTemplate(KubeDNSService, struct{ DNSIP string }{
		DNSIP: dnsip.String(),
	})
	if err != nil {
		return fmt.Errorf("error when parsing kube-proxy configmap template: %v", err)
	}

	if err = createKubeDNSAddon(dnsDeploymentBytes, dnsServiceBytes, client); err != nil {
		return err
	}
	fmt.Println("[addons] Applied essential addon: kube-dns")
	return nil
}

// CreateServiceAccount creates the necessary serviceaccounts that kubeadm uses/might use, if they don't already exist.
func CreateServiceAccount(client clientset.Interface) error {
	sa := v1.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{
			Name:      kubeadmconstants.KubeDNSServiceAccountName,
			Namespace: metav1.NamespaceSystem,
		},
	}
	if _, err := client.CoreV1().ServiceAccounts(metav1.NamespaceSystem).Create(&sa); err != nil {
		if !apierrors.IsAlreadyExists(err) {
			return err
		}
	}
	return nil
}

func createKubeDNSAddon(deploymentBytes, serviceBytes []byte, client clientset.Interface) error {
	kubednsDeployment := &extensions.Deployment{}
	if err := kuberuntime.DecodeInto(api.Codecs.UniversalDecoder(), deploymentBytes, kubednsDeployment); err != nil {
		return fmt.Errorf("unable to decode kube-dns deployment %v", err)
	}

	if _, err := client.ExtensionsV1beta1().Deployments(metav1.NamespaceSystem).Create(kubednsDeployment); err != nil {
		if !apierrors.IsAlreadyExists(err) {
			return fmt.Errorf("unable to create a new kube-dns deployment: %v", err)
		}

		if _, err := client.ExtensionsV1beta1().Deployments(metav1.NamespaceSystem).Update(kubednsDeployment); err != nil {
			return fmt.Errorf("unable to update the kube-dns deployment: %v", err)
		}
	}

	kubednsService := &v1.Service{}
	if err := kuberuntime.DecodeInto(api.Codecs.UniversalDecoder(), serviceBytes, kubednsService); err != nil {
		return fmt.Errorf("unable to decode kube-dns service %v", err)
	}

	if _, err := client.CoreV1().Services(metav1.NamespaceSystem).Create(kubednsService); err != nil {
		// Ignore if the Service is invalid with this error message:
		// 	Service "kube-dns" is invalid: spec.clusterIP: Invalid value: "10.96.0.10": provided IP is already allocated

		if !apierrors.IsAlreadyExists(err) && !apierrors.IsInvalid(err) {
			return fmt.Errorf("unable to create a new kube-dns service: %v", err)
		}

		if _, err := client.CoreV1().Services(metav1.NamespaceSystem).Update(kubednsService); err != nil {
			return fmt.Errorf("unable to create/update the kube-dns service: %v", err)
		}
	}
	return nil
}

// getDNSIP fetches the kubernetes service's ClusterIP and appends a "0" to it in order to get the DNS IP
func getDNSIP(client clientset.Interface) (net.IP, error) {
	k8ssvc, err := client.CoreV1().Services(metav1.NamespaceDefault).Get("kubernetes", metav1.GetOptions{})
	if err != nil {
		return nil, fmt.Errorf("couldn't fetch information about the kubernetes service: %v", err)
	}

	if len(k8ssvc.Spec.ClusterIP) == 0 {
		return nil, fmt.Errorf("couldn't fetch a valid clusterIP from the kubernetes service")
	}

	// Build an IP by taking the kubernetes service's clusterIP and appending a "0" and checking that it's valid
	dnsIP := net.ParseIP(fmt.Sprintf("%s0", k8ssvc.Spec.ClusterIP))
	if dnsIP == nil {
		return nil, fmt.Errorf("could not parse dns ip %q: %v", dnsIP, err)
	}
	return dnsIP, nil
}
