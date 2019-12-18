/*
Copyright 2019 The Kubernetes Authors.

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
	"github.com/pkg/errors"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	errorsutil "k8s.io/apimachinery/pkg/util/errors"
	clientset "k8s.io/client-go/kubernetes"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/addons/dns"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/addons/proxy"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/apiclient"
)

// UpgradeAddons upgrades addons
func UpgradeAddons(client clientset.Interface, cfg *kubeadmapi.InitConfiguration, dryRun bool) error {
	errs := []error{}
	// Upgrade kube-dns/CoreDNS and kube-proxy
	if err := dns.EnsureDNSAddon(&cfg.ClusterConfiguration, client); err != nil {
		errs = append(errs, err)
	}
	// Remove the old DNS deployment if a new DNS service is now used (kube-dns to CoreDNS or vice versa)
	if err := removeOldDNSDeploymentIfAnotherDNSIsUsed(&cfg.ClusterConfiguration, client, dryRun); err != nil {
		errs = append(errs, err)
	}

	if err := proxy.EnsureProxyAddon(&cfg.ClusterConfiguration, &cfg.LocalAPIEndpoint, client); err != nil {
		errs = append(errs, err)
	}
	return errorsutil.NewAggregate(errs)
}

func removeOldDNSDeploymentIfAnotherDNSIsUsed(cfg *kubeadmapi.ClusterConfiguration, client clientset.Interface, dryRun bool) error {
	return apiclient.TryRunCommand(func() error {
		installedDeploymentName := kubeadmconstants.KubeDNSDeploymentName
		deploymentToDelete := kubeadmconstants.CoreDNSDeploymentName

		if cfg.DNS.Type == kubeadmapi.CoreDNS {
			installedDeploymentName = kubeadmconstants.CoreDNSDeploymentName
			deploymentToDelete = kubeadmconstants.KubeDNSDeploymentName
		}

		// If we're dry-running, we don't need to wait for the new DNS addon to become ready
		if !dryRun {
			dnsDeployment, err := client.AppsV1().Deployments(metav1.NamespaceSystem).Get(installedDeploymentName, metav1.GetOptions{})
			if err != nil {
				return err
			}
			if dnsDeployment.Status.ReadyReplicas == 0 {
				return errors.New("the DNS deployment isn't ready yet")
			}
		}

		// We don't want to wait for the DNS deployment above to become ready when dryrunning (as it never will)
		// but here we should execute the DELETE command against the dryrun clientset, as it will only be logged
		err := apiclient.DeleteDeploymentForeground(client, metav1.NamespaceSystem, deploymentToDelete)
		if err != nil && !apierrors.IsNotFound(err) {
			return err
		}
		return nil
	}, 10)
}
