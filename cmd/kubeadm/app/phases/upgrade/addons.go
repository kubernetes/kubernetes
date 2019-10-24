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

package upgrade

import (
	errorsutil "k8s.io/apimachinery/pkg/util/errors"
	clientset "k8s.io/client-go/kubernetes"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/addons/dns"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/addons/proxy"
)

// PerformAddonUpgradeTasks runs all needed addon upgrade tasks
func PerformAddonUpgradeTasks(client clientset.Interface, cfg *kubeadmapi.InitConfiguration, dryRun bool) error {
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
