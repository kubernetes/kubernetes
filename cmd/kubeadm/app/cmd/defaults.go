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

package cmd

import (
	"fmt"
	"strconv"

	netutil "k8s.io/apimachinery/pkg/util/net"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapiext "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1alpha1"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"

	"github.com/blang/semver"
)

var (
	minK8sVersion = semver.MustParse(kubeadmconstants.MinimumControlPlaneVersion)
)

func setInitDynamicDefaults(cfg *kubeadmapi.MasterConfiguration) error {
	// Auto-detect the IP
	if len(cfg.API.AdvertiseAddresses) == 0 {
		ip, err := netutil.ChooseHostInterface()
		if err != nil {
			return err
		}
		cfg.API.AdvertiseAddresses = []string{ip.String()}
	}

	// Validate version argument
	ver, err := kubeadmutil.KubernetesReleaseVersion(cfg.KubernetesVersion)
	if err != nil {
		if cfg.KubernetesVersion != kubeadmapiext.DefaultKubernetesVersion {
			return err
		} else {
			ver = kubeadmapiext.DefaultKubernetesFallbackVersion
		}
	}
	cfg.KubernetesVersion = ver

	// Omit the "v" in the beginning, otherwise semver will fail
	k8sVersion, err := semver.Parse(cfg.KubernetesVersion[1:])
	if err != nil {
		return fmt.Errorf("couldn't parse kubernetes version %q: %v", cfg.KubernetesVersion, err)
	}
	if k8sVersion.LT(minK8sVersion) {
		return fmt.Errorf("this version of kubeadm only supports deploying clusters with the control plane version >= v1.6.0-alpha.1. Current version: %s", cfg.KubernetesVersion)
	}

	fmt.Printf("[init] Using Kubernetes version: %s\n", cfg.KubernetesVersion)
	fmt.Printf("[init] Using Authorization mode: %s\n", cfg.AuthorizationMode)

	// Warn about the limitations with the current cloudprovider solution.
	if cfg.CloudProvider != "" {
		fmt.Println("[init] WARNING: For cloudprovider integrations to work --cloud-provider must be set for all kubelets in the cluster.")
		fmt.Println("\t(/etc/systemd/system/kubelet.service.d/10-kubeadm.conf should be edited for this purpose)")
	}

	// Validate token if any, otherwise generate
	if cfg.Discovery.Token != nil {
		if cfg.Discovery.Token.ID != "" && cfg.Discovery.Token.Secret != "" {
			fmt.Printf("[init] A token has been provided, validating [%s]\n", kubeadmutil.BearerToken(cfg.Discovery.Token))
			if valid, err := kubeadmutil.ValidateToken(cfg.Discovery.Token); valid == false {
				return err
			}
		} else {
			fmt.Println("[init] A token has not been provided, generating one")
			if err := kubeadmutil.GenerateToken(cfg.Discovery.Token); err != nil {
				return err
			}
		}

		// If there aren't any addresses specified, default to the first advertised address which can be user-provided or the default network interface's IP address
		if len(cfg.Discovery.Token.Addresses) == 0 {
			cfg.Discovery.Token.Addresses = []string{cfg.API.AdvertiseAddresses[0] + ":" + strconv.Itoa(kubeadmapiext.DefaultDiscoveryBindPort)}
		}
	}

	return nil
}
