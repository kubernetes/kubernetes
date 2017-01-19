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
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"

	"github.com/blang/semver"
)

var (
	// Maximum version when using AllowAll as the default authz mode. Everything above this will use RBAC by default.
	allowAllMaxVersion = semver.MustParse("1.6.0-alpha.0")
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
	fmt.Println("[init] Using Kubernetes version:", ver)

	// Omit the "v" in the beginning, otherwise semver will fail
	// If the version is newer than the specified version, RBAC v1beta1 support is enabled in the apiserver so we can default to RBAC
	k8sVersion, err := semver.Parse(cfg.KubernetesVersion[1:])
	if k8sVersion.GT(allowAllMaxVersion) {
		cfg.AuthorizationMode = "RBAC"
	}

	fmt.Println("[init] Using Authorization mode:", cfg.AuthorizationMode)

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
