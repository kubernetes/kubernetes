/*
Copyright 2014 The Kubernetes Authors.

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

package options

import (
	"fmt"
	"net"
	"os"
	"strings"
	"time"

	serveroptions "k8s.io/apiserver/pkg/server/options"
	"k8s.io/client-go/util/keyutil"
	_ "k8s.io/component-base/metrics/prometheus/workqueue"
	"k8s.io/klog/v2"
	netutils "k8s.io/utils/net"

	"k8s.io/kubernetes/pkg/controlplane"
	"k8s.io/kubernetes/pkg/kubeapiserver"
	kubeauthenticator "k8s.io/kubernetes/pkg/kubeapiserver/authenticator"
	"k8s.io/kubernetes/pkg/serviceaccount"
)

// completedOptions is a private wrapper that enforces a call of Complete() before Run can be invoked.
type completedOptions struct {
	*ServerRunOptions
}

type CompletedOptions struct {
	completedOptions
}

// Complete set default ServerRunOptions.
// Should be called after kube-apiserver flags parsed.
func Complete(opts *ServerRunOptions) (CompletedOptions, error) {
	// set defaults
	if err := opts.GenericServerRunOptions.DefaultAdvertiseAddress(opts.SecureServing.SecureServingOptions); err != nil {
		return CompletedOptions{}, err
	}

	// process s.ServiceClusterIPRange from list to Primary and Secondary
	// we process secondary only if provided by user
	apiServerServiceIP, primaryServiceIPRange, secondaryServiceIPRange, err := getServiceIPAndRanges(opts.ServiceClusterIPRanges)
	if err != nil {
		return CompletedOptions{}, err
	}
	opts.PrimaryServiceClusterIPRange = primaryServiceIPRange
	opts.SecondaryServiceClusterIPRange = secondaryServiceIPRange
	opts.APIServerServiceIP = apiServerServiceIP

	if err := opts.SecureServing.MaybeDefaultWithSelfSignedCerts(opts.GenericServerRunOptions.AdvertiseAddress.String(), []string{"kubernetes.default.svc", "kubernetes.default", "kubernetes"}, []net.IP{apiServerServiceIP}); err != nil {
		return CompletedOptions{}, fmt.Errorf("error creating self-signed certificates: %v", err)
	}

	if len(opts.GenericServerRunOptions.ExternalHost) == 0 {
		if len(opts.GenericServerRunOptions.AdvertiseAddress) > 0 {
			opts.GenericServerRunOptions.ExternalHost = opts.GenericServerRunOptions.AdvertiseAddress.String()
		} else {
			hostname, err := os.Hostname()
			if err != nil {
				return CompletedOptions{}, fmt.Errorf("error finding host name: %v", err)
			}
			opts.GenericServerRunOptions.ExternalHost = hostname
		}
		klog.Infof("external host was not specified, using %v", opts.GenericServerRunOptions.ExternalHost)
	}

	opts.Authentication.ApplyAuthorization(opts.Authorization)

	// Use (ServiceAccountSigningKeyFile != "") as a proxy to the user enabling
	// TokenRequest functionality. This defaulting was convenient, but messed up
	// a lot of people when they rotated their serving cert with no idea it was
	// connected to their service account keys. We are taking this opportunity to
	// remove this problematic defaulting.
	if opts.ServiceAccountSigningKeyFile == "" {
		// Default to the private server key for service account token signing
		if len(opts.Authentication.ServiceAccounts.KeyFiles) == 0 && opts.SecureServing.ServerCert.CertKey.KeyFile != "" {
			if kubeauthenticator.IsValidServiceAccountKeyFile(opts.SecureServing.ServerCert.CertKey.KeyFile) {
				opts.Authentication.ServiceAccounts.KeyFiles = []string{opts.SecureServing.ServerCert.CertKey.KeyFile}
			} else {
				klog.Warning("No TLS key provided, service account token authentication disabled")
			}
		}
	}

	if opts.ServiceAccountSigningKeyFile != "" && len(opts.Authentication.ServiceAccounts.Issuers) != 0 && opts.Authentication.ServiceAccounts.Issuers[0] != "" {
		sk, err := keyutil.PrivateKeyFromFile(opts.ServiceAccountSigningKeyFile)
		if err != nil {
			return CompletedOptions{}, fmt.Errorf("failed to parse service-account-issuer-key-file: %v", err)
		}
		if opts.Authentication.ServiceAccounts.MaxExpiration != 0 {
			lowBound := time.Hour
			upBound := time.Duration(1<<32) * time.Second
			if opts.Authentication.ServiceAccounts.MaxExpiration < lowBound ||
				opts.Authentication.ServiceAccounts.MaxExpiration > upBound {
				return CompletedOptions{}, fmt.Errorf("the service-account-max-token-expiration must be between 1 hour and 2^32 seconds")
			}
			if opts.Authentication.ServiceAccounts.ExtendExpiration {
				if opts.Authentication.ServiceAccounts.MaxExpiration < serviceaccount.WarnOnlyBoundTokenExpirationSeconds*time.Second {
					klog.Warningf("service-account-extend-token-expiration is true, in order to correctly trigger safe transition logic, service-account-max-token-expiration must be set longer than %d seconds (currently %s)", serviceaccount.WarnOnlyBoundTokenExpirationSeconds, opts.Authentication.ServiceAccounts.MaxExpiration)
				}
				if opts.Authentication.ServiceAccounts.MaxExpiration < serviceaccount.ExpirationExtensionSeconds*time.Second {
					klog.Warningf("service-account-extend-token-expiration is true, enabling tokens valid up to %d seconds, which is longer than service-account-max-token-expiration set to %s seconds", serviceaccount.ExpirationExtensionSeconds, opts.Authentication.ServiceAccounts.MaxExpiration)
				}
			}
		}

		opts.ServiceAccountIssuer, err = serviceaccount.JWTTokenGenerator(opts.Authentication.ServiceAccounts.Issuers[0], sk)
		if err != nil {
			return CompletedOptions{}, fmt.Errorf("failed to build token generator: %v", err)
		}
		opts.ServiceAccountTokenMaxExpiration = opts.Authentication.ServiceAccounts.MaxExpiration
	}

	if opts.Etcd.EnableWatchCache {
		sizes := kubeapiserver.DefaultWatchCacheSizes()
		// Ensure that overrides parse correctly.
		userSpecified, err := serveroptions.ParseWatchCacheSizes(opts.Etcd.WatchCacheSizes)
		if err != nil {
			return CompletedOptions{}, err
		}
		for resource, size := range userSpecified {
			sizes[resource] = size
		}
		opts.Etcd.WatchCacheSizes, err = serveroptions.WriteWatchCacheSizes(sizes)
		if err != nil {
			return CompletedOptions{}, err
		}
	}

	for key, value := range opts.APIEnablement.RuntimeConfig {
		if key == "v1" || strings.HasPrefix(key, "v1/") ||
			key == "api/v1" || strings.HasPrefix(key, "api/v1/") {
			delete(opts.APIEnablement.RuntimeConfig, key)
			opts.APIEnablement.RuntimeConfig["/v1"] = value
		}
		if key == "api/legacy" {
			delete(opts.APIEnablement.RuntimeConfig, key)
		}
	}

	return CompletedOptions{completedOptions: completedOptions{ServerRunOptions: opts}}, nil
}

func getServiceIPAndRanges(serviceClusterIPRanges string) (net.IP, net.IPNet, net.IPNet, error) {
	serviceClusterIPRangeList := []string{}
	if serviceClusterIPRanges != "" {
		serviceClusterIPRangeList = strings.Split(serviceClusterIPRanges, ",")
	}

	var apiServerServiceIP net.IP
	var primaryServiceIPRange net.IPNet
	var secondaryServiceIPRange net.IPNet
	var err error
	// nothing provided by user, use default range (only applies to the Primary)
	if len(serviceClusterIPRangeList) == 0 {
		var primaryServiceClusterCIDR net.IPNet
		primaryServiceIPRange, apiServerServiceIP, err = controlplane.ServiceIPRange(primaryServiceClusterCIDR)
		if err != nil {
			return net.IP{}, net.IPNet{}, net.IPNet{}, fmt.Errorf("error determining service IP ranges: %v", err)
		}
		return apiServerServiceIP, primaryServiceIPRange, net.IPNet{}, nil
	}

	_, primaryServiceClusterCIDR, err := netutils.ParseCIDRSloppy(serviceClusterIPRangeList[0])
	if err != nil {
		return net.IP{}, net.IPNet{}, net.IPNet{}, fmt.Errorf("service-cluster-ip-range[0] is not a valid cidr")
	}

	primaryServiceIPRange, apiServerServiceIP, err = controlplane.ServiceIPRange(*primaryServiceClusterCIDR)
	if err != nil {
		return net.IP{}, net.IPNet{}, net.IPNet{}, fmt.Errorf("error determining service IP ranges for primary service cidr: %v", err)
	}

	// user provided at least two entries
	// note: validation asserts that the list is max of two dual stack entries
	if len(serviceClusterIPRangeList) > 1 {
		_, secondaryServiceClusterCIDR, err := netutils.ParseCIDRSloppy(serviceClusterIPRangeList[1])
		if err != nil {
			return net.IP{}, net.IPNet{}, net.IPNet{}, fmt.Errorf("service-cluster-ip-range[1] is not an ip net")
		}
		secondaryServiceIPRange = *secondaryServiceClusterCIDR
	}
	return apiServerServiceIP, primaryServiceIPRange, secondaryServiceIPRange, nil
}
