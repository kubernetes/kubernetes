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

package controlplane

import (
	"testing"

	genericapiserver "k8s.io/apiserver/pkg/server"
	"k8s.io/client-go/kubernetes"
	netutils "k8s.io/utils/net"

	"k8s.io/kubernetes/pkg/controlplane/reconcilers"
)

func Test_completedConfig_NewBootstrapController(t *testing.T) {
	_, ipv4cidr, err := netutils.ParseCIDRSloppy("192.168.0.0/24")
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}

	_, ipv6cidr, err := netutils.ParseCIDRSloppy("2001:db8::/112")
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}

	ipv4address := netutils.ParseIPSloppy("192.168.1.1")
	ipv6address := netutils.ParseIPSloppy("2001:db8::1")

	type args struct {
		client kubernetes.Interface
	}
	tests := []struct {
		name        string
		config      genericapiserver.Config
		extraConfig *ExtraConfig
		args        args
		wantErr     bool
	}{
		{
			name: "master endpoint reconciler - IPv4 families",
			extraConfig: &ExtraConfig{
				EndpointReconcilerType: reconcilers.MasterCountReconcilerType,
				ServiceIPRange:         *ipv4cidr,
			},
			config: genericapiserver.Config{
				PublicAddress: ipv4address,
				SecureServing: &genericapiserver.SecureServingInfo{Listener: fakeLocalhost443Listener{}},
			},
			wantErr: false,
		},
		{
			name: "master endpoint reconciler - IPv6 families",
			extraConfig: &ExtraConfig{
				EndpointReconcilerType: reconcilers.MasterCountReconcilerType,
				ServiceIPRange:         *ipv6cidr,
			},
			config: genericapiserver.Config{
				PublicAddress: ipv6address,
				SecureServing: &genericapiserver.SecureServingInfo{Listener: fakeLocalhost443Listener{}},
			},
			wantErr: false,
		},
		{
			name: "master endpoint reconciler - wrong IP families",
			extraConfig: &ExtraConfig{
				EndpointReconcilerType: reconcilers.MasterCountReconcilerType,
				ServiceIPRange:         *ipv4cidr,
			},
			config: genericapiserver.Config{
				PublicAddress: ipv6address,
				SecureServing: &genericapiserver.SecureServingInfo{Listener: fakeLocalhost443Listener{}},
			},
			wantErr: true,
		},
		{
			name: "master endpoint reconciler - wrong IP families",
			extraConfig: &ExtraConfig{
				EndpointReconcilerType: reconcilers.MasterCountReconcilerType,
				ServiceIPRange:         *ipv6cidr,
			},
			config: genericapiserver.Config{
				PublicAddress: ipv4address,
				SecureServing: &genericapiserver.SecureServingInfo{Listener: fakeLocalhost443Listener{}},
			},
			wantErr: true,
		},
		{
			name: "lease endpoint reconciler - IPv4 families",
			extraConfig: &ExtraConfig{
				EndpointReconcilerType: reconcilers.LeaseEndpointReconcilerType,
				ServiceIPRange:         *ipv4cidr,
			},
			config: genericapiserver.Config{
				PublicAddress: ipv4address,
				SecureServing: &genericapiserver.SecureServingInfo{Listener: fakeLocalhost443Listener{}},
			},
			wantErr: false,
		},
		{
			name: "lease endpoint reconciler - IPv6 families",
			extraConfig: &ExtraConfig{
				EndpointReconcilerType: reconcilers.LeaseEndpointReconcilerType,
				ServiceIPRange:         *ipv6cidr,
			},
			config: genericapiserver.Config{
				PublicAddress: ipv6address,
				SecureServing: &genericapiserver.SecureServingInfo{Listener: fakeLocalhost443Listener{}},
			},
			wantErr: false,
		},
		{
			name: "lease endpoint reconciler - wrong IP families",
			extraConfig: &ExtraConfig{
				EndpointReconcilerType: reconcilers.LeaseEndpointReconcilerType,
				ServiceIPRange:         *ipv4cidr,
			},
			config: genericapiserver.Config{
				PublicAddress: ipv6address,
				SecureServing: &genericapiserver.SecureServingInfo{Listener: fakeLocalhost443Listener{}},
			},
			wantErr: true,
		},
		{
			name: "lease endpoint reconciler - wrong IP families",
			extraConfig: &ExtraConfig{
				EndpointReconcilerType: reconcilers.LeaseEndpointReconcilerType,
				ServiceIPRange:         *ipv6cidr,
			},
			config: genericapiserver.Config{
				PublicAddress: ipv4address,
				SecureServing: &genericapiserver.SecureServingInfo{Listener: fakeLocalhost443Listener{}},
			},
			wantErr: true,
		},
		{
			name: "none endpoint reconciler - wrong IP families",
			extraConfig: &ExtraConfig{
				EndpointReconcilerType: reconcilers.NoneEndpointReconcilerType,
				ServiceIPRange:         *ipv4cidr,
			},
			config: genericapiserver.Config{
				PublicAddress: ipv6address,
				SecureServing: &genericapiserver.SecureServingInfo{Listener: fakeLocalhost443Listener{}},
			},
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := &completedConfig{
				GenericConfig: tt.config.Complete(nil),
				ExtraConfig:   tt.extraConfig,
			}
			_, err := c.newKubernetesServiceControllerConfig(tt.args.client)
			if (err != nil) != tt.wantErr {
				t.Errorf("completedConfig.NewBootstrapController() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

		})
	}
}
