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
	"net"
	"reflect"
	"testing"
	"time"

	"github.com/spf13/pflag"

	"k8s.io/apimachinery/pkg/util/diff"
	apiserveroptions "k8s.io/apiserver/pkg/server/options"
	"k8s.io/apiserver/pkg/storage/storagebackend"
	utilconfig "k8s.io/apiserver/pkg/util/flag"
	restclient "k8s.io/client-go/rest"
	kapi "k8s.io/kubernetes/pkg/api"
	kubeoptions "k8s.io/kubernetes/pkg/kubeapiserver/options"
	kubeletclient "k8s.io/kubernetes/pkg/kubelet/client"
	"k8s.io/kubernetes/pkg/master/reconcilers"
)

func TestAddFlags(t *testing.T) {
	f := pflag.NewFlagSet("addflagstest", pflag.ContinueOnError)
	s := NewServerRunOptions()
	s.AddFlags(f)

	args := []string{
		"--admission-control=AlwaysDeny",
		"--admission-control-config-file=/admission-control-config",
		"--advertise-address=192.168.10.10",
		"--allow-privileged=false",
		"--alpha-endpoint-reconciler-type=" + string(reconcilers.MasterCountReconcilerType),
		"--anonymous-auth=false",
		"--apiserver-count=5",
		"--audit-log-maxage=11",
		"--audit-log-maxbackup=12",
		"--audit-log-maxsize=13",
		"--audit-log-path=/var/log",
		"--audit-policy-file=/policy",
		"--audit-webhook-config-file=/webhook-config",
		"--audit-webhook-mode=blocking",
		"--authentication-token-webhook-cache-ttl=3m",
		"--authentication-token-webhook-config-file=/token-webhook-config",
		"--authorization-mode=AlwaysDeny",
		"--authorization-policy-file=/policy",
		"--authorization-webhook-cache-authorized-ttl=3m",
		"--authorization-webhook-cache-unauthorized-ttl=1m",
		"--authorization-webhook-config-file=/webhook-config",
		"--bind-address=192.168.10.20",
		"--client-ca-file=/client-ca",
		"--cloud-config=/cloud-config",
		"--cloud-provider=azure",
		"--cors-allowed-origins=10.10.10.100,10.10.10.200",
		"--contention-profiling=true",
		"--enable-aggregator-routing=true",
		"--enable-logs-handler=false",
		"--enable-swagger-ui=true",
		"--etcd-quorum-read=false",
		"--etcd-keyfile=/var/run/kubernetes/etcd.key",
		"--etcd-certfile=/var/run/kubernetes/etcdce.crt",
		"--etcd-cafile=/var/run/kubernetes/etcdca.crt",
		"--kubelet-https=true",
		"--kubelet-read-only-port=10255",
		"--kubelet-timeout=5s",
		"--kubelet-client-certificate=/var/run/kubernetes/ceserver.crt",
		"--kubelet-client-key=/var/run/kubernetes/server.key",
		"--kubelet-certificate-authority=/var/run/kubernetes/caserver.crt",
		"--proxy-client-cert-file=/var/run/kubernetes/proxy.crt",
		"--proxy-client-key-file=/var/run/kubernetes/proxy.key",
		"--request-timeout=2m",
		"--storage-backend=etcd2",
	}
	f.Parse(args)

	// This is a snapshot of expected options parsed by args.
	expected := &ServerRunOptions{
		ServiceNodePortRange:   DefaultServiceNodePortRange,
		MasterCount:            5,
		EndpointReconcilerType: string(reconcilers.MasterCountReconcilerType),
		AllowPrivileged:        false,
		GenericServerRunOptions: &apiserveroptions.ServerRunOptions{
			AdvertiseAddress:            net.ParseIP("192.168.10.10"),
			CorsAllowedOriginList:       []string{"10.10.10.100", "10.10.10.200"},
			MaxRequestsInFlight:         400,
			MaxMutatingRequestsInFlight: 200,
			RequestTimeout:              time.Duration(2) * time.Minute,
			MinRequestTimeout:           1800,
		},
		Admission: &apiserveroptions.AdmissionOptions{
			RecommendedPluginOrder: []string{"NamespaceLifecycle"},
			PluginNames:            []string{"AlwaysDeny"},
			ConfigFile:             "/admission-control-config",
			Plugins:                s.Admission.Plugins,
		},
		Etcd: &apiserveroptions.EtcdOptions{
			StorageConfig: storagebackend.Config{
				Type:       "etcd2",
				ServerList: nil,
				Prefix:     "/registry",
				DeserializationCacheSize: 0,
				Copier:   kapi.Scheme,
				Quorum:   false,
				KeyFile:  "/var/run/kubernetes/etcd.key",
				CAFile:   "/var/run/kubernetes/etcdca.crt",
				CertFile: "/var/run/kubernetes/etcdce.crt",
			},
			DefaultStorageMediaType: "application/vnd.kubernetes.protobuf",
			DeleteCollectionWorkers: 1,
			EnableGarbageCollection: true,
			EnableWatchCache:        true,
			DefaultWatchCacheSize:   100,
		},
		SecureServing: &apiserveroptions.SecureServingOptions{
			BindAddress: net.ParseIP("192.168.10.20"),
			BindPort:    6443,
			ServerCert: apiserveroptions.GeneratableKeyCert{
				CertDirectory: "/var/run/kubernetes",
				PairName:      "apiserver",
			},
		},
		InsecureServing: &kubeoptions.InsecureServingOptions{
			BindAddress: net.ParseIP("127.0.0.1"),
			BindPort:    8080,
		},
		EventTTL: 1 * time.Hour,
		KubeletConfig: kubeletclient.KubeletClientConfig{
			Port:         10250,
			ReadOnlyPort: 10255,
			PreferredAddressTypes: []string{
				string(kapi.NodeHostName),
				string(kapi.NodeInternalDNS),
				string(kapi.NodeInternalIP),
				string(kapi.NodeExternalDNS),
				string(kapi.NodeExternalIP),
			},
			EnableHttps: true,
			HTTPTimeout: time.Duration(5) * time.Second,
			TLSClientConfig: restclient.TLSClientConfig{
				CertFile: "/var/run/kubernetes/ceserver.crt",
				KeyFile:  "/var/run/kubernetes/server.key",
				CAFile:   "/var/run/kubernetes/caserver.crt",
			},
		},
		Audit: &apiserveroptions.AuditOptions{
			LogOptions: apiserveroptions.AuditLogOptions{
				Path:       "/var/log",
				MaxAge:     11,
				MaxBackups: 12,
				MaxSize:    13,
				Format:     "json",
			},
			WebhookOptions: apiserveroptions.AuditWebhookOptions{
				Mode:       "blocking",
				ConfigFile: "/webhook-config",
			},
			PolicyFile: "/policy",
		},
		Features: &apiserveroptions.FeatureOptions{
			EnableSwaggerUI:           true,
			EnableProfiling:           true,
			EnableContentionProfiling: true,
		},
		Authentication: &kubeoptions.BuiltInAuthenticationOptions{
			Anonymous: &kubeoptions.AnonymousAuthenticationOptions{
				Allow: false,
			},
			ClientCert: &apiserveroptions.ClientCertAuthenticationOptions{
				ClientCA: "/client-ca",
			},
			WebHook: &kubeoptions.WebHookAuthenticationOptions{
				CacheTTL:   180000000000,
				ConfigFile: "/token-webhook-config",
			},
			BootstrapToken: &kubeoptions.BootstrapTokenAuthenticationOptions{},
			Keystone:       &kubeoptions.KeystoneAuthenticationOptions{},
			OIDC: &kubeoptions.OIDCAuthenticationOptions{
				UsernameClaim: "sub",
			},
			PasswordFile:  &kubeoptions.PasswordFileAuthenticationOptions{},
			RequestHeader: &apiserveroptions.RequestHeaderAuthenticationOptions{},
			ServiceAccounts: &kubeoptions.ServiceAccountAuthenticationOptions{
				Lookup: true,
			},
			TokenFile:            &kubeoptions.TokenFileAuthenticationOptions{},
			TokenSuccessCacheTTL: 10 * time.Second,
			TokenFailureCacheTTL: 0,
		},
		Authorization: &kubeoptions.BuiltInAuthorizationOptions{
			Mode:                        "AlwaysDeny",
			PolicyFile:                  "/policy",
			WebhookConfigFile:           "/webhook-config",
			WebhookCacheAuthorizedTTL:   180000000000,
			WebhookCacheUnauthorizedTTL: 60000000000,
		},
		CloudProvider: &kubeoptions.CloudProviderOptions{
			CloudConfigFile: "/cloud-config",
			CloudProvider:   "azure",
		},
		StorageSerialization: &kubeoptions.StorageSerializationOptions{
			StorageVersions:        kapi.Registry.AllPreferredGroupVersions(),
			DefaultStorageVersions: kapi.Registry.AllPreferredGroupVersions(),
		},
		APIEnablement: &kubeoptions.APIEnablementOptions{
			RuntimeConfig: utilconfig.ConfigurationMap{},
		},
		EnableLogsHandler:       false,
		EnableAggregatorRouting: true,
		ProxyClientKeyFile:      "/var/run/kubernetes/proxy.key",
		ProxyClientCertFile:     "/var/run/kubernetes/proxy.crt",
	}

	if !reflect.DeepEqual(expected, s) {
		t.Errorf("Got different run options than expected.\nDifference detected on:\n%s", diff.ObjectReflectDiff(expected, s))
	}
}
