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
	auditbuffered "k8s.io/apiserver/plugin/pkg/audit/buffered"
	auditdynamic "k8s.io/apiserver/plugin/pkg/audit/dynamic"
	audittruncate "k8s.io/apiserver/plugin/pkg/audit/truncate"
	restclient "k8s.io/client-go/rest"
	cliflag "k8s.io/component-base/cli/flag"
	kapi "k8s.io/kubernetes/pkg/apis/core"
	kubeoptions "k8s.io/kubernetes/pkg/kubeapiserver/options"
	kubeletclient "k8s.io/kubernetes/pkg/kubelet/client"
	"k8s.io/kubernetes/pkg/master/reconcilers"
)

func TestAddFlags(t *testing.T) {
	fs := pflag.NewFlagSet("addflagstest", pflag.ContinueOnError)
	s := NewServerRunOptions()
	for _, f := range s.Flags().FlagSets {
		fs.AddFlagSet(f)
	}

	args := []string{
		"--enable-admission-plugins=AlwaysDeny",
		"--admission-control-config-file=/admission-control-config",
		"--advertise-address=192.168.10.10",
		"--allow-privileged=false",
		"--anonymous-auth=false",
		"--apiserver-count=5",
		"--audit-log-maxage=11",
		"--audit-log-maxbackup=12",
		"--audit-log-maxsize=13",
		"--audit-log-path=/var/log",
		"--audit-log-mode=blocking",
		"--audit-log-batch-buffer-size=46",
		"--audit-log-batch-max-size=47",
		"--audit-log-batch-max-wait=48s",
		"--audit-log-batch-throttle-enable=true",
		"--audit-log-batch-throttle-qps=49.5",
		"--audit-log-batch-throttle-burst=50",
		"--audit-log-truncate-enabled=true",
		"--audit-log-truncate-max-batch-size=45",
		"--audit-log-truncate-max-event-size=44",
		"--audit-log-version=audit.k8s.io/v1alpha1",
		"--audit-policy-file=/policy",
		"--audit-webhook-config-file=/webhook-config",
		"--audit-webhook-mode=blocking",
		"--audit-webhook-batch-buffer-size=42",
		"--audit-webhook-batch-max-size=43",
		"--audit-webhook-batch-max-wait=1s",
		"--audit-webhook-batch-throttle-enable=false",
		"--audit-webhook-batch-throttle-qps=43.5",
		"--audit-webhook-batch-throttle-burst=44",
		"--audit-webhook-truncate-enabled=true",
		"--audit-webhook-truncate-max-batch-size=43",
		"--audit-webhook-truncate-max-event-size=42",
		"--audit-webhook-initial-backoff=2s",
		"--audit-webhook-version=audit.k8s.io/v1alpha1",
		"--authentication-token-webhook-cache-ttl=3m",
		"--authentication-token-webhook-config-file=/token-webhook-config",
		"--authorization-mode=AlwaysDeny,RBAC",
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
		"--egress-selector-config-file=/var/run/kubernetes/egress-selector/connectivity.yaml",
		"--enable-aggregator-routing=true",
		"--enable-logs-handler=false",
		"--endpoint-reconciler-type=" + string(reconcilers.LeaseEndpointReconcilerType),
		"--etcd-keyfile=/var/run/kubernetes/etcd.key",
		"--etcd-certfile=/var/run/kubernetes/etcdce.crt",
		"--etcd-cafile=/var/run/kubernetes/etcdca.crt",
		"--http2-max-streams-per-connection=42",
		"--kubelet-https=true",
		"--kubelet-read-only-port=10255",
		"--kubelet-timeout=5s",
		"--kubelet-client-certificate=/var/run/kubernetes/ceserver.crt",
		"--kubelet-client-key=/var/run/kubernetes/server.key",
		"--kubelet-certificate-authority=/var/run/kubernetes/caserver.crt",
		"--proxy-client-cert-file=/var/run/kubernetes/proxy.crt",
		"--proxy-client-key-file=/var/run/kubernetes/proxy.key",
		"--request-timeout=2m",
		"--storage-backend=etcd3",
	}
	fs.Parse(args)

	// This is a snapshot of expected options parsed by args.
	expected := &ServerRunOptions{
		ServiceNodePortRange:   kubeoptions.DefaultServiceNodePortRange,
		ServiceClusterIPRange:  kubeoptions.DefaultServiceIPCIDR,
		MasterCount:            5,
		EndpointReconcilerType: string(reconcilers.LeaseEndpointReconcilerType),
		AllowPrivileged:        false,
		GenericServerRunOptions: &apiserveroptions.ServerRunOptions{
			AdvertiseAddress:            net.ParseIP("192.168.10.10"),
			CorsAllowedOriginList:       []string{"10.10.10.100", "10.10.10.200"},
			MaxRequestsInFlight:         400,
			MaxMutatingRequestsInFlight: 200,
			RequestTimeout:              time.Duration(2) * time.Minute,
			MinRequestTimeout:           1800,
			JSONPatchMaxCopyBytes:       int64(100 * 1024 * 1024),
			MaxRequestBodyBytes:         int64(100 * 1024 * 1024),
		},
		Admission: &kubeoptions.AdmissionOptions{
			GenericAdmission: &apiserveroptions.AdmissionOptions{
				RecommendedPluginOrder: s.Admission.GenericAdmission.RecommendedPluginOrder,
				DefaultOffPlugins:      s.Admission.GenericAdmission.DefaultOffPlugins,
				EnablePlugins:          []string{"AlwaysDeny"},
				ConfigFile:             "/admission-control-config",
				Plugins:                s.Admission.GenericAdmission.Plugins,
				Decorators:             s.Admission.GenericAdmission.Decorators,
			},
		},
		Etcd: &apiserveroptions.EtcdOptions{
			StorageConfig: storagebackend.Config{
				Type: "etcd3",
				Transport: storagebackend.TransportConfig{
					ServerList: nil,
					KeyFile:    "/var/run/kubernetes/etcd.key",
					CAFile:     "/var/run/kubernetes/etcdca.crt",
					CertFile:   "/var/run/kubernetes/etcdce.crt",
				},
				Paging:                true,
				Prefix:                "/registry",
				CompactionInterval:    storagebackend.DefaultCompactInterval,
				CountMetricPollPeriod: time.Minute,
			},
			DefaultStorageMediaType: "application/vnd.kubernetes.protobuf",
			DeleteCollectionWorkers: 1,
			EnableGarbageCollection: true,
			EnableWatchCache:        true,
			DefaultWatchCacheSize:   100,
		},
		SecureServing: (&apiserveroptions.SecureServingOptions{
			BindAddress: net.ParseIP("192.168.10.20"),
			BindPort:    6443,
			ServerCert: apiserveroptions.GeneratableKeyCert{
				CertDirectory: "/var/run/kubernetes",
				PairName:      "apiserver",
			},
			HTTP2MaxStreamsPerConnection: 42,
			Required:                     true,
		}).WithLoopback(),
		InsecureServing: (&apiserveroptions.DeprecatedInsecureServingOptions{
			BindAddress: net.ParseIP("127.0.0.1"),
			BindPort:    8080,
		}).WithLoopback(),
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
			EnableHTTPS: true,
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
				BatchOptions: apiserveroptions.AuditBatchOptions{
					Mode: "blocking",
					BatchConfig: auditbuffered.BatchConfig{
						BufferSize:     46,
						MaxBatchSize:   47,
						MaxBatchWait:   48 * time.Second,
						ThrottleEnable: true,
						ThrottleQPS:    49.5,
						ThrottleBurst:  50,
					},
				},
				TruncateOptions: apiserveroptions.AuditTruncateOptions{
					Enabled: true,
					TruncateConfig: audittruncate.Config{
						MaxBatchSize: 45,
						MaxEventSize: 44,
					},
				},
				GroupVersionString: "audit.k8s.io/v1alpha1",
			},
			WebhookOptions: apiserveroptions.AuditWebhookOptions{
				ConfigFile: "/webhook-config",
				BatchOptions: apiserveroptions.AuditBatchOptions{
					Mode: "blocking",
					BatchConfig: auditbuffered.BatchConfig{
						BufferSize:     42,
						MaxBatchSize:   43,
						MaxBatchWait:   1 * time.Second,
						ThrottleEnable: false,
						ThrottleQPS:    43.5,
						ThrottleBurst:  44,
						AsyncDelegate:  true,
					},
				},
				TruncateOptions: apiserveroptions.AuditTruncateOptions{
					Enabled: true,
					TruncateConfig: audittruncate.Config{
						MaxBatchSize: 43,
						MaxEventSize: 42,
					},
				},
				InitialBackoff:     2 * time.Second,
				GroupVersionString: "audit.k8s.io/v1alpha1",
			},
			DynamicOptions: apiserveroptions.AuditDynamicOptions{
				BatchConfig: auditdynamic.NewDefaultWebhookBatchConfig(),
			},
			PolicyFile: "/policy",
		},
		Features: &apiserveroptions.FeatureOptions{
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
			OIDC: &kubeoptions.OIDCAuthenticationOptions{
				UsernameClaim: "sub",
				SigningAlgs:   []string{"RS256"},
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
			Modes:                       []string{"AlwaysDeny", "RBAC"},
			PolicyFile:                  "/policy",
			WebhookConfigFile:           "/webhook-config",
			WebhookCacheAuthorizedTTL:   180000000000,
			WebhookCacheUnauthorizedTTL: 60000000000,
		},
		CloudProvider: &kubeoptions.CloudProviderOptions{
			CloudConfigFile: "/cloud-config",
			CloudProvider:   "azure",
		},
		APIEnablement: &apiserveroptions.APIEnablementOptions{
			RuntimeConfig: cliflag.ConfigurationMap{},
		},
		EgressSelector: &apiserveroptions.EgressSelectorOptions{
			ConfigFile: "/var/run/kubernetes/egress-selector/connectivity.yaml",
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
