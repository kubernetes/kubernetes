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

package options

import (
	"net"
	"reflect"
	"sort"
	"testing"
	"time"

	"github.com/spf13/pflag"

	apimachineryconfig "k8s.io/apimachinery/pkg/apis/config"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/diff"
	apiserverconfig "k8s.io/apiserver/pkg/apis/config"
	apiserveroptions "k8s.io/apiserver/pkg/server/options"
	cmoptions "k8s.io/kubernetes/cmd/controller-manager/app/options"
	kubectrlmgrconfig "k8s.io/kubernetes/pkg/controller/apis/config"
)

func TestAddFlags(t *testing.T) {
	fs := pflag.NewFlagSet("addflagstest", pflag.ContinueOnError)
	s, _ := NewKubeControllerManagerOptions()
	for _, f := range s.Flags([]string{""}, []string{""}).FlagSets {
		fs.AddFlagSet(f)
	}

	args := []string{
		"--address=192.168.4.10",
		"--allocate-node-cidrs=true",
		"--attach-detach-reconcile-sync-period=30s",
		"--cidr-allocator-type=CloudAllocator",
		"--cloud-config=/cloud-config",
		"--cloud-provider=gce",
		"--cluster-cidr=1.2.3.4/24",
		"--cluster-name=k8s",
		"--cluster-signing-cert-file=/cluster-signing-cert",
		"--cluster-signing-key-file=/cluster-signing-key",
		"--concurrent-deployment-syncs=10",
		"--concurrent-endpoint-syncs=10",
		"--concurrent-gc-syncs=30",
		"--concurrent-namespace-syncs=20",
		"--concurrent-replicaset-syncs=10",
		"--concurrent-resource-quota-syncs=10",
		"--concurrent-service-syncs=2",
		"--concurrent-serviceaccount-token-syncs=10",
		"--concurrent_rc_syncs=10",
		"--configure-cloud-routes=false",
		"--contention-profiling=true",
		"--controller-start-interval=2m",
		"--controllers=foo,bar",
		"--deployment-controller-sync-period=45s",
		"--disable-attach-detach-reconcile-sync=true",
		"--enable-dynamic-provisioning=false",
		"--enable-garbage-collector=false",
		"--enable-hostpath-provisioner=true",
		"--enable-taint-manager=false",
		"--experimental-cluster-signing-duration=10h",
		"--flex-volume-plugin-dir=/flex-volume-plugin",
		"--horizontal-pod-autoscaler-downscale-delay=2m",
		"--horizontal-pod-autoscaler-sync-period=45s",
		"--horizontal-pod-autoscaler-upscale-delay=1m",
		"--horizontal-pod-autoscaler-downscale-stabilization=3m",
		"--horizontal-pod-autoscaler-cpu-initialization-period=90s",
		"--horizontal-pod-autoscaler-initial-readiness-delay=50s",
		"--http2-max-streams-per-connection=47",
		"--kube-api-burst=100",
		"--kube-api-content-type=application/json",
		"--kube-api-qps=50.0",
		"--kubeconfig=/kubeconfig",
		"--large-cluster-size-threshold=100",
		"--leader-elect=false",
		"--leader-elect-lease-duration=30s",
		"--leader-elect-renew-deadline=15s",
		"--leader-elect-resource-lock=configmap",
		"--leader-elect-retry-period=5s",
		"--master=192.168.4.20",
		"--min-resync-period=8h",
		"--namespace-sync-period=10m",
		"--node-cidr-mask-size=48",
		"--node-eviction-rate=0.2",
		"--node-monitor-grace-period=30s",
		"--node-monitor-period=10s",
		"--node-startup-grace-period=30s",
		"--pod-eviction-timeout=2m",
		"--port=10000",
		"--profiling=false",
		"--pv-recycler-increment-timeout-nfs=45",
		"--pv-recycler-minimum-timeout-hostpath=45",
		"--pv-recycler-minimum-timeout-nfs=200",
		"--pv-recycler-timeout-increment-hostpath=45",
		"--pvclaimbinder-sync-period=30s",
		"--resource-quota-sync-period=10m",
		"--route-reconciliation-period=30s",
		"--secondary-node-eviction-rate=0.05",
		"--service-account-private-key-file=/service-account-private-key",
		"--terminated-pod-gc-threshold=12000",
		"--unhealthy-zone-threshold=0.6",
		"--use-service-account-credentials=true",
		"--cert-dir=/a/b/c",
		"--bind-address=192.168.4.21",
		"--secure-port=10001",
		"--concurrent-ttl-after-finished-syncs=8",
	}
	fs.Parse(args)
	// Sort GCIgnoredResources because it's built from a map, which means the
	// insertion order is random.
	sort.Sort(sortedGCIgnoredResources(s.GarbageCollectorController.GCIgnoredResources))

	expected := &KubeControllerManagerOptions{
		Generic: &cmoptions.GenericControllerManagerConfigurationOptions{
			Port:            10252,     // Note: InsecureServingOptions.ApplyTo will write the flag value back into the component config
			Address:         "0.0.0.0", // Note: InsecureServingOptions.ApplyTo will write the flag value back into the component config
			MinResyncPeriod: metav1.Duration{Duration: 8 * time.Hour},
			ClientConnection: apimachineryconfig.ClientConnectionConfiguration{
				ContentType: "application/json",
				QPS:         50.0,
				Burst:       100,
			},
			ControllerStartInterval: metav1.Duration{Duration: 2 * time.Minute},
			LeaderElection: apiserverconfig.LeaderElectionConfiguration{
				ResourceLock:  "configmap",
				LeaderElect:   false,
				LeaseDuration: metav1.Duration{Duration: 30 * time.Second},
				RenewDeadline: metav1.Duration{Duration: 15 * time.Second},
				RetryPeriod:   metav1.Duration{Duration: 5 * time.Second},
			},
			Debugging: &cmoptions.DebuggingOptions{
				EnableProfiling:           false,
				EnableContentionProfiling: true,
			},
			Controllers: []string{"foo", "bar"},
		},
		KubeCloudShared: &cmoptions.KubeCloudSharedOptions{
			UseServiceAccountCredentials: true,
			RouteReconciliationPeriod:    metav1.Duration{Duration: 30 * time.Second},
			NodeMonitorPeriod:            metav1.Duration{Duration: 10 * time.Second},
			ClusterName:                  "k8s",
			ClusterCIDR:                  "1.2.3.4/24",
			AllocateNodeCIDRs:            true,
			CIDRAllocatorType:            "CloudAllocator",
			ConfigureCloudRoutes:         false,
			CloudProvider: &cmoptions.CloudProviderOptions{
				Name:            "gce",
				CloudConfigFile: "/cloud-config",
			},
		},
		AttachDetachController: &AttachDetachControllerOptions{
			ReconcilerSyncLoopPeriod:          metav1.Duration{Duration: 30 * time.Second},
			DisableAttachDetachReconcilerSync: true,
		},
		CSRSigningController: &CSRSigningControllerOptions{
			ClusterSigningCertFile: "/cluster-signing-cert",
			ClusterSigningKeyFile:  "/cluster-signing-key",
			ClusterSigningDuration: metav1.Duration{Duration: 10 * time.Hour},
		},
		DaemonSetController: &DaemonSetControllerOptions{
			ConcurrentDaemonSetSyncs: 2,
		},
		DeploymentController: &DeploymentControllerOptions{
			ConcurrentDeploymentSyncs:      10,
			DeploymentControllerSyncPeriod: metav1.Duration{Duration: 45 * time.Second},
		},
		DeprecatedFlags: &DeprecatedControllerOptions{
			DeletingPodsQPS:    0.1,
			RegisterRetryCount: 10,
		},
		EndpointController: &EndpointControllerOptions{
			ConcurrentEndpointSyncs: 10,
		},
		GarbageCollectorController: &GarbageCollectorControllerOptions{
			ConcurrentGCSyncs: 30,
			GCIgnoredResources: []kubectrlmgrconfig.GroupResource{
				{Group: "", Resource: "events"},
			},
			EnableGarbageCollector: false,
		},
		HPAController: &HPAControllerOptions{
			HorizontalPodAutoscalerSyncPeriod:                   metav1.Duration{Duration: 45 * time.Second},
			HorizontalPodAutoscalerUpscaleForbiddenWindow:       metav1.Duration{Duration: 1 * time.Minute},
			HorizontalPodAutoscalerDownscaleForbiddenWindow:     metav1.Duration{Duration: 2 * time.Minute},
			HorizontalPodAutoscalerDownscaleStabilizationWindow: metav1.Duration{Duration: 3 * time.Minute},
			HorizontalPodAutoscalerCPUInitializationPeriod:      metav1.Duration{Duration: 90 * time.Second},
			HorizontalPodAutoscalerInitialReadinessDelay:        metav1.Duration{Duration: 50 * time.Second},
			HorizontalPodAutoscalerTolerance:                    0.1,
			HorizontalPodAutoscalerUseRESTClients:               true,
		},
		JobController: &JobControllerOptions{
			ConcurrentJobSyncs: 5,
		},
		NamespaceController: &NamespaceControllerOptions{
			NamespaceSyncPeriod:      metav1.Duration{Duration: 10 * time.Minute},
			ConcurrentNamespaceSyncs: 20,
		},
		NodeIPAMController: &NodeIPAMControllerOptions{
			NodeCIDRMaskSize: 48,
		},
		NodeLifecycleController: &NodeLifecycleControllerOptions{
			EnableTaintManager:        false,
			NodeEvictionRate:          0.2,
			SecondaryNodeEvictionRate: 0.05,
			NodeMonitorGracePeriod:    metav1.Duration{Duration: 30 * time.Second},
			NodeStartupGracePeriod:    metav1.Duration{Duration: 30 * time.Second},
			PodEvictionTimeout:        metav1.Duration{Duration: 2 * time.Minute},
			LargeClusterSizeThreshold: 100,
			UnhealthyZoneThreshold:    0.6,
		},
		PersistentVolumeBinderController: &PersistentVolumeBinderControllerOptions{
			PVClaimBinderSyncPeriod: metav1.Duration{Duration: 30 * time.Second},
			VolumeConfiguration: kubectrlmgrconfig.VolumeConfiguration{
				EnableDynamicProvisioning:  false,
				EnableHostPathProvisioning: true,
				FlexVolumePluginDir:        "/flex-volume-plugin",
				PersistentVolumeRecyclerConfiguration: kubectrlmgrconfig.PersistentVolumeRecyclerConfiguration{
					MaximumRetry:             3,
					MinimumTimeoutNFS:        200,
					IncrementTimeoutNFS:      45,
					MinimumTimeoutHostPath:   45,
					IncrementTimeoutHostPath: 45,
				},
			},
		},
		PodGCController: &PodGCControllerOptions{
			TerminatedPodGCThreshold: 12000,
		},
		ReplicaSetController: &ReplicaSetControllerOptions{
			ConcurrentRSSyncs: 10,
		},
		ReplicationController: &ReplicationControllerOptions{
			ConcurrentRCSyncs: 10,
		},
		ResourceQuotaController: &ResourceQuotaControllerOptions{
			ResourceQuotaSyncPeriod:      metav1.Duration{Duration: 10 * time.Minute},
			ConcurrentResourceQuotaSyncs: 10,
		},
		SAController: &SAControllerOptions{
			ServiceAccountKeyFile:  "/service-account-private-key",
			ConcurrentSATokenSyncs: 10,
		},
		ServiceController: &cmoptions.ServiceControllerOptions{
			ConcurrentServiceSyncs: 2,
		},
		TTLAfterFinishedController: &TTLAfterFinishedControllerOptions{
			ConcurrentTTLSyncs: 8,
		},
		SecureServing: (&apiserveroptions.SecureServingOptions{
			BindPort:    10001,
			BindAddress: net.ParseIP("192.168.4.21"),
			ServerCert: apiserveroptions.GeneratableKeyCert{
				CertDirectory: "/a/b/c",
				PairName:      "kube-controller-manager",
			},
			HTTP2MaxStreamsPerConnection: 47,
		}).WithLoopback(),
		InsecureServing: (&apiserveroptions.DeprecatedInsecureServingOptions{
			BindAddress: net.ParseIP("192.168.4.10"),
			BindPort:    int(10000),
			BindNetwork: "tcp",
		}).WithLoopback(),
		Authentication: &apiserveroptions.DelegatingAuthenticationOptions{
			CacheTTL:   10 * time.Second,
			ClientCert: apiserveroptions.ClientCertAuthenticationOptions{},
			RequestHeader: apiserveroptions.RequestHeaderAuthenticationOptions{
				UsernameHeaders:     []string{"x-remote-user"},
				GroupHeaders:        []string{"x-remote-group"},
				ExtraHeaderPrefixes: []string{"x-remote-extra-"},
			},
			RemoteKubeConfigFileOptional: true,
		},
		Authorization: &apiserveroptions.DelegatingAuthorizationOptions{
			AllowCacheTTL:                10 * time.Second,
			DenyCacheTTL:                 10 * time.Second,
			RemoteKubeConfigFileOptional: true,
			AlwaysAllowPaths:             []string{"/healthz"}, // note: this does not match /healthz/ or /healthz/*
		},
		Kubeconfig: "/kubeconfig",
		Master:     "192.168.4.20",
	}

	// Sort GCIgnoredResources because it's built from a map, which means the
	// insertion order is random.
	sort.Sort(sortedGCIgnoredResources(expected.GarbageCollectorController.GCIgnoredResources))

	if !reflect.DeepEqual(expected, s) {
		t.Errorf("Got different run options than expected.\nDifference detected on:\n%s", diff.ObjectReflectDiff(expected, s))
	}
}

type sortedGCIgnoredResources []kubectrlmgrconfig.GroupResource

func (r sortedGCIgnoredResources) Len() int {
	return len(r)
}

func (r sortedGCIgnoredResources) Less(i, j int) bool {
	if r[i].Group < r[j].Group {
		return true
	} else if r[i].Group > r[j].Group {
		return false
	}
	return r[i].Resource < r[j].Resource
}

func (r sortedGCIgnoredResources) Swap(i, j int) {
	r[i], r[j] = r[j], r[i]
}
