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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/diff"
	apiserveroptions "k8s.io/apiserver/pkg/server/options"
	cpconfig "k8s.io/cloud-provider/config"
	cpoptions "k8s.io/cloud-provider/options"
	serviceconfig "k8s.io/cloud-provider/service/config"
	componentbaseconfig "k8s.io/component-base/config"
	"k8s.io/component-base/logs"
	"k8s.io/component-base/metrics"
	cmconfig "k8s.io/controller-manager/config"
	cmoptions "k8s.io/controller-manager/options"
	kubecontrollerconfig "k8s.io/kubernetes/cmd/kube-controller-manager/app/config"
	kubectrlmgrconfig "k8s.io/kubernetes/pkg/controller/apis/config"
	csrsigningconfig "k8s.io/kubernetes/pkg/controller/certificates/signer/config"
	daemonconfig "k8s.io/kubernetes/pkg/controller/daemon/config"
	deploymentconfig "k8s.io/kubernetes/pkg/controller/deployment/config"
	endpointconfig "k8s.io/kubernetes/pkg/controller/endpoint/config"
	endpointsliceconfig "k8s.io/kubernetes/pkg/controller/endpointslice/config"
	endpointslicemirroringconfig "k8s.io/kubernetes/pkg/controller/endpointslicemirroring/config"
	garbagecollectorconfig "k8s.io/kubernetes/pkg/controller/garbagecollector/config"
	jobconfig "k8s.io/kubernetes/pkg/controller/job/config"
	namespaceconfig "k8s.io/kubernetes/pkg/controller/namespace/config"
	nodeipamconfig "k8s.io/kubernetes/pkg/controller/nodeipam/config"
	nodelifecycleconfig "k8s.io/kubernetes/pkg/controller/nodelifecycle/config"
	poautosclerconfig "k8s.io/kubernetes/pkg/controller/podautoscaler/config"
	podgcconfig "k8s.io/kubernetes/pkg/controller/podgc/config"
	replicasetconfig "k8s.io/kubernetes/pkg/controller/replicaset/config"
	replicationconfig "k8s.io/kubernetes/pkg/controller/replication/config"
	resourcequotaconfig "k8s.io/kubernetes/pkg/controller/resourcequota/config"
	serviceaccountconfig "k8s.io/kubernetes/pkg/controller/serviceaccount/config"
	statefulsetconfig "k8s.io/kubernetes/pkg/controller/statefulset/config"
	ttlafterfinishedconfig "k8s.io/kubernetes/pkg/controller/ttlafterfinished/config"
	attachdetachconfig "k8s.io/kubernetes/pkg/controller/volume/attachdetach/config"
	persistentvolumeconfig "k8s.io/kubernetes/pkg/controller/volume/persistentvolume/config"
)

var args = []string{
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
	"--cluster-signing-kubelet-serving-cert-file=/cluster-signing-kubelet-serving/cert-file",
	"--cluster-signing-kubelet-serving-key-file=/cluster-signing-kubelet-serving/key-file",
	"--cluster-signing-kubelet-client-cert-file=/cluster-signing-kubelet-client/cert-file",
	"--cluster-signing-kubelet-client-key-file=/cluster-signing-kubelet-client/key-file",
	"--cluster-signing-kube-apiserver-client-cert-file=/cluster-signing-kube-apiserver-client/cert-file",
	"--cluster-signing-kube-apiserver-client-key-file=/cluster-signing-kube-apiserver-client/key-file",
	"--cluster-signing-legacy-unknown-cert-file=/cluster-signing-legacy-unknown/cert-file",
	"--cluster-signing-legacy-unknown-key-file=/cluster-signing-legacy-unknown/key-file",
	"--concurrent-deployment-syncs=10",
	"--concurrent-statefulset-syncs=15",
	"--concurrent-endpoint-syncs=10",
	"--concurrent-service-endpoint-syncs=10",
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
	"--cluster-signing-duration=10h",
	"--flex-volume-plugin-dir=/flex-volume-plugin",
	"--volume-host-cidr-denylist=127.0.0.1/28,feed::/16",
	"--volume-host-allow-local-loopback=false",
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
	"--max-endpoints-per-slice=200",
	"--min-resync-period=8h",
	"--mirroring-concurrent-service-endpoint-syncs=2",
	"--mirroring-max-endpoints-per-subset=1000",
	"--namespace-sync-period=10m",
	"--node-cidr-mask-size=48",
	"--node-cidr-mask-size-ipv4=48",
	"--node-cidr-mask-size-ipv6=108",
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

func TestAddFlags(t *testing.T) {
	fs := pflag.NewFlagSet("addflagstest", pflag.ContinueOnError)
	s, _ := NewKubeControllerManagerOptions()
	for _, f := range s.Flags([]string{""}, []string{""}).FlagSets {
		fs.AddFlagSet(f)
	}

	fs.Parse(args)
	// Sort GCIgnoredResources because it's built from a map, which means the
	// insertion order is random.
	sort.Sort(sortedGCIgnoredResources(s.GarbageCollectorController.GCIgnoredResources))

	expected := &KubeControllerManagerOptions{
		Generic: &cmoptions.GenericControllerManagerConfigurationOptions{
			GenericControllerManagerConfiguration: &cmconfig.GenericControllerManagerConfiguration{
				Port:            10252,     // Note: InsecureServingOptions.ApplyTo will write the flag value back into the component config
				Address:         "0.0.0.0", // Note: InsecureServingOptions.ApplyTo will write the flag value back into the component config
				MinResyncPeriod: metav1.Duration{Duration: 8 * time.Hour},
				ClientConnection: componentbaseconfig.ClientConnectionConfiguration{
					ContentType: "application/json",
					QPS:         50.0,
					Burst:       100,
				},
				ControllerStartInterval: metav1.Duration{Duration: 2 * time.Minute},
				LeaderElection: componentbaseconfig.LeaderElectionConfiguration{
					ResourceLock:      "configmap",
					LeaderElect:       false,
					LeaseDuration:     metav1.Duration{Duration: 30 * time.Second},
					RenewDeadline:     metav1.Duration{Duration: 15 * time.Second},
					RetryPeriod:       metav1.Duration{Duration: 5 * time.Second},
					ResourceName:      "kube-controller-manager",
					ResourceNamespace: "kube-system",
				},
				Controllers: []string{"foo", "bar"},
			},
			Debugging: &cmoptions.DebuggingOptions{
				DebuggingConfiguration: &componentbaseconfig.DebuggingConfiguration{
					EnableProfiling:           false,
					EnableContentionProfiling: true,
				},
			},
		},
		KubeCloudShared: &cpoptions.KubeCloudSharedOptions{
			KubeCloudSharedConfiguration: &cpconfig.KubeCloudSharedConfiguration{
				UseServiceAccountCredentials: true,
				RouteReconciliationPeriod:    metav1.Duration{Duration: 30 * time.Second},
				NodeMonitorPeriod:            metav1.Duration{Duration: 10 * time.Second},
				ClusterName:                  "k8s",
				ClusterCIDR:                  "1.2.3.4/24",
				AllocateNodeCIDRs:            true,
				CIDRAllocatorType:            "CloudAllocator",
				ConfigureCloudRoutes:         false,
			},
			CloudProvider: &cpoptions.CloudProviderOptions{
				CloudProviderConfiguration: &cpconfig.CloudProviderConfiguration{
					Name:            "gce",
					CloudConfigFile: "/cloud-config",
				},
			},
		},
		ServiceController: &cpoptions.ServiceControllerOptions{
			ServiceControllerConfiguration: &serviceconfig.ServiceControllerConfiguration{
				ConcurrentServiceSyncs: 2,
			},
		},
		AttachDetachController: &AttachDetachControllerOptions{
			&attachdetachconfig.AttachDetachControllerConfiguration{
				ReconcilerSyncLoopPeriod:          metav1.Duration{Duration: 30 * time.Second},
				DisableAttachDetachReconcilerSync: true,
			},
		},
		CSRSigningController: &CSRSigningControllerOptions{
			&csrsigningconfig.CSRSigningControllerConfiguration{
				ClusterSigningCertFile: "/cluster-signing-cert",
				ClusterSigningKeyFile:  "/cluster-signing-key",
				ClusterSigningDuration: metav1.Duration{Duration: 10 * time.Hour},
				KubeletServingSignerConfiguration: csrsigningconfig.CSRSigningConfiguration{
					CertFile: "/cluster-signing-kubelet-serving/cert-file",
					KeyFile:  "/cluster-signing-kubelet-serving/key-file",
				},
				KubeletClientSignerConfiguration: csrsigningconfig.CSRSigningConfiguration{
					CertFile: "/cluster-signing-kubelet-client/cert-file",
					KeyFile:  "/cluster-signing-kubelet-client/key-file",
				},
				KubeAPIServerClientSignerConfiguration: csrsigningconfig.CSRSigningConfiguration{
					CertFile: "/cluster-signing-kube-apiserver-client/cert-file",
					KeyFile:  "/cluster-signing-kube-apiserver-client/key-file",
				},
				LegacyUnknownSignerConfiguration: csrsigningconfig.CSRSigningConfiguration{
					CertFile: "/cluster-signing-legacy-unknown/cert-file",
					KeyFile:  "/cluster-signing-legacy-unknown/key-file",
				},
			},
		},
		DaemonSetController: &DaemonSetControllerOptions{
			&daemonconfig.DaemonSetControllerConfiguration{
				ConcurrentDaemonSetSyncs: 2,
			},
		},
		DeploymentController: &DeploymentControllerOptions{
			&deploymentconfig.DeploymentControllerConfiguration{
				ConcurrentDeploymentSyncs:      10,
				DeploymentControllerSyncPeriod: metav1.Duration{Duration: 45 * time.Second},
			},
		},
		StatefulSetController: &StatefulSetControllerOptions{
			&statefulsetconfig.StatefulSetControllerConfiguration{
				ConcurrentStatefulSetSyncs: 15,
			},
		},
		DeprecatedFlags: &DeprecatedControllerOptions{
			&kubectrlmgrconfig.DeprecatedControllerConfiguration{
				DeletingPodsQPS:    0.1,
				RegisterRetryCount: 10,
			},
		},
		EndpointController: &EndpointControllerOptions{
			&endpointconfig.EndpointControllerConfiguration{
				ConcurrentEndpointSyncs: 10,
			},
		},
		EndpointSliceController: &EndpointSliceControllerOptions{
			&endpointsliceconfig.EndpointSliceControllerConfiguration{
				ConcurrentServiceEndpointSyncs: 10,
				MaxEndpointsPerSlice:           200,
			},
		},
		EndpointSliceMirroringController: &EndpointSliceMirroringControllerOptions{
			&endpointslicemirroringconfig.EndpointSliceMirroringControllerConfiguration{
				MirroringConcurrentServiceEndpointSyncs: 2,
				MirroringMaxEndpointsPerSubset:          1000,
			},
		},
		GarbageCollectorController: &GarbageCollectorControllerOptions{
			&garbagecollectorconfig.GarbageCollectorControllerConfiguration{
				ConcurrentGCSyncs: 30,
				GCIgnoredResources: []garbagecollectorconfig.GroupResource{
					{Group: "", Resource: "events"},
				},
				EnableGarbageCollector: false,
			},
		},
		HPAController: &HPAControllerOptions{
			&poautosclerconfig.HPAControllerConfiguration{
				HorizontalPodAutoscalerSyncPeriod:                   metav1.Duration{Duration: 45 * time.Second},
				HorizontalPodAutoscalerUpscaleForbiddenWindow:       metav1.Duration{Duration: 1 * time.Minute},
				HorizontalPodAutoscalerDownscaleForbiddenWindow:     metav1.Duration{Duration: 2 * time.Minute},
				HorizontalPodAutoscalerDownscaleStabilizationWindow: metav1.Duration{Duration: 3 * time.Minute},
				HorizontalPodAutoscalerCPUInitializationPeriod:      metav1.Duration{Duration: 90 * time.Second},
				HorizontalPodAutoscalerInitialReadinessDelay:        metav1.Duration{Duration: 50 * time.Second},
				HorizontalPodAutoscalerTolerance:                    0.1,
				HorizontalPodAutoscalerUseRESTClients:               true,
			},
		},
		JobController: &JobControllerOptions{
			&jobconfig.JobControllerConfiguration{
				ConcurrentJobSyncs: 5,
			},
		},
		NamespaceController: &NamespaceControllerOptions{
			&namespaceconfig.NamespaceControllerConfiguration{
				NamespaceSyncPeriod:      metav1.Duration{Duration: 10 * time.Minute},
				ConcurrentNamespaceSyncs: 20,
			},
		},
		NodeIPAMController: &NodeIPAMControllerOptions{
			&nodeipamconfig.NodeIPAMControllerConfiguration{
				NodeCIDRMaskSize:     48,
				NodeCIDRMaskSizeIPv4: 48,
				NodeCIDRMaskSizeIPv6: 108,
			},
		},
		NodeLifecycleController: &NodeLifecycleControllerOptions{
			&nodelifecycleconfig.NodeLifecycleControllerConfiguration{
				EnableTaintManager:        false,
				NodeEvictionRate:          0.2,
				SecondaryNodeEvictionRate: 0.05,
				NodeMonitorGracePeriod:    metav1.Duration{Duration: 30 * time.Second},
				NodeStartupGracePeriod:    metav1.Duration{Duration: 30 * time.Second},
				PodEvictionTimeout:        metav1.Duration{Duration: 2 * time.Minute},
				LargeClusterSizeThreshold: 100,
				UnhealthyZoneThreshold:    0.6,
			},
		},
		PersistentVolumeBinderController: &PersistentVolumeBinderControllerOptions{
			&persistentvolumeconfig.PersistentVolumeBinderControllerConfiguration{
				PVClaimBinderSyncPeriod: metav1.Duration{Duration: 30 * time.Second},
				VolumeConfiguration: persistentvolumeconfig.VolumeConfiguration{
					EnableDynamicProvisioning:  false,
					EnableHostPathProvisioning: true,
					FlexVolumePluginDir:        "/flex-volume-plugin",
					PersistentVolumeRecyclerConfiguration: persistentvolumeconfig.PersistentVolumeRecyclerConfiguration{
						MaximumRetry:             3,
						MinimumTimeoutNFS:        200,
						IncrementTimeoutNFS:      45,
						MinimumTimeoutHostPath:   45,
						IncrementTimeoutHostPath: 45,
					},
				},
				VolumeHostCIDRDenylist:       []string{"127.0.0.1/28", "feed::/16"},
				VolumeHostAllowLocalLoopback: false,
			},
		},
		PodGCController: &PodGCControllerOptions{
			&podgcconfig.PodGCControllerConfiguration{
				TerminatedPodGCThreshold: 12000,
			},
		},
		ReplicaSetController: &ReplicaSetControllerOptions{
			&replicasetconfig.ReplicaSetControllerConfiguration{
				ConcurrentRSSyncs: 10,
			},
		},
		ReplicationController: &ReplicationControllerOptions{
			&replicationconfig.ReplicationControllerConfiguration{
				ConcurrentRCSyncs: 10,
			},
		},
		ResourceQuotaController: &ResourceQuotaControllerOptions{
			&resourcequotaconfig.ResourceQuotaControllerConfiguration{
				ResourceQuotaSyncPeriod:      metav1.Duration{Duration: 10 * time.Minute},
				ConcurrentResourceQuotaSyncs: 10,
			},
		},
		SAController: &SAControllerOptions{
			&serviceaccountconfig.SAControllerConfiguration{
				ServiceAccountKeyFile:  "/service-account-private-key",
				ConcurrentSATokenSyncs: 10,
			},
		},
		TTLAfterFinishedController: &TTLAfterFinishedControllerOptions{
			&ttlafterfinishedconfig.TTLAfterFinishedControllerConfiguration{
				ConcurrentTTLSyncs: 8,
			},
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
			CacheTTL:            10 * time.Second,
			ClientTimeout:       10 * time.Second,
			WebhookRetryBackoff: apiserveroptions.DefaultAuthWebhookRetryBackoff(),
			ClientCert:          apiserveroptions.ClientCertAuthenticationOptions{},
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
			ClientTimeout:                10 * time.Second,
			WebhookRetryBackoff:          apiserveroptions.DefaultAuthWebhookRetryBackoff(),
			RemoteKubeConfigFileOptional: true,
			AlwaysAllowPaths:             []string{"/healthz"}, // note: this does not match /healthz/ or /healthz/*
		},
		Kubeconfig: "/kubeconfig",
		Master:     "192.168.4.20",
		Metrics:    &metrics.Options{},
		Logs:       logs.NewOptions(),
	}

	// Sort GCIgnoredResources because it's built from a map, which means the
	// insertion order is random.
	sort.Sort(sortedGCIgnoredResources(expected.GarbageCollectorController.GCIgnoredResources))

	if !reflect.DeepEqual(expected, s) {
		t.Errorf("Got different run options than expected.\nDifference detected on:\n%s", diff.ObjectReflectDiff(expected, s))
	}
}

func TestApplyTo(t *testing.T) {
	fs := pflag.NewFlagSet("addflagstest", pflag.ContinueOnError)
	s, _ := NewKubeControllerManagerOptions()
	// flag set to parse the args that are required to start the kube controller manager
	for _, f := range s.Flags([]string{""}, []string{""}).FlagSets {
		fs.AddFlagSet(f)
	}

	fs.Parse(args)
	// Sort GCIgnoredResources because it's built from a map, which means the
	// insertion order is random.
	sort.Sort(sortedGCIgnoredResources(s.GarbageCollectorController.GCIgnoredResources))

	expected := &kubecontrollerconfig.Config{
		ComponentConfig: kubectrlmgrconfig.KubeControllerManagerConfiguration{
			Generic: cmconfig.GenericControllerManagerConfiguration{
				Port:            10252,     // Note: InsecureServingOptions.ApplyTo will write the flag value back into the component config
				Address:         "0.0.0.0", // Note: InsecureServingOptions.ApplyTo will write the flag value back into the component config
				MinResyncPeriod: metav1.Duration{Duration: 8 * time.Hour},
				ClientConnection: componentbaseconfig.ClientConnectionConfiguration{
					ContentType: "application/json",
					QPS:         50.0,
					Burst:       100,
				},
				ControllerStartInterval: metav1.Duration{Duration: 2 * time.Minute},
				LeaderElection: componentbaseconfig.LeaderElectionConfiguration{
					ResourceLock:      "configmap",
					LeaderElect:       false,
					LeaseDuration:     metav1.Duration{Duration: 30 * time.Second},
					RenewDeadline:     metav1.Duration{Duration: 15 * time.Second},
					RetryPeriod:       metav1.Duration{Duration: 5 * time.Second},
					ResourceName:      "kube-controller-manager",
					ResourceNamespace: "kube-system",
				},
				Controllers: []string{"foo", "bar"},
				Debugging: componentbaseconfig.DebuggingConfiguration{
					EnableProfiling:           false,
					EnableContentionProfiling: true,
				},
			},
			KubeCloudShared: cpconfig.KubeCloudSharedConfiguration{
				UseServiceAccountCredentials: true,
				RouteReconciliationPeriod:    metav1.Duration{Duration: 30 * time.Second},
				NodeMonitorPeriod:            metav1.Duration{Duration: 10 * time.Second},
				ClusterName:                  "k8s",
				ClusterCIDR:                  "1.2.3.4/24",
				AllocateNodeCIDRs:            true,
				CIDRAllocatorType:            "CloudAllocator",
				ConfigureCloudRoutes:         false,
				CloudProvider: cpconfig.CloudProviderConfiguration{
					Name:            "gce",
					CloudConfigFile: "/cloud-config",
				},
			},
			ServiceController: serviceconfig.ServiceControllerConfiguration{
				ConcurrentServiceSyncs: 2,
			},
			AttachDetachController: attachdetachconfig.AttachDetachControllerConfiguration{
				ReconcilerSyncLoopPeriod:          metav1.Duration{Duration: 30 * time.Second},
				DisableAttachDetachReconcilerSync: true,
			},
			CSRSigningController: csrsigningconfig.CSRSigningControllerConfiguration{
				ClusterSigningCertFile: "/cluster-signing-cert",
				ClusterSigningKeyFile:  "/cluster-signing-key",
				ClusterSigningDuration: metav1.Duration{Duration: 10 * time.Hour},
				KubeletServingSignerConfiguration: csrsigningconfig.CSRSigningConfiguration{
					CertFile: "/cluster-signing-kubelet-serving/cert-file",
					KeyFile:  "/cluster-signing-kubelet-serving/key-file",
				},
				KubeletClientSignerConfiguration: csrsigningconfig.CSRSigningConfiguration{
					CertFile: "/cluster-signing-kubelet-client/cert-file",
					KeyFile:  "/cluster-signing-kubelet-client/key-file",
				},
				KubeAPIServerClientSignerConfiguration: csrsigningconfig.CSRSigningConfiguration{
					CertFile: "/cluster-signing-kube-apiserver-client/cert-file",
					KeyFile:  "/cluster-signing-kube-apiserver-client/key-file",
				},
				LegacyUnknownSignerConfiguration: csrsigningconfig.CSRSigningConfiguration{
					CertFile: "/cluster-signing-legacy-unknown/cert-file",
					KeyFile:  "/cluster-signing-legacy-unknown/key-file",
				},
			},
			DaemonSetController: daemonconfig.DaemonSetControllerConfiguration{
				ConcurrentDaemonSetSyncs: 2,
			},
			DeploymentController: deploymentconfig.DeploymentControllerConfiguration{
				ConcurrentDeploymentSyncs:      10,
				DeploymentControllerSyncPeriod: metav1.Duration{Duration: 45 * time.Second},
			},
			StatefulSetController: statefulsetconfig.StatefulSetControllerConfiguration{
				ConcurrentStatefulSetSyncs: 15,
			},
			DeprecatedController: kubectrlmgrconfig.DeprecatedControllerConfiguration{
				DeletingPodsQPS:    0.1,
				RegisterRetryCount: 10,
			},
			EndpointController: endpointconfig.EndpointControllerConfiguration{
				ConcurrentEndpointSyncs: 10,
			},
			EndpointSliceController: endpointsliceconfig.EndpointSliceControllerConfiguration{
				ConcurrentServiceEndpointSyncs: 10,
				MaxEndpointsPerSlice:           200,
			},
			EndpointSliceMirroringController: endpointslicemirroringconfig.EndpointSliceMirroringControllerConfiguration{
				MirroringConcurrentServiceEndpointSyncs: 2,
				MirroringMaxEndpointsPerSubset:          1000,
			},
			GarbageCollectorController: garbagecollectorconfig.GarbageCollectorControllerConfiguration{
				ConcurrentGCSyncs: 30,
				GCIgnoredResources: []garbagecollectorconfig.GroupResource{
					{Group: "", Resource: "events"},
				},
				EnableGarbageCollector: false,
			},
			HPAController: poautosclerconfig.HPAControllerConfiguration{
				HorizontalPodAutoscalerSyncPeriod:                   metav1.Duration{Duration: 45 * time.Second},
				HorizontalPodAutoscalerUpscaleForbiddenWindow:       metav1.Duration{Duration: 1 * time.Minute},
				HorizontalPodAutoscalerDownscaleForbiddenWindow:     metav1.Duration{Duration: 2 * time.Minute},
				HorizontalPodAutoscalerDownscaleStabilizationWindow: metav1.Duration{Duration: 3 * time.Minute},
				HorizontalPodAutoscalerCPUInitializationPeriod:      metav1.Duration{Duration: 90 * time.Second},
				HorizontalPodAutoscalerInitialReadinessDelay:        metav1.Duration{Duration: 50 * time.Second},
				HorizontalPodAutoscalerTolerance:                    0.1,
				HorizontalPodAutoscalerUseRESTClients:               true,
			},
			JobController: jobconfig.JobControllerConfiguration{
				ConcurrentJobSyncs: 5,
			},
			NamespaceController: namespaceconfig.NamespaceControllerConfiguration{
				NamespaceSyncPeriod:      metav1.Duration{Duration: 10 * time.Minute},
				ConcurrentNamespaceSyncs: 20,
			},
			NodeIPAMController: nodeipamconfig.NodeIPAMControllerConfiguration{
				NodeCIDRMaskSize:     48,
				NodeCIDRMaskSizeIPv4: 48,
				NodeCIDRMaskSizeIPv6: 108,
			},
			NodeLifecycleController: nodelifecycleconfig.NodeLifecycleControllerConfiguration{
				EnableTaintManager:        false,
				NodeEvictionRate:          0.2,
				SecondaryNodeEvictionRate: 0.05,
				NodeMonitorGracePeriod:    metav1.Duration{Duration: 30 * time.Second},
				NodeStartupGracePeriod:    metav1.Duration{Duration: 30 * time.Second},
				PodEvictionTimeout:        metav1.Duration{Duration: 2 * time.Minute},
				LargeClusterSizeThreshold: 100,
				UnhealthyZoneThreshold:    0.6,
			},
			PersistentVolumeBinderController: persistentvolumeconfig.PersistentVolumeBinderControllerConfiguration{
				PVClaimBinderSyncPeriod: metav1.Duration{Duration: 30 * time.Second},
				VolumeConfiguration: persistentvolumeconfig.VolumeConfiguration{
					EnableDynamicProvisioning:  false,
					EnableHostPathProvisioning: true,
					FlexVolumePluginDir:        "/flex-volume-plugin",
					PersistentVolumeRecyclerConfiguration: persistentvolumeconfig.PersistentVolumeRecyclerConfiguration{
						MaximumRetry:             3,
						MinimumTimeoutNFS:        200,
						IncrementTimeoutNFS:      45,
						MinimumTimeoutHostPath:   45,
						IncrementTimeoutHostPath: 45,
					},
				},
				VolumeHostCIDRDenylist:       []string{"127.0.0.1/28", "feed::/16"},
				VolumeHostAllowLocalLoopback: false,
			},
			PodGCController: podgcconfig.PodGCControllerConfiguration{
				TerminatedPodGCThreshold: 12000,
			},
			ReplicaSetController: replicasetconfig.ReplicaSetControllerConfiguration{
				ConcurrentRSSyncs: 10,
			},
			ReplicationController: replicationconfig.ReplicationControllerConfiguration{
				ConcurrentRCSyncs: 10,
			},
			ResourceQuotaController: resourcequotaconfig.ResourceQuotaControllerConfiguration{
				ResourceQuotaSyncPeriod:      metav1.Duration{Duration: 10 * time.Minute},
				ConcurrentResourceQuotaSyncs: 10,
			},
			SAController: serviceaccountconfig.SAControllerConfiguration{
				ServiceAccountKeyFile:  "/service-account-private-key",
				ConcurrentSATokenSyncs: 10,
			},
			TTLAfterFinishedController: ttlafterfinishedconfig.TTLAfterFinishedControllerConfiguration{
				ConcurrentTTLSyncs: 8,
			},
		},
	}

	// Sort GCIgnoredResources because it's built from a map, which means the
	// insertion order is random.
	sort.Sort(sortedGCIgnoredResources(expected.ComponentConfig.GarbageCollectorController.GCIgnoredResources))

	c := &kubecontrollerconfig.Config{}
	s.ApplyTo(c)

	if !reflect.DeepEqual(expected.ComponentConfig, c.ComponentConfig) {
		t.Errorf("Got different configuration than expected.\nDifference detected on:\n%s", diff.ObjectReflectDiff(expected.ComponentConfig, c.ComponentConfig))
	}
}

type sortedGCIgnoredResources []garbagecollectorconfig.GroupResource

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
