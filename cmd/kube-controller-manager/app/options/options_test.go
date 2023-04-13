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
	"reflect"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/spf13/pflag"

	eventv1 "k8s.io/api/events/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	apiserveroptions "k8s.io/apiserver/pkg/server/options"
	cpconfig "k8s.io/cloud-provider/config"
	serviceconfig "k8s.io/cloud-provider/controllers/service/config"
	cpoptions "k8s.io/cloud-provider/options"
	componentbaseconfig "k8s.io/component-base/config"
	"k8s.io/component-base/logs"
	"k8s.io/component-base/metrics"
	cmconfig "k8s.io/controller-manager/config"
	cmoptions "k8s.io/controller-manager/options"
	migration "k8s.io/controller-manager/pkg/leadermigration/options"
	netutils "k8s.io/utils/net"

	kubecontrollerconfig "k8s.io/kubernetes/cmd/kube-controller-manager/app/config"
	kubectrlmgrconfig "k8s.io/kubernetes/pkg/controller/apis/config"
	csrsigningconfig "k8s.io/kubernetes/pkg/controller/certificates/signer/config"
	cronjobconfig "k8s.io/kubernetes/pkg/controller/cronjob/config"
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
	ephemeralvolumeconfig "k8s.io/kubernetes/pkg/controller/volume/ephemeral/config"
	persistentvolumeconfig "k8s.io/kubernetes/pkg/controller/volume/persistentvolume/config"
)

var args = []string{
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
	"--concurrent-horizontal-pod-autoscaler-syncs=10",
	"--concurrent-statefulset-syncs=15",
	"--concurrent-endpoint-syncs=10",
	"--concurrent-ephemeralvolume-syncs=10",
	"--concurrent-service-endpoint-syncs=10",
	"--concurrent-gc-syncs=30",
	"--concurrent-namespace-syncs=20",
	"--concurrent-job-syncs=10",
	"--concurrent-replicaset-syncs=10",
	"--concurrent-resource-quota-syncs=10",
	"--concurrent-service-syncs=2",
	"--concurrent-serviceaccount-token-syncs=10",
	"--concurrent_rc_syncs=10",
	"--configure-cloud-routes=false",
	"--contention-profiling=true",
	"--controller-start-interval=2m",
	"--controllers=foo,bar",
	"--disable-attach-detach-reconcile-sync=true",
	"--enable-dynamic-provisioning=false",
	"--enable-garbage-collector=false",
	"--enable-hostpath-provisioner=true",
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
				Address:         "0.0.0.0", // Note: This field should have no effect in CM now, and "0.0.0.0" is the default value.
				MinResyncPeriod: metav1.Duration{Duration: 8 * time.Hour},
				ClientConnection: componentbaseconfig.ClientConnectionConfiguration{
					Kubeconfig:  "/kubeconfig",
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
			LeaderMigration: &migration.LeaderMigrationOptions{},
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
				ConcurrentDeploymentSyncs: 10,
			},
		},
		StatefulSetController: &StatefulSetControllerOptions{
			&statefulsetconfig.StatefulSetControllerConfiguration{
				ConcurrentStatefulSetSyncs: 15,
			},
		},
		DeprecatedFlags: &DeprecatedControllerOptions{
			&kubectrlmgrconfig.DeprecatedControllerConfiguration{},
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
		EphemeralVolumeController: &EphemeralVolumeControllerOptions{
			&ephemeralvolumeconfig.EphemeralVolumeControllerConfiguration{
				ConcurrentEphemeralVolumeSyncs: 10,
			},
		},
		GarbageCollectorController: &GarbageCollectorControllerOptions{
			&garbagecollectorconfig.GarbageCollectorControllerConfiguration{
				ConcurrentGCSyncs: 30,
				GCIgnoredResources: []garbagecollectorconfig.GroupResource{
					{Group: "", Resource: "events"},
					{Group: eventv1.GroupName, Resource: "events"},
				},
				EnableGarbageCollector: false,
			},
		},
		HPAController: &HPAControllerOptions{
			&poautosclerconfig.HPAControllerConfiguration{
				ConcurrentHorizontalPodAutoscalerSyncs:              10,
				HorizontalPodAutoscalerSyncPeriod:                   metav1.Duration{Duration: 45 * time.Second},
				HorizontalPodAutoscalerUpscaleForbiddenWindow:       metav1.Duration{Duration: 1 * time.Minute},
				HorizontalPodAutoscalerDownscaleForbiddenWindow:     metav1.Duration{Duration: 2 * time.Minute},
				HorizontalPodAutoscalerDownscaleStabilizationWindow: metav1.Duration{Duration: 3 * time.Minute},
				HorizontalPodAutoscalerCPUInitializationPeriod:      metav1.Duration{Duration: 90 * time.Second},
				HorizontalPodAutoscalerInitialReadinessDelay:        metav1.Duration{Duration: 50 * time.Second},
				HorizontalPodAutoscalerTolerance:                    0.1,
			},
		},
		JobController: &JobControllerOptions{
			&jobconfig.JobControllerConfiguration{
				ConcurrentJobSyncs: 10,
			},
		},
		CronJobController: &CronJobControllerOptions{
			&cronjobconfig.CronJobControllerConfiguration{
				ConcurrentCronJobSyncs: 5,
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
				NodeEvictionRate:          0.2,
				SecondaryNodeEvictionRate: 0.05,
				NodeMonitorGracePeriod:    metav1.Duration{Duration: 30 * time.Second},
				NodeStartupGracePeriod:    metav1.Duration{Duration: 30 * time.Second},
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
			BindAddress: netutils.ParseIPSloppy("192.168.4.21"),
			ServerCert: apiserveroptions.GeneratableKeyCert{
				CertDirectory: "/a/b/c",
				PairName:      "kube-controller-manager",
			},
			HTTP2MaxStreamsPerConnection: 47,
		}).WithLoopback(),
		Authentication: &apiserveroptions.DelegatingAuthenticationOptions{
			CacheTTL:            10 * time.Second,
			TokenRequestTimeout: 10 * time.Second,
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
			AlwaysAllowPaths:             []string{"/healthz", "/readyz", "/livez"}, // note: this does not match /healthz/ or /healthz/*
			AlwaysAllowGroups:            []string{"system:masters"},
		},
		Master:  "192.168.4.20",
		Metrics: &metrics.Options{},
		Logs:    logs.NewOptions(),
	}

	// Sort GCIgnoredResources because it's built from a map, which means the
	// insertion order is random.
	sort.Sort(sortedGCIgnoredResources(expected.GarbageCollectorController.GCIgnoredResources))

	if !reflect.DeepEqual(expected, s) {
		t.Errorf("Got different run options than expected.\nDifference detected on:\n%s", cmp.Diff(expected, s))
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
				Address:         "0.0.0.0", // Note: This field should have no effect in CM now, and "0.0.0.0" is the default value.
				MinResyncPeriod: metav1.Duration{Duration: 8 * time.Hour},
				ClientConnection: componentbaseconfig.ClientConnectionConfiguration{
					Kubeconfig:  "/kubeconfig",
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
				ConcurrentDeploymentSyncs: 10,
			},
			StatefulSetController: statefulsetconfig.StatefulSetControllerConfiguration{
				ConcurrentStatefulSetSyncs: 15,
			},
			DeprecatedController: kubectrlmgrconfig.DeprecatedControllerConfiguration{},
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
			EphemeralVolumeController: ephemeralvolumeconfig.EphemeralVolumeControllerConfiguration{
				ConcurrentEphemeralVolumeSyncs: 10,
			},
			GarbageCollectorController: garbagecollectorconfig.GarbageCollectorControllerConfiguration{
				ConcurrentGCSyncs: 30,
				GCIgnoredResources: []garbagecollectorconfig.GroupResource{
					{Group: "", Resource: "events"},
					{Group: eventv1.GroupName, Resource: "events"},
				},
				EnableGarbageCollector: false,
			},
			HPAController: poautosclerconfig.HPAControllerConfiguration{
				ConcurrentHorizontalPodAutoscalerSyncs:              10,
				HorizontalPodAutoscalerSyncPeriod:                   metav1.Duration{Duration: 45 * time.Second},
				HorizontalPodAutoscalerUpscaleForbiddenWindow:       metav1.Duration{Duration: 1 * time.Minute},
				HorizontalPodAutoscalerDownscaleForbiddenWindow:     metav1.Duration{Duration: 2 * time.Minute},
				HorizontalPodAutoscalerDownscaleStabilizationWindow: metav1.Duration{Duration: 3 * time.Minute},
				HorizontalPodAutoscalerCPUInitializationPeriod:      metav1.Duration{Duration: 90 * time.Second},
				HorizontalPodAutoscalerInitialReadinessDelay:        metav1.Duration{Duration: 50 * time.Second},
				HorizontalPodAutoscalerTolerance:                    0.1,
			},
			JobController: jobconfig.JobControllerConfiguration{
				ConcurrentJobSyncs: 10,
			},
			CronJobController: cronjobconfig.CronJobControllerConfiguration{
				ConcurrentCronJobSyncs: 5,
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
				NodeEvictionRate:          0.2,
				SecondaryNodeEvictionRate: 0.05,
				NodeMonitorGracePeriod:    metav1.Duration{Duration: 30 * time.Second},
				NodeStartupGracePeriod:    metav1.Duration{Duration: 30 * time.Second},
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
		t.Errorf("Got different configuration than expected.\nDifference detected on:\n%s", cmp.Diff(expected.ComponentConfig, c.ComponentConfig))
	}
}

func TestValidateControllersOptions(t *testing.T) {
	testCases := []struct {
		name                   string
		expectErrors           bool
		expectedErrorSubString string
		validate               func() []error
	}{
		{
			name:                   "AttachDetachControllerOptions reconciler sync loop period less than one second",
			expectErrors:           true,
			expectedErrorSubString: "duration time must be greater than one second",
			validate: (&AttachDetachControllerOptions{
				&attachdetachconfig.AttachDetachControllerConfiguration{
					ReconcilerSyncLoopPeriod:          metav1.Duration{Duration: time.Second / 2},
					DisableAttachDetachReconcilerSync: true,
				},
			}).Validate,
		},
		{
			name:                   "CSRSigningControllerOptions KubeletServingSignerConfiguration no cert file",
			expectErrors:           true,
			expectedErrorSubString: "cannot specify key without cert",
			validate: (&CSRSigningControllerOptions{
				&csrsigningconfig.CSRSigningControllerConfiguration{
					ClusterSigningCertFile: "",
					ClusterSigningKeyFile:  "",
					ClusterSigningDuration: metav1.Duration{Duration: 10 * time.Hour},
					KubeletServingSignerConfiguration: csrsigningconfig.CSRSigningConfiguration{
						CertFile: "",
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
			}).Validate,
		},
		{
			name:                   "CSRSigningControllerOptions KubeletServingSignerConfiguration no key file",
			expectErrors:           true,
			expectedErrorSubString: "cannot specify cert without key",
			validate: (&CSRSigningControllerOptions{
				&csrsigningconfig.CSRSigningControllerConfiguration{
					ClusterSigningCertFile: "",
					ClusterSigningKeyFile:  "",
					ClusterSigningDuration: metav1.Duration{Duration: 10 * time.Hour},
					KubeletServingSignerConfiguration: csrsigningconfig.CSRSigningConfiguration{
						CertFile: "/cluster-signing-kubelet-serving/cert-file",
						KeyFile:  "",
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
			}).Validate,
		},
		{
			name:                   "CSRSigningControllerOptions KubeletClientSignerConfiguration no cert file",
			expectErrors:           true,
			expectedErrorSubString: "cannot specify key without cert",
			validate: (&CSRSigningControllerOptions{
				&csrsigningconfig.CSRSigningControllerConfiguration{
					ClusterSigningCertFile: "",
					ClusterSigningKeyFile:  "",
					ClusterSigningDuration: metav1.Duration{Duration: 10 * time.Hour},
					KubeletServingSignerConfiguration: csrsigningconfig.CSRSigningConfiguration{
						CertFile: "/cluster-signing-kubelet-serving/cert-file",
						KeyFile:  "/cluster-signing-kubelet-serving/key-file",
					},
					KubeletClientSignerConfiguration: csrsigningconfig.CSRSigningConfiguration{
						CertFile: "",
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
			}).Validate,
		},
		{
			name:                   "CSRSigningControllerOptions KubeletClientSignerConfiguration no key file",
			expectErrors:           true,
			expectedErrorSubString: "cannot specify cert without key",
			validate: (&CSRSigningControllerOptions{
				&csrsigningconfig.CSRSigningControllerConfiguration{
					ClusterSigningCertFile: "",
					ClusterSigningKeyFile:  "",
					ClusterSigningDuration: metav1.Duration{Duration: 10 * time.Hour},
					KubeletServingSignerConfiguration: csrsigningconfig.CSRSigningConfiguration{
						CertFile: "/cluster-signing-kubelet-serving/cert-file",
						KeyFile:  "/cluster-signing-kubelet-serving/key-file",
					},
					KubeletClientSignerConfiguration: csrsigningconfig.CSRSigningConfiguration{
						CertFile: "/cluster-signing-kubelet-client/cert-file",
						KeyFile:  "",
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
			}).Validate,
		},
		{
			name:                   "CSRSigningControllerOptions KubeAPIServerClientSignerConfiguration no cert file",
			expectErrors:           true,
			expectedErrorSubString: "cannot specify key without cert",
			validate: (&CSRSigningControllerOptions{
				&csrsigningconfig.CSRSigningControllerConfiguration{
					ClusterSigningCertFile: "",
					ClusterSigningKeyFile:  "",
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
						CertFile: "",
						KeyFile:  "/cluster-signing-kube-apiserver-client/key-file",
					},
					LegacyUnknownSignerConfiguration: csrsigningconfig.CSRSigningConfiguration{
						CertFile: "/cluster-signing-legacy-unknown/cert-file",
						KeyFile:  "/cluster-signing-legacy-unknown/key-file",
					},
				},
			}).Validate,
		},
		{
			name:                   "CSRSigningControllerOptions KubeAPIServerClientSignerConfiguration no key file",
			expectErrors:           true,
			expectedErrorSubString: "cannot specify cert without key",
			validate: (&CSRSigningControllerOptions{
				&csrsigningconfig.CSRSigningControllerConfiguration{
					ClusterSigningCertFile: "",
					ClusterSigningKeyFile:  "",
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
						KeyFile:  "",
					},
					LegacyUnknownSignerConfiguration: csrsigningconfig.CSRSigningConfiguration{
						CertFile: "/cluster-signing-legacy-unknown/cert-file",
						KeyFile:  "/cluster-signing-legacy-unknown/key-file",
					},
				},
			}).Validate,
		},
		{
			name:                   "CSRSigningControllerOptions LegacyUnknownSignerConfiguration no cert file",
			expectErrors:           true,
			expectedErrorSubString: "cannot specify key without cert",
			validate: (&CSRSigningControllerOptions{
				&csrsigningconfig.CSRSigningControllerConfiguration{
					ClusterSigningCertFile: "",
					ClusterSigningKeyFile:  "",
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
						CertFile: "",
						KeyFile:  "/cluster-signing-legacy-unknown/key-file",
					},
				},
			}).Validate,
		},
		{
			name:                   "CSRSigningControllerOptions LegacyUnknownSignerConfiguration no key file",
			expectErrors:           true,
			expectedErrorSubString: "cannot specify cert without key",
			validate: (&CSRSigningControllerOptions{
				&csrsigningconfig.CSRSigningControllerConfiguration{
					ClusterSigningCertFile: "",
					ClusterSigningKeyFile:  "",
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
						KeyFile:  "",
					},
				},
			}).Validate,
		},
		{
			name:                   "CSRSigningControllerOptions specific file set along with cluster single signing file",
			expectErrors:           true,
			expectedErrorSubString: "cannot specify --cluster-signing-{cert,key}-file and other --cluster-signing-*-file flags at the same time",
			validate: (&CSRSigningControllerOptions{
				&csrsigningconfig.CSRSigningControllerConfiguration{
					ClusterSigningCertFile: "/cluster-signing-cert-file",
					ClusterSigningKeyFile:  "/cluster-signing-key-file",
					ClusterSigningDuration: metav1.Duration{Duration: 10 * time.Hour},
					KubeletServingSignerConfiguration: csrsigningconfig.CSRSigningConfiguration{
						CertFile: "/cluster-signing-kubelet-serving/cert-file",
						KeyFile:  "",
					},
					KubeletClientSignerConfiguration: csrsigningconfig.CSRSigningConfiguration{
						CertFile: "",
						KeyFile:  "",
					},
					KubeAPIServerClientSignerConfiguration: csrsigningconfig.CSRSigningConfiguration{
						CertFile: "",
						KeyFile:  "",
					},
					LegacyUnknownSignerConfiguration: csrsigningconfig.CSRSigningConfiguration{
						CertFile: "",
						KeyFile:  "",
					},
				},
			}).Validate,
		},
		{
			name:                   "EndpointSliceControllerOptions ConcurrentServiceEndpointSyncs lower than minConcurrentServiceEndpointSyncs (1)",
			expectErrors:           true,
			expectedErrorSubString: "concurrent-service-endpoint-syncs must not be less than 1",
			validate: (&EndpointSliceControllerOptions{
				&endpointsliceconfig.EndpointSliceControllerConfiguration{
					ConcurrentServiceEndpointSyncs: 0,
					MaxEndpointsPerSlice:           200,
				},
			}).Validate,
		},
		{
			name:                   "EndpointSliceControllerOptions ConcurrentServiceEndpointSyncs greater than maxConcurrentServiceEndpointSyncs (50)",
			expectErrors:           true,
			expectedErrorSubString: "concurrent-service-endpoint-syncs must not be more than 50",
			validate: (&EndpointSliceControllerOptions{
				&endpointsliceconfig.EndpointSliceControllerConfiguration{
					ConcurrentServiceEndpointSyncs: 51,
					MaxEndpointsPerSlice:           200,
				},
			}).Validate,
		},
		{
			name:                   "EndpointSliceControllerOptions MaxEndpointsPerSlice lower than minMaxEndpointsPerSlice (1)",
			expectErrors:           true,
			expectedErrorSubString: "max-endpoints-per-slice must not be less than 1",
			validate: (&EndpointSliceControllerOptions{
				&endpointsliceconfig.EndpointSliceControllerConfiguration{
					ConcurrentServiceEndpointSyncs: 10,
					MaxEndpointsPerSlice:           0,
				},
			}).Validate,
		},
		{
			name:                   "EndpointSliceControllerOptions MaxEndpointsPerSlice greater than maxMaxEndpointsPerSlice (1000)",
			expectErrors:           true,
			expectedErrorSubString: "max-endpoints-per-slice must not be more than 1000",
			validate: (&EndpointSliceControllerOptions{
				&endpointsliceconfig.EndpointSliceControllerConfiguration{
					ConcurrentServiceEndpointSyncs: 10,
					MaxEndpointsPerSlice:           1001,
				},
			}).Validate,
		},
		{
			name:                   "EndpointSliceMirroringControllerOptions MirroringConcurrentServiceEndpointSyncs lower than mirroringMinConcurrentServiceEndpointSyncs (1)",
			expectErrors:           true,
			expectedErrorSubString: "mirroring-concurrent-service-endpoint-syncs must not be less than 1",
			validate: (&EndpointSliceMirroringControllerOptions{
				&endpointslicemirroringconfig.EndpointSliceMirroringControllerConfiguration{
					MirroringConcurrentServiceEndpointSyncs: 0,
					MirroringMaxEndpointsPerSubset:          100,
				},
			}).Validate,
		},
		{
			name:                   "EndpointSliceMirroringControllerOptions MirroringConcurrentServiceEndpointSyncs greater than mirroringMaxConcurrentServiceEndpointSyncs (50)",
			expectErrors:           true,
			expectedErrorSubString: "mirroring-concurrent-service-endpoint-syncs must not be more than 50",
			validate: (&EndpointSliceMirroringControllerOptions{
				&endpointslicemirroringconfig.EndpointSliceMirroringControllerConfiguration{
					MirroringConcurrentServiceEndpointSyncs: 51,
					MirroringMaxEndpointsPerSubset:          100,
				},
			}).Validate,
		},
		{
			name:                   "EndpointSliceMirroringControllerOptions MirroringMaxEndpointsPerSubset lower than mirroringMinMaxEndpointsPerSubset (1)",
			expectErrors:           true,
			expectedErrorSubString: "mirroring-max-endpoints-per-subset must not be less than 1",
			validate: (&EndpointSliceMirroringControllerOptions{
				&endpointslicemirroringconfig.EndpointSliceMirroringControllerConfiguration{
					MirroringConcurrentServiceEndpointSyncs: 10,
					MirroringMaxEndpointsPerSubset:          0,
				},
			}).Validate,
		},
		{
			name:                   "EndpointSliceMirroringControllerOptions MirroringMaxEndpointsPerSubset greater than mirroringMaxMaxEndpointsPerSubset (1000)",
			expectErrors:           true,
			expectedErrorSubString: "mirroring-max-endpoints-per-subset must not be more than 1000",
			validate: (&EndpointSliceMirroringControllerOptions{
				&endpointslicemirroringconfig.EndpointSliceMirroringControllerConfiguration{
					MirroringConcurrentServiceEndpointSyncs: 10,
					MirroringMaxEndpointsPerSubset:          1001,
				},
			}).Validate,
		},
		{
			name:                   "EphemeralVolumeControllerOptions ConcurrentEphemeralVolumeSyncs equal 0",
			expectErrors:           true,
			expectedErrorSubString: "concurrent-ephemeralvolume-syncs must be greater than 0",
			validate: (&EphemeralVolumeControllerOptions{
				&ephemeralvolumeconfig.EphemeralVolumeControllerConfiguration{
					ConcurrentEphemeralVolumeSyncs: 0,
				},
			}).Validate,
		},
		{
			name:                   "HPAControllerOptions ConcurrentHorizontalPodAutoscalerSyncs equal 0",
			expectErrors:           true,
			expectedErrorSubString: "concurrent-horizontal-pod-autoscaler-syncs must be greater than 0",
			validate: (&HPAControllerOptions{
				&poautosclerconfig.HPAControllerConfiguration{
					ConcurrentHorizontalPodAutoscalerSyncs:              0,
					HorizontalPodAutoscalerSyncPeriod:                   metav1.Duration{Duration: 45 * time.Second},
					HorizontalPodAutoscalerUpscaleForbiddenWindow:       metav1.Duration{Duration: 1 * time.Minute},
					HorizontalPodAutoscalerDownscaleForbiddenWindow:     metav1.Duration{Duration: 2 * time.Minute},
					HorizontalPodAutoscalerDownscaleStabilizationWindow: metav1.Duration{Duration: 3 * time.Minute},
					HorizontalPodAutoscalerCPUInitializationPeriod:      metav1.Duration{Duration: 90 * time.Second},
					HorizontalPodAutoscalerInitialReadinessDelay:        metav1.Duration{Duration: 50 * time.Second},
					HorizontalPodAutoscalerTolerance:                    0.1,
				},
			}).Validate,
		},
		{
			name:                   "NodeIPAMControllerOptions service cluster ip range more than two entries",
			expectErrors:           true,
			expectedErrorSubString: "--service-cluster-ip-range can not contain more than two entries",
			validate: (&NodeIPAMControllerOptions{
				&nodeipamconfig.NodeIPAMControllerConfiguration{
					ServiceCIDR:          "10.0.0.0/16,244.0.0.0/16,3000::/108",
					NodeCIDRMaskSize:     48,
					NodeCIDRMaskSizeIPv4: 48,
					NodeCIDRMaskSizeIPv6: 108,
				},
			}).Validate,
		},
		{
			name:                   "PersistentVolumeBinderControllerOptions bad cidr deny list",
			expectErrors:           true,
			expectedErrorSubString: "bad --volume-host-ip-denylist/--volume-host-allow-local-loopback invalid CIDR",
			validate: (&PersistentVolumeBinderControllerOptions{
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
					VolumeHostCIDRDenylist:       []string{"127.0.0.1"},
					VolumeHostAllowLocalLoopback: false,
				},
			}).Validate,
		},
		{
			name:                   "StatefulSetControllerOptions ConcurrentStatefulSetSyncs equal 0",
			expectErrors:           true,
			expectedErrorSubString: "concurrent-statefulset-syncs must be greater than 0",
			validate: (&StatefulSetControllerOptions{
				&statefulsetconfig.StatefulSetControllerConfiguration{
					ConcurrentStatefulSetSyncs: 0,
				},
			}).Validate,
		},
		{
			name:                   "JobControllerOptions ConcurrentJobSyncs equal 0",
			expectErrors:           true,
			expectedErrorSubString: "concurrent-job-syncs must be greater than 0",
			validate: (&JobControllerOptions{
				&jobconfig.JobControllerConfiguration{
					ConcurrentJobSyncs: 0,
				},
			}).Validate,
		},
		/* empty errs */
		{
			name:         "CronJobControllerOptions",
			expectErrors: false,
			validate: (&CronJobControllerOptions{
				&cronjobconfig.CronJobControllerConfiguration{
					ConcurrentCronJobSyncs: 5,
				},
			}).Validate,
		},
		{
			name:         "DaemonSetControllerOptions",
			expectErrors: false,
			validate: (&DaemonSetControllerOptions{
				&daemonconfig.DaemonSetControllerConfiguration{
					ConcurrentDaemonSetSyncs: 2,
				},
			}).Validate,
		},
		{
			name:         "DeploymentControllerOptions",
			expectErrors: false,
			validate: (&DeploymentControllerOptions{
				&deploymentconfig.DeploymentControllerConfiguration{
					ConcurrentDeploymentSyncs: 10,
				},
			}).Validate,
		},
		{
			name:         "DeprecatedControllerOptions",
			expectErrors: false,
			validate: (&DeprecatedControllerOptions{
				&kubectrlmgrconfig.DeprecatedControllerConfiguration{},
			}).Validate,
		},
		{
			name:         "EndpointControllerOptions",
			expectErrors: false,
			validate: (&EndpointControllerOptions{
				&endpointconfig.EndpointControllerConfiguration{
					ConcurrentEndpointSyncs: 10,
				},
			}).Validate,
		},
		{
			name:         "GarbageCollectorControllerOptions",
			expectErrors: false,
			validate: (&GarbageCollectorControllerOptions{
				&garbagecollectorconfig.GarbageCollectorControllerConfiguration{
					ConcurrentGCSyncs: 30,
					GCIgnoredResources: []garbagecollectorconfig.GroupResource{
						{Group: "", Resource: "events"},
						{Group: eventv1.GroupName, Resource: "events"},
					},
					EnableGarbageCollector: false,
				},
			}).Validate,
		},
		{
			name:         "JobControllerOptions",
			expectErrors: false,
			validate: (&JobControllerOptions{
				&jobconfig.JobControllerConfiguration{
					ConcurrentJobSyncs: 10,
				},
			}).Validate,
		},
		{
			name:         "NamespaceControllerOptions",
			expectErrors: false,
			validate: (&NamespaceControllerOptions{
				&namespaceconfig.NamespaceControllerConfiguration{
					NamespaceSyncPeriod:      metav1.Duration{Duration: 10 * time.Minute},
					ConcurrentNamespaceSyncs: 20,
				},
			}).Validate,
		},
		{
			name:         "NodeLifecycleControllerOptions",
			expectErrors: false,
			validate: (&NodeLifecycleControllerOptions{
				&nodelifecycleconfig.NodeLifecycleControllerConfiguration{
					NodeEvictionRate:          0.2,
					SecondaryNodeEvictionRate: 0.05,
					NodeMonitorGracePeriod:    metav1.Duration{Duration: 30 * time.Second},
					NodeStartupGracePeriod:    metav1.Duration{Duration: 30 * time.Second},
					LargeClusterSizeThreshold: 100,
					UnhealthyZoneThreshold:    0.6,
				},
			}).Validate,
		},
		{
			name:         "PodGCControllerOptions",
			expectErrors: false,
			validate: (&PodGCControllerOptions{
				&podgcconfig.PodGCControllerConfiguration{
					TerminatedPodGCThreshold: 12000,
				},
			}).Validate,
		},
		{
			name:         "ReplicaSetControllerOptions",
			expectErrors: false,
			validate: (&ReplicaSetControllerOptions{
				&replicasetconfig.ReplicaSetControllerConfiguration{
					ConcurrentRSSyncs: 10,
				},
			}).Validate,
		},
		{
			name:         "ReplicationControllerOptions",
			expectErrors: false,
			validate: (&ReplicationControllerOptions{
				&replicationconfig.ReplicationControllerConfiguration{
					ConcurrentRCSyncs: 10,
				},
			}).Validate,
		},
		{
			name:         "ResourceQuotaControllerOptions",
			expectErrors: false,
			validate: (&ResourceQuotaControllerOptions{
				&resourcequotaconfig.ResourceQuotaControllerConfiguration{
					ResourceQuotaSyncPeriod:      metav1.Duration{Duration: 10 * time.Minute},
					ConcurrentResourceQuotaSyncs: 10,
				},
			}).Validate,
		},
		{
			name:         "SAControllerOptions",
			expectErrors: false,
			validate: (&SAControllerOptions{
				&serviceaccountconfig.SAControllerConfiguration{
					ServiceAccountKeyFile:  "/service-account-private-key",
					ConcurrentSATokenSyncs: 10,
				},
			}).Validate,
		},
		{
			name:         "TTLAfterFinishedControllerOptions",
			expectErrors: false,
			validate: (&TTLAfterFinishedControllerOptions{
				&ttlafterfinishedconfig.TTLAfterFinishedControllerConfiguration{
					ConcurrentTTLSyncs: 8,
				},
			}).Validate,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			errs := tc.validate()
			if len(errs) > 0 && !tc.expectErrors {
				t.Errorf("expected no errors, errors found %+v", errs)
			}

			if len(errs) == 0 && tc.expectErrors {
				t.Errorf("expected errors, no errors found")
			}

			if len(errs) > 0 && tc.expectErrors {
				gotErr := utilerrors.NewAggregate(errs).Error()
				if !strings.Contains(gotErr, tc.expectedErrorSubString) {
					t.Errorf("expected error: %s, got err: %v", tc.expectedErrorSubString, gotErr)
				}
			}
		})
	}
}

func TestValidateControllerManagerOptions(t *testing.T) {
	opts, err := NewKubeControllerManagerOptions()
	if err != nil {
		t.Errorf("expected no error, error found %+v", err)
	}

	opts.EndpointSliceController.MaxEndpointsPerSlice = 1001 // max endpoints per slice should be a positive integer <= 1000

	if err := opts.Validate([]string{"*"}, []string{""}); err == nil {
		t.Error("expected error, no error found")
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
