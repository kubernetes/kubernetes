/*
Copyright 2015 The Kubernetes Authors.

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

package v1beta1

import (
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	componentbaseconfigv1alpha1 "k8s.io/component-base/config/v1alpha1"
	"k8s.io/kubelet/config/v1beta1"
	"k8s.io/kubernetes/pkg/cluster/ports"
	"k8s.io/kubernetes/pkg/kubelet/qos"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
	utilpointer "k8s.io/utils/pointer"
)

func TestSetDefaultsKubeletConfiguration(t *testing.T) {

	tests := []struct {
		name     string
		config   *v1beta1.KubeletConfiguration
		expected *v1beta1.KubeletConfiguration
	}{
		{
			"empty config",
			&v1beta1.KubeletConfiguration{},
			&v1beta1.KubeletConfiguration{
				EnableServer:       utilpointer.BoolPtr(true),
				SyncFrequency:      v1.Duration{Duration: 1 * time.Minute},
				FileCheckFrequency: v1.Duration{Duration: 20 * time.Second},
				HTTPCheckFrequency: v1.Duration{Duration: 20 * time.Second},
				Address:            "0.0.0.0",
				Port:               ports.KubeletPort,
				Authentication: v1beta1.KubeletAuthentication{
					Anonymous: v1beta1.KubeletAnonymousAuthentication{Enabled: utilpointer.BoolPtr(false)},
					Webhook: v1beta1.KubeletWebhookAuthentication{
						Enabled:  utilpointer.BoolPtr(true),
						CacheTTL: v1.Duration{Duration: 2 * time.Minute},
					},
				},
				Authorization: v1beta1.KubeletAuthorization{
					Mode: v1beta1.KubeletAuthorizationModeWebhook,
					Webhook: v1beta1.KubeletWebhookAuthorization{
						CacheAuthorizedTTL:   v1.Duration{Duration: 5 * time.Minute},
						CacheUnauthorizedTTL: v1.Duration{Duration: 30 * time.Second},
					},
				},
				RegistryPullQPS:                           utilpointer.Int32Ptr(5),
				RegistryBurst:                             10,
				EventRecordQPS:                            utilpointer.Int32Ptr(5),
				EventBurst:                                10,
				EnableDebuggingHandlers:                   utilpointer.BoolPtr(true),
				HealthzPort:                               utilpointer.Int32Ptr(10248),
				HealthzBindAddress:                        "127.0.0.1",
				OOMScoreAdj:                               utilpointer.Int32Ptr(int32(qos.KubeletOOMScoreAdj)),
				StreamingConnectionIdleTimeout:            v1.Duration{Duration: 4 * time.Hour},
				NodeStatusUpdateFrequency:                 v1.Duration{Duration: 10 * time.Second},
				NodeStatusReportFrequency:                 v1.Duration{Duration: 5 * time.Minute},
				NodeLeaseDurationSeconds:                  40,
				ImageMinimumGCAge:                         v1.Duration{Duration: 2 * time.Minute},
				ImageGCHighThresholdPercent:               utilpointer.Int32Ptr(85),
				ImageGCLowThresholdPercent:                utilpointer.Int32Ptr(80),
				VolumeStatsAggPeriod:                      v1.Duration{Duration: time.Minute},
				CgroupsPerQOS:                             utilpointer.BoolPtr(true),
				CgroupDriver:                              "cgroupfs",
				CPUManagerPolicy:                          "none",
				CPUManagerReconcilePeriod:                 v1.Duration{Duration: 10 * time.Second},
				MemoryManagerPolicy:                       v1beta1.NoneMemoryManagerPolicy,
				TopologyManagerPolicy:                     v1beta1.NoneTopologyManagerPolicy,
				TopologyManagerScope:                      v1beta1.ContainerTopologyManagerScope,
				RuntimeRequestTimeout:                     v1.Duration{Duration: 2 * time.Minute},
				HairpinMode:                               v1beta1.PromiscuousBridge,
				MaxPods:                                   110,
				PodPidsLimit:                              utilpointer.Int64(-1),
				ResolverConfig:                            utilpointer.String(kubetypes.ResolvConfDefault),
				CPUCFSQuota:                               utilpointer.BoolPtr(true),
				CPUCFSQuotaPeriod:                         &v1.Duration{Duration: 100 * time.Millisecond},
				NodeStatusMaxImages:                       utilpointer.Int32Ptr(50),
				MaxOpenFiles:                              1000000,
				ContentType:                               "application/vnd.kubernetes.protobuf",
				KubeAPIQPS:                                utilpointer.Int32Ptr(5),
				KubeAPIBurst:                              10,
				SerializeImagePulls:                       utilpointer.BoolPtr(true),
				EvictionHard:                              DefaultEvictionHard,
				EvictionPressureTransitionPeriod:          v1.Duration{Duration: 5 * time.Minute},
				EnableControllerAttachDetach:              utilpointer.BoolPtr(true),
				MakeIPTablesUtilChains:                    utilpointer.BoolPtr(true),
				IPTablesMasqueradeBit:                     utilpointer.Int32Ptr(DefaultIPTablesMasqueradeBit),
				IPTablesDropBit:                           utilpointer.Int32Ptr(DefaultIPTablesDropBit),
				FailSwapOn:                                utilpointer.BoolPtr(true),
				ContainerLogMaxSize:                       "10Mi",
				ContainerLogMaxFiles:                      utilpointer.Int32Ptr(5),
				ConfigMapAndSecretChangeDetectionStrategy: v1beta1.WatchChangeDetectionStrategy,
				EnforceNodeAllocatable:                    DefaultNodeAllocatableEnforcement,
				VolumePluginDir:                           DefaultVolumePluginDir,
				Logging: componentbaseconfigv1alpha1.LoggingConfiguration{
					Format: "text",
				},
				EnableSystemLogHandler:  utilpointer.BoolPtr(true),
				EnableProfilingHandler:  utilpointer.BoolPtr(true),
				EnableDebugFlagsHandler: utilpointer.BoolPtr(true),
				SeccompDefault:          utilpointer.BoolPtr(false),
				MemoryThrottlingFactor:  utilpointer.Float64Ptr(DefaultMemoryThrottlingFactor),
			},
		},
		{
			"NodeStatusUpdateFrequency is not zero",
			&v1beta1.KubeletConfiguration{
				NodeStatusUpdateFrequency:                 v1.Duration{Duration: 1 * time.Minute},
			},
			&v1beta1.KubeletConfiguration{
				EnableServer:       utilpointer.BoolPtr(true),
				SyncFrequency:      v1.Duration{Duration: 1 * time.Minute},
				FileCheckFrequency: v1.Duration{Duration: 20 * time.Second},
				HTTPCheckFrequency: v1.Duration{Duration: 20 * time.Second},
				Address:            "0.0.0.0",
				Port:               ports.KubeletPort,
				Authentication: v1beta1.KubeletAuthentication{
					Anonymous: v1beta1.KubeletAnonymousAuthentication{Enabled: utilpointer.BoolPtr(false)},
					Webhook: v1beta1.KubeletWebhookAuthentication{
						Enabled:  utilpointer.BoolPtr(true),
						CacheTTL: v1.Duration{Duration: 2 * time.Minute},
					},
				},
				Authorization: v1beta1.KubeletAuthorization{
					Mode: v1beta1.KubeletAuthorizationModeWebhook,
					Webhook: v1beta1.KubeletWebhookAuthorization{
						CacheAuthorizedTTL:   v1.Duration{Duration: 5 * time.Minute},
						CacheUnauthorizedTTL: v1.Duration{Duration: 30 * time.Second},
					},
				},
				RegistryPullQPS:                           utilpointer.Int32Ptr(5),
				RegistryBurst:                             10,
				EventRecordQPS:                            utilpointer.Int32Ptr(5),
				EventBurst:                                10,
				EnableDebuggingHandlers:                   utilpointer.BoolPtr(true),
				HealthzPort:                               utilpointer.Int32Ptr(10248),
				HealthzBindAddress:                        "127.0.0.1",
				OOMScoreAdj:                               utilpointer.Int32Ptr(int32(qos.KubeletOOMScoreAdj)),
				StreamingConnectionIdleTimeout:            v1.Duration{Duration: 4 * time.Hour},
				NodeStatusUpdateFrequency:                 v1.Duration{Duration: 1 * time.Minute},
				NodeStatusReportFrequency:                 v1.Duration{Duration: 1 * time.Minute},
				NodeLeaseDurationSeconds:                  40,
				ImageMinimumGCAge:                         v1.Duration{Duration: 2 * time.Minute},
				ImageGCHighThresholdPercent:               utilpointer.Int32Ptr(85),
				ImageGCLowThresholdPercent:                utilpointer.Int32Ptr(80),
				VolumeStatsAggPeriod:                      v1.Duration{Duration: time.Minute},
				CgroupsPerQOS:                             utilpointer.BoolPtr(true),
				CgroupDriver:                              "cgroupfs",
				CPUManagerPolicy:                          "none",
				CPUManagerReconcilePeriod:                 v1.Duration{Duration: 10 * time.Second},
				MemoryManagerPolicy:                       v1beta1.NoneMemoryManagerPolicy,
				TopologyManagerPolicy:                     v1beta1.NoneTopologyManagerPolicy,
				TopologyManagerScope:                      v1beta1.ContainerTopologyManagerScope,
				RuntimeRequestTimeout:                     v1.Duration{Duration: 2 * time.Minute},
				HairpinMode:                               v1beta1.PromiscuousBridge,
				MaxPods:                                   110,
				PodPidsLimit:                              utilpointer.Int64(-1),
				ResolverConfig:                            utilpointer.String(kubetypes.ResolvConfDefault),
				CPUCFSQuota:                               utilpointer.BoolPtr(true),
				CPUCFSQuotaPeriod:                         &v1.Duration{Duration: 100 * time.Millisecond},
				NodeStatusMaxImages:                       utilpointer.Int32Ptr(50),
				MaxOpenFiles:                              1000000,
				ContentType:                               "application/vnd.kubernetes.protobuf",
				KubeAPIQPS:                                utilpointer.Int32Ptr(5),
				KubeAPIBurst:                              10,
				SerializeImagePulls:                       utilpointer.BoolPtr(true),
				EvictionHard:                              DefaultEvictionHard,
				EvictionPressureTransitionPeriod:          v1.Duration{Duration: 5 * time.Minute},
				EnableControllerAttachDetach:              utilpointer.BoolPtr(true),
				MakeIPTablesUtilChains:                    utilpointer.BoolPtr(true),
				IPTablesMasqueradeBit:                     utilpointer.Int32Ptr(DefaultIPTablesMasqueradeBit),
				IPTablesDropBit:                           utilpointer.Int32Ptr(DefaultIPTablesDropBit),
				FailSwapOn:                                utilpointer.BoolPtr(true),
				ContainerLogMaxSize:                       "10Mi",
				ContainerLogMaxFiles:                      utilpointer.Int32Ptr(5),
				ConfigMapAndSecretChangeDetectionStrategy: v1beta1.WatchChangeDetectionStrategy,
				EnforceNodeAllocatable:                    DefaultNodeAllocatableEnforcement,
				VolumePluginDir:                           DefaultVolumePluginDir,
				Logging: componentbaseconfigv1alpha1.LoggingConfiguration{
					Format: "text",
				},
				EnableSystemLogHandler:  utilpointer.BoolPtr(true),
				EnableProfilingHandler:  utilpointer.BoolPtr(true),
				EnableDebugFlagsHandler: utilpointer.BoolPtr(true),
				SeccompDefault:          utilpointer.BoolPtr(false),
				MemoryThrottlingFactor:  utilpointer.Float64Ptr(DefaultMemoryThrottlingFactor),
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			SetDefaults_KubeletConfiguration(tc.config)
			if diff := cmp.Diff(tc.expected, tc.config); diff != "" {
				t.Errorf("Got unexpected defaults (-want, +got):\n%s", diff)
			}
		})
	}
}
