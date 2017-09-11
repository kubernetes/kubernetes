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

package v1alpha1

import (
	"path/filepath"
	"runtime"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	kruntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/pkg/kubelet/qos"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/kubernetes/pkg/master/ports"
	utilpointer "k8s.io/kubernetes/pkg/util/pointer"
)

const (
	DefaultRootDir = "/var/lib/kubelet"

	// DEPRECATED: auto detecting cloud providers goes against the initiative
	// for out-of-tree cloud providers as we'll now depend on cAdvisor integrations
	// with cloud providers instead of in the core repo.
	// More details here: https://github.com/kubernetes/kubernetes/issues/50986
	AutoDetectCloudProvider = "auto-detect"

	defaultIPTablesMasqueradeBit = 14
	defaultIPTablesDropBit       = 15
)

var (
	zeroDuration = metav1.Duration{}
	// Refer to [Node Allocatable](https://git.k8s.io/community/contributors/design-proposals/node-allocatable.md) doc for more information.
	defaultNodeAllocatableEnforcement = []string{"pods"}
)

func addDefaultingFuncs(scheme *kruntime.Scheme) error {
	return RegisterDefaults(scheme)
}

func SetDefaults_KubeletConfiguration(obj *KubeletConfiguration) {
	// pointer because the zeroDuration is valid - if you want to skip the trial period
	if obj.ConfigTrialDuration == nil {
		obj.ConfigTrialDuration = &metav1.Duration{Duration: 10 * time.Minute}
	}
	if obj.Authentication.Anonymous.Enabled == nil {
		obj.Authentication.Anonymous.Enabled = boolVar(true)
	}
	if obj.Authentication.Webhook.Enabled == nil {
		obj.Authentication.Webhook.Enabled = boolVar(false)
	}
	if obj.Authentication.Webhook.CacheTTL == zeroDuration {
		obj.Authentication.Webhook.CacheTTL = metav1.Duration{Duration: 2 * time.Minute}
	}
	if obj.Authorization.Mode == "" {
		obj.Authorization.Mode = KubeletAuthorizationModeAlwaysAllow
	}
	if obj.Authorization.Webhook.CacheAuthorizedTTL == zeroDuration {
		obj.Authorization.Webhook.CacheAuthorizedTTL = metav1.Duration{Duration: 5 * time.Minute}
	}
	if obj.Authorization.Webhook.CacheUnauthorizedTTL == zeroDuration {
		obj.Authorization.Webhook.CacheUnauthorizedTTL = metav1.Duration{Duration: 30 * time.Second}
	}

	if obj.Address == "" {
		obj.Address = "0.0.0.0"
	}
	if obj.CAdvisorPort == nil {
		obj.CAdvisorPort = utilpointer.Int32Ptr(4194)
	}
	if obj.VolumeStatsAggPeriod == zeroDuration {
		obj.VolumeStatsAggPeriod = metav1.Duration{Duration: time.Minute}
	}
	if obj.ContainerRuntime == "" {
		obj.ContainerRuntime = kubetypes.DockerContainerRuntime
	}
	if obj.RuntimeRequestTimeout == zeroDuration {
		obj.RuntimeRequestTimeout = metav1.Duration{Duration: 2 * time.Minute}
	}
	if obj.CPUCFSQuota == nil {
		obj.CPUCFSQuota = boolVar(true)
	}
	if obj.EventBurst == 0 {
		obj.EventBurst = 10
	}
	if obj.EventRecordQPS == nil {
		temp := int32(5)
		obj.EventRecordQPS = &temp
	}
	if obj.EnableControllerAttachDetach == nil {
		obj.EnableControllerAttachDetach = boolVar(true)
	}
	if obj.EnableDebuggingHandlers == nil {
		obj.EnableDebuggingHandlers = boolVar(true)
	}
	if obj.EnableServer == nil {
		obj.EnableServer = boolVar(true)
	}
	if obj.FileCheckFrequency == zeroDuration {
		obj.FileCheckFrequency = metav1.Duration{Duration: 20 * time.Second}
	}
	if obj.HealthzBindAddress == "" {
		obj.HealthzBindAddress = "127.0.0.1"
	}
	if obj.HealthzPort == nil {
		obj.HealthzPort = utilpointer.Int32Ptr(10248)
	}
	if obj.HostNetworkSources == nil {
		obj.HostNetworkSources = []string{kubetypes.AllSource}
	}
	if obj.HostPIDSources == nil {
		obj.HostPIDSources = []string{kubetypes.AllSource}
	}
	if obj.HostIPCSources == nil {
		obj.HostIPCSources = []string{kubetypes.AllSource}
	}
	if obj.HTTPCheckFrequency == zeroDuration {
		obj.HTTPCheckFrequency = metav1.Duration{Duration: 20 * time.Second}
	}
	if obj.ImageMinimumGCAge == zeroDuration {
		obj.ImageMinimumGCAge = metav1.Duration{Duration: 2 * time.Minute}
	}
	if obj.ImageGCHighThresholdPercent == nil {
		// default is below docker's default dm.min_free_space of 90%
		temp := int32(85)
		obj.ImageGCHighThresholdPercent = &temp
	}
	if obj.ImageGCLowThresholdPercent == nil {
		temp := int32(80)
		obj.ImageGCLowThresholdPercent = &temp
	}
	if obj.MasterServiceNamespace == "" {
		obj.MasterServiceNamespace = metav1.NamespaceDefault
	}
	if obj.MaxContainerCount == nil {
		temp := int32(-1)
		obj.MaxContainerCount = &temp
	}
	if obj.MaxPerPodContainerCount == 0 {
		obj.MaxPerPodContainerCount = 1
	}
	if obj.MaxOpenFiles == 0 {
		obj.MaxOpenFiles = 1000000
	}
	if obj.MaxPods == 0 {
		obj.MaxPods = 110
	}
	if obj.MinimumGCAge == zeroDuration {
		obj.MinimumGCAge = metav1.Duration{Duration: 0}
	}
	if obj.NonMasqueradeCIDR == "" {
		obj.NonMasqueradeCIDR = "10.0.0.0/8"
	}
	if obj.VolumePluginDir == "" {
		obj.VolumePluginDir = "/usr/libexec/kubernetes/kubelet-plugins/volume/exec/"
	}
	if obj.NodeStatusUpdateFrequency == zeroDuration {
		obj.NodeStatusUpdateFrequency = metav1.Duration{Duration: 10 * time.Second}
	}
	if obj.CPUManagerPolicy == "" {
		obj.CPUManagerPolicy = "none"
	}
	if obj.CPUManagerReconcilePeriod == zeroDuration {
		obj.CPUManagerReconcilePeriod = obj.NodeStatusUpdateFrequency
	}
	if obj.OOMScoreAdj == nil {
		temp := int32(qos.KubeletOOMScoreAdj)
		obj.OOMScoreAdj = &temp
	}
	if obj.Port == 0 {
		obj.Port = ports.KubeletPort
	}
	if obj.ReadOnlyPort == nil {
		obj.ReadOnlyPort = utilpointer.Int32Ptr(ports.KubeletReadOnlyPort)
	}
	if obj.RegisterNode == nil {
		obj.RegisterNode = boolVar(true)
	}
	if obj.RegisterSchedulable == nil {
		obj.RegisterSchedulable = boolVar(true)
	}
	if obj.RegistryBurst == 0 {
		obj.RegistryBurst = 10
	}
	if obj.RegistryPullQPS == nil {
		temp := int32(5)
		obj.RegistryPullQPS = &temp
	}
	if obj.ResolverConfig == "" {
		obj.ResolverConfig = kubetypes.ResolvConfDefault
	}
	if obj.SerializeImagePulls == nil {
		obj.SerializeImagePulls = boolVar(true)
	}
	if obj.SeccompProfileRoot == "" {
		obj.SeccompProfileRoot = filepath.Join(DefaultRootDir, "seccomp")
	}
	if obj.StreamingConnectionIdleTimeout == zeroDuration {
		obj.StreamingConnectionIdleTimeout = metav1.Duration{Duration: 4 * time.Hour}
	}
	if obj.SyncFrequency == zeroDuration {
		obj.SyncFrequency = metav1.Duration{Duration: 1 * time.Minute}
	}
	if obj.ContentType == "" {
		obj.ContentType = "application/vnd.kubernetes.protobuf"
	}
	if obj.KubeAPIQPS == nil {
		temp := int32(5)
		obj.KubeAPIQPS = &temp
	}
	if obj.KubeAPIBurst == 0 {
		obj.KubeAPIBurst = 10
	}
	if string(obj.HairpinMode) == "" {
		obj.HairpinMode = PromiscuousBridge
	}
	if obj.EvictionHard == nil {
		temp := "memory.available<100Mi,nodefs.available<10%,nodefs.inodesFree<5%"
		obj.EvictionHard = &temp
	}
	if obj.EvictionPressureTransitionPeriod == zeroDuration {
		obj.EvictionPressureTransitionPeriod = metav1.Duration{Duration: 5 * time.Minute}
	}
	if obj.ExperimentalKernelMemcgNotification == nil {
		obj.ExperimentalKernelMemcgNotification = boolVar(false)
	}
	if obj.SystemReserved == nil {
		obj.SystemReserved = make(map[string]string)
	}
	if obj.KubeReserved == nil {
		obj.KubeReserved = make(map[string]string)
	}
	if obj.ExperimentalQOSReserved == nil {
		obj.ExperimentalQOSReserved = make(map[string]string)
	}
	if obj.MakeIPTablesUtilChains == nil {
		obj.MakeIPTablesUtilChains = boolVar(true)
	}
	if obj.IPTablesMasqueradeBit == nil {
		temp := int32(defaultIPTablesMasqueradeBit)
		obj.IPTablesMasqueradeBit = &temp
	}
	if obj.IPTablesDropBit == nil {
		temp := int32(defaultIPTablesDropBit)
		obj.IPTablesDropBit = &temp
	}
	if obj.CgroupsPerQOS == nil {
		temp := true
		obj.CgroupsPerQOS = &temp
	}
	if obj.CgroupDriver == "" {
		obj.CgroupDriver = "cgroupfs"
	}
	if obj.EnforceNodeAllocatable == nil {
		obj.EnforceNodeAllocatable = defaultNodeAllocatableEnforcement
	}
	if obj.RemoteRuntimeEndpoint == "" {
		if runtime.GOOS == "linux" {
			obj.RemoteRuntimeEndpoint = "unix:///var/run/dockershim.sock"
		} else if runtime.GOOS == "windows" {
			obj.RemoteRuntimeEndpoint = "tcp://localhost:3735"
		}
	}
}

func boolVar(b bool) *bool {
	return &b
}

var (
	defaultCfg = KubeletConfiguration{}
)
