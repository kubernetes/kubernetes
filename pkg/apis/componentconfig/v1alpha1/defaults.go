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
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/kubelet/qos"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/kubernetes/pkg/master/ports"
	"k8s.io/kubernetes/pkg/runtime"
)

var zeroDuration = unversioned.Duration{}

func addDefaultingFuncs(scheme *runtime.Scheme) {
	scheme.AddDefaultingFuncs(
		SetDefaults_KubeProxyConfiguration,
		SetDefaults_KubeSchedulerConfiguration,
		SetDefaults_LeaderElectionConfiguration,
		SetDefaults_KubeletConfiguration,
	)
}

func SetDefaults_KubeProxyConfiguration(obj *KubeProxyConfiguration) {
	if obj.BindAddress == "" {
		obj.BindAddress = "0.0.0.0"
	}
	if obj.HealthzPort == 0 {
		obj.HealthzPort = 10249
	}
	if obj.HealthzBindAddress == "" {
		obj.HealthzBindAddress = "127.0.0.1"
	}
	if obj.OOMScoreAdj == nil {
		temp := int32(qos.KubeProxyOOMScoreAdj)
		obj.OOMScoreAdj = &temp
	}
	if obj.ResourceContainer == "" {
		obj.ResourceContainer = "/kube-proxy"
	}
	if obj.IPTablesSyncPeriod.Duration == 0 {
		obj.IPTablesSyncPeriod = unversioned.Duration{Duration: 30 * time.Second}
	}
	zero := unversioned.Duration{}
	if obj.UDPIdleTimeout == zero {
		obj.UDPIdleTimeout = unversioned.Duration{Duration: 250 * time.Millisecond}
	}
	if obj.ConntrackMax == 0 {
		obj.ConntrackMax = 256 * 1024 // 4x default (64k)
	}
	if obj.IPTablesMasqueradeBit == nil {
		temp := int32(14)
		obj.IPTablesMasqueradeBit = &temp
	}
	if obj.ConntrackTCPEstablishedTimeout == zero {
		obj.ConntrackTCPEstablishedTimeout = unversioned.Duration{Duration: 24 * time.Hour} // 1 day (1/5 default)
	}
}

func SetDefaults_KubeSchedulerConfiguration(obj *KubeSchedulerConfiguration) {
	if obj.Port == 0 {
		obj.Port = ports.SchedulerPort
	}
	if obj.Address == "" {
		obj.Address = "0.0.0.0"
	}
	if obj.AlgorithmProvider == "" {
		obj.AlgorithmProvider = "DefaultProvider"
	}
	if obj.ContentType == "" {
		obj.ContentType = "application/vnd.kubernetes.protobuf"
	}
	if obj.KubeAPIQPS == 0 {
		obj.KubeAPIQPS = 50.0
	}
	if obj.KubeAPIBurst == 0 {
		obj.KubeAPIBurst = 100
	}
	if obj.SchedulerName == "" {
		obj.SchedulerName = api.DefaultSchedulerName
	}
	if obj.HardPodAffinitySymmetricWeight == 0 {
		obj.HardPodAffinitySymmetricWeight = api.DefaultHardPodAffinitySymmetricWeight
	}
	if obj.FailureDomains == "" {
		obj.FailureDomains = api.DefaultFailureDomains
	}
}

func SetDefaults_LeaderElectionConfiguration(obj *LeaderElectionConfiguration) {
	zero := unversioned.Duration{}
	if obj.LeaseDuration == zero {
		obj.LeaseDuration = unversioned.Duration{Duration: 15 * time.Second}
	}
	if obj.RenewDeadline == zero {
		obj.RenewDeadline = unversioned.Duration{Duration: 10 * time.Second}
	}
	if obj.RetryPeriod == zero {
		obj.RetryPeriod = unversioned.Duration{Duration: 2 * time.Second}
	}
}

func SetDefaults_KubeletConfiguration(obj *KubeletConfiguration) {
	if obj.Address == "" {
		obj.Address = "0.0.0.0"
	}
	if obj.CloudProvider == "" {
		obj.CloudProvider = "auto-detect"
	}
	if obj.CAdvisorPort == 0 {
		obj.CAdvisorPort = 4194
	}
	if obj.CertDirectory == "" {
		obj.CertDirectory = "/var/run/kubernetes"
	}
	if obj.ConfigureCBR0 == nil {
		obj.ConfigureCBR0 = boolVar(false)
	}
	if obj.ContainerRuntime == "" {
		obj.ContainerRuntime = "docker"
	}
	if obj.CPUCFSQuota == nil {
		obj.CPUCFSQuota = boolVar(true)
	}
	if obj.DockerExecHandlerName == "" {
		obj.DockerExecHandlerName = "native"
	}
	if obj.DockerEndpoint == "" {
		obj.DockerEndpoint = "unix:///var/run/docker.sock"
	}
	if obj.EventBurst == 0 {
		obj.EventBurst = 10
	}
	if obj.EventRecordQPS == 0 {
		obj.EventRecordQPS = 5.0
	}
	if obj.EnableDebuggingHandlers == nil {
		obj.EnableDebuggingHandlers = boolVar(true)
	}
	if obj.EnableServer == nil {
		obj.EnableServer = boolVar(true)
	}
	if obj.FileCheckFrequency == zeroDuration {
		obj.FileCheckFrequency = unversioned.Duration{20 * time.Second}
	}
	if obj.HealthzBindAddress == "" {
		obj.HealthzBindAddress = "127.0.0.1"
	}
	if obj.HealthzPort == 0 {
		obj.HealthzPort = 10248
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
		obj.HTTPCheckFrequency = unversioned.Duration{20 * time.Second}
	}
	if obj.ImageGCHighThresholdPercent == 0 {
		obj.ImageGCHighThresholdPercent = 90
	}
	if obj.ImageGCLowThresholdPercent == 0 {
		obj.ImageGCLowThresholdPercent = 80
	}
	if obj.LowDiskSpaceThresholdMB == 0 {
		obj.LowDiskSpaceThresholdMB = 256
	}
	if obj.MasterServiceNamespace == "" {
		obj.MasterServiceNamespace = api.NamespaceDefault
	}
	if obj.MaxContainerCount == nil {
		temp := int64(100)
		obj.MaxContainerCount = &temp
	}
	if obj.MaxPerPodContainerCount == 0 {
		obj.MaxPerPodContainerCount = 2
	}
	if obj.MaxOpenFiles == 0 {
		obj.MaxOpenFiles = 1000000
	}
	if obj.MaxPods == 0 {
		obj.MaxPods = 40
	}
	if obj.MinimumGCAge == zeroDuration {
		obj.MinimumGCAge = unversioned.Duration{1 * time.Minute}
	}
	if obj.VolumeStatsAggPeriod == zeroDuration {
		obj.VolumeStatsAggPeriod = unversioned.Duration{time.Minute}
	}
	if obj.NetworkPluginDir == "" {
		obj.NetworkPluginDir = "/usr/libexec/kubernetes/kubelet-plugins/net/exec/"
	}
	if obj.NonMasqueradeCIDR == "" {
		obj.NonMasqueradeCIDR = "10.0.0.0/8"
	}
	if obj.VolumePluginDir == "" {
		obj.VolumePluginDir = "/usr/libexec/kubernetes/kubelet-plugins/volume/exec/"
	}
	if obj.NodeStatusUpdateFrequency == zeroDuration {
		obj.NodeStatusUpdateFrequency = unversioned.Duration{10 * time.Second}
	}
	if obj.OOMScoreAdj == 0 {
		obj.OOMScoreAdj = int32(qos.KubeletOOMScoreAdj)
	}
	if obj.PodInfraContainerImage == "" {
		//obj.PodInfraContainerImage = kubetypes.PodInfraContainerImage
	}
	if obj.Port == 0 {
		obj.Port = ports.KubeletPort
	}
	if obj.ReadOnlyPort == 0 {
		obj.ReadOnlyPort = ports.KubeletReadOnlyPort
	}
	if obj.RegisterNode == nil {
		obj.RegisterNode = boolVar(true)
	}
	if obj.ResolverConfig == "" {
		obj.ResolverConfig = "/etc/resolv.conf"
	}
	if obj.RegisterSchedulable == nil {
		obj.RegisterSchedulable = boolVar(true)
	}
	if obj.RegistryBurst == 0 {
		obj.RegistryBurst = 10
	}
	if obj.RegistryPullQPS == 0 {
		obj.RegistryPullQPS = 5.0
	}
	if obj.RootDirectory == "" {
		obj.RootDirectory = "/var/lib/kubelet"
	}
	if obj.SerializeImagePulls == nil {
		obj.SerializeImagePulls = boolVar(true)
	}
	if obj.StreamingConnectionIdleTimeout == zeroDuration {
		obj.StreamingConnectionIdleTimeout = unversioned.Duration{4 * time.Hour}
	}
	if obj.SyncFrequency == zeroDuration {
		obj.SyncFrequency = unversioned.Duration{1 * time.Minute}
	}
	if obj.ReconcileCIDR == nil {
		obj.ReconcileCIDR = boolVar(true)
	}
	if obj.KubeAPIQPS == 0 {
		obj.KubeAPIQPS = 5.0
	}
	if obj.KubeAPIBurst == 0 {
		obj.KubeAPIBurst = 10
	}
	if obj.ExperimentalFlannelOverlay == nil {
		obj.ExperimentalFlannelOverlay = boolVar(false)
	}
	if obj.OutOfDiskTransitionFrequency == zeroDuration {
		obj.OutOfDiskTransitionFrequency = unversioned.Duration{5 * time.Minute}
	}
	if string(obj.HairpinMode) == "" {
		obj.HairpinMode = PromiscuousBridge
	}
}

func boolVar(b bool) *bool {
	return &b
}

var (
	defaultCfg = KubeletConfiguration{}
)
