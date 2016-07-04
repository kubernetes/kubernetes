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
	"k8s.io/kubernetes/pkg/master/ports"
	"k8s.io/kubernetes/pkg/runtime"
)

func addDefaultingFuncs(scheme *runtime.Scheme) {
	scheme.AddDefaultingFuncs(
		SetDefaults_KubeProxyConfiguration,
		SetDefaults_KubeSchedulerConfiguration,
		SetDefaults_LeaderElectionConfiguration,
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
