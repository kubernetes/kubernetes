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
	"fmt"
	"strings"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	kruntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/pkg/api"
	kubeletapis "k8s.io/kubernetes/pkg/kubelet/apis"
	"k8s.io/kubernetes/pkg/kubelet/qos"
	"k8s.io/kubernetes/pkg/master/ports"
)

const (
	DefaultRootDir = "/var/lib/kubelet"

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

func SetDefaults_KubeProxyConfiguration(obj *KubeProxyConfiguration) {
	if len(obj.BindAddress) == 0 {
		obj.BindAddress = "0.0.0.0"
	}
	if obj.HealthzBindAddress == "" {
		obj.HealthzBindAddress = fmt.Sprintf("0.0.0.0:%v", ports.ProxyHealthzPort)
	} else if !strings.Contains(obj.HealthzBindAddress, ":") {
		obj.HealthzBindAddress += fmt.Sprintf(":%v", ports.ProxyHealthzPort)
	}
	if obj.MetricsBindAddress == "" {
		obj.MetricsBindAddress = fmt.Sprintf("127.0.0.1:%v", ports.ProxyStatusPort)
	} else if !strings.Contains(obj.MetricsBindAddress, ":") {
		obj.MetricsBindAddress += fmt.Sprintf(":%v", ports.ProxyStatusPort)
	}
	if obj.OOMScoreAdj == nil {
		temp := int32(qos.KubeProxyOOMScoreAdj)
		obj.OOMScoreAdj = &temp
	}
	if obj.ResourceContainer == "" {
		obj.ResourceContainer = "/kube-proxy"
	}
	if obj.IPTables.SyncPeriod.Duration == 0 {
		obj.IPTables.SyncPeriod = metav1.Duration{Duration: 30 * time.Second}
	}
	if obj.IPVS.SyncPeriod.Duration == 0 {
		obj.IPVS.SyncPeriod = metav1.Duration{Duration: 30 * time.Second}
	}
	zero := metav1.Duration{}
	if obj.UDPIdleTimeout == zero {
		obj.UDPIdleTimeout = metav1.Duration{Duration: 250 * time.Millisecond}
	}
	// If ConntrackMax is set, respect it.
	if obj.Conntrack.Max == 0 {
		// If ConntrackMax is *not* set, use per-core scaling.
		if obj.Conntrack.MaxPerCore == 0 {
			obj.Conntrack.MaxPerCore = 32 * 1024
		}
		if obj.Conntrack.Min == 0 {
			obj.Conntrack.Min = 128 * 1024
		}
	}
	if obj.IPTables.MasqueradeBit == nil {
		temp := int32(14)
		obj.IPTables.MasqueradeBit = &temp
	}
	if obj.Conntrack.TCPEstablishedTimeout == zero {
		obj.Conntrack.TCPEstablishedTimeout = metav1.Duration{Duration: 24 * time.Hour} // 1 day (1/5 default)
	}
	if obj.Conntrack.TCPCloseWaitTimeout == zero {
		// See https://github.com/kubernetes/kubernetes/issues/32551.
		//
		// CLOSE_WAIT conntrack state occurs when the the Linux kernel
		// sees a FIN from the remote server. Note: this is a half-close
		// condition that persists as long as the local side keeps the
		// socket open. The condition is rare as it is typical in most
		// protocols for both sides to issue a close; this typically
		// occurs when the local socket is lazily garbage collected.
		//
		// If the CLOSE_WAIT conntrack entry expires, then FINs from the
		// local socket will not be properly SNAT'd and will not reach the
		// remote server (if the connection was subject to SNAT). If the
		// remote timeouts for FIN_WAIT* states exceed the CLOSE_WAIT
		// timeout, then there will be an inconsistency in the state of
		// the connection and a new connection reusing the SNAT (src,
		// port) pair may be rejected by the remote side with RST. This
		// can cause new calls to connect(2) to return with ECONNREFUSED.
		//
		// We set CLOSE_WAIT to one hour by default to better match
		// typical server timeouts.
		obj.Conntrack.TCPCloseWaitTimeout = metav1.Duration{Duration: 1 * time.Hour}
	}
	if obj.ConfigSyncPeriod.Duration == 0 {
		obj.ConfigSyncPeriod.Duration = 15 * time.Minute
	}

	if len(obj.ClientConnection.ContentType) == 0 {
		obj.ClientConnection.ContentType = "application/vnd.kubernetes.protobuf"
	}
	if obj.ClientConnection.QPS == 0.0 {
		obj.ClientConnection.QPS = 5.0
	}
	if obj.ClientConnection.Burst == 0 {
		obj.ClientConnection.Burst = 10
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
		obj.FailureDomains = kubeletapis.DefaultFailureDomains
	}
	if obj.LockObjectNamespace == "" {
		obj.LockObjectNamespace = SchedulerDefaultLockObjectNamespace
	}
	if obj.LockObjectName == "" {
		obj.LockObjectName = SchedulerDefaultLockObjectName
	}
	if obj.PolicyConfigMapNamespace == "" {
		obj.PolicyConfigMapNamespace = api.NamespaceSystem
	}
}

func SetDefaults_LeaderElectionConfiguration(obj *LeaderElectionConfiguration) {
	zero := metav1.Duration{}
	if obj.LeaseDuration == zero {
		obj.LeaseDuration = metav1.Duration{Duration: 15 * time.Second}
	}
	if obj.RenewDeadline == zero {
		obj.RenewDeadline = metav1.Duration{Duration: 10 * time.Second}
	}
	if obj.RetryPeriod == zero {
		obj.RetryPeriod = metav1.Duration{Duration: 2 * time.Second}
	}
	if obj.ResourceLock == "" {
		// obj.ResourceLock = rl.EndpointsResourceLock
		obj.ResourceLock = "endpoints"
	}
}

func boolVar(b bool) *bool {
	return &b
}
