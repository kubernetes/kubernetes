/*
Copyright 2018 The Kubernetes Authors.

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
	apimachineryconfigv1alpha1 "k8s.io/apimachinery/pkg/apis/config/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	apiserverconfigv1alpha1 "k8s.io/apiserver/pkg/apis/config/v1alpha1"
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// GenericControllerManagerConfiguration holds configuration for GenericControllerManagerConfiguration
// related features both in cloud controller manager and kube-controller manager.
type GenericControllerManagerConfiguration struct {
	metav1.TypeMeta `json:",inline"`

	// Port is the port that the controller-manager's http service runs on.
	Port int32
	// Address is the IP address to serve on (set to 0.0.0.0 for all interfaces).
	Address string
	// MinResyncPeriod is the resync period in reflectors; will be random between
	// minResyncPeriod and 2*minResyncPeriod.
	MinResyncPeriod metav1.Duration
	// ClientConnection specifies the kubeconfig file and client connection
	// settings for the proxy server to use when communicating with the apiserver.
	ClientConnection apimachineryconfigv1alpha1.ClientConnectionConfiguration
	// ControllerStartInterval indicate how long to wait between starting controller managers.
	ControllerStartInterval metav1.Duration
	// LeaderElection defines the configuration of leader election client.
	LeaderElection apiserverconfigv1alpha1.LeaderElectionConfiguration
	// Controllers is the list of controllers to enable or disable
	// '*' means "all enabled by default controllers"
	// 'foo' means "enable 'foo'"
	// '-foo' means "disable 'foo'"
	// first item for a particular name wins
	Controllers []string
	// DebuggingConfiguration holds configuration for Debugging related features.
	Debugging apiserverconfigv1alpha1.DebuggingConfiguration
}
