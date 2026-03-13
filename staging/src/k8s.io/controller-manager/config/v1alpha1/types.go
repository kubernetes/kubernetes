/*
Copyright 2020 The Kubernetes Authors.

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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	componentbaseconfigv1alpha1 "k8s.io/component-base/config/v1alpha1"
)

// GenericControllerManagerConfiguration holds configuration for a generic controller-manager.
type GenericControllerManagerConfiguration struct {
	// port is the port that the controller-manager's http service runs on.
	Port int32
	// address is the IP address to serve on (set to 0.0.0.0 for all interfaces).
	Address string
	// minResyncPeriod is the resync period in reflectors; will be random between
	// minResyncPeriod and 2*minResyncPeriod.
	MinResyncPeriod metav1.Duration
	// ClientConnection specifies the kubeconfig file and client connection
	// settings for the proxy server to use when communicating with the apiserver.
	ClientConnection componentbaseconfigv1alpha1.ClientConnectionConfiguration
	// How long to wait between starting controller managers
	ControllerStartInterval metav1.Duration
	// leaderElection defines the configuration of leader election client.
	LeaderElection componentbaseconfigv1alpha1.LeaderElectionConfiguration
	// Controllers is the list of controllers to enable or disable
	// '*' means "all enabled by default controllers"
	// 'foo' means "enable 'foo'"
	// '-foo' means "disable 'foo'"
	// first item for a particular name wins
	Controllers []string
	// DebuggingConfiguration holds configuration for Debugging related features.
	Debugging componentbaseconfigv1alpha1.DebuggingConfiguration
	// LeaderMigrationEnabled indicates whether Leader Migration should be enabled for the controller manager.
	LeaderMigrationEnabled bool
	// LeaderMigration holds the configuration for Leader Migration.
	LeaderMigration LeaderMigrationConfiguration
}

// LeaderMigrationConfiguration provides versioned configuration for all migrating leader locks.
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
type LeaderMigrationConfiguration struct {
	metav1.TypeMeta `json:",inline"`

	// LeaderName is the name of the leader election resource that protects the migration
	// E.g. 1-20-KCM-to-1-21-CCM
	LeaderName string `json:"leaderName"`

	// ResourceLock indicates the resource object type that will be used to lock
	// Should be "leases" or "endpoints"
	ResourceLock string `json:"resourceLock"`

	// ControllerLeaders contains a list of migrating leader lock configurations
	ControllerLeaders []ControllerLeaderConfiguration `json:"controllerLeaders"`
}

// ControllerLeaderConfiguration provides the configuration for a migrating leader lock.
type ControllerLeaderConfiguration struct {
	// Name is the name of the controller being migrated
	// E.g. service-controller, route-controller, cloud-node-controller, etc
	Name string `json:"name"`

	// Component is the name of the component in which the controller should be running.
	// E.g. kube-controller-manager, cloud-controller-manager, etc
	// Or '*' meaning the controller can be run under any component that participates in the migration
	Component string `json:"component"`
}
