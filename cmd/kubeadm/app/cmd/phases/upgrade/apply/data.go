/*
Copyright 2019 The Kubernetes Authors.

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

package phases

import (
	"time"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/version"
	clientset "k8s.io/client-go/kubernetes"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/upgrade"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/apiclient"
)

// Data is the interface to use for upgrade apply phases.
// The "upgradeData" type from "cmd/upgrade/init.go" must satisfy this interface.
type ApplyData interface {
	RenewCerts() bool
	Cfg() *kubeadmapi.InitConfiguration
	DryRun() bool
	IgnorePreflightErrors() sets.String
	UserVersion() *version.Version
	KubeConfigPath() string
	ConfigPath() string
	Client() clientset.Interface
	Waiter() apiclient.Waiter
	KustomizeDir() string
	FeatureGates() string
	UpgradeETCD() bool
	ImagePullTimeout() time.Duration
	AllowExperimentalUpgrades() bool
	AllowRCUpgrades() bool
	Force() bool
	VersionGetter() upgrade.VersionGetter
}
