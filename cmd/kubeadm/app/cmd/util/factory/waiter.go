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

package factory

import (
	"os"
	"time"

	"k8s.io/api/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/apiclient"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/dryrun"
)

// WaiterFactory provides a factory with the responsibility to create/and store the waiter used during kubeadm init
type WaiterFactory struct {
	waiterInstance apiclient.Waiter
}

// Waiter returns a waiter instance.
// The first time this method is called, the waiter instance is created; successive calls, will reuse the same instance
func (f *WaiterFactory) Waiter(cfg *kubeadmapi.MasterConfiguration, client clientset.Interface, dryRun bool) apiclient.Waiter {
	if f.waiterInstance == nil {
		f.waiterInstance = createWaiter(cfg, client, dryRun)
	}
	return f.waiterInstance
}

// createWaiter gets the right waiter implementation
func createWaiter(cfg *kubeadmapi.MasterConfiguration, client clientset.Interface, dryRun bool) apiclient.Waiter {
	if dryRun {
		return dryrun.NewWaiter()
	}

	timeout := 30 * time.Minute

	// No need for a large timeout if we don't expect downloads
	if cfg.ImagePullPolicy == v1.PullNever {
		timeout = 60 * time.Second
	}
	return apiclient.NewKubeWaiter(client, timeout, os.Stdout)
}
