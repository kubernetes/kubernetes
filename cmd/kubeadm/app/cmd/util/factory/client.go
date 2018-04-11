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
	"fmt"
	"os"

	clientset "k8s.io/client-go/kubernetes"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/apiclient"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/kubeconfig"
)

// ClientFactory provides a factory with the responsibility to create/and store the kubernetes client used during kubeadm init
type ClientFactory struct {
	clientInstance clientset.Interface
}

// Client returns a kubernetes client instance.
// The first time this method is called, the kubernetes client instance is created; successive calls, will reuse the same instance
func (f *ClientFactory) Client(cfg *kubeadmapi.MasterConfiguration, dryRun bool) (clientset.Interface, error) {
	if f.clientInstance == nil {
		client, err := createClient(cfg, dryRun)
		if err != nil {
			return nil, fmt.Errorf("error creating client: %v", err)
		}
		f.clientInstance = client
	}
	return f.clientInstance, nil
}

// createClient creates a clientset.Interface object
func createClient(cfg *kubeadmapi.MasterConfiguration, dryRun bool) (clientset.Interface, error) {
	if dryRun {
		// If we're dry-running; we should create a faked client that answers some GETs in order to be able to do the full init flow and just logs the rest of requests
		dryRunGetter := apiclient.NewInitDryRunGetter(cfg.NodeName, cfg.Networking.ServiceSubnet)
		return apiclient.NewDryRunClient(dryRunGetter, os.Stdout), nil
	}

	// If we're acting for real, we should create a connection to the API server and wait for it to come up
	return kubeconfig.ClientSetFromFile(constants.GetAdminKubeConfigPath())
}
