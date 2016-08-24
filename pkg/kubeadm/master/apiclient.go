/*
Copyright 2016 The Kubernetes Authors.

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

package kubemaster

import (
	"fmt"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/client/unversioned/clientcmd"
	clientcmdapi "k8s.io/kubernetes/pkg/client/unversioned/clientcmd/api"
	"k8s.io/kubernetes/pkg/util/wait"
)

func CreateClientAndWaitForAPI(adminConfig *clientcmdapi.Config) error {
	adminClientConfig, err := clientcmd.NewDefaultClientConfig(
		*adminConfig,
		&clientcmd.ConfigOverrides{},
	).ClientConfig()
	if err != nil {
		return err
	}

	client, err := unversioned.New(adminClientConfig)
	if err != nil {
		return err
	}

	wait.PollInfinite(500*time.Millisecond, func() (bool, error) {
		_, err := client.ComponentStatuses().List(api.ListOptions{})
		return err == nil, nil
	})

	cs, err := client.ComponentStatuses().List(api.ListOptions{})
	if err != nil {
		return err
	}

	fmt.Printf("ComponentStatuses: %#v", cs.Items)

	// TODO check if len(cs.Items) < 3 (or what is it supposed to be)
	// TODO check if all components are healthy, and wait if they aren't
	return nil
}
