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

package app

import (
	"fmt"
	"os"
	"path"
	"strings"

	"k8s.io/kubernetes/pkg/kubeadm/cmd"
	"k8s.io/kubernetes/pkg/util/logs"
)

// TODO add this to hyperkube?
// we need some params for testing etc, let's keep these hidden for now
func getEnvParams() map[string]string {
	globalPrefix := os.Getenv("KUBE_PREFIX_ALL")
	if globalPrefix == "" {
		globalPrefix = "/etc/kubernetes"
	}

	envParams := map[string]string{
		"prefix":          globalPrefix,
		"host_pki_path":   path.Join(globalPrefix, "pki"),
		"hyperkube_image": "gcr.io/google_containers/hyperkube:v1.4.0-alpha.3",
		"discovery_image": "dgoodwin/kubediscovery:latest",
	}

	for k, _ := range envParams {
		if v := os.Getenv(fmt.Sprintf("KUBE_%s", strings.ToUpper(k))); v != "" {
			envParams[k] = v
		}
	}

	return envParams
}

func Run() error {
	logs.InitLogs()
	defer logs.FlushLogs()

	cmd := cmd.NewKubeadmCommand(os.Stdin, os.Stdout, os.Stderr, getEnvParams())
	return cmd.Execute()
}
