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

package kubeadm

import (
	"fmt"
	"os"
	"path"
	"runtime"
	"strings"
)

var GlobalEnvParams = SetEnvParams()

// TODO(phase1+) Move these paramaters to the API group
// we need some params for testing etc, let's keep these hidden for now
func SetEnvParams() *EnvParams {

	envParams := map[string]string{
		"kubernetes_dir":  "/etc/kubernetes",
		"host_pki_path":   "/etc/kubernetes/pki",
		"host_etcd_path":  "/var/lib/etcd",
		"hyperkube_image": "",
		"repo_prefix":     "gcr.io/google_containers",
		"discovery_image": fmt.Sprintf("gcr.io/google_containers/kube-discovery-%s:%s", runtime.GOARCH, "1.0"),
		"etcd_image":      "",
	}

	for k := range envParams {
		if v := strings.TrimSpace(os.Getenv(fmt.Sprintf("KUBE_%s", strings.ToUpper(k)))); v != "" {
			envParams[k] = v
		}
	}

	return &EnvParams{
		KubernetesDir:    path.Clean(envParams["kubernetes_dir"]),
		HostPKIPath:      path.Clean(envParams["host_pki_path"]),
		HostEtcdPath:     path.Clean(envParams["host_etcd_path"]),
		HyperkubeImage:   envParams["hyperkube_image"],
		RepositoryPrefix: envParams["repo_prefix"],
		DiscoveryImage:   envParams["discovery_image"],
		EtcdImage:        envParams["etcd_image"],
	}
}
