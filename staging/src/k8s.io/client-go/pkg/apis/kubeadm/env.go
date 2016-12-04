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
	"runtime"
	"strings"
)

var GlobalEnvParams = SetEnvParams()

// TODO(phase2) use componentconfig
// we need some params for testing etc, let's keep these hidden for now
func SetEnvParams() *EnvParams {

	envParams := map[string]string{
		// TODO(phase1+): Mode prefix and host_pki_path to another place as constants, and use them everywhere
		// Right now they're used here and there, but not consequently
		"kubernetes_dir":     "/etc/kubernetes",
		"host_pki_path":      "/etc/kubernetes/pki",
		"host_etcd_path":     "/var/lib/etcd",
		"hyperkube_image":    "",
		"repo_prefix":        "gcr.io/google_containers",
		"discovery_image":    fmt.Sprintf("gcr.io/google_containers/kube-discovery-%s:%s", runtime.GOARCH, "1.0"),
		"etcd_image":         "",
		"component_loglevel": "--v=2",
	}

	for k := range envParams {
		if v := os.Getenv(fmt.Sprintf("KUBE_%s", strings.ToUpper(k))); v != "" {
			envParams[k] = v
		}
	}

	return &EnvParams{
		KubernetesDir:     envParams["kubernetes_dir"],
		HostPKIPath:       envParams["host_pki_path"],
		HostEtcdPath:      envParams["host_etcd_path"],
		HyperkubeImage:    envParams["hyperkube_image"],
		RepositoryPrefix:  envParams["repo_prefix"],
		DiscoveryImage:    envParams["discovery_image"],
		EtcdImage:         envParams["etcd_image"],
		ComponentLoglevel: envParams["component_loglevel"],
	}
}
