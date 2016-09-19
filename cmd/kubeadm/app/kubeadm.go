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
	"strings"

	"github.com/renstrom/dedent"
	"github.com/spf13/pflag"

	"k8s.io/kubernetes/cmd/kubeadm/app/cmd"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/util/logs"
)

var AlphaWarningOnExit = dedent.Dedent(`
	kubeadm: I am an alpha version, my authors welcome your feedback and bug reports
	kubeadm: please create issue an using https://github.com/kubernetes/kubernetes/issues/new
	kubeadm: and make sure to mention @kubernetes/sig-cluster-lifecycle. Thank you!
`)

// TODO(phase2) use componentconfig
// we need some params for testing etc, let's keep these hidden for now
func getEnvParams() map[string]string {

	envParams := map[string]string{
		// TODO(phase1+): Mode prefix and host_pki_path to another place as constants, and use them everywhere
		// Right now they're used here and there, but not consequently
		"kubernetes_dir":     "/etc/kubernetes",
		"host_pki_path":      "/etc/kubernetes/pki",
		"host_etcd_path":     "/var/lib/etcd",
		"hyperkube_image":    "",
		"discovery_image":    "dgoodwin/kubediscovery:latest", // TODO(phase1): fmt.Sprintf("gcr.io/google_containers/kube-discovery-%s:%s", runtime.GOARCH, "1.0"),
		"etcd_image":         "",
		"component_loglevel": "--v=4",
	}

	for k := range envParams {
		if v := os.Getenv(fmt.Sprintf("KUBE_%s", strings.ToUpper(k))); v != "" {
			envParams[k] = v
		}
	}

	return envParams
}

func Run() error {
	logs.InitLogs()
	defer logs.FlushLogs()

	// We do not want these flags to show up in --help
	pflag.CommandLine.MarkHidden("google-json-key")
	pflag.CommandLine.MarkHidden("log-flush-frequency")

	cmd := cmd.NewKubeadmCommand(cmdutil.NewFactory(nil), os.Stdin, os.Stdout, os.Stderr, getEnvParams())
	return cmd.Execute()
}
