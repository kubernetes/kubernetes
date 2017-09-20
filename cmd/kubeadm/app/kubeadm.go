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
	"net/http"
	"os"

	"github.com/spf13/pflag"

	netutil "k8s.io/apimachinery/pkg/util/net"
	_ "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/install"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd"
)

// Run creates and executes new kubeadm command
func Run() error {
	// We do not want these flags to show up in --help
	pflag.CommandLine.MarkHidden("google-json-key")
	pflag.CommandLine.MarkHidden("log-flush-frequency")

	// We want to use for HTTP DefaultTransport better implmentation
	// of ProxyFromEnvironment that supports CIDR notation in NO_PROXY.
	http.DefaultTransport.(*http.Transport).Proxy = netutil.NewProxierWithNoProxyCIDR(http.ProxyFromEnvironment)

	cmd := cmd.NewKubeadmCommand(os.Stdin, os.Stdout, os.Stderr)
	return cmd.Execute()
}
