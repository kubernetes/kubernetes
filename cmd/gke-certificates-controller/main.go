/*
Copyright 2017 The Kubernetes Authors.

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

// The GKE certificates controller is responsible for monitoring certificate
// signing requests and (potentially) auto-approving and signing them within
// GKE.
package main

import (
	"fmt"
	"os"

	"k8s.io/apiserver/pkg/util/flag"
	"k8s.io/kubernetes/cmd/gke-certificates-controller/app"
	"k8s.io/kubernetes/pkg/util/logs"
	"k8s.io/kubernetes/pkg/version/verflag"

	"github.com/spf13/pflag"
)

// TODO(pipejakob): Move this entire cmd directory into its own repo
func main() {
	s := app.NewGKECertificatesController()
	s.AddFlags(pflag.CommandLine)

	flag.InitFlags()
	logs.InitLogs()
	defer logs.FlushLogs()

	verflag.PrintAndExitIfRequested()

	if err := app.Run(s); err != nil {
		fmt.Fprintf(os.Stderr, "%v\n", err)
		os.Exit(1)
	}
}
