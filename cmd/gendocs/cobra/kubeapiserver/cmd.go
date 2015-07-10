/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package kubeapiserver

import (
	"github.com/spf13/cobra"

	"github.com/GoogleCloudPlatform/kubernetes/cmd/kube-apiserver/app"
)

func Cmd() *cobra.Command {
	cmd := &cobra.Command{
		Use: "kube-apiserver",
		Long: `The kubernetes API server validates and configures data
for the api objects which include pods, services, replicationcontrollers, and
others. The API Server services REST operations and provides the frontend to the
cluster's shared state through which all other components interact.
`,
	}
	(&app.APIServer{}).AddFlags(cmd.Flags())
	return cmd
}
