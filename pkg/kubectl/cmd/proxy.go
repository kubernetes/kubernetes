/*
Copyright 2014 Google Inc. All rights reserved.

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

package cmd

import (
	"io"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl"
	"github.com/golang/glog"
	"github.com/spf13/cobra"
)

func NewCmdProxy(out io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "proxy",
		Short: "Run a proxy to the Kubernetes API server",
		Long:  `Run a proxy to the Kubernetes API server.`,
		Run: func(cmd *cobra.Command, args []string) {
			port := getFlagInt(cmd, "port")
			glog.Infof("Starting to serve on localhost:%d", port)
			server, err := kubectl.NewProxyServer(getFlagString(cmd, "www"), getKubeConfig(cmd), port)
			checkErr(err)
			glog.Fatal(server.Serve())
		},
	}
	cmd.Flags().StringP("www", "w", "", "Also serve static files from the given directory under the prefix /static")
	cmd.Flags().IntP("port", "p", 8001, "The port on which to run the proxy")
	return cmd
}
