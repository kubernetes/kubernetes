/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"fmt"
	"io"
	"strings"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl"
	cmdutil "github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl/cmd/util"
	"github.com/golang/glog"
	"github.com/spf13/cobra"
)

const (
	proxy_example = `// Run a proxy to kubernetes apiserver on port 8011, serving static content from ./local/www/
$ kubectl proxy --port=8011 --www=./local/www/

// Run a proxy to kubernetes apiserver, changing the api prefix to k8s-api
// This makes e.g. the pods api available at localhost:8011/k8s-api/v1beta1/pods/
$ kubectl proxy --api-prefix=k8s-api`
)

func NewCmdProxy(f *cmdutil.Factory, out io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:     "proxy [--port=PORT] [--www=static-dir] [--www-prefix=prefix] [--api-prefix=prefix]",
		Short:   "Run a proxy to the Kubernetes API server",
		Long:    `Run a proxy to the Kubernetes API server. `,
		Example: proxy_example,
		Run: func(cmd *cobra.Command, args []string) {
			err := RunProxy(f, out, cmd)
			cmdutil.CheckErr(err)
		},
	}
	cmd.Flags().StringP("www", "w", "", "Also serve static files from the given directory under the specified prefix.")
	cmd.Flags().StringP("www-prefix", "P", "/static/", "Prefix to serve static files under, if static file directory is specified.")
	cmd.Flags().StringP("api-prefix", "", "/api/", "Prefix to serve the proxied API under.")
	cmd.Flags().IntP("port", "p", 8001, "The port on which to run the proxy.")
	return cmd
}

func RunProxy(f *cmdutil.Factory, out io.Writer, cmd *cobra.Command) error {
	port := cmdutil.GetFlagInt(cmd, "port")
	fmt.Fprintf(out, "Starting to serve on localhost:%d", port)

	clientConfig, err := f.ClientConfig()
	if err != nil {
		return err
	}

	staticPrefix := cmdutil.GetFlagString(cmd, "www-prefix")
	if !strings.HasSuffix(staticPrefix, "/") {
		staticPrefix += "/"
	}

	apiProxyPrefix := cmdutil.GetFlagString(cmd, "api-prefix")
	if !strings.HasSuffix(apiProxyPrefix, "/") {
		apiProxyPrefix += "/"
	}
	server, err := kubectl.NewProxyServer(cmdutil.GetFlagString(cmd, "www"), apiProxyPrefix, staticPrefix, clientConfig)
	if err != nil {
		return err
	}

	glog.Fatal(server.Serve(port))
	return nil
}
