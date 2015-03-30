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
	"fmt"
	"io"
	"strings"

	"github.com/GoogleCloudPlatform/lmktfy/pkg/lmktfyctl"
	"github.com/GoogleCloudPlatform/lmktfy/pkg/lmktfyctl/cmd/util"
	"github.com/golang/glog"
	"github.com/spf13/cobra"
)

const (
	proxy_example = `// Run a proxy to lmktfy apiserver on port 8011, serving static content from ./local/www/
$ lmktfyctl proxy --port=8011 --www=./local/www/

// Run a proxy to lmktfy apiserver, changing the api prefix to lmktfy-api
// This makes e.g. the pods api available at localhost:8011/lmktfy-api/v1beta1/pods/
$ lmktfyctl proxy --api-prefix=lmktfy-api`
)

func (f *Factory) NewCmdProxy(out io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:     "proxy [--port=PORT] [--www=static-dir] [--www-prefix=prefix] [--api-prefix=prefix]",
		Short:   "Run a proxy to the LMKTFY API server",
		Long:    `Run a proxy to the LMKTFY API server. `,
		Example: proxy_example,
		Run: func(cmd *cobra.Command, args []string) {
			err := RunProxy(f, out, cmd)
			util.CheckErr(err)
		},
	}
	cmd.Flags().StringP("www", "w", "", "Also serve static files from the given directory under the specified prefix.")
	cmd.Flags().StringP("www-prefix", "P", "/static/", "Prefix to serve static files under, if static file directory is specified.")
	cmd.Flags().StringP("api-prefix", "", "/api/", "Prefix to serve the proxied API under.")
	cmd.Flags().IntP("port", "p", 8001, "The port on which to run the proxy.")
	return cmd
}

func RunProxy(f *Factory, out io.Writer, cmd *cobra.Command) error {
	port := util.GetFlagInt(cmd, "port")
	fmt.Fprintf(out, "Starting to serve on localhost:%d", port)

	clientConfig, err := f.ClientConfig()
	if err != nil {
		return err
	}

	staticPrefix := util.GetFlagString(cmd, "www-prefix")
	if !strings.HasSuffix(staticPrefix, "/") {
		staticPrefix += "/"
	}

	apiProxyPrefix := util.GetFlagString(cmd, "api-prefix")
	if !strings.HasSuffix(apiProxyPrefix, "/") {
		apiProxyPrefix += "/"
	}
	server, err := lmktfyctl.NewProxyServer(util.GetFlagString(cmd, "www"), apiProxyPrefix, staticPrefix, clientConfig)
	if err != nil {
		return err
	}

	glog.Fatal(server.Serve(port))
	return nil
}
