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
// This makes e.g. the pods api available at localhost:8011/k8s-api/v1/pods/
$ kubectl proxy --api-prefix=/k8s-api`
)

func NewCmdProxy(f *cmdutil.Factory, out io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "proxy [--port=PORT] [--www=static-dir] [--www-prefix=prefix] [--api-prefix=prefix]",
		Short: "Run a proxy to the Kubernetes API server",
		Long: `To proxy all of the kubernetes api and nothing else, use:

kubectl proxy --api-prefix=/

To proxy only part of the kubernetes api and also some static files:

kubectl proxy --www=/my/files --www-prefix=/static/ --api-prefix=/api/

The above lets you 'curl localhost:8001/api/v1/pods'.

To proxy the entire kubernetes api at a different root, use:

kubectl proxy --api-prefix=/custom/

The above lets you 'curl localhost:8001/custom/api/v1/pods'
`,
		Example: proxy_example,
		Run: func(cmd *cobra.Command, args []string) {
			err := RunProxy(f, out, cmd)
			cmdutil.CheckErr(err)
		},
	}
	cmd.Flags().StringP("www", "w", "", "Also serve static files from the given directory under the specified prefix.")
	cmd.Flags().StringP("www-prefix", "P", "/static/", "Prefix to serve static files under, if static file directory is specified.")
	cmd.Flags().StringP("api-prefix", "", "/api/", "Prefix to serve the proxied API under.")
	cmd.Flags().String("accept-paths", kubectl.DefaultPathAcceptRE, "Regular expression for paths that the proxy should accept.")
	cmd.Flags().String("reject-paths", kubectl.DefaultPathRejectRE, "Regular expression for paths that the proxy should reject.")
	cmd.Flags().String("accept-hosts", kubectl.DefaultHostAcceptRE, "Regular expression for hosts that the proxy should accept.")
	cmd.Flags().String("reject-methods", kubectl.DefaultMethodRejectRE, "Regular expression for HTTP methods that the proxy should reject.")
	cmd.Flags().IntP("port", "p", 8001, "The port on which to run the proxy.")
	cmd.Flags().Bool("disable-filter", false, "If true, disable request filtering in the proxy. This is dangerous, and can leave you vulnerable to XSRF attacks.  Use with caution.")
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
	filter := &kubectl.FilterServer{
		AcceptPaths: kubectl.MakeRegexpArrayOrDie(cmdutil.GetFlagString(cmd, "accept-paths")),
		RejectPaths: kubectl.MakeRegexpArrayOrDie(cmdutil.GetFlagString(cmd, "reject-paths")),
		AcceptHosts: kubectl.MakeRegexpArrayOrDie(cmdutil.GetFlagString(cmd, "accept-hosts")),
	}
	if cmdutil.GetFlagBool(cmd, "disable-filter") {
		glog.Warning("Request filter disabled, your proxy is vulnerable to XSRF attacks, please be cautious")
		filter = nil
	}

	server, err := kubectl.NewProxyServer(cmdutil.GetFlagString(cmd, "www"), apiProxyPrefix, staticPrefix, filter, clientConfig)
	if err != nil {
		return err
	}

	glog.Fatal(server.Serve(port))
	return nil
}
