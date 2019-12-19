/*
Copyright 2014 The Kubernetes Authors.

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

package proxy

import (
	"errors"
	"fmt"
	"net"
	"os"
	"strings"
	"time"

	"github.com/spf13/cobra"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/client-go/rest"
	"k8s.io/klog"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/proxy"
	"k8s.io/kubectl/pkg/util/i18n"
	"k8s.io/kubectl/pkg/util/templates"
)

// ProxyOptions have the data required to perform the proxy operation
type ProxyOptions struct {
	// Common user flags
	staticDir     string
	staticPrefix  string
	apiPrefix     string
	acceptPaths   string
	rejectPaths   string
	acceptHosts   string
	rejectMethods string
	port          int
	address       string
	disableFilter bool
	unixSocket    string
	keepalive     time.Duration

	clientConfig *rest.Config
	filter       *proxy.FilterServer

	genericclioptions.IOStreams
}

const (
	defaultPort         = 8001
	defaultStaticPrefix = "/static/"
	defaultAPIPrefix    = "/"
	defaultAddress      = "127.0.0.1"
)

var (
	proxyLong = templates.LongDesc(i18n.T(`
		Creates a proxy server or application-level gateway between localhost and
		the Kubernetes API Server. It also allows serving static content over specified
		HTTP path. All incoming data enters through one port and gets forwarded to
		the remote kubernetes API Server port, except for the path matching the static content path.`))

	proxyExample = templates.Examples(i18n.T(`
		# To proxy all of the kubernetes api and nothing else, use:

		    $ kubectl proxy --api-prefix=/

		# To proxy only part of the kubernetes api and also some static files:

		    $ kubectl proxy --www=/my/files --www-prefix=/static/ --api-prefix=/api/

		# The above lets you 'curl localhost:8001/api/v1/pods'.

		# To proxy the entire kubernetes api at a different root, use:

		    $ kubectl proxy --api-prefix=/custom/

		# The above lets you 'curl localhost:8001/custom/api/v1/pods'

		# Run a proxy to kubernetes apiserver on port 8011, serving static content from ./local/www/
		kubectl proxy --port=8011 --www=./local/www/

		# Run a proxy to kubernetes apiserver on an arbitrary local port.
		# The chosen port for the server will be output to stdout.
		kubectl proxy --port=0

		# Run a proxy to kubernetes apiserver, changing the api prefix to k8s-api
		# This makes e.g. the pods api available at localhost:8001/k8s-api/v1/pods/
		kubectl proxy --api-prefix=/k8s-api`))
)

// NewProxyOptions creates the options for proxy
func NewProxyOptions(ioStreams genericclioptions.IOStreams) *ProxyOptions {
	return &ProxyOptions{
		IOStreams:     ioStreams,
		staticPrefix:  defaultStaticPrefix,
		apiPrefix:     defaultAPIPrefix,
		acceptPaths:   proxy.DefaultPathAcceptRE,
		rejectPaths:   proxy.DefaultPathRejectRE,
		acceptHosts:   proxy.DefaultHostAcceptRE,
		rejectMethods: proxy.DefaultMethodRejectRE,
		port:          defaultPort,
		address:       defaultAddress,
	}
}

// NewCmdProxy returns the proxy Cobra command
func NewCmdProxy(f cmdutil.Factory, ioStreams genericclioptions.IOStreams) *cobra.Command {
	o := NewProxyOptions(ioStreams)

	cmd := &cobra.Command{
		Use:                   "proxy [--port=PORT] [--www=static-dir] [--www-prefix=prefix] [--api-prefix=prefix]",
		DisableFlagsInUseLine: true,
		Short:                 i18n.T("Run a proxy to the Kubernetes API server"),
		Long:                  proxyLong,
		Example:               proxyExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(o.Complete(f))
			cmdutil.CheckErr(o.Validate())
			cmdutil.CheckErr(o.RunProxy())
		},
	}

	cmd.Flags().StringVarP(&o.staticDir, "www", "w", o.staticDir, "Also serve static files from the given directory under the specified prefix.")
	cmd.Flags().StringVarP(&o.staticPrefix, "www-prefix", "P", o.staticPrefix, "Prefix to serve static files under, if static file directory is specified.")
	cmd.Flags().StringVarP(&o.apiPrefix, "api-prefix", "", o.apiPrefix, "Prefix to serve the proxied API under.")
	cmd.Flags().StringVar(&o.acceptPaths, "accept-paths", o.acceptPaths, "Regular expression for paths that the proxy should accept.")
	cmd.Flags().StringVar(&o.rejectPaths, "reject-paths", o.rejectPaths, "Regular expression for paths that the proxy should reject. Paths specified here will be rejected even accepted by --accept-paths.")
	cmd.Flags().StringVar(&o.acceptHosts, "accept-hosts", o.acceptHosts, "Regular expression for hosts that the proxy should accept.")
	cmd.Flags().StringVar(&o.rejectMethods, "reject-methods", o.rejectMethods, "Regular expression for HTTP methods that the proxy should reject (example --reject-methods='POST,PUT,PATCH'). ")
	cmd.Flags().IntVarP(&o.port, "port", "p", o.port, "The port on which to run the proxy. Set to 0 to pick a random port.")
	cmd.Flags().StringVarP(&o.address, "address", "", o.address, "The IP address on which to serve on.")
	cmd.Flags().BoolVar(&o.disableFilter, "disable-filter", o.disableFilter, "If true, disable request filtering in the proxy. This is dangerous, and can leave you vulnerable to XSRF attacks, when used with an accessible port.")
	cmd.Flags().StringVarP(&o.unixSocket, "unix-socket", "u", o.unixSocket, "Unix socket on which to run the proxy.")
	cmd.Flags().DurationVar(&o.keepalive, "keepalive", o.keepalive, "keepalive specifies the keep-alive period for an active network connection. Set to 0 to disable keepalive.")
	return cmd
}

// Complete adapts from the command line args and factory to the data required.
func (o *ProxyOptions) Complete(f cmdutil.Factory) error {
	clientConfig, err := f.ToRESTConfig()
	if err != nil {
		return err
	}
	o.clientConfig = clientConfig

	if !strings.HasSuffix(o.staticPrefix, "/") {
		o.staticPrefix += "/"
	}

	if !strings.HasSuffix(o.apiPrefix, "/") {
		o.apiPrefix += "/"
	}

	if o.disableFilter {
		if o.unixSocket == "" {
			klog.Warning("Request filter disabled, your proxy is vulnerable to XSRF attacks, please be cautious")
		}
		o.filter = nil
	} else {
		o.filter = &proxy.FilterServer{
			AcceptPaths:   proxy.MakeRegexpArrayOrDie(o.acceptPaths),
			RejectPaths:   proxy.MakeRegexpArrayOrDie(o.rejectPaths),
			AcceptHosts:   proxy.MakeRegexpArrayOrDie(o.acceptHosts),
			RejectMethods: proxy.MakeRegexpArrayOrDie(o.rejectMethods),
		}
	}
	return nil
}

// Validate checks to the ProxyOptions to see if there is sufficient information to run the command.
func (o ProxyOptions) Validate() error {
	if o.port != defaultPort && o.unixSocket != "" {
		return errors.New("cannot set --unix-socket and --port at the same time")
	}

	if o.staticDir != "" {
		fileInfo, err := os.Stat(o.staticDir)
		if err != nil {
			klog.Warning("Failed to stat static file directory "+o.staticDir+": ", err)
		} else if !fileInfo.IsDir() {
			klog.Warning("Static file directory " + o.staticDir + " is not a directory")
		}
	}

	return nil
}

// RunProxy checks given arguments and executes command
func (o ProxyOptions) RunProxy() error {
	server, err := proxy.NewServer(o.staticDir, o.apiPrefix, o.staticPrefix, o.filter, o.clientConfig, o.keepalive)

	// Separate listening from serving so we can report the bound port
	// when it is chosen by os (eg: port == 0)
	var l net.Listener
	if o.unixSocket == "" {
		l, err = server.Listen(o.address, o.port)
	} else {
		l, err = server.ListenUnix(o.unixSocket)
	}
	if err != nil {
		klog.Fatal(err)
	}
	fmt.Fprintf(o.IOStreams.Out, "Starting to serve on %s\n", l.Addr().String())
	klog.Fatal(server.ServeOnListener(l))
	return nil
}
