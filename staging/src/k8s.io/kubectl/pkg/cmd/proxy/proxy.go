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
	"net/url"
	"os"
	"strings"
	"time"

	"github.com/spf13/cobra"

	"k8s.io/cli-runtime/pkg/genericiooptions"
	"k8s.io/client-go/rest"
	"k8s.io/klog/v2"

	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/proxy"
	"k8s.io/kubectl/pkg/util/i18n"
	"k8s.io/kubectl/pkg/util/templates"
)

// ProxyFlags directly reflect the information that CLI is gathering via flags.
type ProxyFlags struct {
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

	appendServerPath bool

	genericiooptions.IOStreams
}

// NewProxyFlags returns a default ProxyFlags
func NewProxyFlags(streams genericiooptions.IOStreams) *ProxyFlags {
	return &ProxyFlags{
		staticPrefix:  defaultStaticPrefix,
		apiPrefix:     defaultAPIPrefix,
		acceptPaths:   proxy.DefaultPathAcceptRE,
		rejectPaths:   proxy.DefaultPathRejectRE,
		acceptHosts:   proxy.DefaultHostAcceptRE,
		rejectMethods: proxy.DefaultMethodRejectRE,
		port:          defaultPort,
		address:       defaultAddress,

		IOStreams: streams,
	}
}

// ProxyOptions have the data required to perform the proxy operation
type ProxyOptions struct {
	staticDir    string
	staticPrefix string
	apiPrefix    string
	port         int
	address      string
	unixSocket   string
	keepalive    time.Duration

	appendServerPath bool

	clientConfig *rest.Config
	filter       *proxy.FilterServer

	genericiooptions.IOStreams
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
		the Kubernetes API server. It also allows serving static content over specified
		HTTP path. All incoming data enters through one port and gets forwarded to
		the remote Kubernetes API server port, except for the path matching the static content path.`))

	proxyExample = templates.Examples(i18n.T(`
		# To proxy all of the Kubernetes API and nothing else
		kubectl proxy --api-prefix=/

		# To proxy only part of the Kubernetes API and also some static files
		# You can get pods info with 'curl localhost:8001/api/v1/pods'
		kubectl proxy --www=/my/files --www-prefix=/static/ --api-prefix=/api/

		# To proxy the entire Kubernetes API at a different root
		# You can get pods info with 'curl localhost:8001/custom/api/v1/pods'
		kubectl proxy --api-prefix=/custom/

		# Run a proxy to the Kubernetes API server on port 8011, serving static content from ./local/www/
		kubectl proxy --port=8011 --www=./local/www/

		# Run a proxy to the Kubernetes API server on an arbitrary local port
		# The chosen port for the server will be output to stdout
		kubectl proxy --port=0

		# Run a proxy to the Kubernetes API server, changing the API prefix to k8s-api
		# This makes e.g. the pods API available at localhost:8001/k8s-api/v1/pods/
		kubectl proxy --api-prefix=/k8s-api`))
)

// AddFlags registers flags for the proxy command.
func (flags *ProxyFlags) AddFlags(cmd *cobra.Command) {
	cmd.Flags().StringVarP(&flags.staticDir, "www", "w", flags.staticDir, "Also serve static files from the given directory under the specified prefix.")
	cmd.Flags().StringVarP(&flags.staticPrefix, "www-prefix", "P", flags.staticPrefix, "Prefix to serve static files under, if static file directory is specified.")
	cmd.Flags().StringVar(&flags.apiPrefix, "api-prefix", flags.apiPrefix, "Prefix to serve the proxied API under.")
	cmd.Flags().StringVar(&flags.acceptPaths, "accept-paths", flags.acceptPaths, "Regular expression for paths that the proxy should accept.")
	cmd.Flags().StringVar(&flags.rejectPaths, "reject-paths", flags.rejectPaths, "Regular expression for paths that the proxy should reject. Paths specified here will be rejected even accepted by --accept-paths.")
	cmd.Flags().StringVar(&flags.acceptHosts, "accept-hosts", flags.acceptHosts, "Regular expression for hosts that the proxy should accept.")
	cmd.Flags().StringVar(&flags.rejectMethods, "reject-methods", flags.rejectMethods, "Regular expression for HTTP methods that the proxy should reject (example --reject-methods='POST,PUT,PATCH'). ")
	cmd.Flags().IntVarP(&flags.port, "port", "p", flags.port, "The port on which to run the proxy. Set to 0 to pick a random port.")
	cmd.Flags().StringVar(&flags.address, "address", flags.address, "The IP address on which to serve on.")
	cmd.Flags().BoolVar(&flags.disableFilter, "disable-filter", flags.disableFilter, "If true, disable request filtering in the proxy. This is dangerous, and can leave you vulnerable to XSRF attacks, when used with an accessible port.")
	cmd.Flags().StringVarP(&flags.unixSocket, "unix-socket", "u", flags.unixSocket, "Unix socket on which to run the proxy.")
	cmd.Flags().DurationVar(&flags.keepalive, "keepalive", flags.keepalive, "keepalive specifies the keep-alive period for an active network connection. Set to 0 to disable keepalive.")
	cmd.Flags().BoolVar(&flags.appendServerPath, "append-server-path", flags.appendServerPath, "If true, enables automatic path appending of the kube context server path to each request.")
}

// NewCmdProxy returns the proxy Cobra command
func NewCmdProxy(f cmdutil.Factory, ioStreams genericiooptions.IOStreams) *cobra.Command {
	flags := NewProxyFlags(ioStreams)

	cmd := &cobra.Command{
		Use:                   "proxy [--port=PORT] [--www=static-dir] [--www-prefix=prefix] [--api-prefix=prefix]",
		DisableFlagsInUseLine: true,
		Short:                 i18n.T("Run a proxy to the Kubernetes API server"),
		Long:                  proxyLong,
		Example:               proxyExample,
		Run: func(cmd *cobra.Command, args []string) {
			o, err := flags.ToOptions(f)
			cmdutil.CheckErr(err)
			cmdutil.CheckErr(o.Validate())
			cmdutil.CheckErr(o.RunProxy())
		},
	}

	flags.AddFlags(cmd)

	return cmd
}

// ToOptions converts from CLI inputs to runtime inputs.
func (flags *ProxyFlags) ToOptions(f cmdutil.Factory) (*ProxyOptions, error) {
	clientConfig, err := f.ToRESTConfig()
	if err != nil {
		return nil, err
	}

	staticPrefix := flags.staticPrefix
	if !strings.HasSuffix(staticPrefix, "/") {
		staticPrefix += "/"
	}

	apiPrefix := flags.apiPrefix
	if !strings.HasSuffix(apiPrefix, "/") {
		apiPrefix += "/"
	}

	if !flags.appendServerPath {
		target, err := url.Parse(clientConfig.Host)
		if err != nil {
			return nil, err
		}
		if target.Path != "" && target.Path != "/" {
			klog.Warning("Your kube context contains a server path " + target.Path + ", use --append-server-path to automatically append the path to each request")
		}
	}

	var filter *proxy.FilterServer
	if flags.disableFilter {
		if flags.unixSocket == "" {
			klog.Warning("Request filter disabled, your proxy is vulnerable to XSRF attacks, please be cautious")
		}
	} else {
		if flags.rejectPaths == "" || flags.rejectMethods == "" {
			klog.Warning("Request filter has been relaxed, your proxy may be vulnerable to attacks, please be cautious")
		}
		filter = &proxy.FilterServer{
			AcceptPaths:   proxy.MakeRegexpArrayOrDie(flags.acceptPaths),
			RejectPaths:   proxy.MakeRegexpArrayOrDie(flags.rejectPaths),
			AcceptHosts:   proxy.MakeRegexpArrayOrDie(flags.acceptHosts),
			RejectMethods: proxy.MakeRegexpArrayOrDie(flags.rejectMethods),
		}
	}

	o := &ProxyOptions{
		staticDir:    flags.staticDir,
		staticPrefix: staticPrefix,
		apiPrefix:    apiPrefix,
		port:         flags.port,
		address:      flags.address,
		unixSocket:   flags.unixSocket,
		keepalive:    flags.keepalive,

		appendServerPath: flags.appendServerPath,

		clientConfig: clientConfig,
		filter:       filter,

		IOStreams: flags.IOStreams,
	}
	return o, nil
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
	server, err := proxy.NewServer(o.staticDir, o.apiPrefix, o.staticPrefix, o.filter, o.clientConfig, o.keepalive, o.appendServerPath)

	if err != nil {
		return err
	}

	// Separate listening from serving so we can report the bound port
	// when it is chosen by os (eg: port == 0)
	var l net.Listener
	if o.unixSocket == "" {
		l, err = server.Listen(o.address, o.port)
	} else {
		l, err = server.ListenUnix(o.unixSocket)
	}
	if err != nil {
		return err
	}
	fmt.Fprintf(o.IOStreams.Out, "Starting to serve on %s\n", l.Addr().String())
	return server.ServeOnListener(l)
}
