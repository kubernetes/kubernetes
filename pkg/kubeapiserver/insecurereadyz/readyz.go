package insecurereadyz

import (
	"fmt"
	"io/ioutil"
	"net"
	"net/http"
	"strings"

	"github.com/spf13/cobra"
	"github.com/spf13/pflag"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/klog/v2"
)

// ReadyzOpts holds values to drive the insecure /readyz endpoint.
type ReadyzOpts struct {
	BindAddress net.IP
	BindPort    uint16

	KubeconfigFile string
	URL            string
}

func NewReadyzOpts() *ReadyzOpts {
	return &ReadyzOpts{
		BindAddress: net.ParseIP("0.0.0.0"),
		BindPort:    6080,
		URL:         "/readyz",
	}
}

// NewInsecureReadyzCommand creates a insecure-readyz command.
func NewInsecureReadyzCommand() *cobra.Command {
	o := NewReadyzOpts()

	cmd := &cobra.Command{
		Use:   "insecure-readyz",
		Short: "Host an insecure /readyz endpoint insecurely on an HTTP port that mirrors kube-apiserver content",
		// stop printing usage when the command errors
		SilenceUsage: true,
		RunE: func(cmd *cobra.Command, args []string) error {
			if err := o.Validate(); err != nil {
				return err
			}
			config, err := o.ToConfig()
			if err != nil {
				return err
			}

			return config.Run()
		},
	}

	o.AddFlags(cmd.Flags())

	return cmd
}

func (o *ReadyzOpts) AddFlags(fs *pflag.FlagSet) {
	fs.IPVar(&o.BindAddress, "bind-address", o.BindAddress, ""+
		"The IP address on which to listen for the --insecure-port port. The "+
		"associated interface(s) must be reachable by the rest of the cluster, and by CLI/web "+
		"clients. If blank or an unspecified address (0.0.0.0 or ::), all interfaces will be used.")
	fs.Uint16Var(&o.BindPort, "insecure-port", o.BindPort, "Listen on this port")
	fs.StringVar(&o.KubeconfigFile, "kubeconfig", o.KubeconfigFile, "Path to kubeconfig file with authorization and kube-apiserver location information.")
	fs.StringVar(&o.URL, "delegate-url", o.URL, "The URL the insecure /readyz endpoint proxies to")
}

// Validate verifies the inputs.
func (o *ReadyzOpts) Validate() error {
	if len(o.URL) == 0 {
		return fmt.Errorf("delegate-url is required")
	}
	if !strings.HasPrefix(o.URL, "/") {
		return fmt.Errorf("delegate-url must start with '/'")
	}

	if o.BindPort == 0 {
		return fmt.Errorf("insecure-port must be between 1 and 65535")
	}

	if len(o.KubeconfigFile) == 0 {
		return fmt.Errorf("kubeconfig is required")
	}

	return nil
}

// Complete fills in missing values before command execution.
func (o *ReadyzOpts) ToConfig() (*ReadyzConfig, error) {
	clientConfig, err := clientcmd.NewNonInteractiveDeferredLoadingClientConfig(
		&clientcmd.ClientConfigLoadingRules{ExplicitPath: o.KubeconfigFile}, nil).
		ClientConfig()
	if err != nil {
		return nil, err
	}
	// these are required to create the RESTClient. We want the REST client to get the correct authn/authz behavior
	clientConfig.GroupVersion = &schema.GroupVersion{Version: "if-this-is-used-find-the-bug"}
	scheme := runtime.NewScheme()
	codecs := serializer.NewCodecFactory(scheme)
	clientConfig.NegotiatedSerializer = codecs
	rest.SetKubernetesDefaults(clientConfig)

	restClient, err := rest.RESTClientFor(clientConfig)
	if err != nil {
		return nil, err
	}

	// we need the base URL to have the https://host:port for the GET
	defaultBaseURL, _, err := rest.DefaultServerURL(
		clientConfig.Host, clientConfig.APIPath, schema.GroupVersion{}, true /*the only reason to use this is if you have TLS*/)
	if err != nil {
		return nil, err
	}

	return &ReadyzConfig{
		listenAddr: fmt.Sprintf("%s:%d", o.BindAddress.String(), o.BindPort),
		client:     restClient.Client,
		url:        defaultBaseURL.String() + o.URL,
	}, nil
}

// ReadyzConfig holds runtime configuration the insecure /readyz endpoint.
type ReadyzConfig struct {
	listenAddr string
	client     *http.Client
	url        string
}

// Run contains the logic of the insecure-readyz command.
func (o *ReadyzConfig) Run() error {
	mux := http.NewServeMux()
	mux.HandleFunc("/readyz", func(w http.ResponseWriter, req *http.Request) {
		resp, err := o.client.Get(o.url)
		if err != nil {
			http.Error(w, "couldn't contact kube-apiserver", http.StatusInternalServerError)
			klog.Warningf("Failed to get %q: %v", o.url, err)
			return
		}
		defer resp.Body.Close()

		w.Header().Set("Content-Type", "text/plain; charset=utf-8")
		w.Header().Set("X-Content-Type-Options", "nosniff")
		body, err := ioutil.ReadAll(resp.Body)
		if err != nil {
			http.Error(w, "failed to read response from kube-apiserver", http.StatusInternalServerError)
			klog.Warningf("Failed to read the response body: %v", err)
			return
		}

		w.Header().Set("Content-Type", resp.Header.Get("Content-Type"))
		w.WriteHeader(resp.StatusCode)
		w.Write(body)
	})

	klog.Infof("Listening on %s", o.listenAddr)
	klog.Infof("Showing %q on /readyz", o.url)
	return http.ListenAndServe(o.listenAddr, mux)
}
