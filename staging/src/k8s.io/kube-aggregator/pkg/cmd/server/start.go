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

package server

import (
	"fmt"
	"io"
	"io/ioutil"

	"github.com/spf13/cobra"
	"github.com/spf13/pflag"

	"k8s.io/apimachinery/pkg/util/sets"
	genericapiserver "k8s.io/apiserver/pkg/server"
	"k8s.io/apiserver/pkg/server/filters"
	genericoptions "k8s.io/apiserver/pkg/server/options"
	kubeclientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/pkg/api"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/kube-aggregator/pkg/apis/apiregistration/v1alpha1"
	"k8s.io/kube-aggregator/pkg/apiserver"
)

const defaultEtcdPathPrefix = "/registry/kube-aggregator.kubernetes.io/"

type AggregatorOptions struct {
	RecommendedOptions *genericoptions.RecommendedOptions

	// ProxyClientCert/Key are the client cert used to identify this proxy. Backing APIServices use
	// this to confirm the proxy's identity
	ProxyClientCertFile string
	ProxyClientKeyFile  string

	// CoreAPIKubeconfig is a filename for a kubeconfig file to contact the core API server wtih
	// If it is not set, the in cluster config is used
	CoreAPIKubeconfig string

	StdOut io.Writer
	StdErr io.Writer
}

// NewCommandStartAggregator provides a CLI handler for 'start master' command
func NewCommandStartAggregator(out, err io.Writer, stopCh <-chan struct{}) *cobra.Command {
	o := NewDefaultOptions(out, err)

	cmd := &cobra.Command{
		Short: "Launch a API aggregator and proxy server",
		Long:  "Launch a API aggregator and proxy server",
		RunE: func(c *cobra.Command, args []string) error {
			if err := o.Complete(); err != nil {
				return err
			}
			if err := o.Validate(args); err != nil {
				return err
			}
			if err := o.RunAggregator(stopCh); err != nil {
				return err
			}
			return nil
		},
	}

	o.AddFlags(cmd.Flags())
	return cmd
}

// AddFlags is necessary because hyperkube doesn't work using cobra, so we have to have different registration and execution paths
func (o *AggregatorOptions) AddFlags(fs *pflag.FlagSet) {
	o.RecommendedOptions.AddFlags(fs)
	fs.StringVar(&o.ProxyClientCertFile, "proxy-client-cert-file", o.ProxyClientCertFile, "client certificate used identify the proxy to the API server")
	fs.StringVar(&o.ProxyClientKeyFile, "proxy-client-key-file", o.ProxyClientKeyFile, "client certificate key used identify the proxy to the API server")
	fs.StringVar(&o.CoreAPIKubeconfig, "core-kubeconfig", o.CoreAPIKubeconfig, ""+
		"kubeconfig file pointing at the 'core' kubernetes server with enough rights to get,list,watch "+
		" services,endpoints.  If not set, the in-cluster config is used")
}

// NewDefaultOptions builds a "normal" set of options.  You wouldn't normally expose this, but hyperkube isn't cobra compatible
func NewDefaultOptions(out, err io.Writer) *AggregatorOptions {
	o := &AggregatorOptions{
		RecommendedOptions: genericoptions.NewRecommendedOptions(defaultEtcdPathPrefix, api.Scheme, api.Codecs.LegacyCodec(v1alpha1.SchemeGroupVersion)),

		StdOut: out,
		StdErr: err,
	}
	o.RecommendedOptions.SecureServing.ServingOptions.BindPort = 443
	return o
}

func (o AggregatorOptions) Validate(args []string) error {
	return nil
}

func (o *AggregatorOptions) Complete() error {
	return nil
}

func (o AggregatorOptions) RunAggregator(stopCh <-chan struct{}) error {
	// TODO have a "real" external address
	if err := o.RecommendedOptions.SecureServing.MaybeDefaultWithSelfSignedCerts("localhost"); err != nil {
		return fmt.Errorf("error creating self-signed certificates: %v", err)
	}

	serverConfig := genericapiserver.NewConfig().
		WithSerializer(api.Codecs)

	if err := o.RecommendedOptions.ApplyTo(serverConfig); err != nil {
		return err
	}
	serverConfig.LongRunningFunc = filters.BasicLongRunningRequestCheck(
		sets.NewString("watch", "proxy"),
		sets.NewString("attach", "exec", "proxy", "log", "portforward"),
	)

	var kubeconfig *rest.Config
	var err error
	if len(o.CoreAPIKubeconfig) > 0 {
		loadingRules := &clientcmd.ClientConfigLoadingRules{ExplicitPath: o.CoreAPIKubeconfig}
		loader := clientcmd.NewNonInteractiveDeferredLoadingClientConfig(loadingRules, &clientcmd.ConfigOverrides{})

		kubeconfig, err = loader.ClientConfig()

	} else {
		kubeconfig, err = rest.InClusterConfig()
	}
	if err != nil {
		return err
	}

	coreAPIServerClient, err := kubeclientset.NewForConfig(kubeconfig)
	if err != nil {
		return err
	}

	config := apiserver.Config{
		GenericConfig:       serverConfig,
		CoreAPIServerClient: coreAPIServerClient,
	}

	config.ProxyClientCert, err = ioutil.ReadFile(o.ProxyClientCertFile)
	if err != nil {
		return err
	}
	config.ProxyClientKey, err = ioutil.ReadFile(o.ProxyClientKeyFile)
	if err != nil {
		return err
	}

	server, err := config.Complete().New(stopCh)
	if err != nil {
		return err
	}
	return server.GenericAPIServer.PrepareRun().Run(stopCh)
}
