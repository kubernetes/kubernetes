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

	"github.com/pborman/uuid"
	"github.com/spf13/cobra"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/registry/generic"
	"k8s.io/apiserver/pkg/registry/generic/registry"
	genericapiserver "k8s.io/apiserver/pkg/server"
	"k8s.io/apiserver/pkg/server/filters"
	genericoptions "k8s.io/apiserver/pkg/server/options"
	"k8s.io/apiserver/pkg/storage/storagebackend"
	restclient "k8s.io/client-go/rest"
	"k8s.io/kubernetes/cmd/kube-aggregator/pkg/apiserver"
	"k8s.io/kubernetes/pkg/api"
	kubeclientset "k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"

	"k8s.io/kubernetes/cmd/kube-aggregator/pkg/apis/apiregistration/v1alpha1"
)

const defaultEtcdPathPrefix = "/registry/kube-aggregator.kubernetes.io/"

type AggregatorOptions struct {
	Etcd           *genericoptions.EtcdOptions
	SecureServing  *genericoptions.SecureServingOptions
	Authentication *genericoptions.DelegatingAuthenticationOptions
	Authorization  *genericoptions.DelegatingAuthorizationOptions

	// ProxyClientCert/Key are the client cert used to identify this proxy. Backing APIServices use
	// this to confirm the proxy's identity
	ProxyClientCertFile string
	ProxyClientKeyFile  string

	StdOut io.Writer
	StdErr io.Writer
}

// NewCommandStartMaster provides a CLI handler for 'start master' command
func NewCommandStartAggregator(out, err io.Writer) *cobra.Command {
	o := &AggregatorOptions{
		Etcd:           genericoptions.NewEtcdOptions(api.Scheme),
		SecureServing:  genericoptions.NewSecureServingOptions(),
		Authentication: genericoptions.NewDelegatingAuthenticationOptions(),
		Authorization:  genericoptions.NewDelegatingAuthorizationOptions(),

		StdOut: out,
		StdErr: err,
	}
	o.Etcd.StorageConfig.Type = storagebackend.StorageTypeETCD3
	o.Etcd.StorageConfig.Prefix = defaultEtcdPathPrefix
	o.Etcd.StorageConfig.Codec = api.Codecs.LegacyCodec(v1alpha1.SchemeGroupVersion)
	o.SecureServing.ServingOptions.BindPort = 443

	cmd := &cobra.Command{
		Short: "Launch a API aggregator and proxy server",
		Long:  "Launch a API aggregator and proxy server",
		Run: func(c *cobra.Command, args []string) {
			cmdutil.CheckErr(o.Complete())
			cmdutil.CheckErr(o.Validate(args))
			cmdutil.CheckErr(o.RunAggregator())
		},
	}

	flags := cmd.Flags()
	o.Etcd.AddFlags(flags)
	o.SecureServing.AddFlags(flags)
	o.Authentication.AddFlags(flags)
	o.Authorization.AddFlags(flags)
	flags.StringVar(&o.ProxyClientCertFile, "proxy-client-cert-file", o.ProxyClientCertFile, "client certificate used identify the proxy to the API server")
	flags.StringVar(&o.ProxyClientKeyFile, "proxy-client-key-file", o.ProxyClientKeyFile, "client certificate key used identify the proxy to the API server")

	return cmd
}

func (o AggregatorOptions) Validate(args []string) error {
	return nil
}

func (o *AggregatorOptions) Complete() error {
	return nil
}

func (o AggregatorOptions) RunAggregator() error {
	// TODO have a "real" external address
	if err := o.SecureServing.MaybeDefaultWithSelfSignedCerts("localhost"); err != nil {
		return fmt.Errorf("error creating self-signed certificates: %v", err)
	}

	genericAPIServerConfig := genericapiserver.NewConfig().
		WithSerializer(api.Codecs)

	if err := o.SecureServing.ApplyTo(genericAPIServerConfig); err != nil {
		return fmt.Errorf("failed to configure https: %s", err)
	}
	if err := o.Authentication.ApplyTo(genericAPIServerConfig); err != nil {
		return err
	}
	if err := o.Authorization.ApplyTo(genericAPIServerConfig); err != nil {
		return err
	}
	genericAPIServerConfig.LongRunningFunc = filters.BasicLongRunningRequestCheck(
		sets.NewString("watch", "proxy"),
		sets.NewString("attach", "exec", "proxy", "log", "portforward"),
	)

	var err error
	privilegedLoopbackToken := uuid.NewRandom().String()
	if genericAPIServerConfig.LoopbackClientConfig, err = genericAPIServerConfig.SecureServingInfo.NewSelfClientConfig(privilegedLoopbackToken); err != nil {
		return err
	}

	kubeconfig, err := restclient.InClusterConfig()
	if err != nil {
		return err
	}
	coreAPIServerClient, err := kubeclientset.NewForConfig(kubeconfig)
	if err != nil {
		return err
	}

	config := apiserver.Config{
		GenericConfig:       genericAPIServerConfig,
		RESTOptionsGetter:   &restOptionsFactory{storageConfig: &o.Etcd.StorageConfig},
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

	server, err := config.Complete().New()
	if err != nil {
		return err
	}
	server.GenericAPIServer.PrepareRun().Run(wait.NeverStop)

	return nil
}

type restOptionsFactory struct {
	storageConfig *storagebackend.Config
}

func (f *restOptionsFactory) GetRESTOptions(resource schema.GroupResource) (generic.RESTOptions, error) {
	return generic.RESTOptions{
		StorageConfig:           f.storageConfig,
		Decorator:               registry.StorageWithCacher,
		DeleteCollectionWorkers: 1,
		EnableGarbageCollection: false,
		ResourcePrefix:          f.storageConfig.Prefix + "/" + resource.Group + "/" + resource.Resource,
	}, nil
}
