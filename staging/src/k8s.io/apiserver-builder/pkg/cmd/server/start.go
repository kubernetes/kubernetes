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

	"github.com/spf13/cobra"
	flag "github.com/spf13/pflag"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver-builder/pkg/apiserver"
	"k8s.io/apiserver-builder/pkg/builders"
	genericapiserver "k8s.io/apiserver/pkg/server"
	genericmux "k8s.io/apiserver/pkg/server/mux"
	genericoptions "k8s.io/apiserver/pkg/server/options"

	"bytes"
	"github.com/golang/glog"
	"k8s.io/apimachinery/pkg/openapi"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver-builder/pkg/validators"
	"k8s.io/apiserver/pkg/util/logs"
	"k8s.io/client-go/pkg/api"
	"net/http"
	"os"
)

var GetOpenApiDefinition openapi.GetOpenAPIDefinitions

type ServerOptions struct {
	RecommendedOptions *genericoptions.RecommendedOptions

	StdOut      io.Writer
	StdErr      io.Writer
	APIBuilders []*builders.APIGroupBuilder

	PrintBearerToken bool
	PrintOpenapi     bool
	RunDelegatedAuth bool
	BearerToken      string
	PostStartHooks   []PostStartHook
}

type PostStartHook struct {
	Fn   genericapiserver.PostStartHookFunc
	Name string
}

// StartApiServer starts an apiserver hosting the provider apis and openapi definitions.
func StartApiServer(etcdPath string, apis []*builders.APIGroupBuilder, openapidefs openapi.GetOpenAPIDefinitions) {
	logs.InitLogs()
	defer logs.FlushLogs()

	GetOpenApiDefinition = openapidefs

	// To disable providers, manually specify the list provided by getKnownProviders()
	cmd, _ := NewCommandStartServer(etcdPath, os.Stdout, os.Stderr, apis, wait.NeverStop)
	cmd.Flags().AddFlagSet(flag.CommandLine)
	if err := cmd.Execute(); err != nil {
		panic(err)
	}
}

func NewServerOptions(etcdPath string, out, errOut io.Writer, builders []*builders.APIGroupBuilder) *ServerOptions {
	versions := []schema.GroupVersion{}
	for _, b := range builders {
		versions = append(versions, b.GetLegacyCodec()...)
	}

	o := &ServerOptions{
		RecommendedOptions: genericoptions.NewRecommendedOptions(etcdPath, api.Scheme, api.Codecs.LegacyCodec(versions...)),

		StdOut:           out,
		StdErr:           errOut,
		APIBuilders:      builders,
		RunDelegatedAuth: true,
	}
	o.RecommendedOptions.SecureServing.BindPort = 443

	return o
}

// NewCommandStartMaster provides a CLI handler for 'start master' command
func NewCommandStartServer(etcdPath string, out, errOut io.Writer, builders []*builders.APIGroupBuilder,
	stopCh <-chan struct{}) (*cobra.Command, *ServerOptions) {
	o := NewServerOptions(etcdPath, out, errOut, builders)

	// Support overrides
	cmd := &cobra.Command{
		Short: "Launch an API server",
		Long:  "Launch an API server",
		RunE: func(c *cobra.Command, args []string) error {
			if err := o.Complete(); err != nil {
				return err
			}
			if err := o.Validate(args); err != nil {
				return err
			}
			if err := o.RunServer(stopCh); err != nil {
				return err
			}
			return nil
		},
	}

	flags := cmd.Flags()
	flags.BoolVar(&o.PrintBearerToken, "print-bearer-token", false,
		"If true, print a curl command with the bearer token to test the server")
	flags.BoolVar(&o.PrintOpenapi, "print-openapi", false,
		"If true, print the openapi json and exit.")
	flags.BoolVar(&o.RunDelegatedAuth, "delegated-auth", true,
		"If true, setup delegated auth.")
	o.RecommendedOptions.AddFlags(flags)
	return cmd, o
}

func (o ServerOptions) Validate(args []string) error {
	return nil
}

func (o *ServerOptions) Complete() error {
	return nil
}

func (o ServerOptions) Config() (*apiserver.Config, error) {
	// TODO have a "real" external address
	if err := o.RecommendedOptions.SecureServing.MaybeDefaultWithSelfSignedCerts(
		"localhost",
		nil,
		nil); err != nil {
		return nil, fmt.Errorf("error creating self-signed certificates: %v", err)
	}

	serverConfig := genericapiserver.NewConfig(api.Codecs)

	if o.RunDelegatedAuth {
		if err := o.RecommendedOptions.Authentication.ApplyTo(serverConfig); err != nil {
			return nil, err
		}
		if err := o.RecommendedOptions.Authorization.ApplyTo(serverConfig); err != nil {
			return nil, err
		}
	}
	if err := o.RecommendedOptions.Etcd.ApplyTo(serverConfig); err != nil {
		return nil, err
	}
	if err := o.RecommendedOptions.SecureServing.ApplyTo(serverConfig); err != nil {
		return nil, err
	}
	if err := o.RecommendedOptions.Audit.ApplyTo(serverConfig); err != nil {
		return nil, err
	}
	if err := o.RecommendedOptions.Features.ApplyTo(serverConfig); err != nil {
		return nil, err
	}

	config := &apiserver.Config{
		GenericConfig: serverConfig,
	}
	return config, nil
}

func (o *ServerOptions) RunServer(stopCh <-chan struct{}) error {
	config, err := o.Config()
	if err != nil {
		return err
	}

	config.GenericConfig.OpenAPIConfig = genericapiserver.DefaultOpenAPIConfig(GetOpenApiDefinition, api.Scheme)
	config.GenericConfig.OpenAPIConfig.Info.Title = "Api"

	if o.PrintBearerToken {
		glog.Infof("Serving on loopback...")
		glog.Infof("\n\n********************************\nTo test the server run:\n"+
			"curl -k -H \"Authorization: Bearer %s\" %s\n********************************\n\n",
			config.GenericConfig.LoopbackClientConfig.BearerToken,
			config.GenericConfig.LoopbackClientConfig.Host)
	}
	o.BearerToken = config.GenericConfig.LoopbackClientConfig.BearerToken

	for _, provider := range o.APIBuilders {
		config.AddApi(provider)
	}

	server, err := config.Init().Complete().New()
	if err != nil {
		return err
	}

	for _, h := range o.PostStartHooks {
		server.GenericAPIServer.AddPostStartHook(h.Name, h.Fn)
	}

	s := server.GenericAPIServer.PrepareRun()
	err = validators.OpenAPI.SetSchema(readOpenapi(server.GenericAPIServer.HandlerContainer))
	if o.PrintOpenapi {
		fmt.Printf("%s", validators.OpenAPI.OpenApi)
		os.Exit(0)
	}
	if err != nil {
		return err
	}

	s.Run(stopCh)

	return nil
}

func readOpenapi(handler *genericmux.APIContainer) string {
	req, err := http.NewRequest("GET", "/swagger.json", nil)
	if err != nil {
		panic(fmt.Errorf("Could not create openapi request %v", err))
	}
	resp := &BufferedResponse{}
	handler.ServeHTTP(resp, req)
	return resp.String()
}

type BufferedResponse struct {
	bytes.Buffer
}

func (BufferedResponse) Header() http.Header { return http.Header{} }
func (BufferedResponse) WriteHeader(int)     {}
