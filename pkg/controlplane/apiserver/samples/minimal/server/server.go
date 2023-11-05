/*
Copyright 2023 The Kubernetes Authors.

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
	"context"
	"fmt"
	"net"
	"os"
	"path/filepath"

	"github.com/spf13/cobra"

	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	_ "k8s.io/apiserver/pkg/admission"
	genericapifilters "k8s.io/apiserver/pkg/endpoints/filters"
	"k8s.io/apiserver/pkg/registry/generic"
	genericapiserver "k8s.io/apiserver/pkg/server"
	serverstorage "k8s.io/apiserver/pkg/server/storage"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/apiserver/pkg/util/notfoundhandler"
	"k8s.io/client-go/rest"
	cliflag "k8s.io/component-base/cli/flag"
	"k8s.io/component-base/cli/globalflag"
	"k8s.io/component-base/logs"
	logsapi "k8s.io/component-base/logs/api/v1"
	_ "k8s.io/component-base/metrics/prometheus/workqueue"
	"k8s.io/component-base/term"
	"k8s.io/component-base/version"
	"k8s.io/component-base/version/verflag"
	"k8s.io/klog/v2"
	aggregatorapiserver "k8s.io/kube-aggregator/pkg/apiserver"

	controlplaneapiserver "k8s.io/kubernetes/pkg/controlplane/apiserver"
	"k8s.io/kubernetes/pkg/controlplane/apiserver/options"
	_ "k8s.io/kubernetes/pkg/features"
	apiserverinternalrest "k8s.io/kubernetes/pkg/registry/apiserverinternal/rest"
	authenticationrest "k8s.io/kubernetes/pkg/registry/authentication/rest"
	coordinationrest "k8s.io/kubernetes/pkg/registry/coordination/rest"
	corerest "k8s.io/kubernetes/pkg/registry/core/rest"
	eventsrest "k8s.io/kubernetes/pkg/registry/events/rest"
	flowcontrolrest "k8s.io/kubernetes/pkg/registry/flowcontrol/rest"
	// add the kubernetes feature gates
)

func init() {
	utilruntime.Must(logsapi.AddFeatureGates(utilfeature.DefaultMutableFeatureGate))
}

// NewCommand creates a *cobra.Command object with default parameters
func NewCommand() *cobra.Command {
	s := NewOptions()

	cmd := &cobra.Command{
		Use: "sample-minimal-apiserver",
		Long: `The sample minimal apiserver is part of a generic controlplane,
a system serving APIs like Kubernetes, but without the container domain specific
APIs.`,

		// stop printing usage when the command errors
		SilenceUsage: true,
		PersistentPreRunE: func(*cobra.Command, []string) error {
			// silence client-go warnings.
			// kube-apiserver loopback clients should not log self-issued warnings.
			rest.SetDefaultWarningHandler(rest.NoWarnings{})
			return nil
		},
		RunE: func(cmd *cobra.Command, args []string) error {
			verflag.PrintAndExitIfRequested()
			fs := cmd.Flags()

			// Activate logging as soon as possible, after that
			// show flags with the final logging configuration.
			if err := logsapi.ValidateAndApply(s.Logs, utilfeature.DefaultFeatureGate); err != nil {
				return err
			}
			cliflag.PrintFlags(fs)

			// Wire ServiceAccount authentication without relying on pods and nodes.
			s.Authentication.ServiceAccounts.OptionalTokenGetter = genericTokenGetter

			completedOptions, err := s.Complete([]string{}, []net.IP{})
			if err != nil {
				return err
			}

			if errs := completedOptions.Validate(); len(errs) != 0 {
				return utilerrors.NewAggregate(errs)
			}

			// add feature enablement metrics
			utilfeature.DefaultMutableFeatureGate.AddMetrics()
			ctx := genericapiserver.SetupSignalContext()
			return Run(ctx, completedOptions)
		},
		Args: func(cmd *cobra.Command, args []string) error {
			for _, arg := range args {
				if len(arg) > 0 {
					return fmt.Errorf("%q does not take any arguments, got %q", cmd.CommandPath(), args)
				}
			}
			return nil
		},
	}

	var namedFlagSets cliflag.NamedFlagSets
	s.AddFlags(&namedFlagSets)
	verflag.AddFlags(namedFlagSets.FlagSet("global"))
	globalflag.AddGlobalFlags(namedFlagSets.FlagSet("global"), cmd.Name(), logs.SkipLoggingConfigurationFlags())

	fs := cmd.Flags()
	for _, f := range namedFlagSets.FlagSets {
		fs.AddFlagSet(f)
	}

	cols, _, _ := term.TerminalSize(cmd.OutOrStdout())
	cliflag.SetUsageAndHelpFunc(cmd, namedFlagSets, cols)

	return cmd
}

func NewOptions() *options.Options {
	s := options.NewOptions()
	s.Admission.GenericAdmission.DefaultOffPlugins = DefaultOffAdmissionPlugins()

	wd, _ := os.Getwd()
	s.SecureServing.ServerCert.CertDirectory = filepath.Join(wd, ".sample-minimal-controlplane")

	// Wire ServiceAccount authentication without relying on pods and nodes.
	s.Authentication.ServiceAccounts.OptionalTokenGetter = genericTokenGetter

	return s
}

// Run runs the specified APIServer. This should never exit.
func Run(ctx context.Context, opts options.CompletedOptions) error {
	// To help debugging, immediately log version
	klog.Infof("Version: %+v", version.Get())

	klog.InfoS("Golang settings", "GOGC", os.Getenv("GOGC"), "GOMAXPROCS", os.Getenv("GOMAXPROCS"), "GOTRACEBACK", os.Getenv("GOTRACEBACK"))

	config, err := NewConfig(opts)
	if err != nil {
		return err
	}
	completed, err := config.Complete()
	if err != nil {
		return err
	}
	server, err := CreateServerChain(completed)
	if err != nil {
		return err
	}

	prepared, err := server.PrepareRun()
	if err != nil {
		return err
	}

	return prepared.Run(ctx)
}

// CreateServerChain creates the apiservers connected via delegation.
func CreateServerChain(config CompletedConfig) (*aggregatorapiserver.APIAggregator, error) {
	// 1. Natively implemented resources
	notFoundHandler := notfoundhandler.New(config.ControlPlane.Generic.Serializer, genericapifilters.NoMuxAndDiscoveryIncompleteKey)
	nativeAPIs, err := config.ControlPlane.New("sample-generic-controlplane", genericapiserver.NewEmptyDelegateWithCustomHandler(notFoundHandler))
	if err != nil {
		return nil, fmt.Errorf("failed to create generic controlplane apiserver: %w", err)
	}
	storageProviders, err := storageProviders(config.ControlPlane)
	if err != nil {
		return nil, fmt.Errorf("failed to create storage providers: %w", err)
	}
	if err := nativeAPIs.InstallAPIs(storageProviders...); err != nil {
		return nil, fmt.Errorf("failed to install APIs: %w", err)
	}

	// 2. Aggregator for APIServices, discovery and OpenAPI
	aggregatorServer, err := controlplaneapiserver.CreateAggregatorServer(config.Aggregator, nativeAPIs.GenericAPIServer, nil, false, controlplaneapiserver.DefaultGenericAPIServicePriorities())
	if err != nil {
		// we don't need special handling for innerStopCh because the aggregator server doesn't create any go routines
		return nil, fmt.Errorf("failed to create kube-aggregator: %w", err)
	}

	return aggregatorServer, nil
}

func storageProviders(c controlplaneapiserver.CompletedConfig) ([]controlplaneapiserver.RESTStorageProvider, error) {
	return []controlplaneapiserver.RESTStorageProvider{
		withDisabledResources{
			disabled: map[string][]string{
				"v1": {"configmaps", "secrets", "serviceaccounts", "serviceaccounts/token", "resourcequotas", "resourcequotas/status"},
			},
			RESTStorageProvider: &corerest.GenericConfig{
				StorageFactory:              c.Extra.StorageFactory,
				EventTTL:                    c.Extra.EventTTL,
				LoopbackClientConfig:        c.Generic.LoopbackClientConfig,
				ServiceAccountIssuer:        c.Extra.ServiceAccountIssuer,
				ExtendExpiration:            c.Extra.ExtendExpiration,
				ServiceAccountMaxExpiration: c.Extra.ServiceAccountMaxExpiration,
				APIAudiences:                c.Generic.Authentication.APIAudiences,
				Informers:                   c.Extra.VersionedInformers,
			},
		},
		apiserverinternalrest.StorageProvider{},
		authenticationrest.RESTStorageProvider{Authenticator: c.Generic.Authentication.Authenticator, APIAudiences: c.Generic.Authentication.APIAudiences},
		flowcontrolrest.RESTStorageProvider{InformerFactory: c.Generic.SharedInformerFactory},
		coordinationrest.RESTStorageProvider{},
		eventsrest.RESTStorageProvider{TTL: c.EventTTL},
	}, nil
}

type withDisabledResources struct {
	disabled map[string][]string
	controlplaneapiserver.RESTStorageProvider
}

func (s withDisabledResources) NewRESTStorage(apiResourceConfigSource serverstorage.APIResourceConfigSource, restOptionsGetter generic.RESTOptionsGetter) (genericapiserver.APIGroupInfo, error) {
	info, err := s.RESTStorageProvider.NewRESTStorage(apiResourceConfigSource, restOptionsGetter)
	if err != nil {
		return info, err
	}
	for v, res := range info.VersionedResourcesStorageMap {
		disabled := sets.New[string](s.disabled[v]...)
		for r := range res {
			if disabled.Has(r) {
				delete(res, r)
			}
		}
	}
	return info, nil
}
