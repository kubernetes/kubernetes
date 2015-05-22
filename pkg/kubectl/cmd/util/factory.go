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

package util

import (
	"flag"
	"fmt"
	"io"
	"os"
	"strconv"

	"github.com/spf13/cobra"
	"github.com/spf13/pflag"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/latest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/meta"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/validation"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/clientcmd"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl/resource"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

const (
	FlagMatchBinaryVersion = "match-server-version"
)

// Factory provides abstractions that allow the Kubectl command to be extended across multiple types
// of resources and different API sets.
// TODO: make the functions interfaces
// TODO: pass the various interfaces on the factory directly into the command constructors (so the
// commands are decoupled from the factory).
type Factory struct {
	clients    *clientCache
	flags      *pflag.FlagSet
	generators map[string]kubectl.Generator

	// Returns interfaces for dealing with arbitrary runtime.Objects.
	Object func() (meta.RESTMapper, runtime.ObjectTyper)
	// Returns a client for accessing Kubernetes resources or an error.
	Client func() (*client.Client, error)
	// Returns a client.Config for accessing the Kubernetes server.
	ClientConfig func() (*client.Config, error)
	// Returns a RESTClient for working with the specified RESTMapping or an error. This is intended
	// for working with arbitrary resources and is not guaranteed to point to a Kubernetes APIServer.
	RESTClient func(mapping *meta.RESTMapping) (resource.RESTClient, error)
	// Returns a Describer for displaying the specified RESTMapping type or an error.
	Describer func(mapping *meta.RESTMapping) (kubectl.Describer, error)
	// Returns a Printer for formatting objects of the given type or an error.
	Printer func(mapping *meta.RESTMapping, noHeaders, withNamespace bool) (kubectl.ResourcePrinter, error)
	// Returns a Resizer for changing the size of the specified RESTMapping type or an error
	Resizer func(mapping *meta.RESTMapping) (kubectl.Resizer, error)
	// Returns a Reaper for gracefully shutting down resources.
	Reaper func(mapping *meta.RESTMapping) (kubectl.Reaper, error)
	// PodSelectorForObject returns the pod selector associated with the provided object
	PodSelectorForObject func(object runtime.Object) (string, error)
	// PortsForObject returns the ports associated with the provided object
	PortsForObject func(object runtime.Object) ([]string, error)
	// LabelsForObject returns the labels associated with the provided object
	LabelsForObject func(object runtime.Object) (map[string]string, error)
	// Returns a schema that can validate objects stored on disk.
	Validator func() (validation.Schema, error)
	// Returns the default namespace to use in cases where no other namespace is specified
	DefaultNamespace func() (string, error)
	// Returns the generator for the provided generator name
	Generator func(name string) (kubectl.Generator, bool)
}

// NewFactory creates a factory with the default Kubernetes resources defined
// if optionalClientConfig is nil, then flags will be bound to a new clientcmd.ClientConfig.
// if optionalClientConfig is not nil, then this factory will make use of it.
func NewFactory(optionalClientConfig clientcmd.ClientConfig) *Factory {
	mapper := kubectl.ShortcutExpander{latest.RESTMapper}

	flags := pflag.NewFlagSet("", pflag.ContinueOnError)
	flags.SetNormalizeFunc(util.WordSepNormalizeFunc)

	generators := map[string]kubectl.Generator{
		"run-container/v1": kubectl.BasicReplicationController{},
		"service/v1":       kubectl.ServiceGenerator{},
	}

	clientConfig := optionalClientConfig
	if optionalClientConfig == nil {
		clientConfig = DefaultClientConfig(flags)
	}

	clients := &clientCache{
		clients: make(map[string]*client.Client),
		loader:  clientConfig,
	}

	return &Factory{
		clients:    clients,
		flags:      flags,
		generators: generators,

		Object: func() (meta.RESTMapper, runtime.ObjectTyper) {
			cfg, err := clientConfig.ClientConfig()
			CheckErr(err)
			cmdApiVersion := cfg.Version

			return kubectl.OutputVersionMapper{mapper, cmdApiVersion}, api.Scheme
		},
		Client: func() (*client.Client, error) {
			return clients.ClientForVersion("")
		},
		ClientConfig: func() (*client.Config, error) {
			return clients.ClientConfigForVersion("")
		},
		RESTClient: func(mapping *meta.RESTMapping) (resource.RESTClient, error) {
			client, err := clients.ClientForVersion(mapping.APIVersion)
			if err != nil {
				return nil, err
			}
			return client.RESTClient, nil
		},
		Describer: func(mapping *meta.RESTMapping) (kubectl.Describer, error) {
			client, err := clients.ClientForVersion(mapping.APIVersion)
			if err != nil {
				return nil, err
			}
			describer, ok := kubectl.DescriberFor(mapping.Kind, client)
			if !ok {
				return nil, fmt.Errorf("no description has been implemented for %q", mapping.Kind)
			}
			return describer, nil
		},
		Printer: func(mapping *meta.RESTMapping, noHeaders, withNamespace bool) (kubectl.ResourcePrinter, error) {
			return kubectl.NewHumanReadablePrinter(noHeaders, withNamespace), nil
		},
		PodSelectorForObject: func(object runtime.Object) (string, error) {
			// TODO: replace with a swagger schema based approach (identify pod selector via schema introspection)
			switch t := object.(type) {
			case *api.ReplicationController:
				return kubectl.MakeLabels(t.Spec.Selector), nil
			case *api.Pod:
				if len(t.Labels) == 0 {
					return "", fmt.Errorf("the pod has no labels and cannot be exposed")
				}
				return kubectl.MakeLabels(t.Labels), nil
			case *api.Service:
				if t.Spec.Selector == nil {
					return "", fmt.Errorf("the service has no pod selector set")
				}
				return kubectl.MakeLabels(t.Spec.Selector), nil
			default:
				kind, err := meta.NewAccessor().Kind(object)
				if err != nil {
					return "", err
				}
				return "", fmt.Errorf("it is not possible to get a pod selector from %s", kind)
			}
		},
		PortsForObject: func(object runtime.Object) ([]string, error) {
			// TODO: replace with a swagger schema based approach (identify pod selector via schema introspection)
			switch t := object.(type) {
			case *api.ReplicationController:
				return getPorts(t.Spec.Template.Spec), nil
			case *api.Pod:
				return getPorts(t.Spec), nil
			default:
				kind, err := meta.NewAccessor().Kind(object)
				if err != nil {
					return nil, err
				}
				return nil, fmt.Errorf("it is not possible to get ports from %s", kind)
			}
		},
		LabelsForObject: func(object runtime.Object) (map[string]string, error) {
			return meta.NewAccessor().Labels(object)
		},
		Resizer: func(mapping *meta.RESTMapping) (kubectl.Resizer, error) {
			client, err := clients.ClientForVersion(mapping.APIVersion)
			if err != nil {
				return nil, err
			}
			return kubectl.ResizerFor(mapping.Kind, kubectl.NewResizerClient(client))
		},
		Reaper: func(mapping *meta.RESTMapping) (kubectl.Reaper, error) {
			client, err := clients.ClientForVersion(mapping.APIVersion)
			if err != nil {
				return nil, err
			}
			return kubectl.ReaperFor(mapping.Kind, client)
		},
		Validator: func() (validation.Schema, error) {
			if flags.Lookup("validate").Value.String() == "true" {
				client, err := clients.ClientForVersion("")
				if err != nil {
					return nil, err
				}
				return &clientSwaggerSchema{client, api.Scheme}, nil
			}
			return validation.NullSchema{}, nil
		},
		DefaultNamespace: func() (string, error) {
			return clientConfig.Namespace()
		},
		Generator: func(name string) (kubectl.Generator, bool) {
			generator, ok := generators[name]
			return generator, ok
		},
	}
}

// BindFlags adds any flags that are common to all kubectl sub commands.
func (f *Factory) BindFlags(flags *pflag.FlagSet) {
	// any flags defined by external projects (not part of pflags)
	util.AddFlagSetToPFlagSet(flag.CommandLine, flags)

	// This is necessary as github.com/spf13/cobra doesn't support "global"
	// pflags currently.  See https://github.com/spf13/cobra/issues/44.
	util.AddPFlagSetToPFlagSet(pflag.CommandLine, flags)

	// Hack for global access to validation flag.
	// TODO: Refactor out after configuration flag overhaul.
	if f.flags.Lookup("validate") == nil {
		f.flags.Bool("validate", false, "If true, use a schema to validate the input before sending it")
	}

	if f.flags != nil {
		f.flags.VisitAll(func(flag *pflag.Flag) {
			flags.AddFlag(flag)
		})
	}

	// Globally persistent flags across all subcommands.
	// TODO Change flag names to consts to allow safer lookup from subcommands.
	// TODO Add a verbose flag that turns on glog logging. Probably need a way
	// to do that automatically for every subcommand.
	flags.BoolVar(&f.clients.matchVersion, FlagMatchBinaryVersion, false, "Require server version to match client version")
}

func getPorts(spec api.PodSpec) []string {
	result := []string{}
	for _, container := range spec.Containers {
		for _, port := range container.Ports {
			result = append(result, strconv.Itoa(port.ContainerPort))
		}
	}
	return result
}

type clientSwaggerSchema struct {
	c *client.Client
	t runtime.ObjectTyper
}

func (c *clientSwaggerSchema) ValidateBytes(data []byte) error {
	version, _, err := c.t.DataVersionAndKind(data)
	if err != nil {
		return err
	}
	schemaData, err := c.c.RESTClient.Get().
		AbsPath("/swaggerapi/api", version).
		Do().
		Raw()
	if err != nil {
		return err
	}
	schema, err := validation.NewSwaggerSchemaFromBytes(schemaData)
	if err != nil {
		return err
	}
	return schema.ValidateBytes(data)
}

// DefaultClientConfig creates a clientcmd.ClientConfig with the following hierarchy:
//   1.  Use the kubeconfig builder.  The number of merges and overrides here gets a little crazy.  Stay with me.
//       1.  Merge together the kubeconfig itself.  This is done with the following hierarchy rules:
//           1.  CommandLineLocation - this parsed from the command line, so it must be late bound.  If you specify this,
//               then no other kubeconfig files are merged.  This file must exist.
//           2.  If $KUBECONFIG is set, then it is treated as a list of files that should be merged.
//			 3.  HomeDirectoryLocation
//           Empty filenames are ignored.  Files with non-deserializable content produced errors.
//           The first file to set a particular value or map key wins and the value or map key is never changed.
//           This means that the first file to set CurrentContext will have its context preserved.  It also means
//           that if two files specify a "red-user", only values from the first file's red-user are used.  Even
//           non-conflicting entries from the second file's "red-user" are discarded.
//       2.  Determine the context to use based on the first hit in this chain
//           1.  command line argument - again, parsed from the command line, so it must be late bound
//           2.  CurrentContext from the merged kubeconfig file
//           3.  Empty is allowed at this stage
//       3.  Determine the cluster info and auth info to use.  At this point, we may or may not have a context.  They
//           are built based on the first hit in this chain.  (run it twice, once for auth, once for cluster)
//           1.  command line argument
//           2.  If context is present, then use the context value
//           3.  Empty is allowed
//       4.  Determine the actual cluster info to use.  At this point, we may or may not have a cluster info.  Build
//           each piece of the cluster info based on the chain:
//           1.  command line argument
//           2.  If cluster info is present and a value for the attribute is present, use it.
//           3.  If you don't have a server location, bail.
//       5.  Auth info is build using the same rules as cluster info, EXCEPT that you can only have one authentication
//           technique per auth info.  The following conditions result in an error:
//           1.  If there are two conflicting techniques specified from the command line, fail.
//           2.  If the command line does not specify one, and the auth info has conflicting techniques, fail.
//           3.  If the command line specifies one and the auth info specifies another, honor the command line technique.
//   2.  Use default values and potentially prompt for auth information
func DefaultClientConfig(flags *pflag.FlagSet) clientcmd.ClientConfig {
	loadingRules := clientcmd.NewDefaultClientConfigLoadingRules()
	flags.StringVar(&loadingRules.ExplicitPath, "kubeconfig", "", "Path to the kubeconfig file to use for CLI requests.")

	overrides := &clientcmd.ConfigOverrides{}
	flagNames := clientcmd.RecommendedConfigOverrideFlags("")
	// short flagnames are disabled by default.  These are here for compatibility with existing scripts
	flagNames.ClusterOverrideFlags.APIServer.ShortName = "s"

	clientcmd.BindOverrideFlags(overrides, flags, flagNames)
	clientConfig := clientcmd.NewInteractiveDeferredLoadingClientConfig(loadingRules, overrides, os.Stdin)

	return clientConfig
}

// PrintObject prints an api object given command line flags to modify the output format
func (f *Factory) PrintObject(cmd *cobra.Command, obj runtime.Object, out io.Writer) error {
	mapper, _ := f.Object()
	_, kind, err := api.Scheme.ObjectVersionAndKind(obj)
	if err != nil {
		return err
	}

	mapping, err := mapper.RESTMapping(kind)
	if err != nil {
		return err
	}

	printer, err := f.PrinterForMapping(cmd, mapping, false)
	if err != nil {
		return err
	}
	return printer.PrintObj(obj, out)
}

// PrinterForMapping returns a printer suitable for displaying the provided resource type.
// Requires that printer flags have been added to cmd (see AddPrinterFlags).
func (f *Factory) PrinterForMapping(cmd *cobra.Command, mapping *meta.RESTMapping, withNamespace bool) (kubectl.ResourcePrinter, error) {
	printer, ok, err := PrinterForCommand(cmd)
	if err != nil {
		return nil, err
	}
	if ok {
		clientConfig, err := f.ClientConfig()
		if err != nil {
			return nil, err
		}
		defaultVersion := clientConfig.Version

		version := OutputVersion(cmd, defaultVersion)
		if len(version) == 0 {
			version = mapping.APIVersion
		}
		if len(version) == 0 {
			return nil, fmt.Errorf("you must specify an output-version when using this output format")
		}
		printer = kubectl.NewVersionedPrinter(printer, mapping.ObjectConvertor, version, mapping.APIVersion)
	} else {
		printer, err = f.Printer(mapping, GetFlagBool(cmd, "no-headers"), withNamespace)
		if err != nil {
			return nil, err
		}
	}
	return printer, nil
}

// ClientMapperForCommand returns a ClientMapper for the factory.
func (f *Factory) ClientMapperForCommand() resource.ClientMapper {
	return resource.ClientMapperFunc(func(mapping *meta.RESTMapping) (resource.RESTClient, error) {
		return f.RESTClient(mapping)
	})
}
