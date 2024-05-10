/*
Copyright 2018 The Kubernetes Authors.

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
	"io"

	"github.com/spf13/cobra"

	"k8s.io/apiextensions-apiserver/pkg/cmd/server/options"
	genericapiserver "k8s.io/apiserver/pkg/server"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	utilversion "k8s.io/apiserver/pkg/util/version"
)

func NewServerCommand(ctx context.Context, out, errOut io.Writer) *cobra.Command {
	featureGate := utilfeature.DefaultMutableFeatureGate
	// effectiveVersion is used to set what apis and feature gates the generic api server is compatible with.
	// You can also have the flag setting the effectiveVersion of the apiextensions apiserver, and
	// having a mapping from the apiextensions apiserver version to generic apiserver version.
	effectiveVersion := utilversion.DefaultEffectiveVersionRegistry.EffectiveVersionForOrRegister(
		utilversion.ComponentGenericAPIServer, utilversion.DefaultKubeEffectiveVersion())
	featureGate.DeferErrorsToValidation(true)
	o := options.NewCustomResourceDefinitionsServerOptions(out, errOut, featureGate, effectiveVersion)

	cmd := &cobra.Command{
		Short: "Launch an API extensions API server",
		Long:  "Launch an API extensions API server",
		RunE: func(c *cobra.Command, args []string) error {
			if err := featureGate.SetEmulationVersion(effectiveVersion.EmulationVersion()); err != nil {
				return err
			}

			if err := o.Complete(); err != nil {
				return err
			}
			if err := o.Validate(); err != nil {
				return err
			}
			if err := Run(c.Context(), o); err != nil {
				return err
			}
			return nil
		},
	}
	cmd.SetContext(ctx)

	fs := cmd.Flags()
	featureGate.AddFlag(fs)
	effectiveVersion.AddFlags(fs, "")
	o.AddFlags(fs)
	return cmd
}

func Run(ctx context.Context, o *options.CustomResourceDefinitionsServerOptions) error {
	config, err := o.Config()
	if err != nil {
		return err
	}

	server, err := config.Complete().New(genericapiserver.NewEmptyDelegate())
	if err != nil {
		return err
	}
	return server.GenericAPIServer.PrepareRun().RunWithContext(ctx)
}
