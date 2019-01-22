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
	"io"

	"github.com/spf13/cobra"

	"k8s.io/apiextensions-apiserver/pkg/cmd/server/options"
	genericapiserver "k8s.io/apiserver/pkg/server"
)

func NewServerCommand(out, errOut io.Writer, stopCh <-chan struct{}) *cobra.Command {
	o := options.NewCustomResourceDefinitionsServerOptions(out, errOut)

	cmd := &cobra.Command{
		Short: "Launch an API extensions API server",
		Long:  "Launch an API extensions API server",
		RunE: func(c *cobra.Command, args []string) error {
			if err := o.Complete(); err != nil {
				return err
			}
			if err := o.Validate(); err != nil {
				return err
			}
			if err := Run(o, stopCh); err != nil {
				return err
			}
			return nil
		},
	}

	fs := cmd.Flags()
	o.AddFlags(fs)
	return cmd
}

func Run(o *options.CustomResourceDefinitionsServerOptions, stopCh <-chan struct{}) error {
	config, err := o.Config()
	if err != nil {
		return err
	}

	server, err := config.Complete().New(genericapiserver.NewEmptyDelegate())
	if err != nil {
		return err
	}
	return server.GenericAPIServer.PrepareRun().Run(stopCh)
}
