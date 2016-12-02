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
	"io"

	"github.com/spf13/cobra"

	"k8s.io/kubernetes/cmd/kubernetes-discovery/pkg/legacy"
	kcmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
)

type DiscoveryServerOptions struct {
	StdOut io.Writer
	StdErr io.Writer
}

// NewCommandStartMaster provides a CLI handler for 'start master' command
func NewCommandStartDiscoveryServer(out, err io.Writer) *cobra.Command {
	o := &DiscoveryServerOptions{
		StdOut: out,
		StdErr: err,
	}

	cmd := &cobra.Command{
		Short: "Launch a discovery summarizer and proxy server",
		Long:  "Launch a discovery summarizer and proxy server",
		Run: func(c *cobra.Command, args []string) {
			kcmdutil.CheckErr(o.Complete())
			kcmdutil.CheckErr(o.Validate(args))
			kcmdutil.CheckErr(o.RunDiscoveryServer())
		},
	}

	return cmd
}

func (o DiscoveryServerOptions) Validate(args []string) error {
	return nil
}

func (o *DiscoveryServerOptions) Complete() error {
	return nil
}

func (o DiscoveryServerOptions) RunDiscoveryServer() error {
	if true {
		// for now this is the only option.  later, only use this if no etcd is configured
		return o.RunLegacyDiscoveryServer()
	}

	return nil
}

// RunLegacyDiscoveryServer runs the legacy mode of discovery
func (o DiscoveryServerOptions) RunLegacyDiscoveryServer() error {
	configFilePath := "config.json"
	port := "9090"
	s, err := legacy.NewDiscoverySummarizer(configFilePath)
	if err != nil {
		return err
	}
	return s.Run(port)
}
