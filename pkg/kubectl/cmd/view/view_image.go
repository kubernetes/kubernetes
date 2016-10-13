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

package view

import (
	"fmt"
	"io"

	"github.com/renstrom/dedent"
	"github.com/spf13/cobra"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/kubectl"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/resource"
)

type imageOptions struct {
	resource.FilenameOptions
	out      io.Writer
	all      bool
	selector string
}

var (
	image_resources = `
  pod (po), replicationcontroller (rc), deployment (deploy), daemonset (ds), job, replicaset (rs)`

	image_long = dedent.Dedent(`
		View container image(s) of resources.

		Possible resources include (case insensitive):`) + image_resources

	image_example = dedent.Dedent(`
		# View the container image(s) of the nginx deployment
		kubectl view image deployment/nginx`)
)

func NewCmdImage(f *cmdutil.Factory, out io.Writer) *cobra.Command {
	options := &imageOptions{
		out: out,
	}

	cmd := &cobra.Command{
		Use:     "image (-f FILENAME | TYPE NAME)",
		Short:   "View image(s) of a pod template",
		Long:    image_long,
		Example: image_example,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(options.Run(f, cmd, args))
		},
	}

	usage := "identifying the resource to get from a server."
	cmd.Flags().BoolVar(&options.all, "all", false, "select all resources in the namespace of the specified resource types")
	cmd.Flags().StringVarP(&options.selector, "selector", "l", "", "Selector (label query) to filter on")
	cmdutil.AddFilenameOptionFlags(cmd, &options.FilenameOptions, usage)
	return cmd
}

func (o *imageOptions) Run(f *cmdutil.Factory, cmd *cobra.Command, args []string) error {
	mapper, typer := f.Object()
	cmdNamespace, enforceNamespace, err := f.DefaultNamespace()
	if err != nil {
		return err
	}

	builder := resource.NewBuilder(mapper, typer, resource.ClientMapperFunc(f.ClientForMapping), f.Decoder(true)).
		ContinueOnError().
		NamespaceParam(cmdNamespace).DefaultNamespace().
		FilenameParam(enforceNamespace, &o.FilenameOptions).
		SelectorParam(o.selector).
		ResourceTypeOrNameArgs(o.all, args...).
		Flatten().
		Latest()

	infos, err := builder.Do().Infos()
	if err != nil {
		return err
	}

	for _, i := range infos {
		f.VisitPodSpecForObject(i.Object, func(ps *api.PodSpec) error {
			for _, c := range ps.Containers {
				var resourceName string
				if alias, ok := kubectl.ResourceShortFormFor(i.ResourceMapping().Resource); ok {
					resourceName = alias
				} else if resourceName == "" {
					resourceName = "none"
				}
				fmt.Fprintf(o.out, "%s/%s %s\n", resourceName, i.Name, c.Image)
			}
			return nil
		})
	}

	return nil
}
