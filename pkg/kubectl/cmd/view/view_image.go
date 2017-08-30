/*
Copyright 2017 The Kubernetes Authors.

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
	"strings"

	"github.com/spf13/cobra"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/kubectl"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/resource"
	"k8s.io/kubernetes/pkg/kubectl/util/i18n"
	"k8s.io/kubernetes/pkg/printers"
)

// ImageOptions is the start of the data required to perform the operation.  As new fields are added, add them here instead of
// referencing the cmd.Flags()
type ImageOptions struct {
	resource.FilenameOptions

	Out      io.Writer
	All      bool
	Local    bool
	Selector string
}

var (
	imageResources = `
    pod (po), replicationcontroller (rc), deployment (deploy), daemonset (ds), job, replicaset (rs)`

	imageLong = templates.LongDesc(`
		View existing container image(s) of resources.

		Possible resources include (case insensitive):
		` + imageResources)

	imageExample = templates.Examples(`
		# View a deployment's nginx container image.
		kubectl set image deployment/nginx

		# View all deployments' and rc's image
		kubectl set image deployments,rc --all

		# View nginx container image from local file, without hitting the server
		kubectl set image -f path/to/file.yaml`)
)

// NewCmdImage new view image command
func NewCmdImage(f cmdutil.Factory, out io.Writer) *cobra.Command {
	options := &ImageOptions{
		Out: out,
	}

	cmd := &cobra.Command{
		Use:     "image (-f FILENAME | TYPE NAME)",
		Short:   i18n.T("View image of a pod template"),
		Long:    imageLong,
		Example: imageExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(options.Run(f, cmd, args))
		},
	}

	usage := "identifying the resource to get from a server."
	cmdutil.AddFilenameOptionFlags(cmd, &options.FilenameOptions, usage)
	cmdutil.AddNoHeadersFlags(cmd, false)
	cmd.Flags().StringP("output", "o", "", "default name, if not name will show all informations")
	cmd.Flags().BoolVar(&options.All, "all", false, "Select all resources in the namespace of the specified resource types")
	cmd.Flags().StringVarP(&options.Selector, "selector", "l", "", "Selector (label query) to filter on, supports '=', '==', and '!='.(e.g. -l key1=value1,key2=value2)")
	cmd.Flags().BoolVar(&options.Local, "local", false, "If true, set image will NOT contact api-server but run locally.")
	return cmd
}

// Run execute view image command
func (o *ImageOptions) Run(f cmdutil.Factory, cmd *cobra.Command, args []string) error {
	cmdNamespace, enforceNamespace, err := f.DefaultNamespace()
	if err != nil {
		return err
	}

	builder := f.NewBuilder(!o.Local).
		ContinueOnError().
		NamespaceParam(cmdNamespace).DefaultNamespace().
		FilenameParam(enforceNamespace, &o.FilenameOptions).
		Flatten()
	if !o.Local {
		builder = builder.
			SelectorParam(o.Selector).
			ResourceTypeOrNameArgs(o.All, args...).
			Latest()
	}
	infos, err := builder.Do().Infos()
	if err != nil {
		return err
	}
	w := printers.GetNewTabWriter(o.Out)
	defer w.Flush()

	if !cmdutil.GetFlagBool(cmd, "no-headers") {
		if cmdutil.GetFlagString(cmd, "output") == "name" {
			printHeaders(w, true)
		} else {
			printHeaders(w, false)
		}
	}
	for _, info := range infos {
		_, err := f.UpdatePodSpecForObject(info.Object, func(spec *api.PodSpec) error {
			for _, container := range spec.Containers {
				if cmdutil.GetFlagString(cmd, "output") == "name" {
					_, err := fmt.Fprintln(w, container.Image)
					if err != nil {
						return err
					}
					continue
				}
				var resourceName string
				if alias, ok := kubectl.ResourceShortFormFor(info.ResourceMapping().Resource); ok {
					resourceName = alias
				} else if resourceName == "" {
					resourceName = "none"
				}
				Name := resourceName + "/" + info.Name
				_, err := fmt.Fprintf(w, "%s\t%s\t%s\n", Name, container.Name, container.Image)
				if err != nil {
					return err
				}
			}
			return nil
		})
		if err != nil {
			return err
		}
	}
	return nil
}

func printHeaders(out io.Writer, imageOnly bool) error {
	columnNames := []string{"NAME", "CONTAINER(s)", "IMAGE(s)"}
	if imageOnly {
		columnNames = columnNames[2:]
	}
	_, err := fmt.Fprintf(out, "%s\n", strings.Join(columnNames, "\t"))
	return err
}
