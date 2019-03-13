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

package apply

import (
	"bytes"
	"fmt"

	"github.com/spf13/cobra"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/cli-runtime/pkg/printers"
	"k8s.io/cli-runtime/pkg/resource"
	"k8s.io/kubernetes/pkg/kubectl"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/cmd/util/editor"
	"k8s.io/kubernetes/pkg/kubectl/scheme"
	"k8s.io/kubernetes/pkg/kubectl/util/i18n"
	"k8s.io/kubernetes/pkg/kubectl/util/templates"
)

type SetLastAppliedOptions struct {
	CreateAnnotation bool

	PrintFlags *genericclioptions.PrintFlags
	PrintObj   printers.ResourcePrinterFunc

	FilenameOptions resource.FilenameOptions

	infoList                     []*resource.Info
	namespace                    string
	enforceNamespace             bool
	dryRun                       bool
	shortOutput                  bool
	output                       string
	patchBufferList              []PatchBuffer
	builder                      *resource.Builder
	unstructuredClientForMapping func(mapping *meta.RESTMapping) (resource.RESTClient, error)

	genericclioptions.IOStreams
}

type PatchBuffer struct {
	Patch     []byte
	PatchType types.PatchType
}

var (
	applySetLastAppliedLong = templates.LongDesc(i18n.T(`
		Set the latest last-applied-configuration annotations by setting it to match the contents of a file.
		This results in the last-applied-configuration being updated as though 'kubectl apply -f <file>' was run,
		without updating any other parts of the object.`))

	applySetLastAppliedExample = templates.Examples(i18n.T(`
		# Set the last-applied-configuration of a resource to match the contents of a file.
		kubectl apply set-last-applied -f deploy.yaml

		# Execute set-last-applied against each configuration file in a directory.
		kubectl apply set-last-applied -f path/

		# Set the last-applied-configuration of a resource to match the contents of a file, will create the annotation if it does not already exist.
		kubectl apply set-last-applied -f deploy.yaml --create-annotation=true
		`))
)

func NewSetLastAppliedOptions(ioStreams genericclioptions.IOStreams) *SetLastAppliedOptions {
	return &SetLastAppliedOptions{
		PrintFlags: genericclioptions.NewPrintFlags("configured").WithTypeSetter(scheme.Scheme),
		IOStreams:  ioStreams,
	}
}

func NewCmdApplySetLastApplied(f cmdutil.Factory, ioStreams genericclioptions.IOStreams) *cobra.Command {
	o := NewSetLastAppliedOptions(ioStreams)
	cmd := &cobra.Command{
		Use:                   "set-last-applied -f FILENAME",
		DisableFlagsInUseLine: true,
		Short:                 i18n.T("Set the last-applied-configuration annotation on a live object to match the contents of a file."),
		Long:                  applySetLastAppliedLong,
		Example:               applySetLastAppliedExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(o.Complete(f, cmd))
			cmdutil.CheckErr(o.Validate())
			cmdutil.CheckErr(o.RunSetLastApplied())
		},
	}

	o.PrintFlags.AddFlags(cmd)

	cmdutil.AddDryRunFlag(cmd)
	cmd.Flags().BoolVar(&o.CreateAnnotation, "create-annotation", o.CreateAnnotation, "Will create 'last-applied-configuration' annotations if current objects doesn't have one")
	cmdutil.AddJsonFilenameFlag(cmd.Flags(), &o.FilenameOptions.Filenames, "Filename, directory, or URL to files that contains the last-applied-configuration annotations")

	return cmd
}

func (o *SetLastAppliedOptions) Complete(f cmdutil.Factory, cmd *cobra.Command) error {
	o.dryRun = cmdutil.GetDryRunFlag(cmd)
	o.output = cmdutil.GetFlagString(cmd, "output")
	o.shortOutput = o.output == "name"

	var err error
	o.namespace, o.enforceNamespace, err = f.ToRawKubeConfigLoader().Namespace()
	if err != nil {
		return err
	}
	o.builder = f.NewBuilder()
	o.unstructuredClientForMapping = f.UnstructuredClientForMapping

	if o.dryRun {
		// TODO(juanvallejo): This can be cleaned up even further by creating
		// a PrintFlags struct that binds the --dry-run flag, and whose
		// ToPrinter method returns a printer that understands how to print
		// this success message.
		o.PrintFlags.Complete("%s (dry run)")
	}
	printer, err := o.PrintFlags.ToPrinter()
	if err != nil {
		return err
	}
	o.PrintObj = printer.PrintObj

	return nil
}

func (o *SetLastAppliedOptions) Validate() error {
	r := o.builder.
		Unstructured().
		NamespaceParam(o.namespace).DefaultNamespace().
		FilenameParam(o.enforceNamespace, &o.FilenameOptions).
		Flatten().
		Do()

	err := r.Visit(func(info *resource.Info, err error) error {
		if err != nil {
			return err
		}
		patchBuf, diffBuf, patchType, err := editor.GetApplyPatch(info.Object.(runtime.Unstructured))
		if err != nil {
			return err
		}

		// Verify the object exists in the cluster before trying to patch it.
		if err := info.Get(); err != nil {
			if errors.IsNotFound(err) {
				return err
			}
			return cmdutil.AddSourceToErr(fmt.Sprintf("retrieving current configuration of:\n%s\nfrom server for:", info.String()), info.Source, err)
		}
		originalBuf, err := kubectl.GetOriginalConfiguration(info.Object)
		if err != nil {
			return cmdutil.AddSourceToErr(fmt.Sprintf("retrieving current configuration of:\n%s\nfrom server for:", info.String()), info.Source, err)
		}
		if originalBuf == nil && !o.CreateAnnotation {
			return fmt.Errorf("no last-applied-configuration annotation found on resource: %s, to create the annotation, run the command with --create-annotation", info.Name)
		}

		//only add to PatchBufferList when changed
		if !bytes.Equal(cmdutil.StripComments(originalBuf), cmdutil.StripComments(diffBuf)) {
			p := PatchBuffer{Patch: patchBuf, PatchType: patchType}
			o.patchBufferList = append(o.patchBufferList, p)
			o.infoList = append(o.infoList, info)
		} else {
			fmt.Fprintf(o.Out, "set-last-applied %s: no changes required.\n", info.Name)
		}

		return nil
	})
	return err
}

func (o *SetLastAppliedOptions) RunSetLastApplied() error {
	for i, patch := range o.patchBufferList {
		info := o.infoList[i]
		finalObj := info.Object

		if !o.dryRun {
			mapping := info.ResourceMapping()
			client, err := o.unstructuredClientForMapping(mapping)
			if err != nil {
				return err
			}
			helper := resource.NewHelper(client, mapping)
			finalObj, err = helper.Patch(o.namespace, info.Name, patch.PatchType, patch.Patch, nil)
			if err != nil {
				return err
			}
		}
		if err := o.PrintObj(finalObj, o.Out); err != nil {
			return err
		}
	}
	return nil
}
