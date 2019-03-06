/*
Copyright 2014 The Kubernetes Authors.

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

package replace

import (
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"time"

	"github.com/spf13/cobra"

	"k8s.io/klog"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/cli-runtime/pkg/resource"
	"k8s.io/kubernetes/pkg/kubectl"
	"k8s.io/kubernetes/pkg/kubectl/cmd/delete"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/scheme"
	"k8s.io/kubernetes/pkg/kubectl/util/i18n"
	"k8s.io/kubernetes/pkg/kubectl/util/templates"
	"k8s.io/kubernetes/pkg/kubectl/validation"
)

var (
	replaceLong = templates.LongDesc(i18n.T(`
		Replace a resource by filename or stdin.

		JSON and YAML formats are accepted. If replacing an existing resource, the
		complete resource spec must be provided. This can be obtained by

		    $ kubectl get TYPE NAME -o yaml`))

	replaceExample = templates.Examples(i18n.T(`
		# Replace a pod using the data in pod.json.
		kubectl replace -f ./pod.json

		# Replace a pod based on the JSON passed into stdin.
		cat pod.json | kubectl replace -f -

		# Update a single-container pod's image version (tag) to v4
		kubectl get pod mypod -o yaml | sed 's/\(image: myimage\):.*$/\1:v4/' | kubectl replace -f -

		# Force replace, delete and then re-create the resource
		kubectl replace --force -f ./pod.json`))
)

type ReplaceOptions struct {
	PrintFlags  *genericclioptions.PrintFlags
	RecordFlags *genericclioptions.RecordFlags

	DeleteFlags   *delete.DeleteFlags
	DeleteOptions *delete.DeleteOptions

	PrintObj func(obj runtime.Object) error

	createAnnotation bool
	validate         bool

	Schema      validation.Schema
	Builder     func() *resource.Builder
	BuilderArgs []string

	Namespace        string
	EnforceNamespace bool

	Recorder genericclioptions.Recorder

	genericclioptions.IOStreams
}

func NewReplaceOptions(streams genericclioptions.IOStreams) *ReplaceOptions {
	return &ReplaceOptions{
		PrintFlags:  genericclioptions.NewPrintFlags("replaced"),
		DeleteFlags: delete.NewDeleteFlags("to use to replace the resource."),

		IOStreams: streams,
	}
}

func NewCmdReplace(f cmdutil.Factory, streams genericclioptions.IOStreams) *cobra.Command {
	o := NewReplaceOptions(streams)

	cmd := &cobra.Command{
		Use:                   "replace -f FILENAME",
		DisableFlagsInUseLine: true,
		Short:                 i18n.T("Replace a resource by filename or stdin"),
		Long:                  replaceLong,
		Example:               replaceExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(o.Complete(f, cmd, args))
			cmdutil.CheckErr(o.Validate(cmd))
			cmdutil.CheckErr(o.Run())
		},
	}

	o.PrintFlags.AddFlags(cmd)
	o.DeleteFlags.AddFlags(cmd)
	o.RecordFlags.AddFlags(cmd)

	cmdutil.AddValidateFlags(cmd)
	cmdutil.AddApplyAnnotationFlags(cmd)

	return cmd
}

func (o *ReplaceOptions) Complete(f cmdutil.Factory, cmd *cobra.Command, args []string) error {
	var err error

	o.RecordFlags.Complete(cmd)
	o.Recorder, err = o.RecordFlags.ToRecorder()
	if err != nil {
		return err
	}

	o.validate = cmdutil.GetFlagBool(cmd, "validate")
	o.createAnnotation = cmdutil.GetFlagBool(cmd, cmdutil.ApplyAnnotationsFlag)

	printer, err := o.PrintFlags.ToPrinter()
	if err != nil {
		return err
	}
	o.PrintObj = func(obj runtime.Object) error {
		return printer.PrintObj(obj, o.Out)
	}

	dynamicClient, err := f.DynamicClient()
	if err != nil {
		return err
	}
	deleteOpts := o.DeleteFlags.ToOptions(dynamicClient, o.IOStreams)

	//Replace will create a resource if it doesn't exist already, so ignore not found error
	deleteOpts.IgnoreNotFound = true
	if o.PrintFlags.OutputFormat != nil {
		deleteOpts.Output = *o.PrintFlags.OutputFormat
	}
	if deleteOpts.GracePeriod == 0 {
		// To preserve backwards compatibility, but prevent accidental data loss, we convert --grace-period=0
		// into --grace-period=1 and wait until the object is successfully deleted.
		deleteOpts.GracePeriod = 1
		deleteOpts.WaitForDeletion = true
	}
	o.DeleteOptions = deleteOpts

	err = o.DeleteOptions.FilenameOptions.RequireFilenameOrKustomize()
	if err != nil {
		return err
	}

	schema, err := f.Validator(o.validate)
	if err != nil {
		return err
	}

	o.Schema = schema
	o.Builder = f.NewBuilder
	o.BuilderArgs = args

	o.Namespace, o.EnforceNamespace, err = f.ToRawKubeConfigLoader().Namespace()
	if err != nil {
		return err
	}

	return nil
}

func (o *ReplaceOptions) Validate(cmd *cobra.Command) error {
	if o.DeleteOptions.GracePeriod >= 0 && !o.DeleteOptions.ForceDeletion {
		return fmt.Errorf("--grace-period must have --force specified")
	}

	if o.DeleteOptions.Timeout != 0 && !o.DeleteOptions.ForceDeletion {
		return fmt.Errorf("--timeout must have --force specified")
	}

	if cmdutil.IsFilenameSliceEmpty(o.DeleteOptions.FilenameOptions.Filenames, o.DeleteOptions.FilenameOptions.Kustomize) {
		return cmdutil.UsageErrorf(cmd, "Must specify --filename to replace")
	}

	return nil
}

func (o *ReplaceOptions) Run() error {
	if o.DeleteOptions.ForceDeletion {
		return o.forceReplace()
	}

	r := o.Builder().
		Unstructured().
		Schema(o.Schema).
		ContinueOnError().
		NamespaceParam(o.Namespace).DefaultNamespace().
		FilenameParam(o.EnforceNamespace, &o.DeleteOptions.FilenameOptions).
		Flatten().
		Do()
	if err := r.Err(); err != nil {
		return err
	}

	return r.Visit(func(info *resource.Info, err error) error {
		if err != nil {
			return err
		}

		if err := kubectl.CreateOrUpdateAnnotation(o.createAnnotation, info.Object, scheme.DefaultJSONEncoder()); err != nil {
			return cmdutil.AddSourceToErr("replacing", info.Source, err)
		}

		if err := o.Recorder.Record(info.Object); err != nil {
			klog.V(4).Infof("error recording current command: %v", err)
		}

		// Serialize the object with the annotation applied.
		obj, err := resource.NewHelper(info.Client, info.Mapping).Replace(info.Namespace, info.Name, true, info.Object)
		if err != nil {
			return cmdutil.AddSourceToErr("replacing", info.Source, err)
		}

		info.Refresh(obj, true)
		return o.PrintObj(info.Object)
	})
}

func (o *ReplaceOptions) forceReplace() error {
	for i, filename := range o.DeleteOptions.FilenameOptions.Filenames {
		if filename == "-" {
			tempDir, err := ioutil.TempDir("", "kubectl_replace_")
			if err != nil {
				return err
			}
			defer os.RemoveAll(tempDir)
			tempFilename := filepath.Join(tempDir, "resource.stdin")
			err = cmdutil.DumpReaderToFile(os.Stdin, tempFilename)
			if err != nil {
				return err
			}
			o.DeleteOptions.FilenameOptions.Filenames[i] = tempFilename
		}
	}

	r := o.Builder().
		Unstructured().
		ContinueOnError().
		NamespaceParam(o.Namespace).DefaultNamespace().
		ResourceTypeOrNameArgs(false, o.BuilderArgs...).RequireObject(false).
		FilenameParam(o.EnforceNamespace, &o.DeleteOptions.FilenameOptions).
		Flatten().
		Do()
	if err := r.Err(); err != nil {
		return err
	}

	if err := o.DeleteOptions.DeleteResult(r); err != nil {
		return err
	}

	timeout := o.DeleteOptions.Timeout
	if timeout == 0 {
		timeout = 5 * time.Minute
	}
	err := r.Visit(func(info *resource.Info, err error) error {
		if err != nil {
			return err
		}

		return wait.PollImmediate(1*time.Second, timeout, func() (bool, error) {
			if err := info.Get(); !errors.IsNotFound(err) {
				return false, err
			}
			return true, nil
		})
	})
	if err != nil {
		return err
	}

	r = o.Builder().
		Unstructured().
		Schema(o.Schema).
		ContinueOnError().
		NamespaceParam(o.Namespace).DefaultNamespace().
		FilenameParam(o.EnforceNamespace, &o.DeleteOptions.FilenameOptions).
		Flatten().
		Do()
	err = r.Err()
	if err != nil {
		return err
	}

	count := 0
	err = r.Visit(func(info *resource.Info, err error) error {
		if err != nil {
			return err
		}

		if err := kubectl.CreateOrUpdateAnnotation(o.createAnnotation, info.Object, scheme.DefaultJSONEncoder()); err != nil {
			return err
		}

		if err := o.Recorder.Record(info.Object); err != nil {
			klog.V(4).Infof("error recording current command: %v", err)
		}

		obj, err := resource.NewHelper(info.Client, info.Mapping).Create(info.Namespace, true, info.Object, nil)
		if err != nil {
			return err
		}

		count++
		info.Refresh(obj, true)
		return o.PrintObj(info.Object)
	})
	if err != nil {
		return err
	}
	if count == 0 {
		return fmt.Errorf("no objects passed to replace")
	}
	return nil
}
