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

package create

import (
	"fmt"

	"github.com/spf13/cobra"

	batchv1 "k8s.io/api/batch/v1"
	batchv1beta1 "k8s.io/api/batch/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	clientbatchv1 "k8s.io/client-go/kubernetes/typed/batch/v1"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/genericclioptions"
	"k8s.io/kubernetes/pkg/kubectl/genericclioptions/resource"
	"k8s.io/kubernetes/pkg/kubectl/scheme"
	"k8s.io/kubernetes/pkg/kubectl/util/i18n"
)

var (
	jobLong = templates.LongDesc(i18n.T(`
		Create a job with the specified name.`))

	jobExample = templates.Examples(i18n.T(`
		# Create a job from a CronJob named "a-cronjob"
		kubectl create job test-job --from=cronjob/a-cronjob`))
)

type CreateJobOptions struct {
	PrintFlags *PrintFlags

	PrintObj func(obj runtime.Object) error

	Name string
	From string

	Namespace    string
	OutputFormat string
	Client       clientbatchv1.BatchV1Interface
	DryRun       bool
	Builder      *resource.Builder
	Cmd          *cobra.Command

	genericclioptions.IOStreams
}

func NewCreateJobOptions(ioStreams genericclioptions.IOStreams) *CreateJobOptions {
	return &CreateJobOptions{
		PrintFlags: NewPrintFlags("created", legacyscheme.Scheme),
		IOStreams:  ioStreams,
	}
}

// NewCmdCreateJob is a command to ease creating Jobs from CronJobs.
func NewCmdCreateJob(f cmdutil.Factory, ioStreams genericclioptions.IOStreams) *cobra.Command {
	o := NewCreateJobOptions(ioStreams)

	cmd := &cobra.Command{
		Use:     "job NAME [--from=CRONJOB]",
		Short:   jobLong,
		Long:    jobLong,
		Example: jobExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(o.Complete(f, cmd, args))
			cmdutil.CheckErr(o.RunCreateJob())
		},
	}

	o.PrintFlags.AddFlags(cmd)

	cmdutil.AddApplyAnnotationFlags(cmd)
	cmdutil.AddValidateFlags(cmd)
	cmdutil.AddDryRunFlag(cmd)
	cmd.Flags().StringVar(&o.From, "from", o.From, "The name of the resource to create a Job from (only cronjob is supported).")

	return cmd
}

func (o *CreateJobOptions) Complete(f cmdutil.Factory, cmd *cobra.Command, args []string) (err error) {
	if len(args) == 0 {
		return cmdutil.UsageErrorf(cmd, "NAME is required")
	}
	o.Name = args[0]

	o.Namespace, _, err = f.ToRawKubeConfigLoader().Namespace()
	if err != nil {
		return err
	}

	clientset, err := f.KubernetesClientSet()
	if err != nil {
		return err
	}
	o.Client = clientset.BatchV1()
	o.Builder = f.NewBuilder()
	o.DryRun = cmdutil.GetDryRunFlag(cmd)
	o.Cmd = cmd
	o.OutputFormat = cmdutil.GetFlagString(cmd, "output")

	if o.DryRun {
		o.PrintFlags.Complete("%s (dry run)")
	}
	printer, err := o.PrintFlags.ToPrinter()
	if err != nil {
		return err
	}

	o.PrintObj = func(obj runtime.Object) error {
		return printer.PrintObj(obj, o.Out)
	}

	return nil
}

func (o *CreateJobOptions) RunCreateJob() error {
	infos, err := o.Builder.
		Unstructured().
		NamespaceParam(o.Namespace).DefaultNamespace().
		ResourceTypeOrNameArgs(false, o.From).
		Flatten().
		Latest().
		Do().
		Infos()
	if err != nil {
		return err
	}
	if len(infos) != 1 {
		return fmt.Errorf("from must be an existing cronjob")
	}

	uncastVersionedObj, err := scheme.Scheme.ConvertToVersion(infos[0].Object, batchv1beta1.SchemeGroupVersion)
	if err != nil {
		return fmt.Errorf("from must be an existing cronjob: %v", err)
	}
	cronJob, ok := uncastVersionedObj.(*batchv1beta1.CronJob)
	if !ok {
		return fmt.Errorf("from must be an existing cronjob")
	}

	return o.createJob(cronJob)
}

func (o *CreateJobOptions) createJob(cronJob *batchv1beta1.CronJob) error {
	annotations := make(map[string]string)
	annotations["cronjob.kubernetes.io/instantiate"] = "manual"
	for k, v := range cronJob.Spec.JobTemplate.Annotations {
		annotations[k] = v
	}
	job := &batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{
			Name:        o.Name,
			Namespace:   o.Namespace,
			Annotations: annotations,
			Labels:      cronJob.Spec.JobTemplate.Labels,
		},
		Spec: cronJob.Spec.JobTemplate.Spec,
	}

	if !o.DryRun {
		var err error
		job, err = o.Client.Jobs(o.Namespace).Create(job)
		if err != nil {
			return fmt.Errorf("failed to create job: %v", err)
		}
	}

	return o.PrintObj(job)
}
