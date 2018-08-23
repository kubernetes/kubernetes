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
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/cli-runtime/pkg/genericclioptions/resource"
	batchv1client "k8s.io/client-go/kubernetes/typed/batch/v1"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/scheme"
	"k8s.io/kubernetes/pkg/kubectl/util/i18n"
)

var (
	jobLong = templates.LongDesc(i18n.T(`
		Create a job with the specified name.`))

	jobExample = templates.Examples(i18n.T(`
		# Create a job
		kubectl create job my-job --image=busybox

		# Create a job with command
		kubectl create job my-job --image=busybox -- date

		# Create a job from a CronJob named "a-cronjob"
		kubectl create job test-job --from=cronjob/a-cronjob`))
)

type CreateJobOptions struct {
	PrintFlags *genericclioptions.PrintFlags

	PrintObj func(obj runtime.Object) error

	Name    string
	Image   string
	From    string
	Command []string

	Namespace string
	Client    batchv1client.BatchV1Interface
	DryRun    bool
	Builder   *resource.Builder
	Cmd       *cobra.Command

	genericclioptions.IOStreams
}

func NewCreateJobOptions(ioStreams genericclioptions.IOStreams) *CreateJobOptions {
	return &CreateJobOptions{
		PrintFlags: genericclioptions.NewPrintFlags("created").WithTypeSetter(scheme.Scheme),
		IOStreams:  ioStreams,
	}
}

// NewCmdCreateJob is a command to ease creating Jobs from CronJobs.
func NewCmdCreateJob(f cmdutil.Factory, ioStreams genericclioptions.IOStreams) *cobra.Command {
	o := NewCreateJobOptions(ioStreams)
	cmd := &cobra.Command{
		Use:     "job NAME [--image=image --from=cronjob/name] -- [COMMAND] [args...]",
		Short:   jobLong,
		Long:    jobLong,
		Example: jobExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(o.Complete(f, cmd, args))
			cmdutil.CheckErr(o.Validate())
			cmdutil.CheckErr(o.Run())
		},
	}

	o.PrintFlags.AddFlags(cmd)

	cmdutil.AddApplyAnnotationFlags(cmd)
	cmdutil.AddValidateFlags(cmd)
	cmdutil.AddDryRunFlag(cmd)
	cmd.Flags().StringVar(&o.Image, "image", o.Image, "Image name to run.")
	cmd.Flags().StringVar(&o.From, "from", o.From, "The name of the resource to create a Job from (only cronjob is supported).")

	return cmd
}

func (o *CreateJobOptions) Complete(f cmdutil.Factory, cmd *cobra.Command, args []string) error {
	name, err := NameFromCommandArgs(cmd, args)
	if err != nil {
		return err
	}
	o.Name = name
	if len(args) > 1 {
		o.Command = args[1:]
	}

	clientConfig, err := f.ToRESTConfig()
	if err != nil {
		return err
	}
	o.Client, err = batchv1client.NewForConfig(clientConfig)
	if err != nil {
		return err
	}

	o.Namespace, _, err = f.ToRawKubeConfigLoader().Namespace()
	if err != nil {
		return err
	}
	o.Builder = f.NewBuilder()
	o.Cmd = cmd

	o.DryRun = cmdutil.GetDryRunFlag(cmd)
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

func (o *CreateJobOptions) Validate() error {
	if (len(o.Image) == 0 && len(o.From) == 0) || (len(o.Image) != 0 && len(o.From) != 0) {
		return fmt.Errorf("either --image or --from must be specified")
	}
	if o.Command != nil && len(o.Command) != 0 && len(o.From) != 0 {
		return fmt.Errorf("cannot specify --from and command")
	}
	return nil
}

func (o *CreateJobOptions) Run() error {
	var job *batchv1.Job
	if len(o.Image) > 0 {
		job = o.createJob()
	} else {
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

		job = o.createJobFromCronJob(cronJob)
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

func (o *CreateJobOptions) createJob() *batchv1.Job {
	return &batchv1.Job{
		// this is ok because we know exactly how we want to be serialized
		TypeMeta: metav1.TypeMeta{APIVersion: batchv1.SchemeGroupVersion.String(), Kind: "Job"},
		ObjectMeta: metav1.ObjectMeta{
			Name: o.Name,
		},
		Spec: batchv1.JobSpec{
			Template: corev1.PodTemplateSpec{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name:    o.Name,
							Image:   o.Image,
							Command: o.Command,
						},
					},
					RestartPolicy: corev1.RestartPolicyNever,
				},
			},
		},
	}
}

func (o *CreateJobOptions) createJobFromCronJob(cronJob *batchv1beta1.CronJob) *batchv1.Job {
	annotations := make(map[string]string)
	annotations["cronjob.kubernetes.io/instantiate"] = "manual"
	for k, v := range cronJob.Spec.JobTemplate.Annotations {
		annotations[k] = v
	}
	return &batchv1.Job{
		// this is ok because we know exactly how we want to be serialized
		TypeMeta: metav1.TypeMeta{APIVersion: batchv1.SchemeGroupVersion.String(), Kind: "Job"},
		ObjectMeta: metav1.ObjectMeta{
			Name:        o.Name,
			Annotations: annotations,
			Labels:      cronJob.Spec.JobTemplate.Labels,
		},
		Spec: cronJob.Spec.JobTemplate.Spec,
	}
}
