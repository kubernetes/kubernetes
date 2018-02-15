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

package cmd

import (
	"fmt"
	"io"

	"github.com/spf13/cobra"

	batchv1 "k8s.io/api/batch/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/rand"
	clientbatchv1 "k8s.io/client-go/kubernetes/typed/batch/v1"
	clientbatchv1beta1 "k8s.io/client-go/kubernetes/typed/batch/v1beta1"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/util/i18n"
)

var (
	jobLong = templates.LongDesc(i18n.T(`
		Create a Job to execute immediately from a specified CronJob.`))

	jobExample = templates.Examples(i18n.T(`
		# Create a Job from a CronJob named "a-cronjob"
		kubectl create job --from-cronjob=a-cronjob`))
)

type CreateJobOptions struct {
	FromCronJob string

	OutputFormat  string
	Namespace     string
	V1Beta1Client clientbatchv1beta1.BatchV1beta1Interface
	V1Client      clientbatchv1.BatchV1Interface
	Mapper        meta.RESTMapper
	Out           io.Writer
	PrintObject   func(obj runtime.Object) error
	PrintSuccess  func(mapper meta.RESTMapper, shortOutput bool, out io.Writer, resource, name string, dryRun bool, operation string)
}

// NewCmdCreateJob is a command to ease creating Jobs from CronJobs.
func NewCmdCreateJob(f cmdutil.Factory, cmdOut io.Writer) *cobra.Command {
	c := &CreateJobOptions{
		Out: cmdOut,
	}
	cmd := &cobra.Command{
		Use:     "job --from-cronjob=CRONJOB",
		Short:   jobLong,
		Long:    jobLong,
		Example: jobExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(c.Complete(f, cmd, args))
			cmdutil.CheckErr(c.RunCreateJob())
		},
	}
	cmdutil.AddApplyAnnotationFlags(cmd)
	cmdutil.AddValidateFlags(cmd)
	cmdutil.AddPrinterFlags(cmd)
	cmd.Flags().String("from-cronjob", "", "Specify the name of the CronJob to create a Job from.")

	return cmd
}

func (c *CreateJobOptions) Complete(f cmdutil.Factory, cmd *cobra.Command, args []string) (err error) {
	c.FromCronJob = cmdutil.GetFlagString(cmd, "from-cronjob")

	// Complete other options for Run.
	c.Mapper, _ = f.Object()

	c.OutputFormat = cmdutil.GetFlagString(cmd, "output")

	c.Namespace, _, err = f.DefaultNamespace()
	if err != nil {
		return err
	}

	c.PrintObject = func(obj runtime.Object) error {
		return f.PrintObject(cmd, obj, c.Out)
	}

	// need two client sets to deal with the differing versions of CronJobs and Jobs
	clientset, err := f.KubernetesClientSet()
	if err != nil {
		return err
	}
	if c.V1Client == nil {
		c.V1Client = clientset.BatchV1()
	}
	if c.V1Beta1Client == nil {
		c.V1Beta1Client = clientset.BatchV1beta1()
	}

	return nil
}

func (c *CreateJobOptions) RunCreateJob() (err error) {
	cronjob, err := c.V1Beta1Client.CronJobs(c.Namespace).Get(c.FromCronJob, metav1.GetOptions{})

	if err != nil {
		return fmt.Errorf("failed to fetch job: %v", err)
	}

	annotations := make(map[string]string)
	annotations["cronjob.kubernetes.io/instantiate"] = "manual"

	labels := make(map[string]string)
	for k, v := range cronjob.Spec.JobTemplate.Labels {
		labels[k] = v
	}

	jobToCreate := &batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{
			// job name cannot exceed DNS1053LabelMaxLength (52 characters)
			Name:        cronjob.Name + "-manual-" + rand.String(3),
			Namespace:   c.Namespace,
			Annotations: annotations,
			Labels:      labels,
		},
		Spec: cronjob.Spec.JobTemplate.Spec,
	}

	result, err := c.V1Client.Jobs(c.Namespace).Create(jobToCreate)

	if err != nil {
		return fmt.Errorf("failed to create job: %v", err)
	}

	return c.PrintObject(result)
}
