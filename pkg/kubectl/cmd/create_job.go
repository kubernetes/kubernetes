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

package cmd

import (
	"fmt"
	"io"

	"github.com/spf13/cobra"

	"k8s.io/api/batch/v1beta1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/batch/internalversion"
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

	OutputFormat string
	Namespace    string
	Client       internalversion.BatchInterface
	Mapper       meta.RESTMapper
	Out          io.Writer
	PrintObject  func(obj runtime.Object) error
}

// Job is a command to ease creating Jobs from CronJobs.
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
			cmdutil.CheckErr(c.Validate())
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
		return f.PrintObject(cmd, false, c.Mapper, obj, c.Out)
	}

	clientSet, err := f.ClientSetForVersion(&schema.GroupVersion{Group: "batch", Version: "v2alpha1"})
	if err != nil {
		return err
	}
	c.Client = clientSet.Batch()

	return nil
}

func (c *CreateJobOptions) Validate() error {
	_, err := c.Client.CronJobs(c.Namespace).Get(c.FromCronJob, metav1.GetOptions{})
	if err != nil {
		return fmt.Errorf("Could not find source cronjob: %v", err)
	}

	return nil
}

func (c *CreateJobOptions) RunCreateJob() (err error) {
	// the passed-in CJMI is not used for anything but client-gen's code requires it
	result, err := c.Client.CronJobs(c.Namespace).Instantiate(c.FromCronJob, &v1beta1.CronJobManualInstantiation{})
	if err != nil {
		return err
	}

	if useShortOutput := c.OutputFormat == "name"; useShortOutput || len(c.OutputFormat) == 0 {
		cmdutil.PrintSuccess(c.Mapper, useShortOutput, c.Out, "jobs", result.CreatedJob.Name, false, "created")
		return nil
	}

	return c.PrintObject(result)
}
