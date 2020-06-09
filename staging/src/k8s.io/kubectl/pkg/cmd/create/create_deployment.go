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

package create

import (
	"context"
	"fmt"
	"strings"

	"github.com/spf13/cobra"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	utilrand "k8s.io/apimachinery/pkg/util/rand"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/cli-runtime/pkg/resource"
	appsv1client "k8s.io/client-go/kubernetes/typed/apps/v1"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/scheme"
	"k8s.io/kubectl/pkg/util"
	"k8s.io/kubectl/pkg/util/i18n"
	"k8s.io/kubectl/pkg/util/templates"
)

var (
	deploymentLong = templates.LongDesc(i18n.T(`
	Create a deployment with the specified name.`))

	deploymentExample = templates.Examples(i18n.T(`
	# Create a deployment named my-dep that runs the busybox image.
	kubectl create deployment my-dep --image=busybox

	# Create a deployment with command
	kubectl create deployment my-dep --image=busybox -- date

	# Create a deployment named my-dep that runs the nginx image with 3 replicas.
	kubectl create deployment my-dep --image=nginx --replicas=3

	# Create a deployment named my-dep that runs the busybox image and expose port 5701.
	kubectl create deployment my-dep --image=busybox --port=5701`))
)

// CreateDeploymentOptions is returned by NewCmdCreateDeployment
type CreateDeploymentOptions struct {
	PrintFlags *genericclioptions.PrintFlags

	PrintObj func(obj runtime.Object) error

	Name             string
	Images           []string
	Port             int32
	Replicas         int32
	Command          []string
	Namespace        string
	EnforceNamespace bool
	FieldManager     string
	CreateAnnotation bool

	Client         appsv1client.AppsV1Interface
	DryRunStrategy cmdutil.DryRunStrategy
	DryRunVerifier *resource.DryRunVerifier

	genericclioptions.IOStreams
}

func NewCreateDeploymentOptions(ioStreams genericclioptions.IOStreams) *CreateDeploymentOptions {
	return &CreateDeploymentOptions{
		Port:       -1,
		Replicas:   1,
		PrintFlags: genericclioptions.NewPrintFlags("created").WithTypeSetter(scheme.Scheme),
		IOStreams:  ioStreams,
	}
}

// NewCmdCreateDeployment is a macro command to create a new deployment.
// This command is better known to users as `kubectl create deployment`.
func NewCmdCreateDeployment(f cmdutil.Factory, ioStreams genericclioptions.IOStreams) *cobra.Command {
	o := NewCreateDeploymentOptions(ioStreams)
	cmd := &cobra.Command{
		Use:                   "deployment NAME --image=image -- [COMMAND] [args...]",
		DisableFlagsInUseLine: true,
		Aliases:               []string{"deploy"},
		Short:                 deploymentLong,
		Long:                  deploymentLong,
		Example:               deploymentExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(o.Complete(f, cmd, args))
			cmdutil.CheckErr(o.Validate())
			cmdutil.CheckErr(o.Run())
		},
	}

	o.PrintFlags.AddFlags(cmd)

	cmdutil.AddApplyAnnotationFlags(cmd)
	cmdutil.AddValidateFlags(cmd)
	cmdutil.AddGeneratorFlags(cmd, "")
	cmd.Flags().StringSliceVar(&o.Images, "image", o.Images, "Image names to run.")
	cmd.MarkFlagRequired("image")
	cmd.Flags().Int32Var(&o.Port, "port", o.Port, "The port that this container exposes.")
	cmd.Flags().Int32VarP(&o.Replicas, "replicas", "r", o.Replicas, "Number of replicas to create. Default is 1.")
	cmdutil.AddFieldManagerFlagVar(cmd, &o.FieldManager, "kubectl-create")

	return cmd
}

// Complete completes all the options
func (o *CreateDeploymentOptions) Complete(f cmdutil.Factory, cmd *cobra.Command, args []string) error {
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
	o.Client, err = appsv1client.NewForConfig(clientConfig)
	if err != nil {
		return err
	}

	o.Namespace, o.EnforceNamespace, err = f.ToRawKubeConfigLoader().Namespace()
	if err != nil {
		return err
	}

	o.CreateAnnotation = cmdutil.GetFlagBool(cmd, cmdutil.ApplyAnnotationsFlag)

	o.DryRunStrategy, err = cmdutil.GetDryRunStrategy(cmd)
	if err != nil {
		return err
	}
	dynamicClient, err := f.DynamicClient()
	if err != nil {
		return err
	}
	discoveryClient, err := f.ToDiscoveryClient()
	if err != nil {
		return err
	}
	o.DryRunVerifier = resource.NewDryRunVerifier(dynamicClient, discoveryClient)
	cmdutil.PrintFlagsWithDryRunStrategy(o.PrintFlags, o.DryRunStrategy)

	printer, err := o.PrintFlags.ToPrinter()
	if err != nil {
		return err
	}
	o.PrintObj = func(obj runtime.Object) error {
		return printer.PrintObj(obj, o.Out)
	}

	return nil
}

func (o *CreateDeploymentOptions) Validate() error {
	if len(o.Images) > 1 && len(o.Command) > 0 {
		return fmt.Errorf("cannot specify multiple --image options and command")
	}
	return nil
}

// Run performs the execution of 'create deployment' sub command
func (o *CreateDeploymentOptions) Run() error {
	deploy := o.createDeployment()

	if err := util.CreateOrUpdateAnnotation(o.CreateAnnotation, deploy, scheme.DefaultJSONEncoder()); err != nil {
		return err
	}

	if o.DryRunStrategy != cmdutil.DryRunClient {
		createOptions := metav1.CreateOptions{}
		if o.FieldManager != "" {
			createOptions.FieldManager = o.FieldManager
		}
		if o.DryRunStrategy == cmdutil.DryRunServer {
			if err := o.DryRunVerifier.HasSupport(deploy.GroupVersionKind()); err != nil {
				return err
			}
			createOptions.DryRun = []string{metav1.DryRunAll}
		}
		var err error
		deploy, err = o.Client.Deployments(o.Namespace).Create(context.TODO(), deploy, createOptions)
		if err != nil {
			return fmt.Errorf("failed to create deployment: %v", err)
		}
	}

	return o.PrintObj(deploy)
}

func (o *CreateDeploymentOptions) createDeployment() *appsv1.Deployment {
	labels := map[string]string{"app": o.Name}
	selector := metav1.LabelSelector{MatchLabels: labels}
	namespace := ""
	if o.EnforceNamespace {
		namespace = o.Namespace
	}

	deploy := &appsv1.Deployment{
		TypeMeta: metav1.TypeMeta{APIVersion: appsv1.SchemeGroupVersion.String(), Kind: "Deployment"},
		ObjectMeta: metav1.ObjectMeta{
			Name:      o.Name,
			Labels:    labels,
			Namespace: namespace,
		},
		Spec: appsv1.DeploymentSpec{
			Replicas: &o.Replicas,
			Selector: &selector,
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: labels,
				},
				Spec: o.buildPodSpec(),
			},
		},
	}

	if o.Port >= 0 && len(deploy.Spec.Template.Spec.Containers) > 0 {
		deploy.Spec.Template.Spec.Containers[0].Ports = []corev1.ContainerPort{{ContainerPort: o.Port}}
	}
	return deploy
}

// buildPodSpec parses the image strings and assemble them into the Containers
// of a PodSpec. This is all you need to create the PodSpec for a deployment.
func (o *CreateDeploymentOptions) buildPodSpec() corev1.PodSpec {
	podSpec := corev1.PodSpec{Containers: []corev1.Container{}}
	for _, imageString := range o.Images {
		// Retain just the image name
		imageSplit := strings.Split(imageString, "/")
		name := imageSplit[len(imageSplit)-1]
		// Remove any tag or hash
		if strings.Contains(name, ":") {
			name = strings.Split(name, ":")[0]
		}
		if strings.Contains(name, "@") {
			name = strings.Split(name, "@")[0]
		}
		name = sanitizeAndUniquify(name)
		podSpec.Containers = append(podSpec.Containers, corev1.Container{
			Name:    name,
			Image:   imageString,
			Command: o.Command,
		})
	}
	return podSpec
}

// sanitizeAndUniquify replaces characters like "." or "_" into "-" to follow DNS1123 rules.
// Then add random suffix to make it uniquified.
func sanitizeAndUniquify(name string) string {
	if strings.ContainsAny(name, "_.") {
		name = strings.Replace(name, "_", "-", -1)
		name = strings.Replace(name, ".", "-", -1)
		name = fmt.Sprintf("%s-%s", name, utilrand.String(5))
	}
	return name
}
