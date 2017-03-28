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

package auth

import (
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"strings"

	"github.com/spf13/cobra"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	authorizationapi "k8s.io/kubernetes/pkg/apis/authorization"
	internalauthorizationclient "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/authorization/internalversion"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
)

// CanIOptions is the start of the data required to perform the operation.  As new fields are added, add them here instead of
// referencing the cmd.Flags()
type CanIOptions struct {
	AllNamespaces bool
	Quiet         bool
	Namespace     string
	SelfSARClient internalauthorizationclient.SelfSubjectAccessReviewsGetter

	Verb         string
	Resource     schema.GroupVersionResource
	ResourceName string

	Out io.Writer
	Err io.Writer
}

var (
	canILong = templates.LongDesc(`
		Check whether an action is allowed.

		VERB is a logical Kubernetes API verb like 'get', 'list', 'watch', 'delete', etc.
		TYPE is a Kubernetes resource.  Shortcuts and groups will be resolved.
		NAME is the name of a particular Kubernetes resource.`)

	canIExample = templates.Examples(`
		# Check to see if I can create pods in any namespace
		kubectl auth can-i create pods --all-namespaces

		# Check to see if I can list deployments in my current namespace
		kubectl auth can-i list deployments.extensions

		# Check to see if I can get the job named "bar" in namespace "foo"
		kubectl auth can-i list jobs.batch/bar -n foo`)
)

func NewCmdCanI(f cmdutil.Factory, out, err io.Writer) *cobra.Command {
	o := &CanIOptions{
		Out: out,
		Err: err,
	}

	cmd := &cobra.Command{
		Use:     "can-i VERB [TYPE | TYPE/NAME]",
		Short:   "Check whether an action is allowed",
		Long:    canILong,
		Example: canIExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(o.Complete(f, args))
			cmdutil.CheckErr(o.Validate())

			allowed, err := o.RunAccessCheck()
			if err == nil {
				return
			}

			if o.Quiet && !allowed {
				os.Exit(1)
			}

			cmdutil.CheckErr(err)
		},
	}

	cmd.Flags().BoolVar(&o.AllNamespaces, "all-namespaces", o.AllNamespaces, "If true, check the specified action in all namespaces.")
	cmd.Flags().BoolVarP(&o.Quiet, "quiet", "q", o.Quiet, "If true, suppress output and just return the exit code.")
	return cmd
}

func (o *CanIOptions) Complete(f cmdutil.Factory, args []string) error {
	switch len(args) {
	case 2:
		resourceTokens := strings.SplitN(args[1], "/", 2)
		restMapper, _ := f.Object()
		o.Verb = args[0]
		o.Resource = resourceFor(restMapper, resourceTokens[0])
		if len(resourceTokens) > 1 {
			o.ResourceName = resourceTokens[1]
		}
	default:
		return errors.New("you must specify two or three arguments: verb, resource, and optional resourceName")
	}

	var err error
	client, err := f.ClientSet()
	if err != nil {
		return err
	}
	o.SelfSARClient = client.Authorization()

	o.Namespace = ""
	if !o.AllNamespaces {
		o.Namespace, _, err = f.DefaultNamespace()
		if err != nil {
			return err
		}
	}

	if o.Quiet {
		o.Out = ioutil.Discard
	}

	return nil
}

func (o *CanIOptions) Validate() error {
	errors := []error{}
	return utilerrors.NewAggregate(errors)
}

func (o *CanIOptions) RunAccessCheck() (bool, error) {
	sar := &authorizationapi.SelfSubjectAccessReview{
		Spec: authorizationapi.SelfSubjectAccessReviewSpec{
			ResourceAttributes: &authorizationapi.ResourceAttributes{
				Namespace: o.Namespace,
				Verb:      o.Verb,
				Group:     o.Resource.Group,
				Resource:  o.Resource.Resource,
				Name:      o.ResourceName,
			},
		},
	}

	response, err := o.SelfSARClient.SelfSubjectAccessReviews().Create(sar)
	if err != nil {
		return false, err
	}

	if response.Status.Allowed {
		fmt.Fprintln(o.Out, "yes")
	} else {
		fmt.Fprint(o.Out, "no")
		if len(response.Status.Reason) > 0 {
			fmt.Fprintf(o.Out, " - %v", response.Status.Reason)
		}
		if len(response.Status.EvaluationError) > 0 {
			fmt.Fprintf(o.Out, " - %v", response.Status.EvaluationError)
		}
		fmt.Fprintln(o.Out)
	}

	return response.Status.Allowed, nil
}

func resourceFor(mapper meta.RESTMapper, resourceArg string) schema.GroupVersionResource {
	fullySpecifiedGVR, groupResource := schema.ParseResourceArg(strings.ToLower(resourceArg))
	gvr := schema.GroupVersionResource{}
	if fullySpecifiedGVR != nil {
		gvr, _ = mapper.ResourceFor(*fullySpecifiedGVR)
	}
	if gvr.Empty() {
		var err error
		gvr, err = mapper.ResourceFor(groupResource.WithVersion(""))
		if err != nil {
			return schema.GroupVersionResource{Resource: resourceArg}
		}
	}

	return gvr
}
