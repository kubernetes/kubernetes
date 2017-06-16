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

	Verb           string
	Resource       schema.GroupVersionResource
	NonResourceURL string
	Subresource    string
	ResourceName   string

	Out io.Writer
	Err io.Writer
}

var (
	canILong = templates.LongDesc(`
		Check whether an action is allowed.

		VERB is a logical Kubernetes API verb like 'get', 'list', 'watch', 'delete', etc.
		TYPE is a Kubernetes resource. Shortcuts and groups will be resolved.
		NONRESOURCEURL is a partial URL starts with "/".
		NAME is the name of a particular Kubernetes resource.`)

	canIExample = templates.Examples(`
		# Check to see if I can create pods in any namespace
		kubectl auth can-i create pods --all-namespaces

		# Check to see if I can list deployments in my current namespace
		kubectl auth can-i list deployments.extensions

		# Check to see if I can do everything in my current namespace ("*" means all)
		kubectl auth can-i '*' '*'

		# Check to see if I can get the job named "bar" in namespace "foo"
		kubectl auth can-i list jobs.batch/bar -n foo

		# Check to see if I can read pod logs
		kubectl auth can-i get pods --subresource=log

		# Check to see if I can access the URL /logs/
		kubectl auth can-i get /logs/`)
)

func NewCmdCanI(f cmdutil.Factory, out, err io.Writer) *cobra.Command {
	o := &CanIOptions{
		Out: out,
		Err: err,
	}

	cmd := &cobra.Command{
		Use:     "can-i VERB [TYPE | TYPE/NAME | NONRESOURCEURL]",
		Short:   "Check whether an action is allowed",
		Long:    canILong,
		Example: canIExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(o.Complete(f, args))
			cmdutil.CheckErr(o.Validate())

			allowed, err := o.RunAccessCheck()
			if err == nil {
				if o.Quiet && !allowed {
					os.Exit(1)
				}
			}

			cmdutil.CheckErr(err)
		},
	}

	cmd.Flags().BoolVar(&o.AllNamespaces, "all-namespaces", o.AllNamespaces, "If true, check the specified action in all namespaces.")
	cmd.Flags().BoolVarP(&o.Quiet, "quiet", "q", o.Quiet, "If true, suppress output and just return the exit code.")
	cmd.Flags().StringVar(&o.Subresource, "subresource", "", "SubResource such as pod/log or deployment/scale")
	return cmd
}

func (o *CanIOptions) Complete(f cmdutil.Factory, args []string) error {
	if o.Quiet {
		o.Out = ioutil.Discard
	}

	switch len(args) {
	case 2:
		o.Verb = args[0]
		if strings.HasPrefix(args[1], "/") {
			o.NonResourceURL = args[1]
			break
		}
		resourceTokens := strings.SplitN(args[1], "/", 2)
		restMapper, _ := f.Object()
		o.Resource = o.resourceFor(restMapper, resourceTokens[0])
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

	return nil
}

func (o *CanIOptions) Validate() error {
	if o.NonResourceURL != "" {
		if o.Subresource != "" {
			return fmt.Errorf("--subresource can not be used with nonResourceURL")
		}
		if o.Resource != (schema.GroupVersionResource{}) || o.ResourceName != "" {
			return fmt.Errorf("nonResourceURL and Resource can not specified together")
		}
	}
	return nil
}

func (o *CanIOptions) RunAccessCheck() (bool, error) {
	var sar *authorizationapi.SelfSubjectAccessReview
	if o.NonResourceURL == "" {
		sar = &authorizationapi.SelfSubjectAccessReview{
			Spec: authorizationapi.SelfSubjectAccessReviewSpec{
				ResourceAttributes: &authorizationapi.ResourceAttributes{
					Namespace:   o.Namespace,
					Verb:        o.Verb,
					Group:       o.Resource.Group,
					Resource:    o.Resource.Resource,
					Subresource: o.Subresource,
					Name:        o.ResourceName,
				},
			},
		}
	} else {
		sar = &authorizationapi.SelfSubjectAccessReview{
			Spec: authorizationapi.SelfSubjectAccessReviewSpec{
				NonResourceAttributes: &authorizationapi.NonResourceAttributes{
					Verb: o.Verb,
					Path: o.NonResourceURL,
				},
			},
		}

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

func (o *CanIOptions) resourceFor(mapper meta.RESTMapper, resourceArg string) schema.GroupVersionResource {
	if resourceArg == "*" {
		return schema.GroupVersionResource{Resource: resourceArg}
	}

	fullySpecifiedGVR, groupResource := schema.ParseResourceArg(strings.ToLower(resourceArg))
	gvr := schema.GroupVersionResource{}
	if fullySpecifiedGVR != nil {
		gvr, _ = mapper.ResourceFor(*fullySpecifiedGVR)
	}
	if gvr.Empty() {
		var err error
		gvr, err = mapper.ResourceFor(groupResource.WithVersion(""))
		if err != nil {
			if len(groupResource.Group) == 0 {
				fmt.Fprintf(o.Err, "Warning: the server doesn't have a resource type '%s'\n", groupResource.Resource)
			} else {
				fmt.Fprintf(o.Err, "Warning: the server doesn't have a resource type '%s' in group '%s'\n", groupResource.Resource, groupResource.Group)
			}
			return schema.GroupVersionResource{Resource: resourceArg}
		}
	}

	return gvr
}
