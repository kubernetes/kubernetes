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
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"sort"
	"strings"

	"github.com/spf13/cobra"

	authorizationv1 "k8s.io/api/authorization/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/cli-runtime/pkg/genericiooptions"
	"k8s.io/cli-runtime/pkg/printers"
	discovery "k8s.io/client-go/discovery"
	authorizationv1client "k8s.io/client-go/kubernetes/typed/authorization/v1"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/describe"
	rbacutil "k8s.io/kubectl/pkg/util/rbac"
	"k8s.io/kubectl/pkg/util/templates"
	"k8s.io/kubectl/pkg/util/term"
)

// CanIOptions is the start of the data required to perform the operation.  As new fields are added, add them here instead of
// referencing the cmd.Flags()
type CanIOptions struct {
	AllNamespaces   bool
	Quiet           bool
	NoHeaders       bool
	Namespace       string
	AuthClient      authorizationv1client.AuthorizationV1Interface
	DiscoveryClient discovery.DiscoveryInterface

	Verb           string
	Resource       schema.GroupVersionResource
	NonResourceURL string
	Subresource    string
	ResourceName   string
	List           bool

	genericiooptions.IOStreams
	WarningPrinter *printers.WarningPrinter
}

var (
	canILong = templates.LongDesc(`
		Check whether an action is allowed.

		VERB is a logical Kubernetes API verb like 'get', 'list', 'watch', 'delete', etc.
		TYPE is a Kubernetes resource. Shortcuts and groups will be resolved.
		NONRESOURCEURL is a partial URL that starts with "/".
		NAME is the name of a particular Kubernetes resource.
		This command pairs nicely with impersonation. See --as global flag.`)

	canIExample = templates.Examples(`
		# Check to see if I can create pods in any namespace
		kubectl auth can-i create pods --all-namespaces

		# Check to see if I can list deployments in my current namespace
		kubectl auth can-i list deployments.apps

		# Check to see if service account "foo" of namespace "dev" can list pods in the namespace "prod"
		# You must be allowed to use impersonation for the global option "--as"
		kubectl auth can-i list pods --as=system:serviceaccount:dev:foo -n prod

		# Check to see if I can do everything in my current namespace ("*" means all)
		kubectl auth can-i '*' '*'

		# Check to see if I can get the job named "bar" in namespace "foo"
		kubectl auth can-i list jobs.batch/bar -n foo

		# Check to see if I can read pod logs
		kubectl auth can-i get pods --subresource=log

		# Check to see if I can access the URL /logs/
		kubectl auth can-i get /logs/

		# Check to see if I can approve certificates.k8s.io
		kubectl auth can-i approve certificates.k8s.io

		# List all allowed actions in namespace "foo"
		kubectl auth can-i --list --namespace=foo`)

	resourceVerbs       = sets.New[string]("get", "list", "watch", "create", "update", "patch", "delete", "deletecollection", "use", "bind", "impersonate", "*", "approve", "sign", "escalate", "attest")
	nonResourceURLVerbs = sets.New[string]("get", "put", "post", "head", "options", "delete", "patch", "*")
	// holds all the server-supported resources that cannot be discovered by clients. i.e. users and groups for the impersonate verb
	nonStandardResourceNames = sets.New[string]("users", "groups")
)

// NewCmdCanI returns an initialized Command for 'auth can-i' sub command
func NewCmdCanI(f cmdutil.Factory, streams genericiooptions.IOStreams) *cobra.Command {
	o := &CanIOptions{
		IOStreams: streams,
	}

	cmd := &cobra.Command{
		Use:                   "can-i VERB [TYPE | TYPE/NAME | NONRESOURCEURL]",
		DisableFlagsInUseLine: true,
		Short:                 "Check whether an action is allowed",
		Long:                  canILong,
		Example:               canIExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(o.Complete(f, args))
			cmdutil.CheckErr(o.Validate())
			var err error
			if o.List {
				err = o.RunAccessList()
			} else {
				var allowed bool
				allowed, err = o.RunAccessCheck()
				if err == nil {
					if !allowed {
						os.Exit(1)
					}
				}
			}
			cmdutil.CheckErr(err)
		},
	}

	cmd.Flags().BoolVarP(&o.AllNamespaces, "all-namespaces", "A", o.AllNamespaces, "If true, check the specified action in all namespaces.")
	cmd.Flags().BoolVarP(&o.Quiet, "quiet", "q", o.Quiet, "If true, suppress output and just return the exit code.")
	cmd.Flags().StringVar(&o.Subresource, "subresource", o.Subresource, "SubResource such as pod/log or deployment/scale")
	cmd.Flags().BoolVar(&o.List, "list", o.List, "If true, prints all allowed actions.")
	cmd.Flags().BoolVar(&o.NoHeaders, "no-headers", o.NoHeaders, "If true, prints allowed actions without headers")
	return cmd
}

// Complete completes all the required options
func (o *CanIOptions) Complete(f cmdutil.Factory, args []string) error {
	// Set default WarningPrinter if not already set.
	if o.WarningPrinter == nil {
		o.WarningPrinter = printers.NewWarningPrinter(o.ErrOut, printers.WarningPrinterOptions{Color: term.AllowsColorOutput(o.ErrOut)})
	}

	if o.List {
		if len(args) != 0 {
			return errors.New("list option must be specified with no arguments")
		}
	} else {
		if o.Quiet {
			o.Out = io.Discard
		}

		switch len(args) {
		case 2:
			o.Verb = args[0]
			if strings.HasPrefix(args[1], "/") {
				o.NonResourceURL = args[1]
				break
			}
			resourceTokens := strings.SplitN(args[1], "/", 2)
			restMapper, err := f.ToRESTMapper()
			if err != nil {
				return err
			}
			o.Resource = o.resourceFor(restMapper, resourceTokens[0])
			if len(resourceTokens) > 1 {
				o.ResourceName = resourceTokens[1]
			}
		default:
			errString := "you must specify two arguments: verb resource or verb resource/resourceName."
			usageString := "See 'kubectl auth can-i -h' for help and examples."
			return fmt.Errorf("%s\n%s", errString, usageString)
		}
	}

	var err error
	client, err := f.KubernetesClientSet()
	if err != nil {
		return err
	}
	o.AuthClient = client.AuthorizationV1()
	o.DiscoveryClient = client.Discovery()
	o.Namespace = ""
	if !o.AllNamespaces {
		o.Namespace, _, err = f.ToRawKubeConfigLoader().Namespace()
		if err != nil {
			return err
		}
	}

	return nil
}

// Validate makes sure provided values for CanIOptions are valid
func (o *CanIOptions) Validate() error {
	if o.List {
		if o.Quiet || o.AllNamespaces || o.Subresource != "" {
			return errors.New("list option can't be specified with neither quiet, all-namespaces nor subresource options")
		}
		return nil
	}

	if o.WarningPrinter == nil {
		return fmt.Errorf("WarningPrinter can not be used without initialization")
	}

	if o.NonResourceURL != "" {
		if o.Subresource != "" {
			return fmt.Errorf("--subresource can not be used with NonResourceURL")
		}
		if o.Resource != (schema.GroupVersionResource{}) || o.ResourceName != "" {
			return fmt.Errorf("NonResourceURL and ResourceName can not specified together")
		}
		if !isKnownNonResourceVerb(o.Verb) {
			o.WarningPrinter.Print(fmt.Sprintf("verb '%s' is not a known verb\n", o.Verb))
		}
	} else if !o.Resource.Empty() && !o.AllNamespaces && o.DiscoveryClient != nil {
		if namespaced, err := isNamespaced(o.Resource, o.DiscoveryClient); err == nil && !namespaced {
			if len(o.Resource.Group) == 0 {
				o.WarningPrinter.Print(fmt.Sprintf("resource '%s' is not namespace scoped\n", o.Resource.Resource))
			} else {
				o.WarningPrinter.Print(fmt.Sprintf("resource '%s' is not namespace scoped in group '%s'\n", o.Resource.Resource, o.Resource.Group))
			}
		}
		if !isKnownResourceVerb(o.Verb) {
			o.WarningPrinter.Print(fmt.Sprintf("verb '%s' is not a known verb\n", o.Verb))
		}
	}

	if o.NoHeaders {
		return fmt.Errorf("--no-headers cannot be set without --list specified")
	}
	return nil
}

// RunAccessList lists all the access current user has
func (o *CanIOptions) RunAccessList() error {
	sar := &authorizationv1.SelfSubjectRulesReview{
		Spec: authorizationv1.SelfSubjectRulesReviewSpec{
			Namespace: o.Namespace,
		},
	}
	response, err := o.AuthClient.SelfSubjectRulesReviews().Create(context.TODO(), sar, metav1.CreateOptions{})
	if err != nil {
		return err
	}

	return o.printStatus(response.Status)
}

// RunAccessCheck checks if user has access to a certain resource or non resource URL
func (o *CanIOptions) RunAccessCheck() (bool, error) {
	var sar *authorizationv1.SelfSubjectAccessReview
	if o.NonResourceURL == "" {
		sar = &authorizationv1.SelfSubjectAccessReview{
			Spec: authorizationv1.SelfSubjectAccessReviewSpec{
				ResourceAttributes: &authorizationv1.ResourceAttributes{
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
		sar = &authorizationv1.SelfSubjectAccessReview{
			Spec: authorizationv1.SelfSubjectAccessReviewSpec{
				NonResourceAttributes: &authorizationv1.NonResourceAttributes{
					Verb: o.Verb,
					Path: o.NonResourceURL,
				},
			},
		}
	}

	response, err := o.AuthClient.SelfSubjectAccessReviews().Create(context.TODO(), sar, metav1.CreateOptions{})
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
			if !nonStandardResourceNames.Has(groupResource.String()) {
				if len(groupResource.Group) == 0 {
					o.WarningPrinter.Print(fmt.Sprintf("the server doesn't have a resource type '%s'\n", groupResource.Resource))
				} else {
					o.WarningPrinter.Print(fmt.Sprintf("the server doesn't have a resource type '%s' in group '%s'\n", groupResource.Resource, groupResource.Group))
				}
			}
			return schema.GroupVersionResource{Resource: resourceArg}
		}
	}

	return gvr
}

func (o *CanIOptions) printStatus(status authorizationv1.SubjectRulesReviewStatus) error {
	if status.Incomplete {
		o.WarningPrinter.Print(fmt.Sprintf("the list may be incomplete: %v", status.EvaluationError))
	}

	breakdownRules := []rbacv1.PolicyRule{}
	for _, rule := range convertToPolicyRule(status) {
		breakdownRules = append(breakdownRules, rbacutil.BreakdownRule(rule)...)
	}

	compactRules, err := rbacutil.CompactRules(breakdownRules)
	if err != nil {
		return err
	}
	sort.Stable(rbacutil.SortableRuleSlice(compactRules))

	w := printers.GetNewTabWriter(o.Out)
	defer w.Flush()

	allErrs := []error{}
	if !o.NoHeaders {
		if err := printAccessHeaders(w); err != nil {
			allErrs = append(allErrs, err)
		}
	}

	if err := printAccess(w, compactRules); err != nil {
		allErrs = append(allErrs, err)
	}
	return utilerrors.NewAggregate(allErrs)
}

func convertToPolicyRule(status authorizationv1.SubjectRulesReviewStatus) []rbacv1.PolicyRule {
	ret := []rbacv1.PolicyRule{}
	for _, resource := range status.ResourceRules {
		ret = append(ret, rbacv1.PolicyRule{
			Verbs:         resource.Verbs,
			APIGroups:     resource.APIGroups,
			Resources:     resource.Resources,
			ResourceNames: resource.ResourceNames,
		})
	}

	for _, nonResource := range status.NonResourceRules {
		ret = append(ret, rbacv1.PolicyRule{
			Verbs:           nonResource.Verbs,
			NonResourceURLs: nonResource.NonResourceURLs,
		})
	}

	return ret
}

func printAccessHeaders(out io.Writer) error {
	columnNames := []string{"Resources", "Non-Resource URLs", "Resource Names", "Verbs"}
	_, err := fmt.Fprintf(out, "%s\n", strings.Join(columnNames, "\t"))
	return err
}

func printAccess(out io.Writer, rules []rbacv1.PolicyRule) error {
	for _, r := range rules {
		if _, err := fmt.Fprintf(out, "%s\t%v\t%v\t%v\n", describe.CombineResourceGroup(r.Resources, r.APIGroups), r.NonResourceURLs, r.ResourceNames, r.Verbs); err != nil {
			return err
		}
	}
	return nil
}

func isNamespaced(gvr schema.GroupVersionResource, discoveryClient discovery.DiscoveryInterface) (bool, error) {
	if gvr.Resource == "*" {
		return true, nil
	}
	apiResourceList, err := discoveryClient.ServerResourcesForGroupVersion(schema.GroupVersion{
		Group: gvr.Group, Version: gvr.Version,
	}.String())
	if err != nil {
		return true, err
	}

	for _, resource := range apiResourceList.APIResources {
		if resource.Name == gvr.Resource {
			return resource.Namespaced, nil
		}
	}

	return false, fmt.Errorf("the server doesn't have a resource type '%s' in group '%s'", gvr.Resource, gvr.Group)
}

func isKnownResourceVerb(s string) bool {
	return resourceVerbs.Has(s)
}

func isKnownNonResourceVerb(s string) bool {
	return nonResourceURLVerbs.Has(s)
}
