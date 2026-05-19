/*
Copyright 2023 The Kubernetes Authors.

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

package library

import (
	"context"
	"fmt"
	"reflect"
	"strings"

	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	genericfeatures "k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"

	"github.com/google/cel-go/cel"
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"

	apimachineryvalidation "k8s.io/apimachinery/pkg/api/validation"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/authentication/serviceaccount"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
)

// Authz provides a CEL function library extension for performing authorization checks.
// Note that authorization checks are only supported for CEL expression fields in the API
// where an 'authorizer' variable is provided to the CEL expression. See the
// documentation of API fields where CEL expressions are used to learn if the 'authorizer'
// variable is provided.
//
// path
//
// Returns a PathCheck configured to check authorization for a non-resource request
// path (e.g. /healthz). If path is an empty string, an error is returned.
// Note that the leading '/' is not required.
//
//	<Authorizer>.path(<string>) <PathCheck>
//
// Examples:
//
//	authorizer.path('/healthz') // returns a PathCheck for the '/healthz' API path
//	authorizer.path('') // results in "path must not be empty" error
//	authorizer.path('  ') // results in "path must not be empty" error
//
// group
//
// Returns a GroupCheck configured to check authorization for the API resources for
// a particular API group.
// Note that authorization checks are only supported for CEL expression fields in the API
// where an 'authorizer' variable is provided to the CEL expression. Check the
// documentation of API fields where CEL expressions are used to learn if the 'authorizer'
// variable is provided.
//
//	<Authorizer>.group(<string>) <GroupCheck>
//
// Examples:
//
//	authorizer.group('apps') // returns a GroupCheck for the 'apps' API group
//	authorizer.group('') // returns a GroupCheck for the core API group
//	authorizer.group('example.com') // returns a GroupCheck for the custom resources in the 'example.com' API group
//
// serviceAccount
//
// Returns an Authorizer configured to check authorization for the provided service account namespace and name.
// If the name is not a valid DNS subdomain string (as defined by RFC 1123), an error is returned.
// If the namespace is not a valid DNS label (as defined by RFC 1123), an error is returned.
//
//	<Authorizer>.serviceAccount(<string>, <string>) <Authorizer>
//
// Examples:
//
//	authorizer.serviceAccount('default', 'myserviceaccount') // returns an Authorizer for the service account with namespace 'default' and name 'myserviceaccount'
//	authorizer.serviceAccount('not@a#valid!namespace', 'validname') // returns an error
//	authorizer.serviceAccount('valid.example.com', 'invalid@*name') // returns an error
//
// resource
//
// Returns a ResourceCheck configured to check authorization for a particular API resource.
// Note that the provided resource string should be a lower case plural name of a Kubernetes API resource.
//
//	<GroupCheck>.resource(<string>) <ResourceCheck>
//
// Examples:
//
//	authorizer.group('apps').resource('deployments') // returns a ResourceCheck for the 'deployments' resources in the 'apps' group.
//	authorizer.group('').resource('pods') // returns a ResourceCheck for the 'pods' resources in the core group.
//	authorizer.group('apps').resource('') // results in "resource must not be empty" error
//	authorizer.group('apps').resource('  ') // results in "resource must not be empty" error
//
// subresource
//
// Returns a ResourceCheck configured to check authorization for a particular subresource of an API resource.
// If subresource is set to "", the subresource field of this ResourceCheck is considered unset.
//
//	<ResourceCheck>.subresource(<string>) <ResourceCheck>
//
// Examples:
//
//	authorizer.group('').resource('pods').subresource('status') // returns a ResourceCheck the 'status' subresource of 'pods'
//	authorizer.group('apps').resource('deployments').subresource('scale') // returns a ResourceCheck the 'scale' subresource of 'deployments'
//	authorizer.group('example.com').resource('widgets').subresource('scale') // returns a ResourceCheck for the 'scale' subresource of the 'widgets' custom resource
//	authorizer.group('example.com').resource('widgets').subresource('') // returns a ResourceCheck for the 'widgets' resource.
//
// namespace
//
// Returns a ResourceCheck configured to check authorization for a particular namespace.
// For cluster scoped resources, namespace() does not need to be called; namespace defaults
// to "", which is the correct namespace value to use to check cluster scoped resources.
// If namespace is set to "", the ResourceCheck will check authorization for the cluster scope.
//
//	<ResourceCheck>.namespace(<string>) <ResourceCheck>
//
// Examples:
//
//	authorizer.group('apps').resource('deployments').namespace('test') // returns a ResourceCheck for 'deployments' in the 'test' namespace
//	authorizer.group('').resource('pods').namespace('default') // returns a ResourceCheck for 'pods' in the 'default' namespace
//	authorizer.group('').resource('widgets').namespace('') // returns a ResourceCheck for 'widgets' in the cluster scope
//
// name
//
// Returns a ResourceCheck configured to check authorization for a particular resource name.
// If name is set to "", the name field of this ResourceCheck is considered unset.
//
//	<ResourceCheck>.name(<name>) <ResourceCheck>
//
// Examples:
//
//	authorizer.group('apps').resource('deployments').namespace('test').name('backend') // returns a ResourceCheck for the 'backend' 'deployments' resource in the 'test' namespace
//	authorizer.group('apps').resource('deployments').namespace('test').name('') // returns a ResourceCheck for the 'deployments' resource in the 'test' namespace
//
// check
//
// For PathCheck, checks if the principal (user or service account) that sent the request is authorized for the HTTP request verb of the path.
// For ResourceCheck, checks if the principal (user or service account) that sent the request is authorized for the API verb and the configured authorization checks of the ResourceCheck.
// The check operation can be expensive, particularly in clusters using the webhook authorization mode.
//
//	<PathCheck>.check(<check>) <Decision>
//	<ResourceCheck>.check(<check>) <Decision>
//
// Examples:
//
//	authorizer.group('').resource('pods').namespace('default').check('create') // Checks if the principal (user or service account) is authorized create pods in the 'default' namespace.
//	authorizer.path('/healthz').check('get') // Checks if the principal (user or service account) is authorized to make HTTP GET requests to the /healthz API path.
//
// allowed
//
// Returns true if the authorizer's decision for the check is "allow".  Note that if the authorizer's decision is
// "no opinion", that the 'allowed' function will return false.
//
//	<Decision>.allowed() <bool>
//
// Examples:
//
//	authorizer.group('').resource('pods').namespace('default').check('create').allowed() // Returns true if the principal (user or service account) is allowed create pods in the 'default' namespace.
//	authorizer.path('/healthz').check('get').allowed()  // Returns true if the principal (user or service account) is allowed to make HTTP GET requests to the /healthz API path.
//
// reason
//
// Returns a string reason for the authorization decision
//
//	<Decision>.reason() <string>
//
// Examples:
//
//	authorizer.path('/healthz').check('GET').reason()
//
// errored
//
// Returns true if the authorization check resulted in an error.
//
//	<Decision>.errored() <bool>
//
// Examples:
//
//	authorizer.group('').resource('pods').namespace('default').check('create').errored() // Returns true if the authorization check resulted in an error
//
// error
//
// If the authorization check resulted in an error, returns the error. Otherwise, returns the empty string.
//
//	<Decision>.error() <string>
//
// Examples:
//
//	authorizer.group('').resource('pods').namespace('default').check('create').error()
//
// fieldSelector
//
// Takes a string field selector, parses it to field selector requirements, and includes it in the authorization check.
// If the field selector does not parse successfully, no field selector requirements are included in the authorization check.
// Added in Kubernetes 1.31+, Authz library version 1.
//
//	<ResourceCheck>.fieldSelector(<string>) <ResourceCheck>
//
// Examples:
//
//	authorizer.group('').resource('pods').fieldSelector('spec.nodeName=mynode').check('list').allowed()
//
// labelSelector (added in v1, Kubernetes 1.31+)
//
// Takes a string label selector, parses it to label selector requirements, and includes it in the authorization check.
// If the label selector does not parse successfully, no label selector requirements are included in the authorization check.
// Added in Kubernetes 1.31+, Authz library version 1.
//
//	<ResourceCheck>.labelSelector(<string>) <ResourceCheck>
//
// Examples:
//
//	authorizer.group('').resource('pods').labelSelector('app=example').check('list').allowed()
func Authz() cel.EnvOption {
	return cel.Lib(authzLib)
}

var authzLib = &authz{}

type authz struct{}

func (*authz) LibraryName() string {
	return "kubernetes.authz"
}

func (*authz) Types() []*cel.Type {
	return []*cel.Type{
		AuthorizerType,
		PathCheckType,
		GroupCheckType,
		ResourceCheckType,
		DecisionType}
}

func (*authz) declarations() map[string][]cel.FunctionOpt {
	return authzLibraryDecls
}

var authzLibraryDecls = map[string][]cel.FunctionOpt{
	"path": {
		cel.MemberOverload("authorizer_path", []*cel.Type{AuthorizerType, cel.StringType}, PathCheckType,
			cel.BinaryBinding(authorizerPath))},
	"group": {
		cel.MemberOverload("authorizer_group", []*cel.Type{AuthorizerType, cel.StringType}, GroupCheckType,
			cel.BinaryBinding(authorizerGroup))},
	"serviceAccount": {
		cel.MemberOverload("authorizer_serviceaccount", []*cel.Type{AuthorizerType, cel.StringType, cel.StringType}, AuthorizerType,
			cel.FunctionBinding(authorizerServiceAccount))},
	"resource": {
		cel.MemberOverload("groupcheck_resource", []*cel.Type{GroupCheckType, cel.StringType}, ResourceCheckType,
			cel.BinaryBinding(groupCheckResource))},
	"subresource": {
		cel.MemberOverload("resourcecheck_subresource", []*cel.Type{ResourceCheckType, cel.StringType}, ResourceCheckType,
			cel.BinaryBinding(resourceCheckSubresource))},
	"namespace": {
		cel.MemberOverload("resourcecheck_namespace", []*cel.Type{ResourceCheckType, cel.StringType}, ResourceCheckType,
			cel.BinaryBinding(resourceCheckNamespace))},
	"name": {
		cel.MemberOverload("resourcecheck_name", []*cel.Type{ResourceCheckType, cel.StringType}, ResourceCheckType,
			cel.BinaryBinding(resourceCheckName))},
	"check": {
		cel.MemberOverload("pathcheck_check", []*cel.Type{PathCheckType, cel.StringType}, DecisionType,
			cel.BinaryBinding(pathCheckCheck)),
		cel.MemberOverload("resourcecheck_check", []*cel.Type{ResourceCheckType, cel.StringType}, DecisionType,
			cel.BinaryBinding(resourceCheckCheck))},
	"errored": {
		cel.MemberOverload("decision_errored", []*cel.Type{DecisionType}, cel.BoolType,
			cel.UnaryBinding(decisionErrored))},
	"error": {
		cel.MemberOverload("decision_error", []*cel.Type{DecisionType}, cel.StringType,
			cel.UnaryBinding(decisionError))},
	"allowed": {
		cel.MemberOverload("decision_allowed", []*cel.Type{DecisionType}, cel.BoolType,
			cel.UnaryBinding(decisionAllowed))},
	"reason": {
		cel.MemberOverload("decision_reason", []*cel.Type{DecisionType}, cel.StringType,
			cel.UnaryBinding(decisionReason))},
}

func (*authz) CompileOptions() []cel.EnvOption {
	options := make([]cel.EnvOption, 0, len(authzLibraryDecls))
	for name, overloads := range authzLibraryDecls {
		options = append(options, cel.Function(name, overloads...))
	}
	return options
}

func (*authz) ProgramOptions() []cel.ProgramOption {
	return []cel.ProgramOption{}
}

// AuthzSelectors provides a CEL function library extension for adding fieldSelector and
// labelSelector filters to authorization checks. This requires the Authz library.
// See documentation of the Authz library for use and availability of the authorizer variable.
//
// fieldSelector
//
// Takes a string field selector, parses it to field selector requirements, and includes it in the authorization check.
// If the field selector does not parse successfully, no field selector requirements are included in the authorization check.
// Added in Kubernetes 1.31+.
//
//	<ResourceCheck>.fieldSelector(<string>) <ResourceCheck>
//
// Examples:
//
//	authorizer.group('').resource('pods').fieldSelector('spec.nodeName=mynode').check('list').allowed()
//
// labelSelector
//
// Takes a string label selector, parses it to label selector requirements, and includes it in the authorization check.
// If the label selector does not parse successfully, no label selector requirements are included in the authorization check.
// Added in Kubernetes 1.31+.
//
//	<ResourceCheck>.labelSelector(<string>) <ResourceCheck>
//
// Examples:
//
//	authorizer.group('').resource('pods').labelSelector('app=example').check('list').allowed()
func AuthzSelectors() cel.EnvOption {
	return cel.Lib(authzSelectorsLib)
}

var authzSelectorsLib = &authzSelectors{}

type authzSelectors struct{}

func (*authzSelectors) LibraryName() string {
	return "kubernetes.authzSelectors"
}

func (*authzSelectors) Types() []*cel.Type {
	return []*cel.Type{ResourceCheckType}
}

func (*authzSelectors) declarations() map[string][]cel.FunctionOpt {
	return authzSelectorsLibraryDecls
}

var authzSelectorsLibraryDecls = map[string][]cel.FunctionOpt{
	"fieldSelector": {
		cel.MemberOverload("authorizer_fieldselector", []*cel.Type{ResourceCheckType, cel.StringType}, ResourceCheckType,
			cel.BinaryBinding(resourceCheckFieldSelector))},
	"labelSelector": {
		cel.MemberOverload("authorizer_labelselector", []*cel.Type{ResourceCheckType, cel.StringType}, ResourceCheckType,
			cel.BinaryBinding(resourceCheckLabelSelector))},
}

func (*authzSelectors) CompileOptions() []cel.EnvOption {
	options := make([]cel.EnvOption, 0, len(authzSelectorsLibraryDecls))
	for name, overloads := range authzSelectorsLibraryDecls {
		options = append(options, cel.Function(name, overloads...))
	}
	return options
}

func (*authzSelectors) ProgramOptions() []cel.ProgramOption {
	return []cel.ProgramOption{}
}

func authorizerPath(arg1, arg2 ref.Val) ref.Val {
	authz, ok := arg1.(authorizerVal)
	if !ok {
		return types.MaybeNoSuchOverloadErr(arg1)
	}

	path, ok := arg2.Value().(string)
	if !ok {
		return types.MaybeNoSuchOverloadErr(arg1)
	}

	if len(strings.TrimSpace(path)) == 0 {
		return types.NewErr("path must not be empty")
	}

	return authz.pathCheck(path)
}

func authorizerGroup(arg1, arg2 ref.Val) ref.Val {
	authz, ok := arg1.(authorizerVal)
	if !ok {
		return types.MaybeNoSuchOverloadErr(arg1)
	}

	group, ok := arg2.Value().(string)
	if !ok {
		return types.MaybeNoSuchOverloadErr(arg1)
	}

	return authz.groupCheck(group)
}

func authorizerServiceAccount(args ...ref.Val) ref.Val {
	argn := len(args)
	if argn != 3 {
		return types.NoSuchOverloadErr()
	}

	authz, ok := args[0].(authorizerVal)
	if !ok {
		return types.MaybeNoSuchOverloadErr(args[0])
	}

	namespace, ok := args[1].Value().(string)
	if !ok {
		return types.MaybeNoSuchOverloadErr(args[1])
	}

	name, ok := args[2].Value().(string)
	if !ok {
		return types.MaybeNoSuchOverloadErr(args[2])
	}

	if errors := apimachineryvalidation.ValidateServiceAccountName(name, false); len(errors) > 0 {
		return types.NewErr("Invalid service account name")
	}
	if errors := apimachineryvalidation.ValidateNamespaceName(namespace, false); len(errors) > 0 {
		return types.NewErr("Invalid service account namespace")
	}
	return authz.serviceAccount(namespace, name)
}

func groupCheckResource(arg1, arg2 ref.Val) ref.Val {
	groupCheck, ok := arg1.(groupCheckVal)
	if !ok {
		return types.MaybeNoSuchOverloadErr(arg1)
	}

	resource, ok := arg2.Value().(string)
	if !ok {
		return types.MaybeNoSuchOverloadErr(arg1)
	}

	if len(strings.TrimSpace(resource)) == 0 {
		return types.NewErr("resource must not be empty")
	}
	return groupCheck.resourceCheck(resource)
}

func resourceCheckSubresource(arg1, arg2 ref.Val) ref.Val {
	resourceCheck, ok := arg1.(resourceCheckVal)
	if !ok {
		return types.MaybeNoSuchOverloadErr(arg1)
	}

	subresource, ok := arg2.Value().(string)
	if !ok {
		return types.MaybeNoSuchOverloadErr(arg1)
	}

	result := resourceCheck
	result.subresource = subresource
	return result
}

func resourceCheckFieldSelector(arg1, arg2 ref.Val) ref.Val {
	resourceCheck, ok := arg1.(resourceCheckVal)
	if !ok {
		return types.MaybeNoSuchOverloadErr(arg1)
	}

	fieldSelector, ok := arg2.Value().(string)
	if !ok {
		return types.MaybeNoSuchOverloadErr(arg1)
	}

	result := resourceCheck
	result.fieldSelector = fieldSelector
	return result
}

func resourceCheckLabelSelector(arg1, arg2 ref.Val) ref.Val {
	resourceCheck, ok := arg1.(resourceCheckVal)
	if !ok {
		return types.MaybeNoSuchOverloadErr(arg1)
	}

	labelSelector, ok := arg2.Value().(string)
	if !ok {
		return types.MaybeNoSuchOverloadErr(arg1)
	}

	result := resourceCheck
	result.labelSelector = labelSelector
	return result
}

func resourceCheckNamespace(arg1, arg2 ref.Val) ref.Val {
	resourceCheck, ok := arg1.(resourceCheckVal)
	if !ok {
		return types.MaybeNoSuchOverloadErr(arg1)
	}

	namespace, ok := arg2.Value().(string)
	if !ok {
		return types.MaybeNoSuchOverloadErr(arg1)
	}

	result := resourceCheck
	result.namespace = namespace
	return result
}

func resourceCheckName(arg1, arg2 ref.Val) ref.Val {
	resourceCheck, ok := arg1.(resourceCheckVal)
	if !ok {
		return types.MaybeNoSuchOverloadErr(arg1)
	}

	name, ok := arg2.Value().(string)
	if !ok {
		return types.MaybeNoSuchOverloadErr(arg1)
	}

	result := resourceCheck
	result.name = name
	return result
}

func pathCheckCheck(arg1, arg2 ref.Val) ref.Val {
	pathCheck, ok := arg1.(pathCheckVal)
	if !ok {
		return types.MaybeNoSuchOverloadErr(arg1)
	}

	httpRequestVerb, ok := arg2.Value().(string)
	if !ok {
		return types.MaybeNoSuchOverloadErr(arg1)
	}

	return pathCheck.Authorize(context.TODO(), httpRequestVerb)
}

func resourceCheckCheck(arg1, arg2 ref.Val) ref.Val {
	resourceCheck, ok := arg1.(resourceCheckVal)
	if !ok {
		return types.MaybeNoSuchOverloadErr(arg1)
	}

	apiVerb, ok := arg2.Value().(string)
	if !ok {
		return types.MaybeNoSuchOverloadErr(arg1)
	}

	return resourceCheck.Authorize(context.TODO(), apiVerb)
}

func decisionErrored(arg ref.Val) ref.Val {
	decision, ok := arg.(decisionVal)
	if !ok {
		return types.MaybeNoSuchOverloadErr(arg)
	}

	return types.Bool(decision.err != nil)
}

func decisionError(arg ref.Val) ref.Val {
	decision, ok := arg.(decisionVal)
	if !ok {
		return types.MaybeNoSuchOverloadErr(arg)
	}

	if decision.err == nil {
		return types.String("")
	}
	return types.String(decision.err.Error())
}

func decisionAllowed(arg ref.Val) ref.Val {
	decision, ok := arg.(decisionVal)
	if !ok {
		return types.MaybeNoSuchOverloadErr(arg)
	}

	return types.Bool(decision.authDecision == authorizer.DecisionAllow)
}

func decisionReason(arg ref.Val) ref.Val {
	decision, ok := arg.(decisionVal)
	if !ok {
		return types.MaybeNoSuchOverloadErr(arg)
	}

	return types.String(decision.reason)
}

var (
	AuthorizerType    = cel.ObjectType("kubernetes.authorization.Authorizer")
	PathCheckType     = cel.ObjectType("kubernetes.authorization.PathCheck")
	GroupCheckType    = cel.ObjectType("kubernetes.authorization.GroupCheck")
	ResourceCheckType = cel.ObjectType("kubernetes.authorization.ResourceCheck")
	DecisionType      = cel.ObjectType("kubernetes.authorization.Decision")
)

// Resource represents an API resource
type Resource interface {
	// GetName returns the name of the object as presented in the request.  On a CREATE operation, the client
	// may omit name and rely on the server to generate the name.  If that is the case, this method will return
	// the empty string
	GetName() string
	// GetNamespace is the namespace associated with the request (if any)
	GetNamespace() string
	// GetResource is the name of the resource being requested.  This is not the kind.  For example: pods
	GetResource() schema.GroupVersionResource
	// GetSubresource is the name of the subresource being requested.  This is a different resource, scoped to the parent resource, but it may have a different kind.
	// For instance, /pods has the resource "pods" and the kind "Pod", while /pods/foo/status has the resource "pods", the sub resource "status", and the kind "Pod"
	// (because status operates on pods). The binding resource for a pod though may be /pods/foo/binding, which has resource "pods", subresource "binding", and kind "Binding".
	GetSubresource() string
}

func NewAuthorizerVal(userInfo user.Info, authorizer authorizer.Authorizer) ref.Val {
	return authorizerVal{receiverOnlyObjectVal: receiverOnlyVal(AuthorizerType), userInfo: userInfo, authAuthorizer: authorizer}
}

func NewResourceAuthorizerVal(userInfo user.Info, authorizer authorizer.Authorizer, requestResource Resource) ref.Val {
	a := authorizerVal{receiverOnlyObjectVal: receiverOnlyVal(AuthorizerType), userInfo: userInfo, authAuthorizer: authorizer}
	resource := requestResource.GetResource()
	g := a.groupCheck(resource.Group)
	r := g.resourceCheck(resource.Resource)
	r.subresource = requestResource.GetSubresource()
	r.namespace = requestResource.GetNamespace()
	r.name = requestResource.GetName()
	return r
}

type authorizerVal struct {
	receiverOnlyObjectVal
	userInfo       user.Info
	authAuthorizer authorizer.Authorizer
}

func (a authorizerVal) pathCheck(path string) pathCheckVal {
	return pathCheckVal{receiverOnlyObjectVal: receiverOnlyVal(PathCheckType), authorizer: a, path: path}
}

func (a authorizerVal) groupCheck(group string) groupCheckVal {
	return groupCheckVal{receiverOnlyObjectVal: receiverOnlyVal(GroupCheckType), authorizer: a, group: group}
}

func (a authorizerVal) serviceAccount(namespace, name string) authorizerVal {
	sa := &serviceaccount.ServiceAccountInfo{Name: name, Namespace: namespace}
	return authorizerVal{
		receiverOnlyObjectVal: receiverOnlyVal(AuthorizerType),
		userInfo:              sa.UserInfo(),
		authAuthorizer:        a.authAuthorizer,
	}
}

type pathCheckVal struct {
	receiverOnlyObjectVal
	authorizer authorizerVal
	path       string
}

func (a pathCheckVal) Authorize(ctx context.Context, verb string) ref.Val {
	attr := &authorizer.AttributesRecord{
		Path: a.path,
		Verb: verb,
		User: a.authorizer.userInfo,
	}

	decision, reason, err := a.authorizer.authAuthorizer.Authorize(ctx, attr)
	return newDecision(decision, err, reason)
}

type groupCheckVal struct {
	receiverOnlyObjectVal
	authorizer authorizerVal
	group      string
}

func (g groupCheckVal) resourceCheck(resource string) resourceCheckVal {
	return resourceCheckVal{receiverOnlyObjectVal: receiverOnlyVal(ResourceCheckType), groupCheck: g, resource: resource}
}

type resourceCheckVal struct {
	receiverOnlyObjectVal
	groupCheck    groupCheckVal
	resource      string
	subresource   string
	namespace     string
	name          string
	fieldSelector string
	labelSelector string
}

func (a resourceCheckVal) Authorize(ctx context.Context, verb string) ref.Val {
	attr := &authorizer.AttributesRecord{
		ResourceRequest: true,
		APIGroup:        a.groupCheck.group,
		APIVersion:      "*",
		Resource:        a.resource,
		Subresource:     a.subresource,
		Namespace:       a.namespace,
		Name:            a.name,
		Verb:            verb,
		User:            a.groupCheck.authorizer.userInfo,
	}

	if utilfeature.DefaultFeatureGate.Enabled(genericfeatures.AuthorizeWithSelectors) {
		if len(a.fieldSelector) > 0 {
			selector, err := fields.ParseSelector(a.fieldSelector)
			if err != nil {
				attr.FieldSelectorRequirements, attr.FieldSelectorParsingErr = nil, err
			} else {
				attr.FieldSelectorRequirements, attr.FieldSelectorParsingErr = selector.Requirements(), nil
			}
		}
		if len(a.labelSelector) > 0 {
			requirements, err := labels.ParseToRequirements(a.labelSelector)
			if err != nil {
				attr.LabelSelectorRequirements, attr.LabelSelectorParsingErr = nil, err
			} else {
				attr.LabelSelectorRequirements, attr.LabelSelectorParsingErr = requirements, nil
			}
		}
	}

	decision, reason, err := a.groupCheck.authorizer.authAuthorizer.Authorize(ctx, attr)
	return newDecision(decision, err, reason)
}

func newDecision(authDecision authorizer.Decision, err error, reason string) decisionVal {
	return decisionVal{receiverOnlyObjectVal: receiverOnlyVal(DecisionType), authDecision: authDecision, err: err, reason: reason}
}

type decisionVal struct {
	receiverOnlyObjectVal
	err          error
	authDecision authorizer.Decision
	reason       string
}

// receiverOnlyObjectVal provides an implementation of ref.Val for
// any object type that has receiver functions but does not expose any fields to
// CEL.
type receiverOnlyObjectVal struct {
	typeValue *types.Type
}

// receiverOnlyVal returns a receiverOnlyObjectVal for the given type.
func receiverOnlyVal(objectType *cel.Type) receiverOnlyObjectVal {
	return receiverOnlyObjectVal{typeValue: types.NewTypeValue(objectType.String())}
}

// ConvertToNative implements ref.Val.ConvertToNative.
func (a receiverOnlyObjectVal) ConvertToNative(typeDesc reflect.Type) (any, error) {
	return nil, fmt.Errorf("type conversion error from '%s' to '%v'", a.typeValue.String(), typeDesc)
}

// ConvertToType implements ref.Val.ConvertToType.
func (a receiverOnlyObjectVal) ConvertToType(typeVal ref.Type) ref.Val {
	switch typeVal {
	case a.typeValue:
		return a
	case types.TypeType:
		return a.typeValue
	}
	return types.NewErr("type conversion error from '%s' to '%s'", a.typeValue, typeVal)
}

// Equal implements ref.Val.Equal.
func (a receiverOnlyObjectVal) Equal(other ref.Val) ref.Val {
	o, ok := other.(receiverOnlyObjectVal)
	if !ok {
		return types.MaybeNoSuchOverloadErr(other)
	}
	return types.Bool(a == o)
}

// Type implements ref.Val.Type.
func (a receiverOnlyObjectVal) Type() ref.Type {
	return a.typeValue
}

// Value implements ref.Val.Value.
func (a receiverOnlyObjectVal) Value() any {
	return types.NoSuchOverloadErr()
}
