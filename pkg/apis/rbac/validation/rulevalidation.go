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

package validation

import (
	"errors"
	"fmt"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	apierrors "k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/apis/rbac"
	"k8s.io/kubernetes/pkg/auth/user"
	"k8s.io/kubernetes/pkg/serviceaccount"
	utilerrors "k8s.io/kubernetes/pkg/util/errors"
)

type AuthorizationRuleResolver interface {
	// GetRoleReferenceRules attempts to resolve the role reference of a RoleBinding or ClusterRoleBinding.  The passed namespace should be the namepsace
	// of the role binding, the empty string if a cluster role binding.
	GetRoleReferenceRules(ctx api.Context, roleRef api.ObjectReference, namespace string) ([]rbac.PolicyRule, error)

	// GetEffectivePolicyRules returns the list of rules that apply to a given user in a given namespace and error.  If an error is returned, the slice of
	// PolicyRules may not be complete, but it contains all retrievable rules.  This is done because policy rules are purely additive and policy determinations
	// can be made on the basis of those rules that are found.
	GetEffectivePolicyRules(ctx api.Context) ([]rbac.PolicyRule, error)
}

// ConfirmNoEscalation determines if the roles for a given user in a given namespace encompass the provided role.
func ConfirmNoEscalation(ctx api.Context, ruleResolver AuthorizationRuleResolver, rules []rbac.PolicyRule) error {
	ruleResolutionErrors := []error{}

	ownerLocalRules, err := ruleResolver.GetEffectivePolicyRules(ctx)
	if err != nil {
		// As per AuthorizationRuleResolver contract, this may return a non fatal error with an incomplete list of policies. Log the error and continue.
		user, _ := api.UserFrom(ctx)
		glog.V(1).Infof("non-fatal error getting local rules for %v: %v", user, err)
		ruleResolutionErrors = append(ruleResolutionErrors, err)
	}

	masterContext := api.WithNamespace(ctx, "")
	ownerGlobalRules, err := ruleResolver.GetEffectivePolicyRules(masterContext)
	if err != nil {
		// Same case as above. Log error, don't fail.
		user, _ := api.UserFrom(ctx)
		glog.V(1).Infof("non-fatal error getting global rules for %v: %v", user, err)
		ruleResolutionErrors = append(ruleResolutionErrors, err)
	}

	ownerRules := make([]rbac.PolicyRule, 0, len(ownerGlobalRules)+len(ownerLocalRules))
	ownerRules = append(ownerRules, ownerLocalRules...)
	ownerRules = append(ownerRules, ownerGlobalRules...)

	ownerRightsCover, missingRights := Covers(ownerRules, rules)
	if !ownerRightsCover {
		user, _ := api.UserFrom(ctx)
		return apierrors.NewUnauthorized(fmt.Sprintf("attempt to grant extra privileges: %v user=%v ownerrules=%v ruleResolutionErrors=%v", missingRights, user, ownerRules, ruleResolutionErrors))
	}
	return nil
}

type DefaultRuleResolver struct {
	roleGetter               RoleGetter
	roleBindingLister        RoleBindingLister
	clusterRoleGetter        ClusterRoleGetter
	clusterRoleBindingLister ClusterRoleBindingLister
}

func NewDefaultRuleResolver(roleGetter RoleGetter, roleBindingLister RoleBindingLister, clusterRoleGetter ClusterRoleGetter, clusterRoleBindingLister ClusterRoleBindingLister) *DefaultRuleResolver {
	return &DefaultRuleResolver{roleGetter, roleBindingLister, clusterRoleGetter, clusterRoleBindingLister}
}

type RoleGetter interface {
	GetRole(ctx api.Context, id string) (*rbac.Role, error)
}

type RoleBindingLister interface {
	ListRoleBindings(ctx api.Context, options *api.ListOptions) (*rbac.RoleBindingList, error)
}

type ClusterRoleGetter interface {
	GetClusterRole(ctx api.Context, id string) (*rbac.ClusterRole, error)
}

type ClusterRoleBindingLister interface {
	ListClusterRoleBindings(ctx api.Context, options *api.ListOptions) (*rbac.ClusterRoleBindingList, error)
}

// GetRoleReferenceRules attempts resolve the RoleBinding or ClusterRoleBinding.
func (r *DefaultRuleResolver) GetRoleReferenceRules(ctx api.Context, roleRef api.ObjectReference, bindingNamespace string) ([]rbac.PolicyRule, error) {
	switch roleRef.Kind {
	case "Role":
		// Roles can only be referenced by RoleBindings within the same namespace.
		if len(bindingNamespace) == 0 {
			return nil, fmt.Errorf("cluster role binding references role %q in namespace %q", roleRef.Name, roleRef.Namespace)
		} else {
			if bindingNamespace != roleRef.Namespace {
				return nil, fmt.Errorf("role binding in namespace %q references role %q in namespace %q", bindingNamespace, roleRef.Name, roleRef.Namespace)
			}
		}

		role, err := r.roleGetter.GetRole(api.WithNamespace(ctx, roleRef.Namespace), roleRef.Name)
		if err != nil {
			return nil, err
		}
		return role.Rules, nil
	case "ClusterRole":
		clusterRole, err := r.clusterRoleGetter.GetClusterRole(api.WithNamespace(ctx, ""), roleRef.Name)
		if err != nil {
			return nil, err
		}
		return clusterRole.Rules, nil
	default:
		return nil, fmt.Errorf("unsupported role reference kind: %q", roleRef.Kind)
	}
}

func (r *DefaultRuleResolver) GetEffectivePolicyRules(ctx api.Context) ([]rbac.PolicyRule, error) {
	policyRules := []rbac.PolicyRule{}
	errorlist := []error{}

	if namespace := api.NamespaceValue(ctx); len(namespace) == 0 {
		clusterRoleBindings, err := r.clusterRoleBindingLister.ListClusterRoleBindings(ctx, &api.ListOptions{})
		if err != nil {
			return nil, err
		}

		for _, clusterRoleBinding := range clusterRoleBindings.Items {
			if ok, err := appliesTo(ctx, clusterRoleBinding.Subjects); err != nil {
				errorlist = append(errorlist, err)
			} else if !ok {
				continue
			}
			rules, err := r.GetRoleReferenceRules(ctx, clusterRoleBinding.RoleRef, namespace)
			if err != nil {
				errorlist = append(errorlist, err)
				continue
			}
			policyRules = append(policyRules, rules...)
		}
	} else {
		roleBindings, err := r.roleBindingLister.ListRoleBindings(ctx, &api.ListOptions{})
		if err != nil {
			return nil, err
		}

		for _, roleBinding := range roleBindings.Items {
			if ok, err := appliesTo(ctx, roleBinding.Subjects); err != nil {
				errorlist = append(errorlist, err)
			} else if !ok {
				continue
			}
			rules, err := r.GetRoleReferenceRules(ctx, roleBinding.RoleRef, namespace)
			if err != nil {
				errorlist = append(errorlist, err)
				continue
			}
			policyRules = append(policyRules, rules...)
		}
	}

	if len(errorlist) != 0 {
		return policyRules, utilerrors.NewAggregate(errorlist)
	}
	return policyRules, nil
}

func appliesTo(ctx api.Context, subjects []rbac.Subject) (bool, error) {
	user, ok := api.UserFrom(ctx)
	if !ok {
		return false, fmt.Errorf("no user data associated with context")
	}
	for _, subject := range subjects {
		if ok, err := appliesToUser(user, subject); err != nil || ok {
			return ok, err
		}
	}
	return false, nil
}

func appliesToUser(user user.Info, subject rbac.Subject) (bool, error) {
	switch subject.Kind {
	case rbac.UserKind:
		return subject.Name == rbac.UserAll || user.GetName() == subject.Name, nil
	case rbac.GroupKind:
		return has(user.GetGroups(), subject.Name), nil
	case rbac.ServiceAccountKind:
		if subject.Namespace == "" {
			return false, fmt.Errorf("subject of kind service account without specified namespace")
		}
		return serviceaccount.MakeUsername(subject.Namespace, subject.Name) == user.GetName(), nil
	default:
		return false, fmt.Errorf("unknown subject kind: %s", subject.Kind)
	}
}

// NewTestRuleResolver returns a rule resolver from lists of role objects.
func NewTestRuleResolver(roles []rbac.Role, roleBindings []rbac.RoleBinding, clusterRoles []rbac.ClusterRole, clusterRoleBindings []rbac.ClusterRoleBinding) AuthorizationRuleResolver {
	r := staticRoles{
		roles:               roles,
		roleBindings:        roleBindings,
		clusterRoles:        clusterRoles,
		clusterRoleBindings: clusterRoleBindings,
	}
	return newMockRuleResolver(&r)
}

func newMockRuleResolver(r *staticRoles) AuthorizationRuleResolver {
	return NewDefaultRuleResolver(r, r, r, r)
}

type staticRoles struct {
	roles               []rbac.Role
	roleBindings        []rbac.RoleBinding
	clusterRoles        []rbac.ClusterRole
	clusterRoleBindings []rbac.ClusterRoleBinding
}

func (r *staticRoles) GetRole(ctx api.Context, id string) (*rbac.Role, error) {
	namespace, ok := api.NamespaceFrom(ctx)
	if !ok || namespace == "" {
		return nil, errors.New("must provide namespace when getting role")
	}
	for _, role := range r.roles {
		if role.Namespace == namespace && role.Name == id {
			return &role, nil
		}
	}
	return nil, errors.New("role not found")
}

func (r *staticRoles) GetClusterRole(ctx api.Context, id string) (*rbac.ClusterRole, error) {
	namespace, ok := api.NamespaceFrom(ctx)
	if ok && namespace != "" {
		return nil, errors.New("cannot provide namespace when getting cluster role")
	}
	for _, clusterRole := range r.clusterRoles {
		if clusterRole.Namespace == namespace && clusterRole.Name == id {
			return &clusterRole, nil
		}
	}
	return nil, errors.New("role not found")
}

func (r *staticRoles) ListRoleBindings(ctx api.Context, options *api.ListOptions) (*rbac.RoleBindingList, error) {
	namespace, ok := api.NamespaceFrom(ctx)
	if !ok || namespace == "" {
		return nil, errors.New("must provide namespace when listing role bindings")
	}

	roleBindingList := new(rbac.RoleBindingList)
	for _, roleBinding := range r.roleBindings {
		if roleBinding.Namespace != namespace {
			continue
		}
		// TODO(ericchiang): need to implement label selectors?
		roleBindingList.Items = append(roleBindingList.Items, roleBinding)
	}
	return roleBindingList, nil
}

func (r *staticRoles) ListClusterRoleBindings(ctx api.Context, options *api.ListOptions) (*rbac.ClusterRoleBindingList, error) {
	namespace, ok := api.NamespaceFrom(ctx)
	if ok && namespace != "" {
		return nil, errors.New("cannot list cluster role bindings from within a namespace")
	}
	clusterRoleBindings := new(rbac.ClusterRoleBindingList)
	clusterRoleBindings.Items = make([]rbac.ClusterRoleBinding, len(r.clusterRoleBindings))
	copy(clusterRoleBindings.Items, r.clusterRoleBindings)
	return clusterRoleBindings, nil
}
