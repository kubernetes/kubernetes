package restrictusers

import (
	"context"
	"fmt"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	kerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/pkg/apis/rbac"

	authorizationv1 "github.com/openshift/api/authorization/v1"
	userv1 "github.com/openshift/api/user/v1"
	userclient "github.com/openshift/client-go/user/clientset/versioned/typed/user/v1"
)

// SubjectChecker determines whether rolebindings on a subject (user, group, or
// service account) are allowed in a project.
type SubjectChecker interface {
	Allowed(rbac.Subject, *RoleBindingRestrictionContext) (bool, error)
}

// UnionSubjectChecker represents the union of zero or more SubjectCheckers.
type UnionSubjectChecker []SubjectChecker

// NewUnionSubjectChecker returns a new UnionSubjectChecker.
func NewUnionSubjectChecker(checkers []SubjectChecker) UnionSubjectChecker {
	return UnionSubjectChecker(checkers)
}

// Allowed determines whether the given subject is allowed in rolebindings in
// the project.
func (checkers UnionSubjectChecker) Allowed(subject rbac.Subject, ctx *RoleBindingRestrictionContext) (bool, error) {
	errs := []error{}
	for _, checker := range []SubjectChecker(checkers) {
		allowed, err := checker.Allowed(subject, ctx)
		if err != nil {
			errs = append(errs, err)
		} else if allowed {
			return true, nil
		}
	}

	return false, kerrors.NewAggregate(errs)
}

// RoleBindingRestrictionContext holds context that is used when determining
// whether a RoleBindingRestriction allows rolebindings on a particular subject.
type RoleBindingRestrictionContext struct {
	userClient userclient.UserV1Interface
	kclient    kubernetes.Interface

	// groupCache maps user name to groups.
	groupCache GroupCache

	// userToLabels maps user name to labels.Set.
	userToLabelSet map[string]labels.Set

	// groupToLabels maps group name to labels.Set.
	groupToLabelSet map[string]labels.Set

	// namespace is the namespace for which the RoleBindingRestriction makes
	// determinations.
	namespace string
}

// NewRoleBindingRestrictionContext returns a new RoleBindingRestrictionContext
// object.
func newRoleBindingRestrictionContext(ns string, kc kubernetes.Interface, userClient userclient.UserV1Interface, groupCache GroupCache) (*RoleBindingRestrictionContext, error) {
	return &RoleBindingRestrictionContext{
		namespace:       ns,
		kclient:         kc,
		userClient:      userClient,
		groupCache:      groupCache,
		userToLabelSet:  map[string]labels.Set{},
		groupToLabelSet: map[string]labels.Set{},
	}, nil
}

// labelSetForUser returns the label set for the given user subject.
func (ctx *RoleBindingRestrictionContext) labelSetForUser(subject rbac.Subject) (labels.Set, error) {
	if subject.Kind != rbac.UserKind {
		return labels.Set{}, fmt.Errorf("not a user: %q", subject.Name)
	}

	labelSet, ok := ctx.userToLabelSet[subject.Name]
	if ok {
		return labelSet, nil
	}

	user, err := ctx.userClient.Users().Get(context.TODO(), subject.Name, metav1.GetOptions{})
	if err != nil {
		return labels.Set{}, err
	}

	ctx.userToLabelSet[subject.Name] = labels.Set(user.Labels)

	return ctx.userToLabelSet[subject.Name], nil
}

// groupsForUser returns the groups for the given user subject.
func (ctx *RoleBindingRestrictionContext) groupsForUser(subject rbac.Subject) ([]*userv1.Group, error) {
	if subject.Kind != rbac.UserKind {
		return []*userv1.Group{}, fmt.Errorf("not a user: %q", subject.Name)
	}

	err := wait.PollImmediate(1*time.Second, 10*time.Second, func() (bool, error) {
		return ctx.groupCache.HasSynced(), nil
	})
	if err != nil {
		return nil, fmt.Errorf("groups.user.openshift.io cache is not synchronized")
	}

	return ctx.groupCache.GroupsFor(subject.Name)
}

// labelSetForGroup returns the label set for the given group subject.
func (ctx *RoleBindingRestrictionContext) labelSetForGroup(subject rbac.Subject) (labels.Set, error) {
	if subject.Kind != rbac.GroupKind {
		return labels.Set{}, fmt.Errorf("not a group: %q", subject.Name)
	}

	labelSet, ok := ctx.groupToLabelSet[subject.Name]
	if ok {
		return labelSet, nil
	}

	group, err := ctx.userClient.Groups().Get(context.TODO(), subject.Name, metav1.GetOptions{})
	if err != nil {
		return labels.Set{}, err
	}

	ctx.groupToLabelSet[subject.Name] = labels.Set(group.Labels)

	return ctx.groupToLabelSet[subject.Name], nil
}

// UserSubjectChecker determines whether a user subject is allowed in
// rolebindings in the project.
type UserSubjectChecker struct {
	userRestriction *authorizationv1.UserRestriction
}

// NewUserSubjectChecker returns a new UserSubjectChecker.
func NewUserSubjectChecker(userRestriction *authorizationv1.UserRestriction) UserSubjectChecker {
	return UserSubjectChecker{userRestriction: userRestriction}
}

// Allowed determines whether the given user subject is allowed in rolebindings
// in the project.
func (checker UserSubjectChecker) Allowed(subject rbac.Subject, ctx *RoleBindingRestrictionContext) (bool, error) {
	if subject.Kind != rbac.UserKind {
		return false, nil
	}

	for _, userName := range checker.userRestriction.Users {
		if subject.Name == userName {
			return true, nil
		}
	}

	if len(checker.userRestriction.Groups) != 0 {
		subjectGroups, err := ctx.groupsForUser(subject)
		if err != nil {
			return false, err
		}

		for _, groupName := range checker.userRestriction.Groups {
			for _, group := range subjectGroups {
				if group.Name == groupName {
					return true, nil
				}
			}
		}
	}

	if len(checker.userRestriction.Selectors) != 0 {
		labelSet, err := ctx.labelSetForUser(subject)
		if err != nil {
			return false, err
		}

		for _, labelSelector := range checker.userRestriction.Selectors {
			selector, err := metav1.LabelSelectorAsSelector(&labelSelector)
			if err != nil {
				return false, err
			}

			if selector.Matches(labelSet) {
				return true, nil
			}
		}
	}

	return false, nil
}

// GroupSubjectChecker determines whether a group subject is allowed in
// rolebindings in the project.
type GroupSubjectChecker struct {
	groupRestriction *authorizationv1.GroupRestriction
}

// NewGroupSubjectChecker returns a new GroupSubjectChecker.
func NewGroupSubjectChecker(groupRestriction *authorizationv1.GroupRestriction) GroupSubjectChecker {
	return GroupSubjectChecker{groupRestriction: groupRestriction}
}

// Allowed determines whether the given group subject is allowed in rolebindings
// in the project.
func (checker GroupSubjectChecker) Allowed(subject rbac.Subject, ctx *RoleBindingRestrictionContext) (bool, error) {
	if subject.Kind != rbac.GroupKind {
		return false, nil
	}

	for _, groupName := range checker.groupRestriction.Groups {
		if subject.Name == groupName {
			return true, nil
		}
	}

	if len(checker.groupRestriction.Selectors) != 0 {
		labelSet, err := ctx.labelSetForGroup(subject)
		if err != nil {
			return false, err
		}

		for _, labelSelector := range checker.groupRestriction.Selectors {
			selector, err := metav1.LabelSelectorAsSelector(&labelSelector)
			if err != nil {
				return false, err
			}

			if selector.Matches(labelSet) {
				return true, nil
			}
		}
	}

	return false, nil
}

// ServiceAccountSubjectChecker determines whether a serviceaccount subject is
// allowed in rolebindings in the project.
type ServiceAccountSubjectChecker struct {
	serviceAccountRestriction *authorizationv1.ServiceAccountRestriction
}

// NewServiceAccountSubjectChecker returns a new ServiceAccountSubjectChecker.
func NewServiceAccountSubjectChecker(serviceAccountRestriction *authorizationv1.ServiceAccountRestriction) ServiceAccountSubjectChecker {
	return ServiceAccountSubjectChecker{
		serviceAccountRestriction: serviceAccountRestriction,
	}
}

// Allowed determines whether the given serviceaccount subject is allowed in
// rolebindings in the project.
func (checker ServiceAccountSubjectChecker) Allowed(subject rbac.Subject, ctx *RoleBindingRestrictionContext) (bool, error) {
	if subject.Kind != rbac.ServiceAccountKind {
		return false, nil
	}

	subjectNamespace := subject.Namespace
	if len(subjectNamespace) == 0 {
		// If a RoleBinding has a subject that is a ServiceAccount with
		// no namespace specified, the namespace will be defaulted to
		// that of the RoleBinding.  However, admission control plug-ins
		// execute before this happens, so in order not to reject such
		// subjects erroneously, we copy the logic here of using the
		// RoleBinding's namespace if the subject's is empty.
		subjectNamespace = ctx.namespace
	}

	for _, namespace := range checker.serviceAccountRestriction.Namespaces {
		if subjectNamespace == namespace {
			return true, nil
		}
	}

	for _, serviceAccountRef := range checker.serviceAccountRestriction.ServiceAccounts {
		serviceAccountNamespace := serviceAccountRef.Namespace
		if len(serviceAccountNamespace) == 0 {
			serviceAccountNamespace = ctx.namespace
		}

		if subject.Name == serviceAccountRef.Name &&
			subjectNamespace == serviceAccountNamespace {
			return true, nil
		}
	}

	return false, nil
}

// NewSubjectChecker returns a new SubjectChecker.
func NewSubjectChecker(spec *authorizationv1.RoleBindingRestrictionSpec) (SubjectChecker, error) {
	switch {
	case spec.UserRestriction != nil:
		return NewUserSubjectChecker(spec.UserRestriction), nil

	case spec.GroupRestriction != nil:
		return NewGroupSubjectChecker(spec.GroupRestriction), nil

	case spec.ServiceAccountRestriction != nil:
		return NewServiceAccountSubjectChecker(spec.ServiceAccountRestriction), nil
	}

	return nil, fmt.Errorf("invalid RoleBindingRestrictionSpec: %v", spec)
}
