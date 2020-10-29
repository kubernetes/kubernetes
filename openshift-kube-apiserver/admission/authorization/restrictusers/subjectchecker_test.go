package restrictusers

import (
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/kubernetes/pkg/apis/rbac"

	authorizationv1 "github.com/openshift/api/authorization/v1"
	userv1 "github.com/openshift/api/user/v1"
	fakeuserclient "github.com/openshift/client-go/user/clientset/versioned/fake"
)

func mustNewSubjectChecker(t *testing.T, spec *authorizationv1.RoleBindingRestrictionSpec) SubjectChecker {
	checker, err := NewSubjectChecker(spec)
	if err != nil {
		t.Errorf("unexpected error from NewChecker: %v, spec: %#v", err, spec)
	}

	return checker
}

func TestSubjectCheckers(t *testing.T) {
	var (
		userBobRef = rbac.Subject{
			Kind: rbac.UserKind,
			Name: "Bob",
		}
		userAliceRef = rbac.Subject{
			Kind: rbac.UserKind,
			Name: "Alice",
		}
		groupRef = rbac.Subject{
			Kind: rbac.GroupKind,
			Name: "group",
		}
		serviceaccountRef = rbac.Subject{
			Kind:      rbac.ServiceAccountKind,
			Namespace: "namespace",
			Name:      "serviceaccount",
		}
		group = userv1.Group{
			ObjectMeta: metav1.ObjectMeta{
				Name:   "group",
				Labels: map[string]string{"baz": "quux"},
			},
			Users: []string{userBobRef.Name},
		}
		userObjects = []runtime.Object{
			&userv1.User{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "Alice",
					Labels: map[string]string{"foo": "bar"},
				},
			},
			&userv1.User{
				ObjectMeta: metav1.ObjectMeta{Name: "Bob"},
				Groups:     []string{"group"},
			},
			&group,
		}
		kubeObjects = []runtime.Object{
			&corev1.ServiceAccount{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: "namespace",
					Name:      "serviceaccount",
					Labels:    map[string]string{"xyzzy": "thud"},
				},
			},
		}
	)

	testCases := []struct {
		name        string
		checker     SubjectChecker
		subject     rbac.Subject
		shouldAllow bool
	}{
		{
			name: "allow regular user by literal name match",
			checker: mustNewSubjectChecker(t,
				&authorizationv1.RoleBindingRestrictionSpec{
					UserRestriction: &authorizationv1.UserRestriction{
						Users: []string{userAliceRef.Name},
					},
				}),
			subject:     userAliceRef,
			shouldAllow: true,
		},
		{
			name: "allow regular user by group membership",
			checker: mustNewSubjectChecker(t,
				&authorizationv1.RoleBindingRestrictionSpec{
					UserRestriction: &authorizationv1.UserRestriction{
						Groups: []string{groupRef.Name},
					},
				}),
			subject:     userBobRef,
			shouldAllow: true,
		},
		{
			name: "prohibit regular user when another user matches on group membership",
			checker: mustNewSubjectChecker(t,
				&authorizationv1.RoleBindingRestrictionSpec{
					UserRestriction: &authorizationv1.UserRestriction{
						Groups: []string{groupRef.Name},
					},
				}),
			subject:     userAliceRef,
			shouldAllow: false,
		},
		{
			name: "allow regular user by label selector match",
			checker: mustNewSubjectChecker(t,
				&authorizationv1.RoleBindingRestrictionSpec{
					UserRestriction: &authorizationv1.UserRestriction{
						Selectors: []metav1.LabelSelector{
							{MatchLabels: map[string]string{"foo": "bar"}},
						},
					},
				}),
			subject:     userAliceRef,
			shouldAllow: true,
		},
		{
			name: "prohibit regular user when another user matches on label selector",
			checker: mustNewSubjectChecker(t,
				&authorizationv1.RoleBindingRestrictionSpec{
					UserRestriction: &authorizationv1.UserRestriction{
						Selectors: []metav1.LabelSelector{
							{MatchLabels: map[string]string{"foo": "bar"}},
						},
					},
				}),
			subject:     userBobRef,
			shouldAllow: false,
		},
		{
			name: "allow regular group by literal name match",
			checker: mustNewSubjectChecker(t,
				&authorizationv1.RoleBindingRestrictionSpec{
					GroupRestriction: &authorizationv1.GroupRestriction{
						Groups: []string{groupRef.Name},
					},
				}),
			subject:     groupRef,
			shouldAllow: true,
		},
		{
			name: "allow regular group by label selector match",
			checker: mustNewSubjectChecker(t,
				&authorizationv1.RoleBindingRestrictionSpec{
					GroupRestriction: &authorizationv1.GroupRestriction{
						Selectors: []metav1.LabelSelector{
							{MatchLabels: map[string]string{"baz": "quux"}},
						},
					},
				}),
			subject:     groupRef,
			shouldAllow: true,
		},
		{
			name: "allow service account with explicit namespace by match on literal name and explicit namespace",
			checker: mustNewSubjectChecker(t,
				&authorizationv1.RoleBindingRestrictionSpec{
					ServiceAccountRestriction: &authorizationv1.ServiceAccountRestriction{
						ServiceAccounts: []authorizationv1.ServiceAccountReference{
							{
								Name:      serviceaccountRef.Name,
								Namespace: serviceaccountRef.Namespace,
							},
						},
					},
				}),
			subject:     serviceaccountRef,
			shouldAllow: true,
		},
		{
			name: "allow service account with explicit namespace by match on literal name and implicit namespace",
			checker: mustNewSubjectChecker(t,
				&authorizationv1.RoleBindingRestrictionSpec{
					ServiceAccountRestriction: &authorizationv1.ServiceAccountRestriction{
						ServiceAccounts: []authorizationv1.ServiceAccountReference{
							{Name: serviceaccountRef.Name},
						},
					},
				}),
			subject:     serviceaccountRef,
			shouldAllow: true,
		},
		{
			name: "prohibit service account with explicit namespace where literal name matches but explicit namespace does not",
			checker: mustNewSubjectChecker(t,
				&authorizationv1.RoleBindingRestrictionSpec{
					ServiceAccountRestriction: &authorizationv1.ServiceAccountRestriction{
						ServiceAccounts: []authorizationv1.ServiceAccountReference{
							{
								Namespace: serviceaccountRef.Namespace,
								Name:      serviceaccountRef.Name,
							},
						},
					},
				}),
			subject: rbac.Subject{
				Kind:      rbac.ServiceAccountKind,
				Namespace: "othernamespace",
				Name:      serviceaccountRef.Name,
			},
			shouldAllow: false,
		},
		{
			name: "prohibit service account with explicit namespace where literal name matches but implicit namespace does not",
			checker: mustNewSubjectChecker(t,
				&authorizationv1.RoleBindingRestrictionSpec{
					ServiceAccountRestriction: &authorizationv1.ServiceAccountRestriction{
						ServiceAccounts: []authorizationv1.ServiceAccountReference{
							{Name: serviceaccountRef.Name},
						},
					},
				}),
			subject: rbac.Subject{
				Kind:      rbac.ServiceAccountKind,
				Namespace: "othernamespace",
				Name:      serviceaccountRef.Name,
			},
			shouldAllow: false,
		},
		{
			name: "allow service account with implicit namespace by match on literal name and explicit namespace",
			checker: mustNewSubjectChecker(t,
				&authorizationv1.RoleBindingRestrictionSpec{
					ServiceAccountRestriction: &authorizationv1.ServiceAccountRestriction{
						ServiceAccounts: []authorizationv1.ServiceAccountReference{
							{
								Name:      serviceaccountRef.Name,
								Namespace: serviceaccountRef.Namespace,
							},
						},
					},
				}),
			subject: rbac.Subject{
				Kind: rbac.ServiceAccountKind,
				Name: serviceaccountRef.Name,
			},
			shouldAllow: true,
		},
		{
			name: "allow service account with implicit namespace by match on literal name and implicit namespace",
			checker: mustNewSubjectChecker(t,
				&authorizationv1.RoleBindingRestrictionSpec{
					ServiceAccountRestriction: &authorizationv1.ServiceAccountRestriction{
						ServiceAccounts: []authorizationv1.ServiceAccountReference{
							{Name: serviceaccountRef.Name},
						},
					},
				}),
			subject: rbac.Subject{
				Kind: rbac.ServiceAccountKind,
				Name: serviceaccountRef.Name,
			},
			shouldAllow: true,
		},
		{
			name: "prohibit service account with implicit namespace where literal name matches but explicit namespace does not",
			checker: mustNewSubjectChecker(t,
				&authorizationv1.RoleBindingRestrictionSpec{
					ServiceAccountRestriction: &authorizationv1.ServiceAccountRestriction{
						ServiceAccounts: []authorizationv1.ServiceAccountReference{
							{
								Namespace: "othernamespace",
								Name:      serviceaccountRef.Name,
							},
						},
					},
				}),
			subject: rbac.Subject{
				Kind: rbac.ServiceAccountKind,
				Name: serviceaccountRef.Name,
			},
			shouldAllow: false,
		},
		{
			name: "prohibit service account with explicit namespace where explicit namespace matches but literal name does not",
			checker: mustNewSubjectChecker(t,
				&authorizationv1.RoleBindingRestrictionSpec{
					ServiceAccountRestriction: &authorizationv1.ServiceAccountRestriction{
						ServiceAccounts: []authorizationv1.ServiceAccountReference{
							{
								Namespace: serviceaccountRef.Namespace,
								Name:      "othername",
							},
						},
					},
				}),
			subject:     serviceaccountRef,
			shouldAllow: false,
		},
		{
			name: "allow service account by match on namespace",
			checker: mustNewSubjectChecker(t,
				&authorizationv1.RoleBindingRestrictionSpec{
					ServiceAccountRestriction: &authorizationv1.ServiceAccountRestriction{
						Namespaces: []string{serviceaccountRef.Namespace},
					},
				}),
			subject:     serviceaccountRef,
			shouldAllow: true,
		},
	}

	stopCh := make(chan struct{})
	defer close(stopCh)

	kclient := fake.NewSimpleClientset(kubeObjects...)
	fakeUserClient := fakeuserclient.NewSimpleClientset(userObjects...)
	groupCache := fakeGroupCache{groups: []userv1.Group{group}}
	// This is a terrible, horrible, no-good, very bad hack to avoid a race
	// condition between the test "allow regular user by group membership"
	// and the group cache's initialisation.
	for {
		if groups, _ := groupCache.GroupsFor(group.Users[0]); len(groups) == 1 {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}

	ctx, err := newRoleBindingRestrictionContext("namespace",
		kclient, fakeUserClient.UserV1(), groupCache)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	for _, tc := range testCases {
		allowed, err := tc.checker.Allowed(tc.subject, ctx)
		if err != nil {
			t.Errorf("test case %v: unexpected error: %v", tc.name, err)
		}
		if allowed && !tc.shouldAllow {
			t.Errorf("test case %v: subject allowed but should be prohibited", tc.name)
		}
		if !allowed && tc.shouldAllow {
			t.Errorf("test case %v: subject prohibited but should be allowed", tc.name)
		}
	}
}
