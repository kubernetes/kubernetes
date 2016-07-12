/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package protectedattributes_test

import (
	"fmt"
	"testing"

	"k8s.io/kubernetes/pkg/admission"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/rbac"
	kuser "k8s.io/kubernetes/pkg/auth/user"
	clientsetfake "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/fake"
	"k8s.io/kubernetes/pkg/runtime"
	utilrand "k8s.io/kubernetes/pkg/util/rand"
	"k8s.io/kubernetes/plugin/pkg/admission/protectedattributes"
)

func TestProtectedAttributesOnCreate(t *testing.T) {
	testCases := map[string]struct {
		objects []runtime.Object
		sa      *api.ServiceAccount
		allowed []rbac.Subject
		denied  []rbac.Subject
	}{
		"protected label": {
			[]runtime.Object{
				clusterRole("admin"),
				clusterRoleBinding("admin", user("alice"), user("bob")),
				roleBindingToClusterRole("devNS", "admin", user("joe")),
				roleBindingToClusterRole("devNS", "tester", user("karl")),

				clusterRole("tester"),
				clusterRoleBinding("tester", user("ivan")),

				role("devNS", "admin"),
				roleBinding("devNS", "admin", user("george"), group("hr")),

				role("devNS", "developer"),
				roleBinding("devNS", "developer", user("carol"), group("devs")),

				role("devNS", "tester"),
				roleBinding("devNS", "tester", user("eve"), group("finance")),

				clusterProtectedLabel("env", "admin"),
				protectedLabel("devNS", "env", "developer"),
				protectedLabelForClusterRole("devNS", "env", "tester"),
			},
			serviceAccount("devNS", map[string]string{"env": "prod"}, nil),
			[]rbac.Subject{ // Allowed.
				user("alice"), user("bob"), user("joe"), // Cluster admins.
				user("carol"), group("devs"), // devNS developers.
				user("ivan"), user("karl"), // Cluster testers.
			},
			[]rbac.Subject{ // Denied.
				user("eve"), group("finance"), // devNS testers.
				user("george"), group("hr"), // devNS admins.
				user("foobar"), // User with no roles.
			},
		},
		"protected annotation": {
			[]runtime.Object{
				clusterRole("admin"),
				clusterRoleBinding("admin", user("alice"), user("bob")),
				roleBindingToClusterRole("devNS", "admin", user("joe")),
				roleBindingToClusterRole("devNS", "tester", user("karl")),

				clusterRole("tester"),
				clusterRoleBinding("tester", user("ivan")),

				role("devNS", "admin"),
				roleBinding("devNS", "admin", user("george"), group("hr")),

				role("devNS", "developer"),
				roleBinding("devNS", "developer", user("carol"), group("devs")),

				role("devNS", "tester"),
				roleBinding("devNS", "tester", user("eve"), group("finance")),

				clusterProtectedAnnotation("k8s.io/foo", "admin"),
				protectedAnnotation("devNS", "k8s.io/foo", "developer"),
				protectedAnnotationForClusterRole("devNS", "k8s.io/foo", "tester"),
			},
			serviceAccount("devNS", nil, map[string]string{"k8s.io/foo": "prod"}),
			[]rbac.Subject{ // Allowed.
				user("alice"), user("bob"), user("joe"), // Cluster admins.
				user("carol"), group("devs"), // devNS developers.
				user("ivan"), user("karl"), // Cluster testers.
			},
			[]rbac.Subject{ // Denied.
				user("eve"), group("finance"), // devNS testers.
				user("george"), group("hr"), // devNS admins.
				user("foobar"), // User with no roles.
			},
		},
		"protected label values": {
			[]runtime.Object{
				clusterRole("admin"),
				clusterRoleBinding("admin", user("alice"), user("bob")),
				roleBindingToClusterRole("devNS", "admin", user("joe")),
				roleBindingToClusterRole("devNS", "tester", user("karl")),

				clusterRole("tester"),
				clusterRoleBinding("tester", user("ivan")),

				role("devNS", "admin"),
				roleBinding("devNS", "admin", user("george"), group("hr")),

				role("devNS", "developer"),
				roleBinding("devNS", "developer", user("carol"), group("devs")),

				role("devNS", "tester"),
				roleBinding("devNS", "tester", user("eve"), group("finance")),

				clusterProtectedLabel("env", "admin", "dev", "prod"),
				protectedLabel("devNS", "env", "developer", "dev", "test"),
				protectedLabelForClusterRole("devNS", "env", "tester", "prod", "test"),
			},
			serviceAccount("devNS", map[string]string{"env": "prod"}, nil),
			[]rbac.Subject{ // Allowed.
				user("alice"), user("bob"), user("joe"), // Cluster admins.
				user("ivan"), user("karl"), // Cluster testers.
			},
			[]rbac.Subject{ // Denied.
				user("carol"), group("devs"), // devNS developers.
				user("eve"), group("finance"), // devNS testers.
				user("george"), group("hr"), // devNS admins.
				user("foobar"), // User with no roles.
			},
		},
		"protected annotation values": {
			[]runtime.Object{
				clusterRole("admin"),
				clusterRoleBinding("admin", user("alice"), user("bob")),
				roleBindingToClusterRole("devNS", "admin", user("joe")),
				roleBindingToClusterRole("devNS", "tester", user("karl")),

				clusterRole("tester"),
				clusterRoleBinding("tester", user("ivan")),

				role("devNS", "admin"),
				roleBinding("devNS", "admin", user("george"), group("hr")),

				role("devNS", "developer"),
				roleBinding("devNS", "developer", user("carol"), group("devs")),

				role("devNS", "tester"),
				roleBinding("devNS", "tester", user("eve"), group("finance")),

				clusterProtectedAnnotation("k8s.io/foo", "admin", "dev", "prod"),
				protectedAnnotation("devNS", "k8s.io/foo", "developer", "dev", "test"),
				protectedAnnotationForClusterRole("devNS", "k8s.io/foo", "tester", "prod", "test"),
			},
			serviceAccount("devNS", nil, map[string]string{"k8s.io/foo": "prod"}),
			[]rbac.Subject{ // Allowed.
				user("alice"), user("bob"), user("joe"), // Cluster admins.
				user("ivan"), user("karl"), // Cluster testers.
			},
			[]rbac.Subject{ // Denied.
				user("carol"), group("devs"), // devNS developers.
				user("eve"), group("finance"), // devNS testers.
				user("george"), group("hr"), // devNS admins.
				user("foobar"), // User with no roles.
			},
		},
		"namespace scope": {
			[]runtime.Object{
				role("devNS", "admin"),
				roleBinding("devNS", "admin", user("alice")),

				protectedLabel("devNS", "env", "admin"),
			},
			// This service account is in testNS, so devNS protected
			// attributes must not apply to it.
			serviceAccount("testNS", map[string]string{"env": "prod"}, nil),
			[]rbac.Subject{ // Allowed.
				user("alice"), user("bob"), user("carol"), group("devs"),
			},
			[]rbac.Subject{},
		},
		"cluster scope": {
			[]runtime.Object{
				clusterRole("admin"),
				clusterRole("developer"),
				clusterRoleBinding("admin", user("alice")),
				clusterRoleBinding("developer", user("bob")),

				clusterProtectedLabel("env", "admin"),
				clusterProtectedLabel("env", "developer"),

				role("devNS", "admin"),
				role("devNS", "developer"),
				roleBinding("devNS", "admin", user("carol")),
				roleBinding("devNS", "developer", group("devs")),
				roleBindingToClusterRole("testNS", "admin", user("carol"), group("devs")),
				roleBindingToClusterRole("testNS", "developer", user("carol"), group("devs")),
			},
			// Only cluster admins and developers can add 'env' in
			// 'devNS'. Role bindings to cluster roles in testNS do
			// not matter in devNS.
			serviceAccount("devNS", map[string]string{"env": "prod"}, nil),
			[]rbac.Subject{ // Allowed.
				user("alice"), user("bob"),
			},
			[]rbac.Subject{ // Denied.
				user("carol"), group("devs"), user("ivan"),
			},
		},
		"label/annotation mix": {
			[]runtime.Object{
				clusterRole("admin"),
				clusterRoleBinding("admin", user("alice"), user("bob")),

				role("devNS", "admin"),
				role("devNS", "developer"),
				role("devNS", "tester"),
				roleBinding("devNS", "admin", user("alice"), user("carol"), user("bob"), user("ivan")),
				roleBinding("devNS", "developer", user("alice"), group("devs"), user("ivan")),
				roleBinding("devNS", "tester", group("devs"), user("carol")),

				protectedLabel("devNS", "env", "admin", "value1", "value2"),
				protectedLabel("devNS", "env", "developer"),
				protectedLabel("devNS", "app", "developer"),
				protectedLabel("devNS", "env", "tester"),
				protectedLabel("devNS", "app", "tester"),
				protectedLabelForClusterRole("devNS", "env", "admin"),

				clusterProtectedAnnotation("version", "developer"),
				clusterProtectedAnnotation("version", "admin", "42"),

				protectedAnnotation("devNS", "k8s.io/foo", "admin"),
				protectedAnnotation("devNS", "k8s.io/foo", "developer"),
				protectedAnnotation("devNS", "k8s.io/foo", "tester"),
				protectedAnnotation("devNS", "version", "tester", "43"),
			},
			// Only someone who has all three of "admin", "developer"
			// and cluster "admin" roles can set the given mix of
			// labels and annotations (devNS/tester is almost there
			// but can only use annotation "version" with value "43",
			// not "42").
			serviceAccount("devNS",
				map[string]string{"env": "prod", "app": "nginx"},        // Labels.
				map[string]string{"k8s.io/foo": "bar", "version": "42"}, // Annotations.
			),
			[]rbac.Subject{ // Allowed.
				user("alice"),
			},
			[]rbac.Subject{ // Denied.
				user("bob"), user("carol"), group("devs"), user("ivan"),
			},
		},
	}

	for tcName, tc := range testCases {
		clientset := clientsetfake.NewSimpleClientset(tc.objects...)
		ac := protectedattributes.NewProtectedAttributes(clientset)

		parseSubject := func(subject rbac.Subject) (string, []string) {
			switch subject.Kind {
			case "User":
				return subject.Name, []string{}
			case "Group":
				return "nobody", []string{subject.Name}
			default:
				return "nobody", []string{}
			}
		}

		for _, subject := range tc.allowed {
			user, groups := parseSubject(subject)
			err := ac.Admit(admissionAttrs(admission.Create, tc.sa, nil, user, groups))
			if err != nil {
				t.Errorf("Expected %q to succeed for user %q (groups: %q), got error: %q", tcName, user, groups, err)
			}
		}

		for _, subject := range tc.denied {
			user, groups := parseSubject(subject)
			err := ac.Admit(admissionAttrs(admission.Create, tc.sa, nil, user, groups))
			if err == nil {
				t.Errorf("Expected %q to error for user %q (groups: %q), but it was allowed", tcName, user, groups)
			}
		}
	}
}

func TestAddProtectedAttributesOnUpdate(t *testing.T) {
	clientset := clientsetfake.NewSimpleClientset(
		clusterRole("admin"),
		clusterRoleBinding("admin", user("alice")),

		role("devNS", "admin"),
		roleBinding("devNS", "admin", user("bob")),

		protectedLabelForClusterRole("devNS", "env", "admin"),
	)
	ac := protectedattributes.NewProtectedAttributes(clientset)

	oldSA := serviceAccount("devNS", nil, nil)
	newSA := serviceAccount("devNS", map[string]string{"env": "prod"}, nil)

	err := ac.Admit(admissionAttrs(admission.Update, newSA, oldSA, "alice", nil))
	if err != nil {
		t.Errorf("Expected no error on update for alice, got: %s", err)
	}

	err = ac.Admit(admissionAttrs(admission.Update, newSA, oldSA, "bob", nil))
	if err == nil {
		t.Error("Expected an error on update for bob, got no error")
	}
}

func TestRemoveProtectedAttributesOnUpdate(t *testing.T) {
	clientset := clientsetfake.NewSimpleClientset(
		clusterRole("admin"),
		clusterRoleBinding("admin", user("alice")),

		role("devNS", "admin"),
		roleBinding("devNS", "admin", user("bob")),

		protectedLabelForClusterRole("devNS", "env", "admin"),
		protectedLabel("devNS", "env", "admin", "dev"),
	)
	ac := protectedattributes.NewProtectedAttributes(clientset)

	oldSA := serviceAccount("devNS", map[string]string{"env": "prod"}, nil)
	newSA := serviceAccount("devNS", map[string]string{"foo": "bar"}, nil)

	err := ac.Admit(admissionAttrs(admission.Update, newSA, oldSA, "alice", nil))
	if err != nil {
		t.Errorf("Expected no error on update for alice, got: %s", err)
	}

	err = ac.Admit(admissionAttrs(admission.Update, newSA, oldSA, "bob", nil))
	if err == nil {
		t.Error("Expected an error on update for bob, got no error")
	}
}

func TestProtectedAttributesOnDelete(t *testing.T) {
	clientset := clientsetfake.NewSimpleClientset(
		role("devNS", "admin"),
		roleBinding("devNS", "admin", user("alice")),

		protectedLabel("devNS", "env", "admin"),
	)
	ac := protectedattributes.NewProtectedAttributes(clientset)

	sa := serviceAccount("devNS", map[string]string{"env": "prod"}, nil)

	err := ac.Admit(admissionAttrsOnDelete(sa, "alice", nil))
	if err != nil {
		t.Errorf("Expected no error on update for alice, got: %s", err)
	}

	err = ac.Admit(admissionAttrsOnDelete(sa, "bob", nil))
	if err == nil {
		t.Error("Expected an error on update for bob, got no error")
	}
}

func clusterRole(name string) *rbac.ClusterRole {
	return &rbac.ClusterRole{
		ObjectMeta: api.ObjectMeta{
			Name: name,
		},
	}
}

func role(ns, name string) *rbac.Role {
	return &rbac.Role{
		ObjectMeta: api.ObjectMeta{
			Namespace: ns,
			Name:      name,
		},
	}
}

func clusterRoleBinding(clusterRoleName string, subjects ...rbac.Subject) *rbac.ClusterRoleBinding {
	return &rbac.ClusterRoleBinding{
		ObjectMeta: api.ObjectMeta{
			Name: randomName("cluster-role-binding-"),
		},
		RoleRef: api.ObjectReference{
			Kind: "ClusterRole",
			Name: clusterRoleName,
		},
		Subjects: subjects,
	}
}

func roleBinding(ns, roleName string, subjects ...rbac.Subject) *rbac.RoleBinding {
	return &rbac.RoleBinding{
		ObjectMeta: api.ObjectMeta{
			Namespace: ns,
			Name:      randomName("role-binding-"),
		},
		RoleRef: api.ObjectReference{
			Kind:      "Role",
			Namespace: ns,
			Name:      roleName,
		},
		Subjects: subjects,
	}
}

func roleBindingToClusterRole(ns, clusterRoleName string, subjects ...rbac.Subject) *rbac.RoleBinding {
	return &rbac.RoleBinding{
		ObjectMeta: api.ObjectMeta{
			Namespace: ns,
			Name:      randomName("role-binding-"),
		},
		RoleRef: api.ObjectReference{
			Kind: "ClusterRole",
			Name: clusterRoleName,
		},
		Subjects: subjects,
	}
}

func clusterProtectedLabel(name, roleName string, values ...string) *rbac.ClusterProtectedAttribute {
	return &rbac.ClusterProtectedAttribute{
		ObjectMeta: api.ObjectMeta{
			Name: randomName("cpa-"),
		},
		AttributeKind: "Label",
		AttributeName: name,
		RoleRef: api.ObjectReference{
			Kind: "ClusterRole",
			Name: roleName,
		},
	}
}

func clusterProtectedAnnotation(name, roleName string, values ...string) *rbac.ClusterProtectedAttribute {
	return &rbac.ClusterProtectedAttribute{
		ObjectMeta: api.ObjectMeta{
			Name: randomName("cpa-"),
		},
		AttributeKind: "Annotation",
		AttributeName: name,
		RoleRef: api.ObjectReference{
			Kind: "ClusterRole",
			Name: roleName,
		},
		ProtectedValues: values,
	}
}

func protectedLabel(ns, name, roleName string, values ...string) *rbac.ProtectedAttribute {
	return &rbac.ProtectedAttribute{
		ObjectMeta: api.ObjectMeta{
			Namespace: ns,
			Name:      randomName("pa-"),
		},
		AttributeKind: "Label",
		AttributeName: name,
		RoleRef: api.ObjectReference{
			Kind:      "Role",
			Namespace: ns,
			Name:      roleName,
		},
		ProtectedValues: values,
	}
}

func protectedLabelForClusterRole(ns, name, clusterRoleName string, values ...string) *rbac.ProtectedAttribute {
	return &rbac.ProtectedAttribute{
		ObjectMeta: api.ObjectMeta{
			Namespace: ns,
			Name:      randomName("pa-"),
		},
		AttributeKind: "Label",
		AttributeName: name,
		RoleRef: api.ObjectReference{
			Kind: "ClusterRole",
			Name: clusterRoleName,
		},
		ProtectedValues: values,
	}
}

func protectedAnnotation(ns, name, roleName string, values ...string) *rbac.ProtectedAttribute {
	return &rbac.ProtectedAttribute{
		ObjectMeta: api.ObjectMeta{
			Namespace: ns,
			Name:      randomName("pa-"),
		},
		AttributeKind: "Annotation",
		AttributeName: name,
		RoleRef: api.ObjectReference{
			Kind:      "Role",
			Namespace: ns,
			Name:      roleName,
		},
		ProtectedValues: values,
	}
}

func protectedAnnotationForClusterRole(ns, name, clusterRoleName string, values ...string) *rbac.ProtectedAttribute {
	return &rbac.ProtectedAttribute{
		ObjectMeta: api.ObjectMeta{
			Namespace: ns,
			Name:      randomName("pa-"),
		},
		AttributeKind: "Annotation",
		AttributeName: name,
		RoleRef: api.ObjectReference{
			Kind: "ClusterRole",
			Name: clusterRoleName,
		},
		ProtectedValues: values,
	}
}

func user(name string) rbac.Subject {
	return rbac.Subject{Kind: "User", Name: name}
}

func group(name string) rbac.Subject {
	return rbac.Subject{Kind: "Group", Name: name}
}

// Using ServiceAccount as a resource being admitted purely out of
// convenience: any resource with labels/annotations can be used.
func serviceAccount(ns string, labels map[string]string, annotations map[string]string) *api.ServiceAccount {
	return &api.ServiceAccount{
		ObjectMeta: api.ObjectMeta{
			Namespace:   ns,
			Name:        "testServiceAccount",
			Labels:      labels,
			Annotations: annotations,
		},
	}
}

func admissionAttrs(op admission.Operation, newSA, oldSA *api.ServiceAccount, userName string, groups []string) admission.Attributes {
	return admission.NewAttributesRecord(
		newSA, oldSA,
		newSA.GroupVersionKind(),
		newSA.Namespace,
		newSA.Name,
		api.Resource("serviceaccounts").WithVersion("version"),
		"", // No subresource.
		op,
		&kuser.DefaultInfo{Name: userName, Groups: groups},
	)
}

func admissionAttrsOnDelete(sa *api.ServiceAccount, userName string, groups []string) admission.Attributes {
	return admission.NewAttributesRecord(
		nil, sa,
		sa.GroupVersionKind(),
		sa.Namespace,
		sa.Name,
		api.Resource("serviceaccounts").WithVersion("version"),
		"", // No subresource.
		admission.Delete,
		&kuser.DefaultInfo{Name: userName, Groups: groups},
	)
}

func randomName(prefix string) string {
	return fmt.Sprintf("%s%s", prefix, utilrand.String(6))
}
