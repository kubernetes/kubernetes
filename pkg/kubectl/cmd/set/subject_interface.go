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

package set

import (
	rbacv1 "k8s.io/api/rbac/v1"
	rbacv1alpha1 "k8s.io/api/rbac/v1alpha1"
	rbacv1beta1 "k8s.io/api/rbac/v1beta1"
)

type subjectInterface interface {
	GetKind() string
	GetAPIGroup() string
	GetName() string
	GetNamespace() string
}

func equals(s1, s2 subjectInterface) bool {
	if s1.GetKind() != s2.GetKind() {
		return false
	}
	if s1.GetAPIGroup() != s2.GetAPIGroup() {
		return false
	}
	if s1.GetName() != s2.GetName() {
		return false
	}
	if s1.GetNamespace() != s2.GetNamespace() {
		return false
	}
	return true
}

func V1SubjectsToSubjectInterfaces(s []rbacv1.Subject) []subjectInterface {
	si := []subjectInterface{}
	for _, subject := range s {
		si = append(si, v1subject{subject})
	}
	return si
}

func SubjectInterfacesToV1Subjects(si []subjectInterface) []rbacv1.Subject {
	s := []rbacv1.Subject{}
	for _, subject := range si {
		v1subj := subject.(v1subject)
		s = append(s, v1subj.Subject)
	}
	return s
}

var _ subjectInterface = v1subject{}

type v1subject struct {
	rbacv1.Subject
}

func (s v1subject) GetKind() string {
	return s.Kind
}

func (s v1subject) GetAPIGroup() string {
	return s.APIGroup
}

func (s v1subject) GetName() string {
	return s.Name
}

func (s v1subject) GetNamespace() string {
	return s.Namespace
}

func V1Beta1SubjectsToSubjectInterfaces(s []rbacv1beta1.Subject) []subjectInterface {
	si := []subjectInterface{}
	for _, subject := range s {
		si = append(si, v1beta1subject{subject})
	}
	return si
}

func SubjectInterfacesToV1Beta1Subjects(si []subjectInterface) []rbacv1beta1.Subject {
	s := []rbacv1beta1.Subject{}
	for _, subject := range si {
		v1beta1subj := subject.(v1beta1subject)
		s = append(s, v1beta1subj.Subject)
	}
	return s
}

var _ subjectInterface = v1beta1subject{}

type v1beta1subject struct {
	rbacv1beta1.Subject
}

func (s v1beta1subject) GetKind() string {
	return s.Kind
}

func (s v1beta1subject) GetAPIGroup() string {
	return s.APIGroup
}

func (s v1beta1subject) GetName() string {
	return s.Name
}

func (s v1beta1subject) GetNamespace() string {
	return s.Namespace
}

func V1Alpha1SubjectsToSubjectInterfaces(s []rbacv1alpha1.Subject) []subjectInterface {
	si := []subjectInterface{}
	for _, subject := range s {
		si = append(si, v1alpha1subject{subject})
	}
	return si
}

func SubjectInterfacesToV1Alpha1Subjects(si []subjectInterface) []rbacv1alpha1.Subject {
	s := []rbacv1alpha1.Subject{}
	for _, subject := range si {
		v1alpha1subj := subject.(v1alpha1subject)
		s = append(s, v1alpha1subj.Subject)
	}
	return s
}

var _ subjectInterface = v1alpha1subject{}

type v1alpha1subject struct {
	rbacv1alpha1.Subject
}

func (s v1alpha1subject) GetKind() string {
	return s.Kind
}

func (s v1alpha1subject) GetAPIGroup() string {
	return s.APIVersion
}

func (s v1alpha1subject) GetName() string {
	return s.Name
}

func (s v1alpha1subject) GetNamespace() string {
	return s.Namespace
}
