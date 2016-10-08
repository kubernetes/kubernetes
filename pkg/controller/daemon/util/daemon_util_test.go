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

package daemon

import (
	"fmt"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apis/extensions"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/fake"
	"k8s.io/kubernetes/pkg/client/testing/core"
	"k8s.io/kubernetes/pkg/runtime"
	labelsutil "k8s.io/kubernetes/pkg/util/labels"
	podutil "k8s.io/kubernetes/pkg/util/pod"
)

var (
	simpleDaemonSetLabel = map[string]string{"name": "simple-daemon", "type": "production"}
)

func newPodTemplateSpec(image string) api.PodTemplateSpec {
	return api.PodTemplateSpec{
		ObjectMeta: api.ObjectMeta{
			Labels:      simpleDaemonSetLabel,
			Annotations: map[string]string{},
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Image: image,
				},
			},
		},
	}
}

func newPodTemplate(image, revision string) *api.PodTemplate {
	template := newPodTemplateSpec(image)
	hash := podutil.GetPodTemplateSpecHash(template)

	podTemplate := api.PodTemplate{
		ObjectMeta: api.ObjectMeta{
			Name:        fmt.Sprintf("image-%s", image),
			Labels:      map[string]string{},
			Annotations: map[string]string{},
		},
		Template: template,
	}
	podTemplate.ObjectMeta.Annotations[RevisionAnnotation] = revision
	podTemplate.ObjectMeta.Labels = labelsutil.CloneAndAddLabel(
		simpleDaemonSetLabel,
		extensions.DefaultDaemonSetUniqueLabelKey,
		hash,
	)
	return &podTemplate
}

func newPodTemplates(templates []*api.PodTemplate) *api.PodTemplateList {
	podTemplateList := api.PodTemplateList{}
	for _, template := range templates {
		podTemplateList.Items = append(podTemplateList.Items, *template)
	}
	return &podTemplateList
}

func newDaemonSet(name, image string) *extensions.DaemonSet {
	return &extensions.DaemonSet{
		ObjectMeta: api.ObjectMeta{
			Name:      name,
			Namespace: api.NamespaceDefault,
		},
		Spec: extensions.DaemonSetSpec{
			Selector: &unversioned.LabelSelector{MatchLabels: simpleDaemonSetLabel},
			Template: newPodTemplateSpec(image),
		},
	}
}

func NewPodTemplateController(c clientset.Interface) *PodTemplateController {
	return &PodTemplateController{KubeClient: c}
}

func addListPodTemplatesReactor(fakeClient *fake.Clientset, obj runtime.Object) *fake.Clientset {
	fakeClient.AddReactor("list", "podtemplates", func(action core.Action) (handled bool, ret runtime.Object, err error) {
		return true, obj, nil
	})
	return fakeClient
}

func addCreatePodTemplateReactor(fakeClient *fake.Clientset) *fake.Clientset {
	fakeClient.AddReactor("create", "podtemplates", func(action core.Action) (handled bool, ret runtime.Object, err error) {
		obj := action.(core.CreateAction).GetObject().(*api.PodTemplate)
		return true, obj, nil
	})
	return fakeClient
}

func TestGetOrCreatePodTemplate(t *testing.T) {
	tests := []struct {
		test     string
		objs     []runtime.Object
		expected string
	}{
		{
			test: "no matching element in the list",
			objs: []runtime.Object{
				newPodTemplates([]*api.PodTemplate{
					newPodTemplate("xx", "4"),
					newPodTemplate("yy", "2")}),
			},
			expected: "5",
		},
	}
	for _, test := range tests {
		fakeClient := &fake.Clientset{}
		fakeClient = addListPodTemplatesReactor(fakeClient, test.objs[0])
		fakeClient = addCreatePodTemplateReactor(fakeClient)

		ptc := NewPodTemplateController(fakeClient)
		ds := newDaemonSet("foo", "nginx")
		podTmpl, err := GetOrCreatePodTemplate(ptc, ds, fakeClient)
		if err != nil {
			t.Errorf("In test case %s, got unexpected error %v", test.test, err)
		}

		podTmplRevision := podTmpl.ObjectMeta.Annotations[RevisionAnnotation]
		if podTmplRevision != test.expected {
			t.Errorf("In test case %s, expected: %s got:%s", test.test, test.expected, podTmplRevision)
		}
	}
}

func TestRevisionParsing(t *testing.T) {
	tests := []struct {
		name             string
		podTemplate      api.PodTemplate
		expectedRevision int64
		expectError      bool
	}{
		{
			name:             "positive number to int64",
			podTemplate:      *newPodTemplate("foo", "1"),
			expectedRevision: 1,
			expectError:      false,
		},
		{
			name:        "wrong value",
			podTemplate: *newPodTemplate("foo", ""),
			expectError: true,
		},
	}
	for _, test := range tests {
		revision, err := Revision(&test.podTemplate)
		if test.expectError {
			if err == nil {
				t.Errorf("[%s] Unexpected succes when converting revision. Excpected error, got %d", test.name, revision)
			}
		} else {
			if err != nil {
				t.Errorf("[%s] Unexpected error when converting revision. Err: %s", test.name, err)
			}
			if revision != test.expectedRevision {
				t.Errorf("[%s] Error when parsing revision: expexted %d, got %d", test.name, test.expectedRevision, revision)
			}
		}

	}
}

func TestMaxRevision(t *testing.T) {
	tests := []struct {
		name             string
		podTemplates     api.PodTemplateList
		expectedRevision int64
	}{
		{
			name:             "one element",
			podTemplates:     *newPodTemplates([]*api.PodTemplate{newPodTemplate("foo", "1")}),
			expectedRevision: 1,
		},
		{
			name:             "empty list",
			podTemplates:     *newPodTemplates([]*api.PodTemplate{}),
			expectedRevision: 0,
		},
		{
			name: "empty list",
			podTemplates: *newPodTemplates([]*api.PodTemplate{
				newPodTemplate("x", "2"),
				newPodTemplate("y", "5"),
				newPodTemplate("z", "1"),
			}),
			expectedRevision: 5,
		},
	}
	for _, test := range tests {
		revision := MaxRevision(&test.podTemplates)
		if revision != test.expectedRevision {
			t.Errorf("[%s] Error when searching max revision: expexted %d, got %d", test.name, test.expectedRevision, revision)
		}

	}
}
