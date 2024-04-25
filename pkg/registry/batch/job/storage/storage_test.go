/*
Copyright 2015 The Kubernetes Authors.

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

package storage

import (
	"testing"

	"k8s.io/utils/ptr"

	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/apis/meta/internalversion"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/generic"
	genericregistrytest "k8s.io/apiserver/pkg/registry/generic/testing"
	"k8s.io/apiserver/pkg/registry/rest"
	etcd3testing "k8s.io/apiserver/pkg/storage/etcd3/testing"
	"k8s.io/apiserver/pkg/warning"
	podtest "k8s.io/kubernetes/pkg/api/pod/testing"
	"k8s.io/kubernetes/pkg/apis/batch"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/registry/registrytest"
)

func newStorage(t *testing.T) (*JobStorage, *etcd3testing.EtcdTestServer) {
	etcdStorage, server := registrytest.NewEtcdStorage(t, batch.GroupName)
	restOptions := generic.RESTOptions{
		StorageConfig:           etcdStorage,
		Decorator:               generic.UndecoratedStorage,
		DeleteCollectionWorkers: 1,
		ResourcePrefix:          "jobs",
	}
	jobStorage, err := NewStorage(restOptions)
	if err != nil {
		t.Fatalf("unexpected error from REST storage: %v", err)
	}
	return &jobStorage, server
}

func validNewJob() *batch.Job {
	completions := int32(1)
	parallelism := int32(1)
	return &batch.Job{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "foo",
			Namespace: "default",
		},
		Spec: batch.JobSpec{
			Completions: &completions,
			Parallelism: &parallelism,
			Selector: &metav1.LabelSelector{
				MatchLabels: map[string]string{"a": "b"},
			},
			ManualSelector: newBool(true),
			Template: api.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{"a": "b"},
				},
				Spec: podtest.MakePodSpec(api.RestartPolicyOnFailure),
			},
		},
	}
}

func validNewV1Job() *batchv1.Job {
	completions := int32(1)
	parallelism := int32(1)
	return &batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "foo",
			Namespace: "default",
		},
		Spec: batchv1.JobSpec{
			Completions: &completions,
			Parallelism: &parallelism,
			Selector: &metav1.LabelSelector{
				MatchLabels: map[string]string{"a": "b"},
			},
			ManualSelector: newBool(true),
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{"a": "b"},
				},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name:                     "test",
							Image:                    "test_image",
							ImagePullPolicy:          corev1.PullIfNotPresent,
							TerminationMessagePolicy: corev1.TerminationMessageReadFile,
						},
					},
					RestartPolicy: corev1.RestartPolicyOnFailure,
					DNSPolicy:     corev1.DNSClusterFirst,
				},
			},
		},
	}
}

func TestCreate(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Job.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Job.Store)
	validJob := validNewJob()
	validJob.ObjectMeta = metav1.ObjectMeta{}
	test.TestCreate(
		// valid
		validJob,
		// invalid (empty selector)
		&batch.Job{
			Spec: batch.JobSpec{
				ManualSelector: ptr.To(false),
				Completions:    validJob.Spec.Completions,
				Selector:       &metav1.LabelSelector{},
				Template:       validJob.Spec.Template,
			},
		},
	)
}

func TestUpdate(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Job.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Job.Store)
	two := int32(2)
	test.TestUpdate(
		// valid
		validNewJob(),
		// updateFunc
		func(obj runtime.Object) runtime.Object {
			object := obj.(*batch.Job)
			object.Spec.Parallelism = &two
			return object
		},
		// invalid updateFunc
		func(obj runtime.Object) runtime.Object {
			object := obj.(*batch.Job)
			object.Spec.Selector = &metav1.LabelSelector{}
			return object
		},
		func(obj runtime.Object) runtime.Object {
			object := obj.(*batch.Job)
			object.Spec.Completions = &two
			return object
		},
	)
}

func TestDelete(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Job.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Job.Store)
	test.TestDelete(validNewJob())
}

type dummyRecorder struct {
	agent string
	text  string
}

func (r *dummyRecorder) AddWarning(agent, text string) {
	r.agent = agent
	r.text = text
	return
}

func (r *dummyRecorder) getWarning() string {
	return r.text
}

var _ warning.Recorder = &dummyRecorder{}

func TestJobDeletion(t *testing.T) {
	orphanDependents := true
	orphanDeletionPropagation := metav1.DeletePropagationOrphan
	backgroundDeletionPropagation := metav1.DeletePropagationBackground
	job := validNewV1Job()
	ctx := genericapirequest.NewDefaultContext()
	key := "/jobs/" + metav1.NamespaceDefault + "/foo"
	tests := []struct {
		description   string
		expectWarning bool
		deleteOptions *metav1.DeleteOptions
		listOptions   *internalversion.ListOptions
		requestInfo   *genericapirequest.RequestInfo
	}{
		{
			description:   "deletion: no policy, v1, warning",
			expectWarning: true,
			deleteOptions: &metav1.DeleteOptions{},
			requestInfo:   &genericapirequest.RequestInfo{APIGroup: "batch", APIVersion: "v1"},
		},
		{
			description:   "deletion: no policy, v2, no warning",
			expectWarning: false,
			deleteOptions: &metav1.DeleteOptions{},
			requestInfo:   &genericapirequest.RequestInfo{APIGroup: "batch", APIVersion: "v2"},
		},
		{
			description:   "deletion: no policy, no APIVersion, no warning",
			expectWarning: false,
			deleteOptions: &metav1.DeleteOptions{},
			requestInfo:   &genericapirequest.RequestInfo{APIGroup: "batch", APIVersion: ""},
		},
		{
			description:   "deletion: orphan dependents, no warnings",
			expectWarning: false,
			deleteOptions: &metav1.DeleteOptions{OrphanDependents: &orphanDependents},
			requestInfo:   &genericapirequest.RequestInfo{APIGroup: "batch", APIVersion: "v1"},
		},
		{
			description:   "deletion: orphan deletion, no warnings",
			expectWarning: false,
			deleteOptions: &metav1.DeleteOptions{PropagationPolicy: &orphanDeletionPropagation},
			requestInfo:   &genericapirequest.RequestInfo{APIGroup: "batch", APIVersion: "v1"},
		},
		{
			description:   "deletion: background deletion, no warnings",
			expectWarning: false,
			deleteOptions: &metav1.DeleteOptions{PropagationPolicy: &backgroundDeletionPropagation},
			requestInfo:   &genericapirequest.RequestInfo{APIGroup: "batch", APIVersion: "v1"},
		},
		{
			description:   "deleteCollection: no policy, v1, warning",
			expectWarning: true,
			deleteOptions: &metav1.DeleteOptions{},
			listOptions:   &internalversion.ListOptions{},
			requestInfo:   &genericapirequest.RequestInfo{APIGroup: "batch", APIVersion: "v1"},
		},
		{
			description:   "deleteCollection: no policy, v2, no warning",
			expectWarning: false,
			deleteOptions: &metav1.DeleteOptions{},
			listOptions:   &internalversion.ListOptions{},
			requestInfo:   &genericapirequest.RequestInfo{APIGroup: "batch", APIVersion: "v2"},
		},
		{
			description:   "deleteCollection: no policy, no APIVersion, no warning",
			expectWarning: false,
			deleteOptions: &metav1.DeleteOptions{},
			listOptions:   &internalversion.ListOptions{},
			requestInfo:   &genericapirequest.RequestInfo{APIGroup: "batch", APIVersion: ""},
		},
		{
			description:   "deleteCollection: orphan dependents, no warnings",
			expectWarning: false,
			deleteOptions: &metav1.DeleteOptions{OrphanDependents: &orphanDependents},
			listOptions:   &internalversion.ListOptions{},
			requestInfo:   &genericapirequest.RequestInfo{APIGroup: "batch", APIVersion: "v1"},
		},
		{
			description:   "deletionCollection: orphan deletion, no warnings",
			expectWarning: false,
			deleteOptions: &metav1.DeleteOptions{PropagationPolicy: &orphanDeletionPropagation},
			listOptions:   &internalversion.ListOptions{},
			requestInfo:   &genericapirequest.RequestInfo{APIGroup: "batch", APIVersion: "v1"},
		},
		{
			description:   "deletionCollection: background deletion, no warnings",
			expectWarning: false,
			deleteOptions: &metav1.DeleteOptions{PropagationPolicy: &backgroundDeletionPropagation},
			listOptions:   &internalversion.ListOptions{},
			requestInfo:   &genericapirequest.RequestInfo{APIGroup: "batch", APIVersion: "v1"},
		},
	}
	for _, test := range tests {
		t.Run(test.description, func(t *testing.T) {
			storage, server := newStorage(t)
			defer server.Terminate(t)
			defer storage.Job.Store.DestroyFunc()
			dc := dummyRecorder{agent: "", text: ""}
			ctx = genericapirequest.WithRequestInfo(ctx, test.requestInfo)
			ctxWithRecorder := warning.WithWarningRecorder(ctx, &dc)
			// Create the object
			if err := storage.Job.Storage.Create(ctxWithRecorder, key, job, nil, 0, false); err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			_, _, err := storage.Job.Delete(ctxWithRecorder, job.Name, rest.ValidateAllObjectFunc, test.deleteOptions)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			_, err = storage.Job.DeleteCollection(ctxWithRecorder, rest.ValidateAllObjectFunc, test.deleteOptions, test.listOptions)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if test.expectWarning {
				if dc.getWarning() != deleteOptionWarnings {
					t.Fatalf("expected delete option warning but did not get one")
				}
			}
		})
	}
}

func TestGet(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Job.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Job.Store)
	test.TestGet(validNewJob())
}

func TestList(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Job.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Job.Store)
	test.TestList(validNewJob())
}

func TestWatch(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Job.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Job.Store)
	test.TestWatch(
		validNewJob(),
		// matching labels
		[]labels.Set{},
		// not matching labels
		[]labels.Set{
			{"x": "y"},
		},
		// matching fields
		[]fields.Set{},
		// not matching fields
		[]fields.Set{
			{"metadata.name": "xyz"},
			{"name": "foo"},
		},
	)
}

// TODO: test update /status

func newBool(val bool) *bool {
	p := new(bool)
	*p = val
	return p
}

func TestCategories(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Job.Store.DestroyFunc()
	expected := []string{"all"}
	registrytest.AssertCategories(t, storage.Job, expected)
}
