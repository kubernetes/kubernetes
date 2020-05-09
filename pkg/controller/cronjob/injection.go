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

package cronjob

import (
	"context"
	"fmt"
	"sync"

	batchv1 "k8s.io/api/batch/v1"
	batchv1beta1 "k8s.io/api/batch/v1beta1"
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/record"
)

// cjControlInterface is an interface that knows how to update CronJob status
// created as an interface to allow testing.
type cjControlInterface interface {
	UpdateStatus(cj *batchv1beta1.CronJob) (*batchv1beta1.CronJob, error)
}

// realCJControl is the default implementation of cjControlInterface.
type realCJControl struct {
	KubeClient clientset.Interface
}

var _ cjControlInterface = &realCJControl{}

func (c *realCJControl) UpdateStatus(cj *batchv1beta1.CronJob) (*batchv1beta1.CronJob, error) {
	return c.KubeClient.BatchV1beta1().CronJobs(cj.Namespace).UpdateStatus(context.TODO(), cj, metav1.UpdateOptions{})
}

// fakeCJControl is the default implementation of cjControlInterface.
type fakeCJControl struct {
	Updates []batchv1beta1.CronJob
}

var _ cjControlInterface = &fakeCJControl{}

func (c *fakeCJControl) UpdateStatus(cj *batchv1beta1.CronJob) (*batchv1beta1.CronJob, error) {
	c.Updates = append(c.Updates, *cj)
	return cj, nil
}

// ------------------------------------------------------------------ //

// jobControlInterface is an interface that knows how to add or delete jobs
// created as an interface to allow testing.
type jobControlInterface interface {
	// GetJob retrieves a Job.
	GetJob(namespace, name string) (*batchv1.Job, error)
	// CreateJob creates new Jobs according to the spec.
	CreateJob(namespace string, job *batchv1.Job) (*batchv1.Job, error)
	// UpdateJob updates a Job.
	UpdateJob(namespace string, job *batchv1.Job) (*batchv1.Job, error)
	// PatchJob patches a Job.
	PatchJob(namespace string, name string, pt types.PatchType, data []byte, subresources ...string) (*batchv1.Job, error)
	// DeleteJob deletes the Job identified by name.
	// TODO: delete by UID?
	DeleteJob(namespace string, name string) error
}

// realJobControl is the default implementation of jobControlInterface.
type realJobControl struct {
	KubeClient clientset.Interface
	Recorder   record.EventRecorder
}

var _ jobControlInterface = &realJobControl{}

func copyLabels(template *batchv1beta1.JobTemplateSpec) labels.Set {
	l := make(labels.Set)
	for k, v := range template.Labels {
		l[k] = v
	}
	return l
}

func copyAnnotations(template *batchv1beta1.JobTemplateSpec) labels.Set {
	a := make(labels.Set)
	for k, v := range template.Annotations {
		a[k] = v
	}
	return a
}

func (r realJobControl) GetJob(namespace, name string) (*batchv1.Job, error) {
	return r.KubeClient.BatchV1().Jobs(namespace).Get(context.TODO(), name, metav1.GetOptions{})
}

func (r realJobControl) UpdateJob(namespace string, job *batchv1.Job) (*batchv1.Job, error) {
	return r.KubeClient.BatchV1().Jobs(namespace).Update(context.TODO(), job, metav1.UpdateOptions{})
}

func (r realJobControl) PatchJob(namespace string, name string, pt types.PatchType, data []byte, subresources ...string) (*batchv1.Job, error) {
	return r.KubeClient.BatchV1().Jobs(namespace).Patch(context.TODO(), name, pt, data, metav1.PatchOptions{}, subresources...)
}

func (r realJobControl) CreateJob(namespace string, job *batchv1.Job) (*batchv1.Job, error) {
	return r.KubeClient.BatchV1().Jobs(namespace).Create(context.TODO(), job, metav1.CreateOptions{})
}

func (r realJobControl) DeleteJob(namespace string, name string) error {
	background := metav1.DeletePropagationBackground
	return r.KubeClient.BatchV1().Jobs(namespace).Delete(context.TODO(), name, metav1.DeleteOptions{PropagationPolicy: &background})
}

type fakeJobControl struct {
	sync.Mutex
	Job           *batchv1.Job
	Jobs          []batchv1.Job
	DeleteJobName []string
	Err           error
	UpdateJobName []string
	PatchJobName  []string
	Patches       [][]byte
}

var _ jobControlInterface = &fakeJobControl{}

func (f *fakeJobControl) CreateJob(namespace string, job *batchv1.Job) (*batchv1.Job, error) {
	f.Lock()
	defer f.Unlock()
	if f.Err != nil {
		return nil, f.Err
	}
	job.SelfLink = fmt.Sprintf("/apis/batch/v1/namespaces/%s/jobs/%s", namespace, job.Name)
	f.Jobs = append(f.Jobs, *job)
	job.UID = "test-uid"
	return job, nil
}

func (f *fakeJobControl) GetJob(namespace, name string) (*batchv1.Job, error) {
	f.Lock()
	defer f.Unlock()
	if f.Err != nil {
		return nil, f.Err
	}
	return f.Job, nil
}

func (f *fakeJobControl) UpdateJob(namespace string, job *batchv1.Job) (*batchv1.Job, error) {
	f.Lock()
	defer f.Unlock()
	if f.Err != nil {
		return nil, f.Err
	}
	f.UpdateJobName = append(f.UpdateJobName, job.Name)
	return job, nil
}

func (f *fakeJobControl) PatchJob(namespace string, name string, pt types.PatchType, data []byte, subresources ...string) (*batchv1.Job, error) {
	f.Lock()
	defer f.Unlock()
	if f.Err != nil {
		return nil, f.Err
	}
	f.PatchJobName = append(f.PatchJobName, name)
	f.Patches = append(f.Patches, data)
	// We don't have anything to return. Just return something non-nil.
	return &batchv1.Job{}, nil
}

func (f *fakeJobControl) DeleteJob(namespace string, name string) error {
	f.Lock()
	defer f.Unlock()
	if f.Err != nil {
		return f.Err
	}
	f.DeleteJobName = append(f.DeleteJobName, name)
	return nil
}

func (f *fakeJobControl) Clear() {
	f.Lock()
	defer f.Unlock()
	f.DeleteJobName = []string{}
	f.Jobs = []batchv1.Job{}
	f.Err = nil
}

// ------------------------------------------------------------------ //

// podControlInterface is an interface that knows how to list or delete pods
// created as an interface to allow testing.
type podControlInterface interface {
	// ListPods list pods
	ListPods(namespace string, opts metav1.ListOptions) (*v1.PodList, error)
	// DeleteJob deletes the pod identified by name.
	// TODO: delete by UID?
	DeletePod(namespace string, name string) error
}

// realPodControl is the default implementation of podControlInterface.
type realPodControl struct {
	KubeClient clientset.Interface
	Recorder   record.EventRecorder
}

var _ podControlInterface = &realPodControl{}

func (r realPodControl) ListPods(namespace string, opts metav1.ListOptions) (*v1.PodList, error) {
	return r.KubeClient.CoreV1().Pods(namespace).List(context.TODO(), opts)
}

func (r realPodControl) DeletePod(namespace string, name string) error {
	return r.KubeClient.CoreV1().Pods(namespace).Delete(context.TODO(), name, metav1.DeleteOptions{})
}

type fakePodControl struct {
	sync.Mutex
	Pods          []v1.Pod
	DeletePodName []string
	Err           error
}

var _ podControlInterface = &fakePodControl{}

func (f *fakePodControl) ListPods(namespace string, opts metav1.ListOptions) (*v1.PodList, error) {
	f.Lock()
	defer f.Unlock()
	if f.Err != nil {
		return nil, f.Err
	}
	return &v1.PodList{Items: f.Pods}, nil
}

func (f *fakePodControl) DeletePod(namespace string, name string) error {
	f.Lock()
	defer f.Unlock()
	if f.Err != nil {
		return f.Err
	}
	f.DeletePodName = append(f.DeletePodName, name)
	return nil
}
