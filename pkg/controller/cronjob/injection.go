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
	"fmt"
	"sync"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/client-go/tools/record"
	"k8s.io/kubernetes/pkg/api/v1"
	batchv1 "k8s.io/kubernetes/pkg/apis/batch/v1"
	batchv2alpha1 "k8s.io/kubernetes/pkg/apis/batch/v2alpha1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
)

// sjControlInterface is an interface that knows how to update CronJob status
// created as an interface to allow testing.
type sjControlInterface interface {
	UpdateStatus(sj *batchv2alpha1.CronJob) (*batchv2alpha1.CronJob, error)
}

// realSJControl is the default implementation of sjControlInterface.
type realSJControl struct {
	KubeClient clientset.Interface
}

var _ sjControlInterface = &realSJControl{}

func (c *realSJControl) UpdateStatus(sj *batchv2alpha1.CronJob) (*batchv2alpha1.CronJob, error) {
	return c.KubeClient.BatchV2alpha1().CronJobs(sj.Namespace).UpdateStatus(sj)
}

// fakeSJControl is the default implementation of sjControlInterface.
type fakeSJControl struct {
	Updates []batchv2alpha1.CronJob
}

var _ sjControlInterface = &fakeSJControl{}

func (c *fakeSJControl) UpdateStatus(sj *batchv2alpha1.CronJob) (*batchv2alpha1.CronJob, error) {
	c.Updates = append(c.Updates, *sj)
	return sj, nil
}

// ------------------------------------------------------------------ //

// jobControlInterface is an interface that knows how to add or delete jobs
// created as an interface to allow testing.
type jobControlInterface interface {
	// GetJob retrieves a job
	GetJob(namespace, name string) (*batchv1.Job, error)
	// CreateJob creates new jobs according to the spec
	CreateJob(namespace string, job *batchv1.Job) (*batchv1.Job, error)
	// UpdateJob updates a job
	UpdateJob(namespace string, job *batchv1.Job) (*batchv1.Job, error)
	// DeleteJob deletes the job identified by name.
	// TODO: delete by UID?
	DeleteJob(namespace string, name string) error
}

// realJobControl is the default implementation of jobControlInterface.
type realJobControl struct {
	KubeClient clientset.Interface
	Recorder   record.EventRecorder
}

var _ jobControlInterface = &realJobControl{}

func copyLabels(template *batchv2alpha1.JobTemplateSpec) labels.Set {
	l := make(labels.Set)
	for k, v := range template.Labels {
		l[k] = v
	}
	return l
}

func copyAnnotations(template *batchv2alpha1.JobTemplateSpec) labels.Set {
	a := make(labels.Set)
	for k, v := range template.Annotations {
		a[k] = v
	}
	return a
}

func (r realJobControl) GetJob(namespace, name string) (*batchv1.Job, error) {
	return r.KubeClient.BatchV1().Jobs(namespace).Get(name, metav1.GetOptions{})
}

func (r realJobControl) UpdateJob(namespace string, job *batchv1.Job) (*batchv1.Job, error) {
	return r.KubeClient.BatchV1().Jobs(namespace).Update(job)
}

func (r realJobControl) CreateJob(namespace string, job *batchv1.Job) (*batchv1.Job, error) {
	return r.KubeClient.BatchV1().Jobs(namespace).Create(job)
}

func (r realJobControl) DeleteJob(namespace string, name string) error {
	return r.KubeClient.BatchV1().Jobs(namespace).Delete(name, nil)
}

type fakeJobControl struct {
	sync.Mutex
	Job           *batchv1.Job
	Jobs          []batchv1.Job
	DeleteJobName []string
	Err           error
}

var _ jobControlInterface = &fakeJobControl{}

func (f *fakeJobControl) CreateJob(namespace string, job *batchv1.Job) (*batchv1.Job, error) {
	f.Lock()
	defer f.Unlock()
	if f.Err != nil {
		return nil, f.Err
	}
	job.SelfLink = fmt.Sprintf("/api/batch/v1/namespaces/%s/jobs/%s", namespace, job.Name)
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
	return job, nil
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
	return r.KubeClient.Core().Pods(namespace).List(opts)
}

func (r realPodControl) DeletePod(namespace string, name string) error {
	return r.KubeClient.Core().Pods(namespace).Delete(name, nil)
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
