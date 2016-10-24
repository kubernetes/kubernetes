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

package scheduledjob

import (
	"fmt"
	"sync"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/batch"
	"k8s.io/kubernetes/pkg/client/record"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/labels"
)

// sjControlInterface is an interface that knows how to update ScheduledJob status
// created as an interface to allow testing.
type sjControlInterface interface {
	UpdateStatus(sj *batch.ScheduledJob) (*batch.ScheduledJob, error)
}

// realSJControl is the default implementation of sjControlInterface.
type realSJControl struct {
	KubeClient *client.Client
}

var _ sjControlInterface = &realSJControl{}

func (c *realSJControl) UpdateStatus(sj *batch.ScheduledJob) (*batch.ScheduledJob, error) {
	return c.KubeClient.Batch().ScheduledJobs(sj.Namespace).UpdateStatus(sj)
}

// fakeSJControl is the default implementation of sjControlInterface.
type fakeSJControl struct {
	Updates []batch.ScheduledJob
}

var _ sjControlInterface = &fakeSJControl{}

func (c *fakeSJControl) UpdateStatus(sj *batch.ScheduledJob) (*batch.ScheduledJob, error) {
	c.Updates = append(c.Updates, *sj)
	return sj, nil
}

// ------------------------------------------------------------------ //

// jobControlInterface is an interface that knows how to add or delete jobs
// created as an interface to allow testing.
type jobControlInterface interface {
	// GetJob retrieves a job
	GetJob(namespace, name string) (*batch.Job, error)
	// CreateJob creates new jobs according to the spec
	CreateJob(namespace string, job *batch.Job) (*batch.Job, error)
	// UpdateJob updates a job
	UpdateJob(namespace string, job *batch.Job) (*batch.Job, error)
	// DeleteJob deletes the job identified by name.
	// TODO: delete by UID?
	DeleteJob(namespace string, name string) error
}

// realJobControl is the default implementation of jobControlInterface.
type realJobControl struct {
	KubeClient *client.Client
	Recorder   record.EventRecorder
}

var _ jobControlInterface = &realJobControl{}

func copyLabels(template *batch.JobTemplateSpec) labels.Set {
	l := make(labels.Set)
	for k, v := range template.Labels {
		l[k] = v
	}
	return l
}

func copyAnnotations(template *batch.JobTemplateSpec) labels.Set {
	a := make(labels.Set)
	for k, v := range template.Annotations {
		a[k] = v
	}
	return a
}

func (r realJobControl) GetJob(namespace, name string) (*batch.Job, error) {
	return r.KubeClient.Batch().Jobs(namespace).Get(name)
}

func (r realJobControl) UpdateJob(namespace string, job *batch.Job) (*batch.Job, error) {
	return r.KubeClient.Batch().Jobs(namespace).Update(job)
}

func (r realJobControl) CreateJob(namespace string, job *batch.Job) (*batch.Job, error) {
	return r.KubeClient.Batch().Jobs(namespace).Create(job)
}

func (r realJobControl) DeleteJob(namespace string, name string) error {
	return r.KubeClient.Batch().Jobs(namespace).Delete(name, nil)
}

type fakeJobControl struct {
	sync.Mutex
	Job           *batch.Job
	Jobs          []batch.Job
	DeleteJobName []string
	Err           error
}

var _ jobControlInterface = &fakeJobControl{}

func (f *fakeJobControl) CreateJob(namespace string, job *batch.Job) (*batch.Job, error) {
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

func (f *fakeJobControl) GetJob(namespace, name string) (*batch.Job, error) {
	f.Lock()
	defer f.Unlock()
	if f.Err != nil {
		return nil, f.Err
	}
	return f.Job, nil
}

func (f *fakeJobControl) UpdateJob(namespace string, job *batch.Job) (*batch.Job, error) {
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
	f.Jobs = []batch.Job{}
	f.Err = nil
}

// ------------------------------------------------------------------ //

// podControlInterface is an interface that knows how to list or delete pods
// created as an interface to allow testing.
type podControlInterface interface {
	// ListPods list pods
	ListPods(namespace string, opts api.ListOptions) (*api.PodList, error)
	// DeleteJob deletes the pod identified by name.
	// TODO: delete by UID?
	DeletePod(namespace string, name string) error
}

// realPodControl is the default implementation of podControlInterface.
type realPodControl struct {
	KubeClient *client.Client
	Recorder   record.EventRecorder
}

var _ podControlInterface = &realPodControl{}

func (r realPodControl) ListPods(namespace string, opts api.ListOptions) (*api.PodList, error) {
	return r.KubeClient.Pods(namespace).List(opts)
}

func (r realPodControl) DeletePod(namespace string, name string) error {
	return r.KubeClient.Pods(namespace).Delete(name, nil)
}

type fakePodControl struct {
	sync.Mutex
	Pods          []api.Pod
	DeletePodName []string
	Err           error
}

var _ podControlInterface = &fakePodControl{}

func (f *fakePodControl) ListPods(namespace string, opts api.ListOptions) (*api.PodList, error) {
	f.Lock()
	defer f.Unlock()
	return &api.PodList{Items: f.Pods}, nil
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
