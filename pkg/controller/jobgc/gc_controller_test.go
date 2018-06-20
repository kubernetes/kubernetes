/*
Copyright 2018 The Kubernetes Authors.

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

package jobgc

import (
	"sync"
	"testing"
	"time"

	"k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/client-go/informers"
	batchinformers "k8s.io/client-go/informers/batch/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/gengo/examples/set-gen/sets"
	"k8s.io/kubernetes/pkg/controller"
)

type FakeController struct{}

func (*FakeController) Run(<-chan struct{}) {}

func (*FakeController) HasSynced() bool {
	return true
}

func (*FakeController) LastSyncResourceVersion() string {
	return ""
}

func alwaysReady() bool { return true }

func NewFromClient(kubeClient clientset.Interface, finishedJobThreshold int) (*JobGCController, batchinformers.JobInformer) {
	informerFactory := informers.NewSharedInformerFactory(kubeClient, controller.NoResyncPeriodFunc())
	jobInformer := informerFactory.Batch().V1().Jobs()
	controller := NewJobGC(kubeClient, jobInformer, finishedJobThreshold)
	controller.jobListerSynced = alwaysReady
	return controller, jobInformer
}

type nameToStatus struct {
	name   string
	status v1.JobStatus
}

var (
	finishedJobStatus = v1.JobStatus{
		Conditions: []v1.JobCondition{
			{
				Type:   v1.JobComplete,
				Status: corev1.ConditionTrue,
			},
		},
	}
	runningJobStatus = v1.JobStatus{
		Conditions: []v1.JobCondition{},
	}
)

func TestGCJobFinished(t *testing.T) {

	testCases := []struct {
		jobs            []nameToStatus
		threshold       int
		deletedJobNames sets.String
	}{
		{
			// No job GC, threshold = 0
			jobs: []nameToStatus{
				{name: "finishedJob1", status: finishedJobStatus},
				{name: "finishedJob2", status: finishedJobStatus},
			},
			threshold:       0,
			deletedJobNames: sets.NewString(),
		},
		{
			// only GC on finished jobs
			jobs: []nameToStatus{
				{name: "finishedJob1", status: finishedJobStatus},
				{name: "activeJob1", status: runningJobStatus},
				{name: "finishedJob2", status: finishedJobStatus},
				{name: "finishedJob3", status: finishedJobStatus},
			},
			threshold:       1,
			deletedJobNames: sets.NewString("finishedJob1", "finishedJob2"),
		},
		{
			// No job GC, the number of finished jobs is less than threshold
			jobs: []nameToStatus{
				{name: "finishedJob1", status: finishedJobStatus},
			},
			threshold:       2,
			deletedJobNames: sets.NewString(),
		},
	}

	for i, test := range testCases {
		client := fake.NewSimpleClientset()
		gcc, jobInformer := NewFromClient(client, test.threshold)
		deletedJobNames := make([]string, 0)
		var lock sync.Mutex
		gcc.deleteJob = func(_, name string) error {
			lock.Lock()
			defer lock.Unlock()
			deletedJobNames = append(deletedJobNames, name)
			return nil
		}

		creationTime := time.Unix(0, 0)
		for _, job := range test.jobs {
			creationTime = creationTime.Add(1 * time.Hour)
			jobInformer.Informer().GetStore().Add(&v1.Job{
				ObjectMeta: metav1.ObjectMeta{Name: job.name, CreationTimestamp: metav1.Time{Time: creationTime}},
				Status:     job.status,
			})
		}

		gcc.gc()

		pass := true
		for _, job := range deletedJobNames {
			if !test.deletedJobNames.Has(job) {
				pass = false
			}
		}
		if len(deletedJobNames) != len(test.deletedJobNames) {
			pass = false
		}
		if !pass {
			t.Errorf("[%v]job's deleted expected and actual did not match.\n\texpected: %v\n\tactual: %v", i, test.deletedJobNames, deletedJobNames)
		}
	}
}

func TestGCJobFinishedSameCreateionTime(t *testing.T) {
	testCases := []struct {
		jobs            []nameToStatus
		threshold       int
		deletedJobNames sets.String
	}{
		// Jobs' CreationTime are the same, compared by name
		{
			jobs: []nameToStatus{
				{name: "finishedJob2", status: finishedJobStatus},
				{name: "finishedJob1", status: finishedJobStatus},
			},
			threshold:       1,
			deletedJobNames: sets.NewString("finishedJob1"),
		},
	}

	for i, test := range testCases {
		client := fake.NewSimpleClientset()
		gcc, jobInformer := NewFromClient(client, test.threshold)
		deletedJobNames := make([]string, 0)
		var lock sync.Mutex
		gcc.deleteJob = func(_, name string) error {
			lock.Lock()
			defer lock.Unlock()
			deletedJobNames = append(deletedJobNames, name)
			return nil
		}

		creationTime := time.Unix(0, 0)
		for _, job := range test.jobs {
			jobInformer.Informer().GetStore().Add(&v1.Job{
				ObjectMeta: metav1.ObjectMeta{Name: job.name, CreationTimestamp: metav1.Time{Time: creationTime}},
				Status:     job.status,
			})
		}

		jobs, err := jobInformer.Lister().List(labels.Everything())
		if err != nil {
			t.Errorf("Error while listing all Jobs: %v", err)
			return
		}
		gcc.gcFinished(jobs)

		pass := true
		for _, job := range deletedJobNames {
			if !test.deletedJobNames.Has(job) {
				pass = false
			}
		}
		if len(deletedJobNames) != len(test.deletedJobNames) {
			pass = false
		}
		if !pass {
			t.Errorf("[%v]job's deleted expected and actual did not match.\n\texpected: %v\n\tactual: %v", i, test.deletedJobNames, deletedJobNames)
		}
	}
}
