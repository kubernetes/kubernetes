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
	"sort"
	"sync"
	"time"

	"k8s.io/api/batch/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	batchinformers "k8s.io/client-go/informers/batch/v1"
	clientset "k8s.io/client-go/kubernetes"
	batchlisters "k8s.io/client-go/listers/batch/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/kubernetes/pkg/controller"
	jobctrl "k8s.io/kubernetes/pkg/controller/job"
	"k8s.io/kubernetes/pkg/util/metrics"

	"github.com/golang/glog"
)

const (
	// gcCheckPeriod is the time period to wait before processing a job garbage collection.
	gcCheckPeriod = 20 * time.Second
)

// JobGCController is responsible for finished job garbage collecting.
type JobGCController struct {
	kubeClient clientset.Interface

	jobLister       batchlisters.JobLister
	jobListerSynced cache.InformerSynced

	deleteJob            func(namespace, name string) error
	finishedJobThreshold int
}

// NewJobGC creates a new JobGCController.
func NewJobGC(kubeClient clientset.Interface, jobInformer batchinformers.JobInformer, finishedJobThreshold int) *JobGCController {
	if kubeClient != nil && kubeClient.BatchV1().RESTClient().GetRateLimiter() != nil {
		metrics.RegisterMetricAndTrackRateLimiterUsage("gc_controller", kubeClient.BatchV1().RESTClient().GetRateLimiter())
	}
	gcc := &JobGCController{
		kubeClient:           kubeClient,
		finishedJobThreshold: finishedJobThreshold,
		deleteJob: func(namespace, name string) error {
			glog.Infof("JobGC is force deleting Job: %v:%v", namespace, name)
			return kubeClient.BatchV1().Jobs(namespace).Delete(name, metav1.NewDeleteOptions(0))
		},
	}

	gcc.jobLister = jobInformer.Lister()
	gcc.jobListerSynced = jobInformer.Informer().HasSynced

	return gcc
}

func (gcc *JobGCController) Run(stop <-chan struct{}) {
	defer utilruntime.HandleCrash()

	glog.Infof("Starting GC controller")
	defer glog.Infof("Shutting down GC controller")

	if !controller.WaitForCacheSync("GC", stop, gcc.jobListerSynced) {
		return
	}

	go wait.Until(gcc.gc, gcCheckPeriod, stop)

	<-stop
}

func (gcc *JobGCController) gc() {
	jobs, err := gcc.jobLister.List(labels.Everything())
	if err != nil {
		glog.Errorf("Error while listing all Jobs: %v", err)
		return
	}
	if gcc.finishedJobThreshold > 0 {
		gcc.gcFinished(jobs)
	}
}

// gcFinished deletes jobs that are finished.
func (gcc *JobGCController) gcFinished(jobs []*v1.Job) {
	finishedJobs := []*v1.Job{}
	for _, job := range jobs {
		if jobctrl.IsJobFinished(job) {
			finishedJobs = append(finishedJobs, job)
		}
	}

	finishedJobCount := len(finishedJobs)
	sort.Slice(
		finishedJobs,
		func(i, j int) bool {
			if finishedJobs[i].CreationTimestamp.Equal(&finishedJobs[j].CreationTimestamp) {
				return finishedJobs[i].Name < finishedJobs[j].Name
			}
			return finishedJobs[i].CreationTimestamp.Before(&finishedJobs[j].CreationTimestamp)
		},
	)

	deleteCount := finishedJobCount - gcc.finishedJobThreshold

	if deleteCount > finishedJobCount {
		deleteCount = finishedJobCount
	}
	if deleteCount > 0 {
		glog.Infof("garbage collecting %v jobs", deleteCount)
	}

	var wait sync.WaitGroup
	for i := 0; i < deleteCount; i++ {
		wait.Add(1)
		go func(namespace string, name string) {
			defer wait.Done()
			if err := gcc.deleteJob(namespace, name); err != nil {
				// ignore not founds
				defer utilruntime.HandleError(err)
			}
		}(finishedJobs[i].Namespace, finishedJobs[i].Name)
	}
	wait.Wait()
}
