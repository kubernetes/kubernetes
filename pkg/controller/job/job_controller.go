/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package job

import (
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/cache"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/controller"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/controller/framework"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/workqueue"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
	"github.com/golang/glog"
)

type JobManager struct {
	kubeClient client.Interface
	podControl controller.PodControlInterface

	// An rc is temporarily suspended after creating/deleting these many jobs.
	// It resumes normal action after observing the watch events for them.
	burstJobs int
	// To allow injection for testing
	syncHandler func(jobKey string) error

	// A TTLCache of pod creates/deletes each rc expects to see
	expectations controller.ControllerExpectationsInterface

	// A store of jobs, populated by the jobController
	jobStore cache.StoreToJobLister
	// A store of pods, populated by the podController
	podStore cache.StoreToPodLister
	// Watches changes to all jobs
	jobController *framework.Controller
	// Watches changes to all pods
	podController *framework.Controller
	// Controllers that need to be updated
	queue *workqueue.Type
}

var (
	burstJobs        = 500
	fullResyncPeriod = 30 * time.Second
	podRelistPeriod  = 5 * time.Minute
)

func NewJobManager(kubeClient client.Interface, burstJobs int) *JobManager {
	jm := &JobManager{
		burstJobs: burstJobs,
		syncHandler: func(jobKey string) error {
			return nil
		},
		expectations: controller.NewControllerExpectations(),
	}

	jm.jobStore.Store, jm.jobController = framework.NewInformer(
		&cache.ListWatch{
			ListFunc: func() (runtime.Object, error) {
				return jm.kubeClient.Jobs(api.NamespaceAll).List(labels.Everything())
			},
			WatchFunc: func(rv string) (watch.Interface, error) {
				return jm.kubeClient.Jobs(api.NamespaceAll).Watch(labels.Everything(), fields.Everything(), rv)
			},
		},
		&api.Job{},
		fullResyncPeriod,
		framework.ResourceEventHandlerFuncs{
			AddFunc: jm.enqueueJob,
			UpdateFunc: func(old, cur interface{}) {
				jm.enqueueJob(cur)
			},
			// This will enter the sync loop and no-op, becuase the controller has been deleted from the store.
			// Note that deleting a controller immediately after scaling it to 0 will not work. The recommended
			// way of achieving this is by performing a `stop` operation on the controller.
			DeleteFunc: jm.enqueueJob,
		},
	)

	jm.podStore.Store, jm.podController = framework.NewInformer(
		&cache.ListWatch{
			ListFunc: func() (runtime.Object, error) {
				return jm.kubeClient.Pods(api.NamespaceAll).List(labels.Everything(), fields.Everything())
			},
			WatchFunc: func(rv string) (watch.Interface, error) {
				return jm.kubeClient.Pods(api.NamespaceAll).Watch(labels.Everything(), fields.Everything(), rv)
			},
		},
		&api.Pod{},
		podRelistPeriod,
		framework.ResourceEventHandlerFuncs{
			AddFunc:    jm.addPod,
			UpdateFunc: jm.updatePod,
			DeleteFunc: jm.deletePod,
		},
	)

	return jm
}

func (jm *JobManager) Run(workers int, stopCh <-chan struct{}) {
	defer util.HandleCrash()
	go jm.jobController.Run(stopCh)
	go jm.podController.Run(stopCh)
	for i := 0; i < workers; i++ {
		go util.Until(jm.worker, time.Second, stopCh)
	}
	<-stopCh
}

func (jm *JobManager) worker() {
	for {
		func() {
			key, quit := jm.queue.Get()
			if quit {
				return
			}
			defer jm.queue.Done(key)
			err := jm.syncHandler(key.(string))
			if err != nil {
				glog.Errorf("Error syncing replication controller: %v", err)
			}
		}()
	}
}

func (jm *JobManager) enqueueJob(obj interface{}) {

}

func (jm *JobManager) addPod(obj interface{}) {

}

func (jm *JobManager) updatePod(oldObj, newObj interface{}) {

}

func (jm *JobManager) deletePod(obj interface{}) {

}
