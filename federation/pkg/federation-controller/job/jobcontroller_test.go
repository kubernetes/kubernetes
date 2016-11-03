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

package job

import (
	"flag"
	"fmt"
	"testing"
	"time"

	fedv1 "k8s.io/kubernetes/federation/apis/federation/v1beta1"
	fedclientfake "k8s.io/kubernetes/federation/client/clientset_generated/federation_release_1_5/fake"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util/test"
	"k8s.io/kubernetes/pkg/api/unversioned"
	apiv1 "k8s.io/kubernetes/pkg/api/v1"
	batchv1 "k8s.io/kubernetes/pkg/apis/batch/v1"
	kubeclientset "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5"
	kubeclientfake "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/fake"
	"k8s.io/kubernetes/pkg/client/testing/core"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/watch"

	"github.com/stretchr/testify/assert"
)

func installWatchReactor(fakeClien *core.Fake, resource string) {
	fakeWatch := watch.NewRaceFreeFake()
	fakeClien.PrependWatchReactor(resource, core.DefaultWatchReactor(fakeWatch, nil))
	fakeClien.PrependReactor("create", resource, func(action core.Action) (handled bool, ret runtime.Object, err error) {
		obj := action.(core.CreateAction).GetObject()
		fakeWatch.Add(obj)
		return false, nil, nil
	})
	fakeClien.PrependReactor("update", resource, func(action core.Action) (handled bool, ret runtime.Object, err error) {
		fakeWatch.Modify(action.(core.UpdateAction).GetObject())
		return false, nil, nil
	})
	fakeClien.PrependReactor("delete", resource, func(action core.Action) (handled bool, ret runtime.Object, err error) {
		fakeWatch.Delete(&batchv1.Job{
			ObjectMeta: apiv1.ObjectMeta{
				Name:      action.(core.DeleteAction).GetName(),
				Namespace: action.GetNamespace(),
			},
		})
		return false, nil, nil
	})
}

func TestJobController(t *testing.T) {
	flag.Set("logtostderr", "true")
	flag.Set("v", "5")
	flag.Parse()

	jobReviewDelay = 50 * time.Millisecond
	clusterAvailableDelay = 200 * time.Millisecond
	clusterUnavailableDelay = 200 * time.Millisecond
	allJobReviewDelay = 500 * time.Millisecond

	fedclientset := fedclientfake.NewSimpleClientset()
	installWatchReactor(&fedclientset.Fake, "jobs")

	fedclientset.Federation().Clusters().Create(testutil.NewCluster("k8s-1", apiv1.ConditionTrue))
	fedclientset.Federation().Clusters().Create(testutil.NewCluster("k8s-2", apiv1.ConditionTrue))

	kube1clientset := kubeclientfake.NewSimpleClientset()
	installWatchReactor(&kube1clientset.Fake, "jobs")
	kube2clientset := kubeclientfake.NewSimpleClientset()
	installWatchReactor(&kube2clientset.Fake, "jobs")

	fedInformerClientFactory := func(cluster *fedv1.Cluster) (kubeclientset.Interface, error) {
		switch cluster.Name {
		case "k8s-1":
			return kube1clientset, nil
		case "k8s-2":
			return kube2clientset, nil
		default:
			return nil, fmt.Errorf("Unknown cluster: %v", cluster.Name)
		}
	}
	jobController := NewJobController(fedclientset)
	fedjobinformer := testutil.ToFederatedInformerForTestOnly(jobController.fedJobInformer)
	fedjobinformer.SetClientFactory(fedInformerClientFactory)

	stopChan := make(chan struct{})
	defer close(stopChan)
	go jobController.Run(5, stopChan)

	test := func(job *batchv1.Job) {
		job, _ = fedclientset.Batch().Jobs(apiv1.NamespaceDefault).Create(job)
		time.Sleep(1 * time.Second)

		job1, _ := kube1clientset.Batch().Jobs(apiv1.NamespaceDefault).Get(job.Name)
		finishJob(job1, 100*time.Millisecond)
		job1, _ = kube1clientset.Batch().Jobs(apiv1.NamespaceDefault).UpdateStatus(job1)

		job2, _ := kube2clientset.Batch().Jobs(apiv1.NamespaceDefault).Get(job.Name)
		finishJob(job2, 100*time.Millisecond)
		job2, _ = kube2clientset.Batch().Jobs(apiv1.NamespaceDefault).UpdateStatus(job2)

		time.Sleep(1 * time.Second)
		job, _ = fedclientset.Batch().Jobs(apiv1.NamespaceDefault).Get(job.Name)
		assert.Equal(t, *job.Spec.Parallelism, *job1.Spec.Parallelism+*job2.Spec.Parallelism)
		if job.Spec.Completions != nil {
			assert.Equal(t, *job.Spec.Completions, *job1.Spec.Completions+*job2.Spec.Completions)
		}
		assert.Equal(t, job.Status.Succeeded, job1.Status.Succeeded+job2.Status.Succeeded)

		fedclientset.Batch().Jobs(apiv1.NamespaceDefault).Delete(job.Name, &apiv1.DeleteOptions{})
		time.Sleep(1 * time.Second)
		job1, _ = kube1clientset.Batch().Jobs(apiv1.NamespaceDefault).Get(job.Name)
		job2, _ = kube2clientset.Batch().Jobs(apiv1.NamespaceDefault).Get(job.Name)
		job, _ = fedclientset.Batch().Jobs(apiv1.NamespaceDefault).Get(job.Name)
		assert.Nil(t, job1)
		assert.Nil(t, job2)
		assert.Nil(t, job)
	}

	job := newJob("job1", 2, 7)
	test(job)
	job = newJob("job2", 2, -1)
	test(job)
	job = newJob("job3", 7, 2)
	test(job)
}

func newJob(name string, parallelism int32, completions int32) *batchv1.Job {
	job := batchv1.Job{
		ObjectMeta: apiv1.ObjectMeta{
			Name:      name,
			Namespace: apiv1.NamespaceDefault,
			SelfLink:  "/api/v1/namespaces/default/jobs/name",
		},
		Spec: batchv1.JobSpec{
			Parallelism: &parallelism,
			Completions: &completions,
			Template: apiv1.PodTemplateSpec{
				ObjectMeta: apiv1.ObjectMeta{
					Labels: map[string]string{
						"foo": name,
					},
				},
				Spec: apiv1.PodSpec{
					Containers: []apiv1.Container{
						{Image: "foo/bar"},
					},
				},
			},
		},
	}
	if parallelism < 0 {
		job.Spec.Parallelism = nil
	}
	if completions < 0 {
		job.Spec.Completions = nil
	}

	batchv1.SetDefaults_Job(&job)
	return &job
}

func newCondition(conditionType batchv1.JobConditionType, reason, message string) batchv1.JobCondition {
	return batchv1.JobCondition{
		Type:               conditionType,
		Status:             apiv1.ConditionTrue,
		LastProbeTime:      unversioned.Now(),
		LastTransitionTime: unversioned.Now(),
		Reason:             reason,
		Message:            message,
	}
}

func finishJob(job *batchv1.Job, duration time.Duration) {
	job.Status.Conditions = append(job.Status.Conditions, newCondition(batchv1.JobComplete, "", ""))
	if job.Spec.Completions == nil {
		job.Status.Succeeded = 1
	} else {
		job.Status.Succeeded = *job.Spec.Completions
	}
	now := unversioned.Now()
	job.Status.StartTime = &now
	time.Sleep(duration)
	now = unversioned.Now()
	job.Status.CompletionTime = &now
}
