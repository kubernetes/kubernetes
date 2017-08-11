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

	batchv1 "k8s.io/api/batch/v1"
	apiv1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	kubeclientset "k8s.io/client-go/kubernetes"
	kubeclientfake "k8s.io/client-go/kubernetes/fake"
	core "k8s.io/client-go/testing"
	fedv1 "k8s.io/kubernetes/federation/apis/federation/v1beta1"
	fedclientfake "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset/fake"
	fedutil "k8s.io/kubernetes/federation/pkg/federation-controller/util"
	finalizersutil "k8s.io/kubernetes/federation/pkg/federation-controller/util/finalizers"
	testutil "k8s.io/kubernetes/federation/pkg/federation-controller/util/test"
	batchv1internal "k8s.io/kubernetes/pkg/apis/batch/v1"

	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/util/sets"
	"reflect"
	"strings"
)

func installWatchReactor(fakeClien *core.Fake, resource string) chan runtime.Object {
	objChan := make(chan runtime.Object, 100)

	fakeWatch := watch.NewRaceFreeFake()
	fakeClien.PrependWatchReactor(resource, core.DefaultWatchReactor(fakeWatch, nil))
	fakeClien.PrependReactor("create", resource, func(action core.Action) (handled bool, ret runtime.Object, err error) {
		obj := action.(core.CreateAction).GetObject()
		batchv1internal.SetDefaults_Job(obj.(*batchv1.Job))
		fakeWatch.Add(obj)
		objChan <- obj
		return false, nil, nil
	})
	fakeClien.PrependReactor("update", resource, func(action core.Action) (handled bool, ret runtime.Object, err error) {
		obj := action.(core.UpdateAction).GetObject()
		fakeWatch.Modify(obj)
		objChan <- obj
		return false, nil, nil
	})
	fakeClien.PrependReactor("delete", resource, func(action core.Action) (handled bool, ret runtime.Object, err error) {
		obj := &batchv1.Job{
			ObjectMeta: metav1.ObjectMeta{
				Name:      action.(core.DeleteAction).GetName(),
				Namespace: action.GetNamespace(),
			},
		}
		fakeWatch.Delete(obj)
		objChan <- obj
		return false, nil, nil
	})

	return objChan
}

func TestJobController(t *testing.T) {
	flag.Set("logtostderr", "true")
	flag.Set("v", "5")
	flag.Parse()

	jobReviewDelay = 50 * time.Millisecond
	clusterAvailableDelay = 200 * time.Millisecond
	clusterUnavailableDelay = 200 * time.Millisecond

	fedclientset := fedclientfake.NewSimpleClientset()
	fedChan := installWatchReactor(&fedclientset.Fake, "jobs")

	fedclientset.Federation().Clusters().Create(testutil.NewCluster("k8s-1", apiv1.ConditionTrue))
	fedclientset.Federation().Clusters().Create(testutil.NewCluster("k8s-2", apiv1.ConditionTrue))

	kube1clientset := kubeclientfake.NewSimpleClientset()
	kube1Chan := installWatchReactor(&kube1clientset.Fake, "jobs")
	kube2clientset := kubeclientfake.NewSimpleClientset()
	kube2Chan := installWatchReactor(&kube2clientset.Fake, "jobs")

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

	test := func(job *batchv1.Job, parallelism1, parallelism2, completions1, completions2 int32) {
		job, _ = fedclientset.Batch().Jobs(metav1.NamespaceDefault).Create(job)

		joinErrors := func(errors []error) error {
			if len(errors) == 0 {
				return nil
			}
			errorStrings := []string{}
			for _, err := range errors {
				errorStrings = append(errorStrings, err.Error())
			}
			return fmt.Errorf("%s", strings.Join(errorStrings, "\n"))
		}

		// check local jobs are created with correct spec
		checkLocalJob := func(parallelism, completions int32) testutil.CheckingFunction {
			return func(obj runtime.Object) error {
				errors := []error{}
				ljob := obj.(*batchv1.Job)
				if !fedutil.ObjectMetaEquivalent(job.ObjectMeta, ljob.ObjectMeta) {
					errors = append(errors, fmt.Errorf("Job meta un-equivalent: %#v (expected) != %#v (actual)", job.ObjectMeta, ljob.ObjectMeta))
				}
				if err := checkEqual(t, *ljob.Spec.Parallelism, parallelism, "Spec.Parallelism"); err != nil {
					errors = append(errors, err)
				}
				if ljob.Spec.Completions != nil {
					if err := checkEqual(t, *ljob.Spec.Completions, completions, "Spec.Completions"); err != nil {
						errors = append(errors, err)
					}
				}
				return joinErrors(errors)
			}
		}
		checkFedJob := func(obj runtime.Object) error {
			errors := []error{}
			return joinErrors(errors)
		}
		assert.NoError(t, testutil.CheckObjectFromChan(kube1Chan, checkLocalJob(parallelism1, completions1)))
		assert.NoError(t, testutil.CheckObjectFromChan(kube2Chan, checkLocalJob(parallelism2, completions2)))
		assert.NoError(t, testutil.CheckObjectFromChan(fedChan, checkFedJob))

		// finish local jobs
		job1, _ := kube1clientset.Batch().Jobs(metav1.NamespaceDefault).Get(job.Name, metav1.GetOptions{})
		finishJob(job1, 100*time.Millisecond)
		job1, _ = kube1clientset.Batch().Jobs(metav1.NamespaceDefault).UpdateStatus(job1)
		job2, _ := kube2clientset.Batch().Jobs(metav1.NamespaceDefault).Get(job.Name, metav1.GetOptions{})
		finishJob(job2, 100*time.Millisecond)
		job2, _ = kube2clientset.Batch().Jobs(metav1.NamespaceDefault).UpdateStatus(job2)

		// check fed job status updated
		assert.NoError(t, testutil.CheckObjectFromChan(fedChan, func(obj runtime.Object) error {
			errors := []error{}
			job := obj.(*batchv1.Job)
			if err := checkEqual(t, *job.Spec.Parallelism, *job1.Spec.Parallelism+*job2.Spec.Parallelism, "Spec.Parallelism"); err != nil {
				errors = append(errors, err)
			}
			if job.Spec.Completions != nil {
				if err := checkEqual(t, *job.Spec.Completions, *job1.Spec.Completions+*job2.Spec.Completions, "Spec.Completions"); err != nil {
					errors = append(errors, err)
				}
			}
			if err := checkEqual(t, job.Status.Succeeded, job1.Status.Succeeded+job2.Status.Succeeded, "Status.Succeeded"); err != nil {
				errors = append(errors, err)
			}
			return joinErrors(errors)
		}))

		// delete fed job by set deletion time, and remove orphan finalizer
		job, _ = fedclientset.Batch().Jobs(metav1.NamespaceDefault).Get(job.Name, metav1.GetOptions{})
		deletionTimestamp := metav1.Now()
		job.DeletionTimestamp = &deletionTimestamp
		finalizersutil.RemoveFinalizers(job, sets.NewString(metav1.FinalizerOrphanDependents))
		fedclientset.Batch().Jobs(metav1.NamespaceDefault).Update(job)

		// check jobs are deleted
		checkDeleted := func(obj runtime.Object) error {
			djob := obj.(*batchv1.Job)
			deletedJob := &batchv1.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:      djob.Name,
					Namespace: djob.Namespace,
				},
			}
			if !reflect.DeepEqual(djob, deletedJob) {
				return fmt.Errorf("%s/%s should be deleted", djob.Namespace, djob.Name)
			}
			return nil
		}
		assert.NoError(t, testutil.CheckObjectFromChan(kube1Chan, checkDeleted))
		assert.NoError(t, testutil.CheckObjectFromChan(kube2Chan, checkDeleted))
		assert.NoError(t, testutil.CheckObjectFromChan(fedChan, checkDeleted))
	}

	test(newJob("job1", 2, 7), 1, 1, 4, 3)
	test(newJob("job2", 2, -1), 1, 1, -1, -1)
	test(newJob("job3", 7, 2), 4, 3, 1, 1)
	test(newJob("job4", 7, 1), 4, 3, 1, 0)
}

func checkEqual(_ *testing.T, expected, actual interface{}, msg string) error {
	if !assert.ObjectsAreEqual(expected, actual) {
		return fmt.Errorf("%s not equal: %#v (expected) != %#v (actual)", msg, expected, actual)
	}
	return nil
}

func newJob(name string, parallelism int32, completions int32) *batchv1.Job {
	job := batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: metav1.NamespaceDefault,
			SelfLink:  "/api/v1/namespaces/default/jobs/name",
		},
		Spec: batchv1.JobSpec{
			Parallelism: &parallelism,
			Completions: &completions,
			Template: apiv1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"foo": name,
					},
				},
				Spec: apiv1.PodSpec{
					Containers: []apiv1.Container{
						{Image: "foo/bar"},
					},
					RestartPolicy: apiv1.RestartPolicyNever,
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

	batchv1internal.SetDefaults_Job(&job)
	return &job
}

func newCondition(conditionType batchv1.JobConditionType, reason, message string) batchv1.JobCondition {
	return batchv1.JobCondition{
		Type:               conditionType,
		Status:             apiv1.ConditionTrue,
		LastProbeTime:      metav1.Now(),
		LastTransitionTime: metav1.Now(),
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
	now := metav1.Now()
	job.Status.StartTime = &now
	time.Sleep(duration)
	now = metav1.Now()
	job.Status.CompletionTime = &now
}
