/*
Copyright 2017 The Kubernetes Authors.

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

// +build ignore
package federatedtypes

import (
	batchv1 "k8s.io/api/batch/v1"
	apiv1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	pkgruntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	kubeclientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	federationclientset "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset"
	fedutil "k8s.io/kubernetes/federation/pkg/federation-controller/util"
)

const (
	JobKind                     = "job"
	JobControllerName           = "jobs"
	FedJobPreferencesAnnotation = "federation.kubernetes.io/job-preferences"
)

func init() {
	//RegisterFederatedType(JobKind, JobControllerName, []schema.GroupVersionResource{extensionsv1.SchemeGroupVersion.WithResource(JobControllerName)}, NewJobAdapter)
}

type JobAdapter struct {
	//*jobSchedulingAdapter
	client federationclientset.Interface
}

func NewJobAdapter(client federationclientset.Interface, config *restclient.Config, adapterSpecificArgs map[string]interface{}) bool {
	//return &JobAdapter{client}
	return true
}

func (a *JobAdapter) Kind() string {
	return JobKind
}

func (a *JobAdapter) ObjectType() pkgruntime.Object {
	return &batchv1.Job{}
}

func (a *JobAdapter) IsExpectedType(obj interface{}) bool {
	_, ok := obj.(*batchv1.Job)
	return ok
}

func (a *JobAdapter) Copy(obj pkgruntime.Object) pkgruntime.Object {
	job := obj.(*batchv1.Job)
	return &batchv1.Job{
		ObjectMeta: fedutil.DeepCopyRelevantObjectMeta(job.ObjectMeta),
		Spec:       *fedutil.DeepCopyApiTypeOrPanic(&job.Spec).(*batchv1.JobSpec),
	}
}

func (a *JobAdapter) Equivalent(obj1, obj2 pkgruntime.Object) bool {
	return fedutil.ObjectMetaAndSpecEquivalent(obj1, obj2)
}

func (a *JobAdapter) QualifiedName(obj pkgruntime.Object) QualifiedName {
	job := obj.(*batchv1.Job)
	return QualifiedName{Namespace: job.Namespace, Name: job.Name}
}

func (a *JobAdapter) ObjectMeta(obj pkgruntime.Object) *metav1.ObjectMeta {
	return &obj.(*batchv1.Job).ObjectMeta
}

func (a *JobAdapter) FedCreate(obj pkgruntime.Object) (pkgruntime.Object, error) {
	job := obj.(*batchv1.Job)
	return a.client.BatchV1().Jobs(job.Namespace).Create(job)
}

func (a *JobAdapter) FedDelete(qualifiedName QualifiedName, options *metav1.DeleteOptions) error {
	return a.client.BatchV1().Jobs(qualifiedName.Namespace).Delete(qualifiedName.Name, options)
}

func (a *JobAdapter) FedGet(qualifiedName QualifiedName) (pkgruntime.Object, error) {
	return a.client.BatchV1().Jobs(qualifiedName.Namespace).Get(qualifiedName.Name, metav1.GetOptions{})
}

func (a *JobAdapter) FedList(namespace string, options metav1.ListOptions) (pkgruntime.Object, error) {
	return a.client.BatchV1().Jobs(namespace).List(options)
}

func (a *JobAdapter) FedUpdate(obj pkgruntime.Object) (pkgruntime.Object, error) {
	job := obj.(*batchv1.Job)
	return a.client.BatchV1().Jobs(job.Namespace).Update(job)
}

func (a *JobAdapter) FedWatch(namespace string, options metav1.ListOptions) (watch.Interface, error) {
	return a.client.BatchV1().Jobs(namespace).Watch(options)
}

func (a *JobAdapter) ClusterCreate(client kubeclientset.Interface, obj pkgruntime.Object) (pkgruntime.Object, error) {
	job := obj.(*batchv1.Job)
	return client.BatchV1().Jobs(job.Namespace).Create(job)
}

func (a *JobAdapter) ClusterDelete(client kubeclientset.Interface, qualifiedName QualifiedName, options *metav1.DeleteOptions) error {
	return client.BatchV1().Jobs(qualifiedName.Namespace).Delete(qualifiedName.Name, options)
}

func (a *JobAdapter) ClusterGet(client kubeclientset.Interface, qualifiedName QualifiedName) (pkgruntime.Object, error) {
	return client.BatchV1().Jobs(qualifiedName.Namespace).Get(qualifiedName.Name, metav1.GetOptions{})
}

func (a *JobAdapter) ClusterList(client kubeclientset.Interface, namespace string, options metav1.ListOptions) (pkgruntime.Object, error) {
	return client.BatchV1().Jobs(namespace).List(options)
}

func (a *JobAdapter) ClusterUpdate(client kubeclientset.Interface, obj pkgruntime.Object) (pkgruntime.Object, error) {
	job := obj.(*batchv1.Job)
	return client.Batch().Jobs(job.Namespace).Update(job)
}

func (a *JobAdapter) ClusterWatch(client kubeclientset.Interface, namespace string, options metav1.ListOptions) (watch.Interface, error) {
	return client.Batch().Jobs(namespace).Watch(options)
}

func (a *JobAdapter) EquivalentIgnoringSchedule(obj1, obj2 pkgruntime.Object) bool {
	job1 := obj1.(*batchv1.Job)
	job2 := a.Copy(obj2).(*batchv1.Job)
	//job2.Spec.Replicas = job1.Spec.Replicas // why do we need to copy Replicas address?
	return fedutil.ObjectMetaAndSpecEquivalent(job1, job2)
}

func (a *JobAdapter) NewTestObject(namespace string) pkgruntime.Object {
	parallelism := int32(3)
	completions := int32(3)
	zero := int64(0)
	return &batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "test-job-",
			Namespace:    namespace,
		},
		Spec: batchv1.JobSpec{
			Parallelism: &parallelism,
			Completions: &completions,
			Template: apiv1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{"foo": "bar"},
				},
				Spec: apiv1.PodSpec{
					TerminationGracePeriodSeconds: &zero,
					Containers: []apiv1.Container{
						{
							Name:  "busybox",
							Image: "busybox",
							Command: []string{
								"echo",
								"Hello FederatedType Jobs!",
							},
						},
					},
				},
			},
		},
	}
}
