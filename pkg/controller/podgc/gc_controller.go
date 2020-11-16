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

package podgc

import (
	"context"
	"sort"
	"sync"
	"time"

	batch "k8s.io/api/batch/v1"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	batchinformers "k8s.io/client-go/informers/batch/v1"
	coreinformers "k8s.io/client-go/informers/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	batchv1listers "k8s.io/client-go/listers/batch/v1"
	corelisters "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/component-base/metrics/prometheus/ratelimiter"

	"k8s.io/klog/v2"
)

const (
	// gcCheckPeriod defines frequency of running main controller loop
	gcCheckPeriod = 20 * time.Second
	// quarantineTime defines how long Orphaned GC waits for nodes to show up
	// in an informer before issuing a GET call to check if they are truly gone
	quarantineTime = 40 * time.Second
)

type PodGCController struct {
	kubeClient clientset.Interface

	podLister        corelisters.PodLister
	podListerSynced  cache.InformerSynced
	nodeLister       corelisters.NodeLister
	nodeListerSynced cache.InformerSynced

	nodeQueue workqueue.DelayingInterface

	jobLister       batchv1listers.JobLister
	jobListerSynced cache.InformerSynced

	deletePod              func(namespace, name string) error
	terminatedPodThreshold int
}

func NewPodGC(kubeClient clientset.Interface,
	podInformer coreinformers.PodInformer,
	nodeInformer coreinformers.NodeInformer,
	jobInformer batchinformers.JobInformer,
	terminatedPodThreshold int) *PodGCController {
	if kubeClient != nil && kubeClient.CoreV1().RESTClient().GetRateLimiter() != nil {
		ratelimiter.RegisterMetricAndTrackRateLimiterUsage("gc_controller", kubeClient.CoreV1().RESTClient().GetRateLimiter())
	}
	gcc := &PodGCController{
		kubeClient:             kubeClient,
		terminatedPodThreshold: terminatedPodThreshold,
		podLister:              podInformer.Lister(),
		podListerSynced:        podInformer.Informer().HasSynced,
		nodeLister:             nodeInformer.Lister(),
		nodeListerSynced:       nodeInformer.Informer().HasSynced,
		nodeQueue:              workqueue.NewNamedDelayingQueue("orphaned_pods_nodes"),
		deletePod: func(namespace, name string) error {
			klog.Infof("PodGC is force deleting Pod: %v/%v", namespace, name)
			return kubeClient.CoreV1().Pods(namespace).Delete(context.TODO(), name, *metav1.NewDeleteOptions(0))
		},
	}

	gcc.podLister = podInformer.Lister()
	gcc.podListerSynced = podInformer.Informer().HasSynced

	gcc.jobLister = jobInformer.Lister()
	gcc.jobListerSynced = jobInformer.Informer().HasSynced

	return gcc
}

func (gcc *PodGCController) Run(stop <-chan struct{}) {
	defer utilruntime.HandleCrash()

	klog.Infof("Starting GC controller")
	defer gcc.nodeQueue.ShutDown()
	defer klog.Infof("Shutting down GC controller")

	if !cache.WaitForNamedCacheSync("GC", stop, gcc.podListerSynced, gcc.jobListerSynced, gcc.nodeListerSynced) {
		return
	}

	go wait.Until(gcc.gc, gcCheckPeriod, stop)

	<-stop
}

func (gcc *PodGCController) gc() {
	pods, err := gcc.podLister.List(labels.Everything())
	if err != nil {
		klog.Errorf("Error while listing all pods: %v", err)
		return
	}
	nodes, err := gcc.nodeLister.List(labels.Everything())
	if err != nil {
		klog.Errorf("Error while listing all nodes: %v", err)
		return
	}
	if gcc.terminatedPodThreshold > 0 {
		gcc.gcTerminated(pods)
	}
	gcc.gcOrphaned(pods, nodes)
	gcc.gcUnscheduledTerminating(pods)
}

func isPodTerminated(pod *v1.Pod) bool {
	if phase := pod.Status.Phase; phase != v1.PodPending && phase != v1.PodRunning && phase != v1.PodUnknown {
		return true
	}
	return false
}

func (gcc *PodGCController) gcTerminated(pods []*v1.Pod) {
	terminatedPods := []*v1.Pod{}
	for _, pod := range pods {
		if isPodTerminated(pod) {
			terminatedPods = append(terminatedPods, pod)
		}
	}

	terminatedPodCount := len(terminatedPods)
	deleteCount := terminatedPodCount - gcc.terminatedPodThreshold

	if deleteCount <= 0 {
		return
	}

	klog.Infof("garbage collecting %v pods", deleteCount)
	// sort only when necessary
	sort.Sort(byCreationTimestamp(terminatedPods))
	var wait sync.WaitGroup
	for i := 0; i < deleteCount; i++ {
		wait.Add(1)
		go func(namespace string, name string) {
			defer wait.Done()
			if err := gcc.deletePod(namespace, name); err != nil {
				// ignore not founds
				defer utilruntime.HandleError(err)
			}
		}(terminatedPods[i].Namespace, terminatedPods[i].Name)
	}
	wait.Wait()
}

// gcOrphaned deletes pods that are bound to nodes that don't exist.
func (gcc *PodGCController) gcOrphaned(pods []*v1.Pod, nodes []*v1.Node) {
	klog.V(4).Infof("GC'ing orphaned")
	existingNodeNames := sets.NewString()
	for _, node := range nodes {
		existingNodeNames.Insert(node.Name)
	}
	// Add newly found unknown nodes to quarantine
	for _, pod := range pods {
		if pod.Spec.NodeName != "" && !existingNodeNames.Has(pod.Spec.NodeName) {
			gcc.nodeQueue.AddAfter(pod.Spec.NodeName, quarantineTime)
		}
	}
	// Check if nodes are still missing after quarantine period
	deletedNodesNames, quit := gcc.discoverDeletedNodes(existingNodeNames)
	if quit {
		return
	}
	// Delete orphaned pods
	for _, pod := range pods {
		if !deletedNodesNames.Has(pod.Spec.NodeName) {
			continue
		}
		if !gcc.orphanedPodDeletable(pod) {
			continue
		}
		klog.V(2).Infof("Found orphaned Pod %v/%v assigned to the Node %v. Deleting.", pod.Namespace, pod.Name, pod.Spec.NodeName)
		if err := gcc.deletePod(pod.Namespace, pod.Name); err != nil {
			utilruntime.HandleError(err)
		} else {
			klog.V(0).Infof("Forced deletion of orphaned Pod %v/%v succeeded", pod.Namespace, pod.Name)
		}
	}
}

func (gcc *PodGCController) discoverDeletedNodes(existingNodeNames sets.String) (sets.String, bool) {
	deletedNodesNames := sets.NewString()
	for gcc.nodeQueue.Len() > 0 {
		item, quit := gcc.nodeQueue.Get()
		if quit {
			return nil, true
		}
		nodeName := item.(string)
		if !existingNodeNames.Has(nodeName) {
			exists, err := gcc.checkIfNodeExists(nodeName)
			switch {
			case err != nil:
				klog.Errorf("Error while getting node %q: %v", nodeName, err)
				// Node will be added back to the queue in the subsequent loop if still needed
			case !exists:
				deletedNodesNames.Insert(nodeName)
			}
		}
		gcc.nodeQueue.Done(item)
	}
	return deletedNodesNames, false
}

func (gcc *PodGCController) checkIfNodeExists(name string) (bool, error) {
	_, fetchErr := gcc.kubeClient.CoreV1().Nodes().Get(context.TODO(), name, metav1.GetOptions{})
	if errors.IsNotFound(fetchErr) {
		return false, nil
	}
	return fetchErr == nil, fetchErr
}

// gcUnscheduledTerminating deletes pods that are terminating and haven't been scheduled to a particular node.
func (gcc *PodGCController) gcUnscheduledTerminating(pods []*v1.Pod) {
	klog.V(4).Infof("GC'ing unscheduled pods which are terminating.")

	for _, pod := range pods {
		if pod.DeletionTimestamp == nil || len(pod.Spec.NodeName) > 0 {
			continue
		}

		klog.V(2).Infof("Found unscheduled terminating Pod %v/%v not assigned to any Node. Deleting.", pod.Namespace, pod.Name)
		if err := gcc.deletePod(pod.Namespace, pod.Name); err != nil {
			utilruntime.HandleError(err)
		} else {
			klog.V(0).Infof("Forced deletion of unscheduled terminating Pod %v/%v succeeded", pod.Namespace, pod.Name)
		}
	}
}

func isJobFinished(j *batch.Job) bool {
	for _, c := range j.Status.Conditions {
		if (c.Type == batch.JobComplete || c.Type == batch.JobFailed) && c.Status == v1.ConditionTrue {
			return true
		}
	}
	return false
}

// We should not delete a pod if it's terminated
// and belongs to a job and the job has not finished yet.
func (gcc *PodGCController) orphanedPodDeletable(pod *v1.Pod) bool {
	if !isPodTerminated(pod) {
		return true
	}

	controllerRef := metav1.GetControllerOf(pod)
	if controllerRef == nil {
		return true
	}
	job := gcc.resolveControllerRef(pod.Namespace, controllerRef)
	if job == nil || isJobFinished(job) {
		return true
	}

	return false
}

var concernedKind = batch.SchemeGroupVersion.WithKind("Job")

// resolveControllerRef returns the controller referenced by a ControllerRef,
// or nil if the ControllerRef could not be resolved to a matching controller
// of the concerned Kind.
func (gcc *PodGCController) resolveControllerRef(namespace string, controllerRef *metav1.OwnerReference) *batch.Job {
	// We can't look up by UID, so look up by Name and then verify UID.
	// Don't even try to look up by Name if it's the wrong Kind.
	if controllerRef.Kind != concernedKind.Kind {
		return nil
	}
	job, err := gcc.jobLister.Jobs(namespace).Get(controllerRef.Name)
	if err != nil {
		return nil
	}
	if job.UID != controllerRef.UID {
		// The controller we found with this Name is not the same one that the
		// ControllerRef points to.
		return nil
	}
	return job
}

// byCreationTimestamp sorts a list by creation timestamp, using their names as a tie breaker.
type byCreationTimestamp []*v1.Pod

func (o byCreationTimestamp) Len() int      { return len(o) }
func (o byCreationTimestamp) Swap(i, j int) { o[i], o[j] = o[j], o[i] }

func (o byCreationTimestamp) Less(i, j int) bool {
	if o[i].CreationTimestamp.Equal(&o[j].CreationTimestamp) {
		return o[i].Name < o[j].Name
	}
	return o[i].CreationTimestamp.Before(&o[j].CreationTimestamp)
}
