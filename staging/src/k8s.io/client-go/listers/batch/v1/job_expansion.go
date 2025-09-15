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

package v1

import (
	"fmt"

	batch "k8s.io/api/batch/v1"
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
)

// JobListerExpansion allows custom methods to be added to
// JobLister.
type JobListerExpansion interface {
	// GetPodJobs returns a list of Jobs that potentially
	// match a Pod. Only the one specified in the Pod's ControllerRef
	// will actually manage it.
	// Returns an error only if no matching Jobs are found.
	GetPodJobs(pod *v1.Pod) (jobs []batch.Job, err error)
}

// GetPodJobs returns a list of Jobs that potentially
// match a Pod. Only the one specified in the Pod's ControllerRef
// will actually manage it.
// Returns an error only if no matching Jobs are found.
func (l *jobLister) GetPodJobs(pod *v1.Pod) (jobs []batch.Job, err error) {
	if len(pod.Labels) == 0 {
		err = fmt.Errorf("no jobs found for pod %v because it has no labels", pod.Name)
		return
	}

	var list []*batch.Job
	list, err = l.Jobs(pod.Namespace).List(labels.Everything())
	if err != nil {
		return
	}
	for _, job := range list {
		selector, err := metav1.LabelSelectorAsSelector(job.Spec.Selector)
		if err != nil {
			// This object has an invalid selector, it does not match the pod
			continue
		}
		if !selector.Matches(labels.Set(pod.Labels)) {
			continue
		}
		jobs = append(jobs, *job)
	}
	if len(jobs) == 0 {
		err = fmt.Errorf("could not find jobs for pod %s in namespace %s with labels: %v", pod.Name, pod.Namespace, pod.Labels)
	}
	return
}

// JobNamespaceListerExpansion allows custom methods to be added to
// JobNamespaceLister.
type JobNamespaceListerExpansion interface{}
