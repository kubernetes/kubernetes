/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package algorithm

import (
	"testing"

	extensions "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
)

// Some functions used by multiple federated-scheduler tests.

type schedulerTester struct {
	t          *testing.T
	scheduler  ScheduleAlgorithm
	clusterLister ClusterLister
}

// Call if you know exactly where rc should get scheduled.
func (st *schedulerTester) expectSchedule(rc *extensions.ReplicaSet, expected string) {
	actual, err := st.scheduler.Schedule(rc, st.clusterLister)
	if err != nil {
		st.t.Errorf("Unexpected error %v\nTried to schedule: %#v", err, rc)
		return
	}
	if actual != expected {
		st.t.Errorf("Unexpected scheduling value: %v, expected %v", actual, expected)
	}
}

// Call if you can't predict where rc will be scheduled.
func (st *schedulerTester) expectSuccess(rc *extensions.ReplicaSet) {
	_, err := st.scheduler.Schedule(rc, st.clusterLister)
	if err != nil {
		st.t.Errorf("Unexpected error %v\nTried to federated-scheduler: %#v", err, rc)
		return
	}
}

// Call if rc should *not* schedule.
func (st *schedulerTester) expectFailure(rc *extensions.ReplicaSet) {
	_, err := st.scheduler.Schedule(rc, st.clusterLister)
	if err == nil {
		st.t.Error("Unexpected non-error")
	}
}
