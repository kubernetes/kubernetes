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

package scheduler

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/golang/glog"

	fed "k8s.io/kubernetes/federation/apis/federation"
	fedv1 "k8s.io/kubernetes/federation/apis/federation/v1beta1"
	//kubeclientset "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_4"
	planner "k8s.io/kubernetes/federation/pkg/federation-controller/replicaset/planner"
	"k8s.io/kubernetes/pkg/api/meta"
	extensionsv1 "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
)

const (
	// schedule result was put into annotation in a format of "clusterName:replicas[/clusterName:replicas]..."
	ExpectedReplicasAnnotation = "kubernetes.io/expected-replicas"
)

type Scheduler struct {
	planner *planner.Planner
}

func NewScheduler(preferences *fed.FederatedReplicaSetPreferences) *Scheduler {
	return &Scheduler{
		planner: planner.NewPlanner(preferences),
	}
}

func (scheduler *Scheduler) Schedule(frs *extensionsv1.ReplicaSet, clusters []*fedv1.Cluster) map[string]int64 {
	var clusterNames []string
	for _, cluster := range clusters {
		clusterNames = append(clusterNames, cluster.Name)
	}
	scheduleResult := scheduler.planner.Plan(int64(*frs.Spec.Replicas), clusterNames)
	result := make(map[string]int64)
	for clusterName, replicas := range scheduleResult {
		result[clusterName] = replicas
	}

	return result
}

func (scheduler *Scheduler) ScheduleFromAnnotation(frs *extensionsv1.ReplicaSet) map[string]int64 {
	accessor, err := meta.Accessor(frs)
	if err != nil {
		panic(err) // should never happen
	}
	anno := accessor.GetAnnotations()
	scheduleResultString, found := anno[ExpectedReplicasAnnotation]
	scheduleResult := make(map[string]int64)
	if found {
		scheduleResult, err = decodeScheduleResult(scheduleResultString)
	}

	return scheduleResult
}

func decodeScheduleResult(scheduleResultString string) (map[string]int64, error) {
	var scheduleResult = make(map[string]int64)
	clusterReplicas := strings.Split(scheduleResultString, "/")
	for _, clusterReplica := range clusterReplicas {
		cr := strings.Split(clusterReplica, ":")
		if len(cr) != 2 {
			glog.Errorf("Failed decode schedule result: %v", cr)
			return nil, fmt.Errorf("Failed decode schedule result: %v", cr)
		}
		replicas, err := strconv.ParseInt(cr[1], 10, 64)
		if err != nil {
			glog.Errorf("Failed parse scheduled replcias: %v:%v", cr[0], cr[1])
			return nil, err
		}
		scheduleResult[cr[0]] = replicas
	}

	return scheduleResult, nil
}
