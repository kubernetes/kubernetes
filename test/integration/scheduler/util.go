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

package scheduler

import (
	testutils "k8s.io/kubernetes/test/integration/util"
)

var (
	createPausePod                  = testutils.CreatePausePod
	createPausePodWithResource      = testutils.CreatePausePodWithResource
	createNode                      = testutils.CreateNode
	initPausePod                    = testutils.InitPausePod
	runPausePod                     = testutils.RunPausePod
	waitForPodUnschedulable         = testutils.WaitForPodUnschedulable
	waitForPodToScheduleWithTimeout = testutils.WaitForPodToScheduleWithTimeout
	waitCachedPodsStable            = testutils.WaitCachedPodsStable
	waitForPDBsStable               = testutils.WaitForPDBsStable
	waitForReflection               = testutils.WaitForReflection
	waitForNodesInCache             = testutils.WaitForNodesInCache
	createAndWaitForNodesInCache    = testutils.CreateAndWaitForNodesInCache
	getPod                          = testutils.GetPod
	deletePod                       = testutils.DeletePod
	updateNode                      = testutils.UpdateNode
	podSchedulingError              = testutils.PodSchedulingError
	podScheduledIn                  = testutils.PodScheduledIn
	podUnschedulable                = testutils.PodUnschedulable
	podIsGettingEvicted             = testutils.PodIsGettingEvicted
	initTest                        = testutils.InitTestSchedulerWithNS
	initTestDisablePreemption       = testutils.InitTestDisablePreemption
	initDisruptionController        = testutils.InitDisruptionController
	createNamespacesWithLabels      = testutils.CreateNamespacesWithLabels
	runPodWithContainers            = testutils.RunPodWithContainers
	initPodWithContainers           = testutils.InitPodWithContainers
	nextPodOrDie                    = testutils.NextPodOrDie
	nextPod                         = testutils.NextPod
)
