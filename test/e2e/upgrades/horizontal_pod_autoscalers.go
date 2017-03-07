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

package upgrades

import (
	"fmt"

	"k8s.io/kubernetes/test/e2e/common"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
)

// HPAUpgradeTest tests that HPA rescales target resource correctly before and after a cluster upgrade.
type HPAUpgradeTest struct {
	rc *common.ResourceConsumer
}

func (HPAUpgradeTest) Name() string { return "hpa-upgrade" }

// Creates a resource consumer and an HPA object that autoscales the consumer.
func (t *HPAUpgradeTest) Setup(f *framework.Framework) {
	t.rc = common.NewDynamicResourceConsumer(
		"res-cons-upgrade",
		common.KindRC,
		1,   /* replicas */
		250, /* initCPUTotal */
		0,
		0,
		500, /* cpuLimit */
		200, /* memLimit */
		f)
	common.CreateCPUHorizontalPodAutoscaler(
		t.rc,
		20, /* targetCPUUtilizationPercent */
		1,  /* minPods */
		5)  /* maxPods */

	t.rc.Pause()
	t.test()
}

// Test waits for upgrade to complete and verifies if HPA works correctly.
func (t *HPAUpgradeTest) Test(f *framework.Framework, done <-chan struct{}, upgrade UpgradeType) {
	// Block until upgrade is done
	By(fmt.Sprintf("Waiting for upgrade to finish before checking HPA"))
	<-done
	t.test()
}

// Teardown cleans up any remaining resources.
func (t *HPAUpgradeTest) Teardown(f *framework.Framework) {
	// rely on the namespace deletion to clean up everything
	t.rc.CleanUp()
}

func (t *HPAUpgradeTest) test() {
	t.rc.Resume()

	By(fmt.Sprintf("HPA scales to 1 replica: consume 10 millicores, target per pod 100 millicores, min pods 1."))
	t.rc.ConsumeCPU(10) /* millicores */
	By(fmt.Sprintf("HPA waits for 1 replica"))
	t.rc.WaitForReplicas(1)

	By(fmt.Sprintf("HPA scales to 3 replicas: consume 250 millicores, target per pod 100 millicores."))
	t.rc.ConsumeCPU(250) /* millicores */
	By(fmt.Sprintf("HPA waits for 3 replicas"))
	t.rc.WaitForReplicas(3)

	By(fmt.Sprintf("HPA scales to 5 replicas: consume 700 millicores, target per pod 100 millicores, max pods 5."))
	t.rc.ConsumeCPU(700) /* millicores */
	By(fmt.Sprintf("HPA waits for 5 replicas"))
	t.rc.WaitForReplicas(5)

	// We need to pause background goroutines as during upgrade master is unavailable and requests issued by them fail.
	t.rc.Pause()
}
