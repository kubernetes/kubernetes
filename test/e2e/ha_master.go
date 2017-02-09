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

package e2e

import (
	"fmt"
	"os/exec"
	"path"
	"strconv"
	"strings"
	"time"

	. "github.com/onsi/ginkgo"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	"k8s.io/kubernetes/test/e2e/framework"
)

func addMasterReplica(zone string) error {
	framework.Logf(fmt.Sprintf("Adding a new master replica, zone: %s", zone))
	v, _, err := framework.RunCmd(path.Join(framework.TestContext.RepoRoot, "hack/e2e-internal/e2e-grow-cluster.sh"), zone, "true", "true", "false")
	framework.Logf("%s", v)
	if err != nil {
		return err
	}
	return nil
}

func removeMasterReplica(zone string) error {
	framework.Logf(fmt.Sprintf("Removing an existing master replica, zone: %s", zone))
	v, _, err := framework.RunCmd(path.Join(framework.TestContext.RepoRoot, "hack/e2e-internal/e2e-shrink-cluster.sh"), zone, "true", "false", "false")
	framework.Logf("%s", v)
	if err != nil {
		return err
	}
	return nil
}

func addWorkerNodes(zone string) error {
	framework.Logf(fmt.Sprintf("Adding worker nodes, zone: %s", zone))
	v, _, err := framework.RunCmd(path.Join(framework.TestContext.RepoRoot, "hack/e2e-internal/e2e-grow-cluster.sh"), zone, "true", "false", "true")
	framework.Logf("%s", v)
	if err != nil {
		return err
	}
	return nil
}

func removeWorkerNodes(zone string) error {
	framework.Logf(fmt.Sprintf("Removing worker nodes, zone: %s", zone))
	v, _, err := framework.RunCmd(path.Join(framework.TestContext.RepoRoot, "hack/e2e-internal/e2e-shrink-cluster.sh"), zone, "true", "true", "true")
	framework.Logf("%s", v)
	if err != nil {
		return err
	}
	return nil
}

func verifyRCs(c clientset.Interface, ns string, names []string) {
	for _, name := range names {
		framework.ExpectNoError(framework.VerifyPods(c, ns, name, true, 1))
	}
}

func createNewRC(c clientset.Interface, ns string, name string) {
	_, err := newRCByName(c, ns, name, 1, nil)
	framework.ExpectNoError(err)
}

func findRegionForZone(zone string) string {
	region, err := exec.Command("gcloud", "compute", "zones", "list", zone, "--quiet", "--format=[no-heading](region)").CombinedOutput()
	framework.ExpectNoError(err)
	if string(region) == "" {
		framework.Failf("Region not found; zone: %s", zone)
	}
	return string(region)
}

func findZonesForRegion(region string) []string {
	output, err := exec.Command("gcloud", "compute", "zones", "list", "--filter=region="+region,
		"--quiet", "--format=[no-heading](name)").CombinedOutput()
	framework.ExpectNoError(err)
	zones := strings.Split(string(output), "\n")
	return zones
}

// removeZoneFromZones removes zone from zones slide.
// Please note that entries in zones can be repeated. In such situation only one replica is removed.
func removeZoneFromZones(zones []string, zone string) []string {
	idx := -1
	for j, z := range zones {
		if z == zone {
			idx = j
			break
		}
	}
	if idx >= 0 {
		return zones[:idx+copy(zones[idx:], zones[idx+1:])]
	}
	return zones
}

var _ = framework.KubeDescribe("HA-master [Feature:HAMaster]", func() {
	f := framework.NewDefaultFramework("ha-master")
	var c clientset.Interface
	var ns string
	var additionalReplicaZones []string
	var additionalNodesZones []string
	var existingRCs []string

	BeforeEach(func() {
		framework.SkipUnlessProviderIs("gce")
		c = f.ClientSet
		ns = f.Namespace.Name
		framework.ExpectNoError(framework.WaitForMasters(framework.TestContext.CloudConfig.MasterName, c, 1, 10*time.Minute))
		additionalReplicaZones = make([]string, 0)
		existingRCs = make([]string, 0)
	})

	AfterEach(func() {
		// Clean-up additional worker nodes if the test execution was broken.
		for _, zone := range additionalNodesZones {
			removeWorkerNodes(zone)
		}
		framework.ExpectNoError(framework.AllNodesReady(c, 5*time.Minute))

		// Clean-up additional master replicas if the test execution was broken.
		for _, zone := range additionalReplicaZones {
			removeMasterReplica(zone)
		}
		framework.ExpectNoError(framework.WaitForMasters(framework.TestContext.CloudConfig.MasterName, c, 1, 10*time.Minute))
	})

	type Action int
	const (
		None Action = iota
		AddReplica
		RemoveReplica
		AddNodes
		RemoveNodes
	)

	step := func(action Action, zone string) {
		switch action {
		case None:
		case AddReplica:
			framework.ExpectNoError(addMasterReplica(zone))
			additionalReplicaZones = append(additionalReplicaZones, zone)
		case RemoveReplica:
			framework.ExpectNoError(removeMasterReplica(zone))
			additionalReplicaZones = removeZoneFromZones(additionalReplicaZones, zone)
		case AddNodes:
			framework.ExpectNoError(addWorkerNodes(zone))
			additionalNodesZones = append(additionalNodesZones, zone)
		case RemoveNodes:
			framework.ExpectNoError(removeWorkerNodes(zone))
			additionalNodesZones = removeZoneFromZones(additionalNodesZones, zone)
		}
		framework.ExpectNoError(framework.WaitForMasters(framework.TestContext.CloudConfig.MasterName, c, len(additionalReplicaZones)+1, 10*time.Minute))
		framework.ExpectNoError(framework.AllNodesReady(c, 5*time.Minute))

		// Verify that API server works correctly with HA master.
		rcName := "ha-master-" + strconv.Itoa(len(existingRCs))
		createNewRC(c, ns, rcName)
		existingRCs = append(existingRCs, rcName)
		verifyRCs(c, ns, existingRCs)
	}

	It("survive addition/removal replicas same zone [Serial][Disruptive]", func() {
		zone := framework.TestContext.CloudConfig.Zone
		step(None, "")
		numAdditionalReplicas := 2
		for i := 0; i < numAdditionalReplicas; i++ {
			step(AddReplica, zone)
		}
		for i := 0; i < numAdditionalReplicas; i++ {
			step(RemoveReplica, zone)
		}
	})

	It("survive addition/removal replicas different zones [Serial][Disruptive]", func() {
		zone := framework.TestContext.CloudConfig.Zone
		region := findRegionForZone(zone)
		zones := findZonesForRegion(region)
		zones = removeZoneFromZones(zones, zone)

		step(None, "")
		// If numAdditionalReplicas is larger then the number of remaining zones in the region,
		// we create a few masters in the same zone and zone entry is repeated in additionalReplicaZones.
		numAdditionalReplicas := 2
		for i := 0; i < numAdditionalReplicas; i++ {
			step(AddReplica, zones[i%len(zones)])
		}
		for i := 0; i < numAdditionalReplicas; i++ {
			step(RemoveReplica, zones[i%len(zones)])
		}
	})

	It("survive addition/removal replicas multizone workers [Serial][Disruptive]", func() {
		zone := framework.TestContext.CloudConfig.Zone
		region := findRegionForZone(zone)
		zones := findZonesForRegion(region)
		zones = removeZoneFromZones(zones, zone)

		step(None, "")
		numAdditionalReplicas := 2

		// Add worker nodes.
		for i := 0; i < numAdditionalReplicas && i < len(zones); i++ {
			step(AddNodes, zones[i])
		}

		// Add master repilcas.
		//
		// If numAdditionalReplicas is larger then the number of remaining zones in the region,
		// we create a few masters in the same zone and zone entry is repeated in additionalReplicaZones.
		for i := 0; i < numAdditionalReplicas; i++ {
			step(AddReplica, zones[i%len(zones)])
		}

		// Remove master repilcas.
		for i := 0; i < numAdditionalReplicas; i++ {
			step(RemoveReplica, zones[i%len(zones)])
		}

		// Remove worker nodes.
		for i := 0; i < numAdditionalReplicas && i < len(zones); i++ {
			step(RemoveNodes, zones[i])
		}
	})
})
