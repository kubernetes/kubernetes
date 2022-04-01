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

package gcp

import (
	"context"
	"fmt"
	"os/exec"
	"path"
	"regexp"
	"strconv"
	"strings"
	"time"

	e2econfig "k8s.io/kubernetes/test/e2e/framework/config"
	e2eutils "k8s.io/kubernetes/test/e2e/framework/utils"

	"github.com/onsi/ginkgo"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/common"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
)

func addMasterReplica(zone string) error {
	e2eutils.Logf(fmt.Sprintf("Adding a new master replica, zone: %s", zone))
	_, _, err := e2eutils.RunCmd(path.Join(e2econfig.TestContext.RepoRoot, "hack/e2e-internal/e2e-grow-cluster.sh"), zone, "true", "true", "false")
	if err != nil {
		return err
	}
	return nil
}

func removeMasterReplica(zone string) error {
	e2eutils.Logf(fmt.Sprintf("Removing an existing master replica, zone: %s", zone))
	_, _, err := e2eutils.RunCmd(path.Join(e2econfig.TestContext.RepoRoot, "hack/e2e-internal/e2e-shrink-cluster.sh"), zone, "true", "false", "false")
	if err != nil {
		return err
	}
	return nil
}

func addWorkerNodes(zone string) error {
	e2eutils.Logf(fmt.Sprintf("Adding worker nodes, zone: %s", zone))
	_, _, err := e2eutils.RunCmd(path.Join(e2econfig.TestContext.RepoRoot, "hack/e2e-internal/e2e-grow-cluster.sh"), zone, "true", "false", "true")
	if err != nil {
		return err
	}
	return nil
}

func removeWorkerNodes(zone string) error {
	e2eutils.Logf(fmt.Sprintf("Removing worker nodes, zone: %s", zone))
	_, _, err := e2eutils.RunCmd(path.Join(e2econfig.TestContext.RepoRoot, "hack/e2e-internal/e2e-shrink-cluster.sh"), zone, "true", "true", "true")
	if err != nil {
		return err
	}
	return nil
}

func verifyRCs(c clientset.Interface, ns string, names []string) {
	for _, name := range names {
		e2eutils.ExpectNoError(e2epod.VerifyPods(c, ns, name, true, 1))
	}
}

func createNewRC(c clientset.Interface, ns string, name string) {
	_, err := common.NewRCByName(c, ns, name, 1, nil, nil)
	e2eutils.ExpectNoError(err)
}

func findRegionForZone(zone string) string {
	region, err := exec.Command("gcloud", "compute", "zones", "list", zone, "--quiet", "--format=csv[no-heading](region)").Output()
	e2eutils.ExpectNoError(err)
	if string(region) == "" {
		e2eutils.Failf("Region not found; zone: %s", zone)
	}
	return string(region)
}

func findZonesForRegion(region string) []string {
	output, err := exec.Command("gcloud", "compute", "zones", "list", "--filter=region="+region,
		"--quiet", "--format=csv[no-heading](name)").Output()
	e2eutils.ExpectNoError(err)
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

// generateMasterRegexp returns a regex for matching master node name.
func generateMasterRegexp(prefix string) string {
	return prefix + "(-...)?"
}

// waitForMasters waits until the cluster has the desired number of ready masters in it.
func waitForMasters(masterPrefix string, c clientset.Interface, size int, timeout time.Duration) error {
	for start := time.Now(); time.Since(start) < timeout; time.Sleep(20 * time.Second) {
		nodes, err := c.CoreV1().Nodes().List(context.TODO(), metav1.ListOptions{})
		if err != nil {
			e2eutils.Logf("Failed to list nodes: %v", err)
			continue
		}

		// Filter out nodes that are not master replicas
		e2enode.Filter(nodes, func(node v1.Node) bool {
			res, err := regexp.Match(generateMasterRegexp(masterPrefix), ([]byte)(node.Name))
			if err != nil {
				e2eutils.Logf("Failed to match regexp to node name: %v", err)
				return false
			}
			return res
		})

		numNodes := len(nodes.Items)

		// Filter out not-ready nodes.
		e2enode.Filter(nodes, func(node v1.Node) bool {
			return e2enode.IsConditionSetAsExpected(&node, v1.NodeReady, true)
		})

		numReady := len(nodes.Items)

		if numNodes == size && numReady == size {
			e2eutils.Logf("Cluster has reached the desired number of masters %d", size)
			return nil
		}
		e2eutils.Logf("Waiting for the number of masters %d, current %d, not ready master nodes %d", size, numNodes, numNodes-numReady)
	}
	return fmt.Errorf("timeout waiting %v for the number of masters to be %d", timeout, size)
}

var _ = SIGDescribe("HA-master [Feature:HAMaster]", func() {
	f := framework.NewDefaultFramework("ha-master")
	var c clientset.Interface
	var ns string
	var additionalReplicaZones []string
	var additionalNodesZones []string
	var existingRCs []string

	ginkgo.BeforeEach(func() {
		e2eskipper.SkipUnlessProviderIs("gce")
		c = f.ClientSet
		ns = f.Namespace.Name
		e2eutils.ExpectNoError(waitForMasters(e2econfig.TestContext.CloudConfig.MasterName, c, 1, 10*time.Minute))
		additionalReplicaZones = make([]string, 0)
		existingRCs = make([]string, 0)
	})

	ginkgo.AfterEach(func() {
		// Clean-up additional worker nodes if the test execution was broken.
		for _, zone := range additionalNodesZones {
			removeWorkerNodes(zone)
		}
		e2eutils.ExpectNoError(e2eutils.AllNodesReady(c, 5*time.Minute))

		// Clean-up additional master replicas if the test execution was broken.
		for _, zone := range additionalReplicaZones {
			removeMasterReplica(zone)
		}
		e2eutils.ExpectNoError(waitForMasters(e2econfig.TestContext.CloudConfig.MasterName, c, 1, 10*time.Minute))
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
			e2eutils.ExpectNoError(addMasterReplica(zone))
			additionalReplicaZones = append(additionalReplicaZones, zone)
		case RemoveReplica:
			e2eutils.ExpectNoError(removeMasterReplica(zone))
			additionalReplicaZones = removeZoneFromZones(additionalReplicaZones, zone)
		case AddNodes:
			e2eutils.ExpectNoError(addWorkerNodes(zone))
			additionalNodesZones = append(additionalNodesZones, zone)
		case RemoveNodes:
			e2eutils.ExpectNoError(removeWorkerNodes(zone))
			additionalNodesZones = removeZoneFromZones(additionalNodesZones, zone)
		}
		e2eutils.ExpectNoError(waitForMasters(e2econfig.TestContext.CloudConfig.MasterName, c, len(additionalReplicaZones)+1, 10*time.Minute))
		e2eutils.ExpectNoError(e2eutils.AllNodesReady(c, 5*time.Minute))

		// Verify that API server works correctly with HA master.
		rcName := "ha-master-" + strconv.Itoa(len(existingRCs))
		createNewRC(c, ns, rcName)
		existingRCs = append(existingRCs, rcName)
		verifyRCs(c, ns, existingRCs)
	}

	ginkgo.It("survive addition/removal replicas same zone [Serial][Disruptive]", func() {
		zone := e2econfig.TestContext.CloudConfig.Zone
		step(None, "")
		numAdditionalReplicas := 2
		for i := 0; i < numAdditionalReplicas; i++ {
			step(AddReplica, zone)
		}
		for i := 0; i < numAdditionalReplicas; i++ {
			step(RemoveReplica, zone)
		}
	})

	ginkgo.It("survive addition/removal replicas different zones [Serial][Disruptive]", func() {
		zone := e2econfig.TestContext.CloudConfig.Zone
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

	ginkgo.It("survive addition/removal replicas multizone workers [Serial][Disruptive]", func() {
		zone := e2econfig.TestContext.CloudConfig.Zone
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
