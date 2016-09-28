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
	"bytes"
	"fmt"
	. "github.com/onsi/ginkgo"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/test/e2e/framework"
	"os/exec"
	"path"
	"strconv"
)

func addMasterReplica() error {
	framework.Logf(fmt.Sprintf("Adding a new master replica:"))
	v, _, err := framework.RunCmd(path.Join(framework.TestContext.RepoRoot, "hack/e2e-internal/e2e-add-master.sh"))
	framework.Logf("%s", v)
	if err != nil {
		return err
	}
	return nil
}

func removeMasterReplica() error {
	framework.Logf(fmt.Sprintf("Removing an existing master replica:"))
	v, _, err := framework.RunCmd(path.Join(framework.TestContext.RepoRoot, "hack/e2e-internal/e2e-remove-master.sh"))
	framework.Logf("%s", v)
	if err != nil {
		return err
	}
	return nil
}

func verifyRCs(c *client.Client, ns string, names []string) {
	for _, name := range names {
		framework.ExpectNoError(framework.VerifyPods(c, ns, name, true, 1))
	}
}

func createNewRC(c *client.Client, ns string, name string) {
	_, err := newRCByName(c, ns, name, 1)
	framework.ExpectNoError(err)
}

func verifyNumberOfMasterReplicas(expected int) {
	output, err := exec.Command("gcloud", "compute", "instances", "list",
		"--project="+framework.TestContext.CloudConfig.ProjectID,
		"--zones="+framework.TestContext.CloudConfig.Zone,
		"--regexp="+framework.TestContext.CloudConfig.MasterName+"(-...)?",
		"--filter=status=RUNNING",
		"--format=[no-heading]").CombinedOutput()
	framework.Logf("%s", output)
	framework.ExpectNoError(err)
	newline := []byte("\n")
	replicas := bytes.Count(output, newline)
	framework.Logf("Num master replicas/expected: %d/%d", replicas, expected)
	if replicas != expected {
		framework.Failf("Wrong number of master replicas")
	}
}

var _ = framework.KubeDescribe("HA-master [Feature:HAMaster]", func() {
	f := framework.NewDefaultFramework("ha-master")
	var c *client.Client
	var ns string
	var additionalReplicas int
	var existingRCs []string

	BeforeEach(func() {
		framework.SkipUnlessProviderIs("gce")
		c = f.Client
		ns = f.Namespace.Name
		verifyNumberOfMasterReplicas(1)
		additionalReplicas = 0
		existingRCs = make([]string, 0)
	})

	AfterEach(func() {
		// Clean-up additional master replicas if the test execution was broken.
		for i := 0; i < additionalReplicas; i++ {
			removeMasterReplica()
		}
	})

	type Action int
	const (
		None Action = iota
		AddReplica
		RemoveReplica
	)

	step := func(action Action) {
		switch action {
		case None:
		case AddReplica:
			framework.ExpectNoError(addMasterReplica())
			additionalReplicas++
		case RemoveReplica:
			framework.ExpectNoError(removeMasterReplica())
			additionalReplicas--
		}
		verifyNumberOfMasterReplicas(additionalReplicas + 1)

		// Verify that API server works correctly with HA master.
		rcName := "ha-master-" + strconv.Itoa(len(existingRCs))
		createNewRC(c, ns, rcName)
		existingRCs = append(existingRCs, rcName)
		verifyRCs(c, ns, existingRCs)
	}

	It("pods survive addition/removal [Slow]", func() {
		step(None)
		step(AddReplica)
		step(AddReplica)
		step(RemoveReplica)
		step(RemoveReplica)
	})
})
