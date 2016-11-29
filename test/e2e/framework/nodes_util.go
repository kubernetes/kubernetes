/*
Copyright 2014 The Kubernetes Authors.

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

package framework

import (
	"fmt"
	"path"
	"strings"
	"time"

	"k8s.io/kubernetes/pkg/api/v1"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/util/wait"
)

// The following upgrade functions are passed into the framework below and used
// to do the actual upgrades.
var MasterUpgrade = func(v string) error {
	switch TestContext.Provider {
	case "gce":
		return masterUpgradeGCE(v)
	case "gke":
		return masterUpgradeGKE(v)
	default:
		return fmt.Errorf("MasterUpgrade() is not implemented for provider %s", TestContext.Provider)
	}
}

func masterUpgradeGCE(rawV string) error {
	v := "v" + rawV
	_, _, err := RunCmd(path.Join(TestContext.RepoRoot, "cluster/gce/upgrade.sh"), "-M", v)
	return err
}

func masterUpgradeGKE(v string) error {
	Logf("Upgrading master to %q", v)
	_, _, err := RunCmd("gcloud", "container",
		"clusters",
		fmt.Sprintf("--project=%s", TestContext.CloudConfig.ProjectID),
		fmt.Sprintf("--zone=%s", TestContext.CloudConfig.Zone),
		"upgrade",
		TestContext.CloudConfig.Cluster,
		"--master",
		fmt.Sprintf("--cluster-version=%s", v),
		"--quiet")
	return err
}

var NodeUpgrade = func(f *Framework, v string, img string) error {
	// Perform the upgrade.
	var err error
	switch TestContext.Provider {
	case "gce":
		// TODO(maisem): add GCE support for upgrading to different images.
		err = nodeUpgradeGCE(v)
	case "gke":
		err = nodeUpgradeGKE(v, img)
	default:
		err = fmt.Errorf("NodeUpgrade() is not implemented for provider %s", TestContext.Provider)
	}
	if err != nil {
		return err
	}

	// Wait for it to complete and validate nodes are healthy.
	//
	// TODO(ihmccreery) We shouldn't have to wait for nodes to be ready in
	// GKE; the operation shouldn't return until they all are.
	Logf("Waiting up to %v for all nodes to be ready after the upgrade", RestartNodeReadyAgainTimeout)
	if _, err := CheckNodesReady(f.ClientSet, RestartNodeReadyAgainTimeout, TestContext.CloudConfig.NumNodes); err != nil {
		return err
	}
	return nil
}

func nodeUpgradeGCE(rawV string) error {
	v := "v" + rawV
	_, _, err := RunCmd(path.Join(TestContext.RepoRoot, "cluster/gce/upgrade.sh"), "-N", v)
	return err
}

func cleanupNodeUpgradeGCE(tmplBefore string) {
	Logf("Cleaning up any unused node templates")
	tmplAfter, err := MigTemplate()
	if err != nil {
		Logf("Could not get node template post-upgrade; may have leaked template %s", tmplBefore)
		return
	}
	if tmplBefore == tmplAfter {
		// The node upgrade failed so there's no need to delete
		// anything.
		Logf("Node template %s is still in use; not cleaning up", tmplBefore)
		return
	}
	Logf("Deleting node template %s", tmplBefore)
	if _, _, err := retryCmd("gcloud", "compute", "instance-templates",
		fmt.Sprintf("--project=%s", TestContext.CloudConfig.ProjectID),
		"delete",
		tmplBefore); err != nil {
		Logf("gcloud compute instance-templates delete %s call failed with err: %v", tmplBefore, err)
		Logf("May have leaked instance template %q", tmplBefore)
	}
}

func nodeUpgradeGKE(v string, img string) error {
	Logf("Upgrading nodes to version %q and image %q", v, img)
	args := []string{
		"container",
		"clusters",
		fmt.Sprintf("--project=%s", TestContext.CloudConfig.ProjectID),
		fmt.Sprintf("--zone=%s", TestContext.CloudConfig.Zone),
		"upgrade",
		TestContext.CloudConfig.Cluster,
		fmt.Sprintf("--cluster-version=%s", v),
		"--quiet",
	}
	if len(img) > 0 {
		args = append(args, fmt.Sprintf("--image-type=%s", img))
	}
	_, _, err := RunCmd("gcloud", args...)
	return err
}

// CheckNodesReady waits up to nt for expect nodes accessed by c to be ready,
// returning an error if this doesn't happen in time. It returns the names of
// nodes it finds.
func CheckNodesReady(c clientset.Interface, nt time.Duration, expect int) ([]string, error) {
	// First, keep getting all of the nodes until we get the number we expect.
	var nodeList *v1.NodeList
	var errLast error
	start := time.Now()
	found := wait.Poll(Poll, nt, func() (bool, error) {
		// A rolling-update (GCE/GKE implementation of restart) can complete before the apiserver
		// knows about all of the nodes. Thus, we retry the list nodes call
		// until we get the expected number of nodes.
		nodeList, errLast = c.Core().Nodes().List(v1.ListOptions{
			FieldSelector: fields.Set{"spec.unschedulable": "false"}.AsSelector().String()})
		if errLast != nil {
			return false, nil
		}
		if len(nodeList.Items) != expect {
			errLast = fmt.Errorf("expected to find %d nodes but found only %d (%v elapsed)",
				expect, len(nodeList.Items), time.Since(start))
			Logf("%v", errLast)
			return false, nil
		}
		return true, nil
	}) == nil
	nodeNames := make([]string, len(nodeList.Items))
	for i, n := range nodeList.Items {
		nodeNames[i] = n.ObjectMeta.Name
	}
	if !found {
		return nodeNames, fmt.Errorf("couldn't find %d nodes within %v; last error: %v",
			expect, nt, errLast)
	}
	Logf("Successfully found %d nodes", expect)

	// Next, ensure in parallel that all the nodes are ready. We subtract the
	// time we spent waiting above.
	timeout := nt - time.Since(start)
	result := make(chan bool, len(nodeList.Items))
	for _, n := range nodeNames {
		n := n
		go func() { result <- WaitForNodeToBeReady(c, n, timeout) }()
	}
	failed := false
	// TODO(mbforbes): Change to `for range` syntax once we support only Go
	// >= 1.4.
	for i := range nodeList.Items {
		_ = i
		if !<-result {
			failed = true
		}
	}
	if failed {
		return nodeNames, fmt.Errorf("at least one node failed to be ready")
	}
	return nodeNames, nil
}

// MigTemplate (GCE-only) returns the name of the MIG template that the
// nodes of the cluster use.
func MigTemplate() (string, error) {
	var errLast error
	var templ string
	key := "instanceTemplate"
	if wait.Poll(Poll, SingleCallTimeout, func() (bool, error) {
		// TODO(mikedanese): make this hit the compute API directly instead of
		// shelling out to gcloud.
		// An `instance-groups managed describe` call outputs what we want to stdout.
		output, _, err := retryCmd("gcloud", "compute", "instance-groups", "managed",
			fmt.Sprintf("--project=%s", TestContext.CloudConfig.ProjectID),
			"describe",
			fmt.Sprintf("--zone=%s", TestContext.CloudConfig.Zone),
			TestContext.CloudConfig.NodeInstanceGroup)
		if err != nil {
			errLast = fmt.Errorf("gcloud compute instance-groups managed describe call failed with err: %v", err)
			return false, nil
		}

		// The 'describe' call probably succeeded; parse the output and try to
		// find the line that looks like "instanceTemplate: url/to/<templ>" and
		// return <templ>.
		if val := ParseKVLines(output, key); len(val) > 0 {
			url := strings.Split(val, "/")
			templ = url[len(url)-1]
			Logf("MIG group %s using template: %s", TestContext.CloudConfig.NodeInstanceGroup, templ)
			return true, nil
		}
		errLast = fmt.Errorf("couldn't find %s in output to get MIG template. Output: %s", key, output)
		return false, nil
	}) != nil {
		return "", fmt.Errorf("MigTemplate() failed with last error: %v", errLast)
	}
	return templ, nil
}
