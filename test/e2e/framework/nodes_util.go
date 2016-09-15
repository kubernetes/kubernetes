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

	"k8s.io/kubernetes/pkg/api"
	client "k8s.io/kubernetes/pkg/client/unversioned"
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
	if _, err := CheckNodesReady(f.Client, RestartNodeReadyAgainTimeout, TestContext.CloudConfig.NumNodes); err != nil {
		return err
	}
	return nil
}

func nodeUpgradeGCE(rawV string) error {
	v := "v" + rawV
	_, _, err := RunCmd(path.Join(TestContext.RepoRoot, "cluster/gce/upgrade.sh"), "-N", v)
	return err
}

// MigRollingUpdate starts a MIG rolling update, upgrading the nodes to a new
// instance template named tmpl, and waits up to nt times the number of nodes
// for it to complete.
func MigRollingUpdate(tmpl string, nt time.Duration) error {
	Logf(fmt.Sprintf("starting the MIG rolling update to %s", tmpl))
	id, err := migRollingUpdateStart(tmpl, nt)
	if err != nil {
		return fmt.Errorf("couldn't start the MIG rolling update: %v", err)
	}

	Logf(fmt.Sprintf("polling the MIG rolling update (%s) until it completes", id))
	if err := migRollingUpdatePoll(id, nt); err != nil {
		return fmt.Errorf("err waiting until update completed: %v", err)
	}

	return nil
}

// migRollingUpdateStart (GCE/GKE-only) starts a MIG rolling update using templ
// as the new template, waiting up to nt per node, and returns the ID of that
// update.
func migRollingUpdateStart(templ string, nt time.Duration) (string, error) {
	var errLast error
	var id string
	prefix, suffix := "Started [", "]."
	if err := wait.Poll(Poll, SingleCallTimeout, func() (bool, error) {
		// TODO(mikedanese): make this hit the compute API directly instead of
		//                 shelling out to gcloud.
		// NOTE(mikedanese): If you are changing this gcloud command, update
		//                 cluster/gce/upgrade.sh to match this EXACTLY.
		// A `rolling-updates start` call outputs what we want to stderr.
		_, output, err := retryCmd("gcloud", "alpha", "compute",
			"rolling-updates",
			fmt.Sprintf("--project=%s", TestContext.CloudConfig.ProjectID),
			fmt.Sprintf("--zone=%s", TestContext.CloudConfig.Zone),
			"start",
			// Required args.
			fmt.Sprintf("--group=%s", TestContext.CloudConfig.NodeInstanceGroup),
			fmt.Sprintf("--template=%s", templ),
			// Optional args to fine-tune behavior.
			fmt.Sprintf("--instance-startup-timeout=%ds", int(nt.Seconds())),
			// NOTE: We can speed up this process by increasing
			//       --max-num-concurrent-instances.
			fmt.Sprintf("--max-num-concurrent-instances=%d", 1),
			fmt.Sprintf("--max-num-failed-instances=%d", 0),
			fmt.Sprintf("--min-instance-update-time=%ds", 0))
		if err != nil {
			errLast = fmt.Errorf("rolling-updates call failed with err: %v", err)
			return false, nil
		}

		// The 'start' call probably succeeded; parse the output and try to find
		// the line that looks like "Started [url/to/<id>]." and return <id>.
		for _, line := range strings.Split(output, "\n") {
			// As a sanity check, ensure the line starts with prefix and ends
			// with suffix.
			if strings.Index(line, prefix) != 0 || strings.Index(line, suffix) != len(line)-len(suffix) {
				continue
			}
			url := strings.Split(strings.TrimSuffix(strings.TrimPrefix(line, prefix), suffix), "/")
			id = url[len(url)-1]
			Logf("Started MIG rolling update; ID: %s", id)
			return true, nil
		}
		errLast = fmt.Errorf("couldn't find line like '%s ... %s' in output to MIG rolling-update start. Output: %s",
			prefix, suffix, output)
		return false, nil
	}); err != nil {
		return "", fmt.Errorf("migRollingUpdateStart() failed with last error: %v", errLast)
	}
	return id, nil
}

// migRollingUpdatePoll (CKE/GKE-only) polls the progress of the MIG rolling
// update with ID id until it is complete. It returns an error if this takes
// longer than nt times the number of nodes.
func migRollingUpdatePoll(id string, nt time.Duration) error {
	// Two keys and a val.
	status, progress, done := "status", "statusMessage", "ROLLED_OUT"
	start, timeout := time.Now(), nt*time.Duration(TestContext.CloudConfig.NumNodes)
	var errLast error
	Logf("Waiting up to %v for MIG rolling update to complete.", timeout)
	if wait.Poll(RestartPoll, timeout, func() (bool, error) {
		// A `rolling-updates describe` call outputs what we want to stdout.
		output, _, err := retryCmd("gcloud", "alpha", "compute",
			"rolling-updates",
			fmt.Sprintf("--project=%s", TestContext.CloudConfig.ProjectID),
			fmt.Sprintf("--zone=%s", TestContext.CloudConfig.Zone),
			"describe",
			id)
		if err != nil {
			errLast = fmt.Errorf("Error calling rolling-updates describe %s: %v", id, err)
			Logf("%v", errLast)
			return false, nil
		}

		// The 'describe' call probably succeeded; parse the output and try to
		// find the line that looks like "status: <status>" and see whether it's
		// done.
		Logf("Waiting for MIG rolling update: %s (%v elapsed)",
			ParseKVLines(output, progress), time.Since(start))
		if st := ParseKVLines(output, status); st == done {
			return true, nil
		}
		return false, nil
	}) != nil {
		return fmt.Errorf("timeout waiting %v for MIG rolling update to complete. Last error: %v", timeout, errLast)
	}
	Logf("MIG rolling update complete after %v", time.Since(start))
	return nil
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
func CheckNodesReady(c *client.Client, nt time.Duration, expect int) ([]string, error) {
	// First, keep getting all of the nodes until we get the number we expect.
	var nodeList *api.NodeList
	var errLast error
	start := time.Now()
	found := wait.Poll(Poll, nt, func() (bool, error) {
		// A rolling-update (GCE/GKE implementation of restart) can complete before the apiserver
		// knows about all of the nodes. Thus, we retry the list nodes call
		// until we get the expected number of nodes.
		nodeList, errLast = c.Nodes().List(api.ListOptions{
			FieldSelector: fields.Set{"spec.unschedulable": "false"}.AsSelector()})
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
