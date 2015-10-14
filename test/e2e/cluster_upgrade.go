/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"io"
	"io/ioutil"
	"net/http"
	"os"
	"os/exec"
	"path"
	"strings"
	"sync"
	"time"

	"k8s.io/kubernetes/pkg/api"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/wait"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

const (
	// version applies to upgrades; kube-push always pushes local binaries.
	versionURLFmt = "https://storage.googleapis.com/kubernetes-release/%s/%s.txt"
)

// realVersion turns a version constant s--one accepted by cluster/gce/upgrade.sh--
// into a deployable version string. If the s is not known to be a version
// constant, it will assume it is already a valid version, and return s directly.
//
// NOTE: KEEP THIS LIST UP-TO-DATE WITH THE CODE BELOW.
// The version strings supported are:
// - "latest_stable"   (returns a string like "0.18.2")
// - "latest_release"  (returns a string like "0.19.1")
// - "latest_ci"       (returns a string like "0.19.1-669-gabac8c8")
func realVersion(s string) (string, error) {
	bucket, file := "", ""
	switch s {
	// NOTE: IF YOU CHANGE THE FOLLOWING LIST, ALSO UPDATE cluster/gce/upgrade.sh
	case "latest_stable":
		bucket, file = "release", "stable"
	case "latest_release":
		bucket, file = "release", "latest"
	case "latest_ci":
		bucket, file = "ci", "latest"
	default:
		// If we don't match one of the above, we assume that the passed version
		// is already valid (such as "0.19.1" or "0.19.1-669-gabac8c8").
		Logf("Assuming %q is already a valid version.", s)
		return s, nil
	}

	url := fmt.Sprintf(versionURLFmt, bucket, file)
	var v string
	Logf("Fetching version from %s", url)
	c := &http.Client{Timeout: 2 * time.Second}
	if err := wait.Poll(poll, singleCallTimeout, func() (bool, error) {
		r, err := c.Get(url)
		if err != nil {
			Logf("Error reaching %s: %v", url, err)
			return false, nil
		}
		if r.StatusCode != http.StatusOK {
			Logf("Bad response; status: %d, response: %v", r.StatusCode, r)
			return false, nil
		}
		defer r.Body.Close()
		b, err := ioutil.ReadAll(r.Body)
		if err != nil {
			Logf("Could not read response body: %v", err)
			return false, nil
		}
		v = strings.TrimSpace(string(b))
		return true, nil
	}); err != nil {
		return "", fmt.Errorf("failed to fetch real version from %s", url)
	}
	// Versions start with "v", so remove that.
	return strings.TrimPrefix(v, "v"), nil
}

// The following upgrade functions are passed into the framework below and used
// to do the actual upgrades.
var masterUpgrade = func(v string) error {
	switch testContext.Provider {
	case "gce":
		return masterUpgradeGCE(v)
	case "gke":
		return masterUpgradeGKE(v)
	default:
		return fmt.Errorf("masterUpgrade() is not implemented for provider %s", testContext.Provider)
	}
}

func masterUpgradeGCE(rawV string) error {
	v := "v" + rawV
	_, _, err := runCmd(path.Join(testContext.RepoRoot, "hack/e2e-internal/e2e-upgrade.sh"), "-M", v)
	return err
}

func masterUpgradeGKE(v string) error {
	Logf("Upgrading master to %q", v)
	_, _, err := runCmd("gcloud", "container",
		fmt.Sprintf("--project=%s", testContext.CloudConfig.ProjectID),
		fmt.Sprintf("--zone=%s", testContext.CloudConfig.Zone),
		"clusters",
		"upgrade",
		testContext.CloudConfig.Cluster,
		"--master",
		fmt.Sprintf("--cluster-version=%s", v))
	return err
}

var masterPush = func(_ string) error {
	// TODO(mikedanese): Make master push use the provided version.
	_, _, err := runCmd(path.Join(testContext.RepoRoot, "hack/e2e-internal/e2e-push.sh"), "-m")
	return err
}

var nodeUpgrade = func(f Framework, replicas int, v string) error {
	// Perform the upgrade.
	var err error
	switch testContext.Provider {
	case "gce":
		err = nodeUpgradeGCE(v)
	case "gke":
		err = nodeUpgradeGKE(v)
	default:
		err = fmt.Errorf("nodeUpgrade() is not implemented for provider %s", testContext.Provider)
	}
	if err != nil {
		return err
	}

	// Wait for it to complete and validate nodes and pods are healthy.
	Logf("Waiting up to %v for all nodes to be ready after the upgrade", restartNodeReadyAgainTimeout)
	if _, err := checkNodesReady(f.Client, restartNodeReadyAgainTimeout, testContext.CloudConfig.NumNodes); err != nil {
		return err
	}
	Logf("Waiting up to %v for all pods to be running and ready after the upgrade", restartPodReadyAgainTimeout)
	return waitForPodsRunningReady(f.Namespace.Name, replicas, restartPodReadyAgainTimeout)
}

func nodeUpgradeGCE(rawV string) error {
	v := "v" + rawV
	Logf("Preparing node upgarde by creating new instance template for %q", v)
	stdout, _, err := runCmd(path.Join(testContext.RepoRoot, "hack/e2e-internal/e2e-upgrade.sh"), "-P", v)
	if err != nil {
		return err
	}
	tmpl := strings.TrimSpace(stdout)

	Logf("Performing a node upgrade to %q; waiting at most %v per node", tmpl, restartPerNodeTimeout)
	if err := migRollingUpdate(tmpl, restartPerNodeTimeout); err != nil {
		return fmt.Errorf("error doing node upgrade via a migRollingUpdate to %s: %v", tmpl, err)
	}
	return nil
}

func nodeUpgradeGKE(v string) error {
	Logf("Upgrading nodes to %q", v)
	_, _, err := runCmd("gcloud", "container",
		fmt.Sprintf("--project=%s", testContext.CloudConfig.ProjectID),
		fmt.Sprintf("--zone=%s", testContext.CloudConfig.Zone),
		"clusters",
		"upgrade",
		testContext.CloudConfig.Cluster,
		fmt.Sprintf("--cluster-version=%s", v))
	return err
}

var _ = Describe("Skipped", func() {

	Describe("Cluster upgrade", func() {
		svcName, replicas := "baz", 2
		var rcName, ip, v string
		var ingress api.LoadBalancerIngress
		f := Framework{BaseName: "cluster-upgrade"}
		var w *WebserverTest

		BeforeEach(func() {
			// The version is determined once at the beginning of the test so that
			// the master and nodes won't be skewed if the value changes during the
			// test.
			By(fmt.Sprintf("Getting real version for %q", testContext.UpgradeTarget))
			var err error
			v, err = realVersion(testContext.UpgradeTarget)
			expectNoError(err)
			Logf("Version for %q is %s", testContext.UpgradeTarget, v)

			By("Setting up the service, RC, and pods")
			f.beforeEach()
			w = NewWebserverTest(f.Client, f.Namespace.Name, svcName)
			rc := w.CreateWebserverRC(replicas)
			rcName = rc.ObjectMeta.Name
			svc := w.BuildServiceSpec()
			svc.Spec.Type = api.ServiceTypeLoadBalancer
			w.CreateService(svc)

			By("Waiting for the service to become reachable")
			result, err := waitForLoadBalancerIngress(f.Client, svcName, f.Namespace.Name)
			Expect(err).NotTo(HaveOccurred())
			ingresses := result.Status.LoadBalancer.Ingress
			if len(ingresses) != 1 {
				Failf("Was expecting only 1 ingress IP but got %d (%v): %v", len(ingresses), ingresses, result)
			}
			ingress = ingresses[0]
			Logf("Got load balancer ingress point %v", ingress)
			ip = ingress.IP
			if ip == "" {
				ip = ingress.Hostname
			}
			testLoadBalancerReachable(ingress, 80)

			// TODO(mikedanese): Add setup, validate, and teardown for:
			//  - secrets
			//  - volumes
			//  - persistent volumes
		})

		AfterEach(func() {
			f.afterEach()
			w.Cleanup()
		})

		Describe("kube-push", func() {
			BeforeEach(func() {
				SkipUnlessProviderIs("gce")
			})

			It("of master should maintain responsive services", func() {
				By("Validating cluster before master upgrade")
				expectNoError(validate(f, svcName, rcName, ingress, replicas))
				By("Performing a master upgrade")
				testMasterUpgrade(ip, v, masterPush)
				By("Validating cluster after master upgrade")
				expectNoError(validate(f, svcName, rcName, ingress, replicas))
			})
		})

		Describe("upgrade-master", func() {
			BeforeEach(func() {
				SkipUnlessProviderIs("gce", "gke")
			})

			It("should maintain responsive services", func() {
				By("Validating cluster before master upgrade")
				expectNoError(validate(f, svcName, rcName, ingress, replicas))
				By("Performing a master upgrade")
				testMasterUpgrade(ip, v, masterUpgrade)
				By("Checking master version")
				expectNoError(checkMasterVersion(f.Client, v))
				By("Validating cluster after master upgrade")
				expectNoError(validate(f, svcName, rcName, ingress, replicas))
			})
		})

		Describe("upgrade-cluster", func() {
			var tmplBefore, tmplAfter string
			BeforeEach(func() {
				if providerIs("gce") {
					By("Getting the node template before the upgrade")
					var err error
					tmplBefore, err = migTemplate()
					expectNoError(err)
				}
			})

			AfterEach(func() {
				if providerIs("gce") {
					By("Cleaning up any unused node templates")
					var err error
					tmplAfter, err = migTemplate()
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
						fmt.Sprintf("--project=%s", testContext.CloudConfig.ProjectID),
						"delete",
						tmplBefore); err != nil {
						Logf("gcloud compute instance-templates delete %s call failed with err: %v", tmplBefore, err)
						Logf("May have leaked instance template %q", tmplBefore)
					}
				}
			})

			It("should maintain a functioning cluster", func() {
				SkipUnlessProviderIs("gce", "gke")

				By("Validating cluster before master upgrade")
				expectNoError(validate(f, svcName, rcName, ingress, replicas))
				By("Performing a master upgrade")
				testMasterUpgrade(ip, v, masterUpgrade)
				By("Checking master version")
				expectNoError(checkMasterVersion(f.Client, v))
				By("Validating cluster after master upgrade")
				expectNoError(validate(f, svcName, rcName, ingress, replicas))
				By("Performing a node upgrade")
				testNodeUpgrade(f, nodeUpgrade, replicas, v)
				By("Validating cluster after node upgrade")
				expectNoError(validate(f, svcName, rcName, ingress, replicas))
			})
		})
	})
})

func testMasterUpgrade(ip, v string, mUp func(v string) error) {
	Logf("Starting async validation")
	httpClient := http.Client{Timeout: 2 * time.Second}
	done := make(chan struct{}, 1)
	// Let's make sure we've finished the heartbeat before shutting things down.
	var wg sync.WaitGroup
	go util.Until(func() {
		defer GinkgoRecover()
		wg.Add(1)
		defer wg.Done()

		if err := wait.Poll(poll, singleCallTimeout, func() (bool, error) {
			r, err := httpClient.Get("http://" + ip)
			if err != nil {
				Logf("Error reaching %s: %v", ip, err)
				return false, nil
			}
			if r.StatusCode < http.StatusOK || r.StatusCode >= http.StatusNotFound {
				Logf("Bad response; status: %d, response: %v", r.StatusCode, r)
				return false, nil
			}
			return true, nil
		}); err != nil {
			// We log the error here because the test will fail at the very end
			// because this validation runs in another goroutine. Without this,
			// a failure is very confusing to track down because from the logs
			// everything looks fine.
			msg := fmt.Sprintf("Failed to contact service during master upgrade: %v", err)
			Logf(msg)
			Failf(msg)
		}
	}, 200*time.Millisecond, done)

	Logf("Starting master upgrade")
	expectNoError(mUp(v))
	done <- struct{}{}
	Logf("Stopping async validation")
	wg.Wait()
	Logf("Master upgrade complete")
}

func checkMasterVersion(c *client.Client, want string) error {
	v, err := c.ServerVersion()
	if err != nil {
		return fmt.Errorf("checkMasterVersion() couldn't get the master version: %v", err)
	}
	// We do prefix trimming and then matching because:
	// want looks like:  0.19.3-815-g50e67d4
	// got  looks like: v0.19.3-815-g50e67d4034e858-dirty
	got := strings.TrimPrefix(v.GitVersion, "v")
	if !strings.HasPrefix(got, want) {
		return fmt.Errorf("master had kube-apiserver version %s which does not start with %s",
			got, want)
	}
	Logf("Master is at version %s", want)
	return nil
}

func testNodeUpgrade(f Framework, nUp func(f Framework, n int, v string) error, replicas int, v string) {
	Logf("Starting node upgrade")
	expectNoError(nUp(f, replicas, v))
	Logf("Node upgrade complete")
	By("Checking node versions")
	expectNoError(checkNodesVersions(f.Client, v))
	Logf("All nodes are at version %s", v)
}

func checkNodesVersions(c *client.Client, want string) error {
	l, err := listNodes(c, labels.Everything(), fields.Everything())
	if err != nil {
		return fmt.Errorf("checkNodesVersions() failed to list nodes: %v", err)
	}
	for _, n := range l.Items {
		// We do prefix trimming and then matching because:
		// want   looks like:  0.19.3-815-g50e67d4
		// kv/kvp look  like: v0.19.3-815-g50e67d4034e858-dirty
		kv, kpv := strings.TrimPrefix(n.Status.NodeInfo.KubeletVersion, "v"),
			strings.TrimPrefix(n.Status.NodeInfo.KubeProxyVersion, "v")
		if !strings.HasPrefix(kv, want) {
			return fmt.Errorf("node %s had kubelet version %s which does not start with %s",
				n.ObjectMeta.Name, kv, want)
		}
		if !strings.HasPrefix(kpv, want) {
			return fmt.Errorf("node %s had kube-proxy version %s which does not start with %s",
				n.ObjectMeta.Name, kpv, want)
		}
	}
	return nil
}

// retryCmd runs cmd using args and retries it for up to singleCallTimeout if
// it returns an error. It returns stdout and stderr.
func retryCmd(command string, args ...string) (string, string, error) {
	var err error
	stdout, stderr := "", ""
	wait.Poll(poll, singleCallTimeout, func() (bool, error) {
		stdout, stderr, err = runCmd(command, args...)
		if err != nil {
			Logf("Got %v", err)
			return false, nil
		}
		return true, nil
	})
	return stdout, stderr, err
}

// runCmd runs cmd using args and returns its stdout and stderr. It also outputs
// cmd's stdout and stderr to their respective OS streams.
func runCmd(command string, args ...string) (string, string, error) {
	Logf("Running %s %v", command, args)
	var bout, berr bytes.Buffer
	cmd := exec.Command(command, args...)
	// We also output to the OS stdout/stderr to aid in debugging in case cmd
	// hangs and never retruns before the test gets killed.
	cmd.Stdout = io.MultiWriter(os.Stdout, &bout)
	cmd.Stderr = io.MultiWriter(os.Stderr, &berr)
	err := cmd.Run()
	stdout, stderr := bout.String(), berr.String()
	if err != nil {
		return "", "", fmt.Errorf("error running %s %v; got error %v, stdout %q, stderr %q",
			command, args, err, stdout, stderr)
	}
	return stdout, stderr, nil
}

func validate(f Framework, svcNameWant, rcNameWant string, ingress api.LoadBalancerIngress, podsWant int) error {
	Logf("Beginning cluster validation")
	// Verify RC.
	rcs, err := f.Client.ReplicationControllers(f.Namespace.Name).List(labels.Everything(), fields.Everything())
	if err != nil {
		return fmt.Errorf("error listing RCs: %v", err)
	}
	if len(rcs.Items) != 1 {
		return fmt.Errorf("wanted 1 RC with name %s, got %d", rcNameWant, len(rcs.Items))
	}
	if got := rcs.Items[0].Name; got != rcNameWant {
		return fmt.Errorf("wanted RC name %q, got %q", rcNameWant, got)
	}

	// Verify pods.
	if err := verifyPods(f.Client, f.Namespace.Name, rcNameWant, false, podsWant); err != nil {
		return fmt.Errorf("failed to find %d %q pods: %v", podsWant, rcNameWant, err)
	}

	// Verify service.
	svc, err := f.Client.Services(f.Namespace.Name).Get(svcNameWant)
	if err != nil {
		return fmt.Errorf("error getting service %s: %v", svcNameWant, err)
	}
	if svcNameWant != svc.Name {
		return fmt.Errorf("wanted service name %q, got %q", svcNameWant, svc.Name)
	}
	// TODO(mikedanese): Make testLoadBalancerReachable return an error.
	testLoadBalancerReachable(ingress, 80)

	Logf("Cluster validation succeeded")
	return nil
}

// migRollingUpdate starts a MIG rolling update, upgrading the nodes to a new
// instance template named tmpl, and waits up to nt times the nubmer of nodes
// for it to complete.
func migRollingUpdate(tmpl string, nt time.Duration) error {
	By(fmt.Sprintf("starting the MIG rolling update to %s", tmpl))
	id, err := migRollingUpdateStart(tmpl, nt)
	if err != nil {
		return fmt.Errorf("couldn't start the MIG rolling update: %v", err)
	}

	By(fmt.Sprintf("polling the MIG rolling update (%s) until it completes", id))
	if err := migRollingUpdatePoll(id, nt); err != nil {
		return fmt.Errorf("err waiting until update completed: %v", err)
	}

	return nil
}

// migTemlate (GCE/GKE-only) returns the name of the MIG template that the
// nodes of the cluster use.
func migTemplate() (string, error) {
	var errLast error
	var templ string
	key := "instanceTemplate"
	if wait.Poll(poll, singleCallTimeout, func() (bool, error) {
		// TODO(mikedanese): make this hit the compute API directly instead of
		// shelling out to gcloud.
		// An `instance-groups managed describe` call outputs what we want to stdout.
		output, _, err := retryCmd("gcloud", "compute", "instance-groups", "managed",
			fmt.Sprintf("--project=%s", testContext.CloudConfig.ProjectID),
			"describe",
			fmt.Sprintf("--zone=%s", testContext.CloudConfig.Zone),
			testContext.CloudConfig.NodeInstanceGroup)
		if err != nil {
			errLast = fmt.Errorf("gcloud compute instance-groups managed describe call failed with err: %v", err)
			return false, nil
		}

		// The 'describe' call probably succeeded; parse the output and try to
		// find the line that looks like "instanceTemplate: url/to/<templ>" and
		// return <templ>.
		if val := parseKVLines(output, key); len(val) > 0 {
			url := strings.Split(val, "/")
			templ = url[len(url)-1]
			Logf("MIG group %s using template: %s", testContext.CloudConfig.NodeInstanceGroup, templ)
			return true, nil
		}
		errLast = fmt.Errorf("couldn't find %s in output to get MIG template. Output: %s", key, output)
		return false, nil
	}) != nil {
		return "", fmt.Errorf("migTemplate() failed with last error: %v", errLast)
	}
	return templ, nil
}

// migRollingUpdateStart (GCE/GKE-only) starts a MIG rolling update using templ
// as the new template, waiting up to nt per node, and returns the ID of that
// update.
func migRollingUpdateStart(templ string, nt time.Duration) (string, error) {
	var errLast error
	var id string
	prefix, suffix := "Started [", "]."
	if err := wait.Poll(poll, singleCallTimeout, func() (bool, error) {
		// TODO(mikedanese): make this hit the compute API directly instead of
		//                 shelling out to gcloud.
		// NOTE(mikedanese): If you are changing this gcloud command, update
		//                 cluster/gce/upgrade.sh to match this EXACTLY.
		// A `rolling-updates start` call outputs what we want to stderr.
		_, output, err := retryCmd("gcloud", append(migUdpateCmdBase(),
			"rolling-updates",
			fmt.Sprintf("--project=%s", testContext.CloudConfig.ProjectID),
			fmt.Sprintf("--zone=%s", testContext.CloudConfig.Zone),
			"start",
			// Required args.
			fmt.Sprintf("--group=%s", testContext.CloudConfig.NodeInstanceGroup),
			fmt.Sprintf("--template=%s", templ),
			// Optional args to fine-tune behavior.
			fmt.Sprintf("--instance-startup-timeout=%ds", int(nt.Seconds())),
			// NOTE: We can speed up this process by increasing
			//       --max-num-concurrent-instances.
			fmt.Sprintf("--max-num-concurrent-instances=%d", 1),
			fmt.Sprintf("--max-num-failed-instances=%d", 0),
			fmt.Sprintf("--min-instance-update-time=%ds", 0))...)
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

// migUpdateCmdBase gets the base of the MIG rolling update command--i.e., all
// pieces of the gcloud command that come after "gcloud" but before
// "rolling-updates". Examples of returned values are:
//
//   {preview"}
//
//   {"alpha", "compute"}
//
// TODO(mikedanese): Remove this hack on July 29, 2015 when the migration to
//                 `gcloud alpha compute rolling-updates` is complete.
func migUdpateCmdBase() []string {
	b := []string{"preview"}
	a := []string{"rolling-updates", "-h"}
	if err := exec.Command("gcloud", append(b, a...)...).Run(); err != nil {
		b = []string{"alpha", "compute"}
	}
	return b
}

// migRollingUpdatePoll (CKE/GKE-only) polls the progress of the MIG rolling
// update with ID id until it is complete. It returns an error if this takes
// longer than nt times the number of nodes.
func migRollingUpdatePoll(id string, nt time.Duration) error {
	// Two keys and a val.
	status, progress, done := "status", "statusMessage", "ROLLED_OUT"
	start, timeout := time.Now(), nt*time.Duration(testContext.CloudConfig.NumNodes)
	var errLast error
	Logf("Waiting up to %v for MIG rolling update to complete.", timeout)
	if wait.Poll(restartPoll, timeout, func() (bool, error) {
		// A `rolling-updates describe` call outputs what we want to stdout.
		output, _, err := retryCmd("gcloud", append(migUdpateCmdBase(),
			"rolling-updates",
			fmt.Sprintf("--project=%s", testContext.CloudConfig.ProjectID),
			fmt.Sprintf("--zone=%s", testContext.CloudConfig.Zone),
			"describe",
			id)...)
		if err != nil {
			errLast = fmt.Errorf("Error calling rolling-updates describe %s: %v", id, err)
			Logf("%v", errLast)
			return false, nil
		}

		// The 'describe' call probably succeeded; parse the output and try to
		// find the line that looks like "status: <status>" and see whether it's
		// done.
		Logf("Waiting for MIG rolling update: %s (%v elapsed)",
			parseKVLines(output, progress), time.Since(start))
		if st := parseKVLines(output, status); st == done {
			return true, nil
		}
		return false, nil
	}) != nil {
		return fmt.Errorf("timeout waiting %v for MIG rolling update to complete. Last error: %v", timeout, errLast)
	}
	Logf("MIG rolling update complete after %v", time.Since(start))
	return nil
}
