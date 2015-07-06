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

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/wait"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

const (
	// version applies to upgrades; kube-push always pushes local binaries.
	version       = "latest_ci"
	versionURLFmt = "https://storage.googleapis.com/kubernetes-release/%s/%s.txt"
)

// realVersion turns a version constant--one accepted by cluster/gce/upgrade.sh--
// into a deployable version string.
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
		return "", fmt.Errorf("version %s is not supported", s)
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
	_, _, err := runCmd("gcloud", "beta", "container",
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
	// TODO(mbforbes): Make master push use the provided version.
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
	_, _, err := runCmd("gcloud", "beta", "container",
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
			By(fmt.Sprintf("Getting real version for %q", version))
			var err error
			v, err = realVersion(version)
			expectNoError(err)
			Logf("Version for %q is %s", version, v)

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

			// TODO(mbforbes): Add setup, validate, and teardown for:
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
					// TODO(mbforbes): Distinguish between transient failures
					// and "cannot delete--in use" errors and retry on the
					// former.
					// TODO(mbforbes): Call this with retryCmd().
					Logf("Deleting node template %s", tmplBefore)
					o, err := exec.Command("gcloud", "compute", "instance-templates",
						fmt.Sprintf("--project=%s", testContext.CloudConfig.ProjectID),
						"delete",
						tmplBefore).CombinedOutput()
					if err != nil {
						Logf("gcloud compute instance-templates delete %s call failed with err: %v, output: %s",
							tmplBefore, err, string(o))
						Logf("May have leaked %s", tmplBefore)
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
	rcs, err := f.Client.ReplicationControllers(f.Namespace.Name).List(labels.Everything())
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
	// TODO(mbforbes): Make testLoadBalancerReachable return an error.
	testLoadBalancerReachable(ingress, 80)

	Logf("Cluster validation succeeded")
	return nil
}
