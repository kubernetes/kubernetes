/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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
	"os"
	"os/exec"
	"path"
	"strings"
	"time"

	"k8s.io/kubernetes/pkg/api"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/test/e2e/chaosmonkey"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
)

// TODO(mikedanese): Add setup, validate, and teardown for:
//  - secrets
//  - volumes
//  - persistent volumes
var _ = framework.KubeDescribe("Upgrade [Feature:Upgrade]", func() {
	f := framework.NewDefaultFramework("cluster-upgrade")

	framework.KubeDescribe("master upgrade", func() {
		It("should maintain responsive services [Feature:MasterUpgrade]", func() {
			cm := chaosmonkey.New(func() {
				v, err := realVersion(framework.TestContext.UpgradeTarget)
				framework.ExpectNoError(err)
				framework.ExpectNoError(masterUpgrade(v))
				framework.ExpectNoError(checkMasterVersion(f.Client, v))
			})
			cm.Register(func(sem *chaosmonkey.Semaphore) {
				// Close over f.
				testServiceRemainsUp(f, sem)
			})
			cm.Do()
		})
	})

	framework.KubeDescribe("node upgrade", func() {
		It("should maintain a functioning cluster [Feature:NodeUpgrade]", func() {
			cm := chaosmonkey.New(func() {
				v, err := realVersion(framework.TestContext.UpgradeTarget)
				framework.ExpectNoError(err)
				framework.ExpectNoError(nodeUpgrade(f, v))
				framework.ExpectNoError(checkNodesVersions(f.Client, v))
			})
			cm.Register(func(sem *chaosmonkey.Semaphore) {
				// Close over f.
				testServiceUpBeforeAndAfter(f, sem)
			})
			cm.Do()
		})

		It("should maintain responsive services [Feature:ExperimentalNodeUpgrade]", func() {
			cm := chaosmonkey.New(func() {
				v, err := realVersion(framework.TestContext.UpgradeTarget)
				framework.ExpectNoError(err)
				framework.ExpectNoError(nodeUpgrade(f, v))
				framework.ExpectNoError(checkNodesVersions(f.Client, v))
			})
			cm.Register(func(sem *chaosmonkey.Semaphore) {
				// Close over f.
				testServiceRemainsUp(f, sem)
			})
			cm.Do()
		})
	})

	framework.KubeDescribe("cluster upgrade", func() {
		It("should maintain a functioning cluster [Feature:ClusterUpgrade]", func() {
			cm := chaosmonkey.New(func() {
				v, err := realVersion(framework.TestContext.UpgradeTarget)
				framework.ExpectNoError(err)
				framework.ExpectNoError(masterUpgrade(v))
				framework.ExpectNoError(checkMasterVersion(f.Client, v))
				framework.ExpectNoError(nodeUpgrade(f, v))
				framework.ExpectNoError(checkNodesVersions(f.Client, v))
			})
			cm.Register(func(sem *chaosmonkey.Semaphore) {
				// Close over f.
				testServiceUpBeforeAndAfter(f, sem)
			})
			cm.Do()
		})

		It("should maintain responsive services [Feature:ExperimentalClusterUpgrade]", func() {
			cm := chaosmonkey.New(func() {
				v, err := realVersion(framework.TestContext.UpgradeTarget)
				framework.ExpectNoError(err)
				framework.ExpectNoError(masterUpgrade(v))
				framework.ExpectNoError(checkMasterVersion(f.Client, v))
				framework.ExpectNoError(nodeUpgrade(f, v))
				framework.ExpectNoError(checkNodesVersions(f.Client, v))
			})
			cm.Register(func(sem *chaosmonkey.Semaphore) {
				// Close over f.
				testServiceRemainsUp(f, sem)
			})
			cm.Do()
		})
	})
})

// realVersion turns a version constant s into a version string deployable on
// GKE.  See hack/get-build.sh for more information.
func realVersion(s string) (string, error) {
	framework.Logf(fmt.Sprintf("Getting real version for %q", s))
	v, _, err := runCmd(path.Join(framework.TestContext.RepoRoot, "hack/get-build.sh"), "-v", s)
	if err != nil {
		return v, err
	}
	framework.Logf("Version for %q is %q", s, v)
	return strings.TrimPrefix(strings.TrimSpace(v), "v"), nil
}

// The following upgrade functions are passed into the framework below and used
// to do the actual upgrades.
var masterUpgrade = func(v string) error {
	switch framework.TestContext.Provider {
	case "gce":
		return masterUpgradeGCE(v)
	case "gke":
		return masterUpgradeGKE(v)
	default:
		return fmt.Errorf("masterUpgrade() is not implemented for provider %s", framework.TestContext.Provider)
	}
}

func masterUpgradeGCE(rawV string) error {
	v := "v" + rawV
	_, _, err := runCmd(path.Join(framework.TestContext.RepoRoot, "cluster/gce/upgrade.sh"), "-M", v)
	return err
}

func masterUpgradeGKE(v string) error {
	framework.Logf("Upgrading master to %q", v)
	_, _, err := runCmd("gcloud", "container",
		fmt.Sprintf("--project=%s", framework.TestContext.CloudConfig.ProjectID),
		fmt.Sprintf("--zone=%s", framework.TestContext.CloudConfig.Zone),
		"clusters",
		"upgrade",
		framework.TestContext.CloudConfig.Cluster,
		"--master",
		fmt.Sprintf("--cluster-version=%s", v),
		"--quiet")
	return err
}

var nodeUpgrade = func(f *framework.Framework, v string) error {
	// Perform the upgrade.
	var err error
	switch framework.TestContext.Provider {
	case "gce":
		err = nodeUpgradeGCE(v)
	case "gke":
		err = nodeUpgradeGKE(v)
	default:
		err = fmt.Errorf("nodeUpgrade() is not implemented for provider %s", framework.TestContext.Provider)
	}
	if err != nil {
		return err
	}

	// Wait for it to complete and validate nodes are healthy.
	//
	// TODO(ihmccreery) We shouldn't have to wait for nodes to be ready in
	// GKE; the operation shouldn't return until they all are.
	framework.Logf("Waiting up to %v for all nodes to be ready after the upgrade", restartNodeReadyAgainTimeout)
	if _, err := checkNodesReady(f.Client, restartNodeReadyAgainTimeout, framework.TestContext.CloudConfig.NumNodes); err != nil {
		return err
	}
	return nil
}

func nodeUpgradeGCE(rawV string) error {
	// TODO(ihmccreery) This code path should be identical to how a user
	// would trigger a node update; right now it's very different.
	v := "v" + rawV

	framework.Logf("Getting the node template before the upgrade")
	tmplBefore, err := migTemplate()
	if err != nil {
		return fmt.Errorf("error getting the node template before the upgrade: %v", err)
	}

	framework.Logf("Preparing node upgrade by creating new instance template for %q", v)
	stdout, _, err := runCmd(path.Join(framework.TestContext.RepoRoot, "cluster/gce/upgrade.sh"), "-P", v)
	if err != nil {
		cleanupNodeUpgradeGCE(tmplBefore)
		return fmt.Errorf("error preparing node upgrade: %v", err)
	}
	tmpl := strings.TrimSpace(stdout)

	framework.Logf("Performing a node upgrade to %q; waiting at most %v per node", tmpl, restartPerNodeTimeout)
	if err := migRollingUpdate(tmpl, restartPerNodeTimeout); err != nil {
		cleanupNodeUpgradeGCE(tmplBefore)
		return fmt.Errorf("error doing node upgrade via a migRollingUpdate to %s: %v", tmpl, err)
	}
	return nil
}

func cleanupNodeUpgradeGCE(tmplBefore string) {
	framework.Logf("Cleaning up any unused node templates")
	tmplAfter, err := migTemplate()
	if err != nil {
		framework.Logf("Could not get node template post-upgrade; may have leaked template %s", tmplBefore)
		return
	}
	if tmplBefore == tmplAfter {
		// The node upgrade failed so there's no need to delete
		// anything.
		framework.Logf("Node template %s is still in use; not cleaning up", tmplBefore)
		return
	}
	framework.Logf("Deleting node template %s", tmplBefore)
	if _, _, err := retryCmd("gcloud", "compute", "instance-templates",
		fmt.Sprintf("--project=%s", framework.TestContext.CloudConfig.ProjectID),
		"delete",
		tmplBefore); err != nil {
		framework.Logf("gcloud compute instance-templates delete %s call failed with err: %v", tmplBefore, err)
		framework.Logf("May have leaked instance template %q", tmplBefore)
	}
}

func nodeUpgradeGKE(v string) error {
	framework.Logf("Upgrading nodes to %q", v)
	_, _, err := runCmd("gcloud", "container",
		fmt.Sprintf("--project=%s", framework.TestContext.CloudConfig.ProjectID),
		fmt.Sprintf("--zone=%s", framework.TestContext.CloudConfig.Zone),
		"clusters",
		"upgrade",
		framework.TestContext.CloudConfig.Cluster,
		fmt.Sprintf("--cluster-version=%s", v),
		"--quiet")
	return err
}

func testServiceUpBeforeAndAfter(f *framework.Framework, sem *chaosmonkey.Semaphore) {
	testService(f, sem, false)
}

func testServiceRemainsUp(f *framework.Framework, sem *chaosmonkey.Semaphore) {
	testService(f, sem, true)
}

// testService is a helper for testServiceUpBeforeAndAfter and testServiceRemainsUp with a flag for testDuringDisruption
//
// TODO(ihmccreery) remove this abstraction once testServiceUpBeforeAndAfter is no longer needed, because node upgrades
// maintain a responsive service.
func testService(f *framework.Framework, sem *chaosmonkey.Semaphore, testDuringDisruption bool) {
	// Setup
	serviceName := "service-test"

	jig := NewServiceTestJig(f.Client, serviceName)
	// nodeIP := pickNodeIP(jig.Client) // for later

	By("creating a TCP service " + serviceName + " with type=LoadBalancer in namespace " + f.Namespace.Name)
	// TODO it's weird that we have to do this and then wait WaitForLoadBalancer which changes
	// tcpService.
	tcpService := jig.CreateTCPServiceOrFail(f.Namespace.Name, func(s *api.Service) {
		s.Spec.Type = api.ServiceTypeLoadBalancer
	})
	tcpService = jig.WaitForLoadBalancerOrFail(f.Namespace.Name, tcpService.Name)
	jig.SanityCheckService(tcpService, api.ServiceTypeLoadBalancer)

	// Get info to hit it with
	tcpIngressIP := getIngressPoint(&tcpService.Status.LoadBalancer.Ingress[0])
	svcPort := int(tcpService.Spec.Ports[0].Port)

	By("creating pod to be part of service " + serviceName)
	// TODO newRCTemplate only allows for the creation of one replica... that probably won't
	// work so well.
	jig.RunOrFail(f.Namespace.Name, nil)

	// Hit it once before considering ourselves ready
	By("hitting the pod through the service's LoadBalancer")
	jig.TestReachableHTTP(tcpIngressIP, svcPort, loadBalancerLagTimeoutDefault)

	sem.Ready()

	if testDuringDisruption {
		// Continuous validation
		wait.Until(func() {
			By("hitting the pod through the service's LoadBalancer")
			jig.TestReachableHTTP(tcpIngressIP, svcPort, framework.Poll)
		}, framework.Poll, sem.StopCh)
	} else {
		// Block until chaosmonkey is done
		By("waiting for upgrade to finish without checking if service remains up")
		<-sem.StopCh
	}

	// Sanity check and hit it once more
	By("hitting the pod through the service's LoadBalancer")
	jig.TestReachableHTTP(tcpIngressIP, svcPort, loadBalancerLagTimeoutDefault)
	jig.SanityCheckService(tcpService, api.ServiceTypeLoadBalancer)
}

func checkMasterVersion(c *client.Client, want string) error {
	framework.Logf("Checking master version")
	v, err := c.Discovery().ServerVersion()
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
	framework.Logf("Master is at version %s", want)
	return nil
}

func checkNodesVersions(c *client.Client, want string) error {
	l := framework.ListSchedulableNodesOrDie(c)
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

// retryCmd runs cmd using args and retries it for up to framework.SingleCallTimeout if
// it returns an error. It returns stdout and stderr.
func retryCmd(command string, args ...string) (string, string, error) {
	var err error
	stdout, stderr := "", ""
	wait.Poll(framework.Poll, framework.SingleCallTimeout, func() (bool, error) {
		stdout, stderr, err = runCmd(command, args...)
		if err != nil {
			framework.Logf("Got %v", err)
			return false, nil
		}
		return true, nil
	})
	return stdout, stderr, err
}

// runCmd runs cmd using args and returns its stdout and stderr. It also outputs
// cmd's stdout and stderr to their respective OS streams.
//
// TODO(ihmccreery) This function should either be moved into util.go or
// removed; other e2e's use bare exe.Command.
func runCmd(command string, args ...string) (string, string, error) {
	framework.Logf("Running %s %v", command, args)
	var bout, berr bytes.Buffer
	cmd := exec.Command(command, args...)
	// We also output to the OS stdout/stderr to aid in debugging in case cmd
	// hangs and never returns before the test gets killed.
	//
	// This creates some ugly output because gcloud doesn't always provide
	// newlines.
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

// migRollingUpdate starts a MIG rolling update, upgrading the nodes to a new
// instance template named tmpl, and waits up to nt times the number of nodes
// for it to complete.
func migRollingUpdate(tmpl string, nt time.Duration) error {
	framework.Logf(fmt.Sprintf("starting the MIG rolling update to %s", tmpl))
	id, err := migRollingUpdateStart(tmpl, nt)
	if err != nil {
		return fmt.Errorf("couldn't start the MIG rolling update: %v", err)
	}

	framework.Logf(fmt.Sprintf("polling the MIG rolling update (%s) until it completes", id))
	if err := migRollingUpdatePoll(id, nt); err != nil {
		return fmt.Errorf("err waiting until update completed: %v", err)
	}

	return nil
}

// migTemplate (GCE-only) returns the name of the MIG template that the
// nodes of the cluster use.
func migTemplate() (string, error) {
	var errLast error
	var templ string
	key := "instanceTemplate"
	if wait.Poll(framework.Poll, framework.SingleCallTimeout, func() (bool, error) {
		// TODO(mikedanese): make this hit the compute API directly instead of
		// shelling out to gcloud.
		// An `instance-groups managed describe` call outputs what we want to stdout.
		output, _, err := retryCmd("gcloud", "compute", "instance-groups", "managed",
			fmt.Sprintf("--project=%s", framework.TestContext.CloudConfig.ProjectID),
			"describe",
			fmt.Sprintf("--zone=%s", framework.TestContext.CloudConfig.Zone),
			framework.TestContext.CloudConfig.NodeInstanceGroup)
		if err != nil {
			errLast = fmt.Errorf("gcloud compute instance-groups managed describe call failed with err: %v", err)
			return false, nil
		}

		// The 'describe' call probably succeeded; parse the output and try to
		// find the line that looks like "instanceTemplate: url/to/<templ>" and
		// return <templ>.
		if val := framework.ParseKVLines(output, key); len(val) > 0 {
			url := strings.Split(val, "/")
			templ = url[len(url)-1]
			framework.Logf("MIG group %s using template: %s", framework.TestContext.CloudConfig.NodeInstanceGroup, templ)
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
	if err := wait.Poll(framework.Poll, framework.SingleCallTimeout, func() (bool, error) {
		// TODO(mikedanese): make this hit the compute API directly instead of
		//                 shelling out to gcloud.
		// NOTE(mikedanese): If you are changing this gcloud command, update
		//                 cluster/gce/upgrade.sh to match this EXACTLY.
		// A `rolling-updates start` call outputs what we want to stderr.
		_, output, err := retryCmd("gcloud", "alpha", "compute",
			"rolling-updates",
			fmt.Sprintf("--project=%s", framework.TestContext.CloudConfig.ProjectID),
			fmt.Sprintf("--zone=%s", framework.TestContext.CloudConfig.Zone),
			"start",
			// Required args.
			fmt.Sprintf("--group=%s", framework.TestContext.CloudConfig.NodeInstanceGroup),
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
			framework.Logf("Started MIG rolling update; ID: %s", id)
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
	start, timeout := time.Now(), nt*time.Duration(framework.TestContext.CloudConfig.NumNodes)
	var errLast error
	framework.Logf("Waiting up to %v for MIG rolling update to complete.", timeout)
	if wait.Poll(restartPoll, timeout, func() (bool, error) {
		// A `rolling-updates describe` call outputs what we want to stdout.
		output, _, err := retryCmd("gcloud", "alpha", "compute",
			"rolling-updates",
			fmt.Sprintf("--project=%s", framework.TestContext.CloudConfig.ProjectID),
			fmt.Sprintf("--zone=%s", framework.TestContext.CloudConfig.Zone),
			"describe",
			id)
		if err != nil {
			errLast = fmt.Errorf("Error calling rolling-updates describe %s: %v", id, err)
			framework.Logf("%v", errLast)
			return false, nil
		}

		// The 'describe' call probably succeeded; parse the output and try to
		// find the line that looks like "status: <status>" and see whether it's
		// done.
		framework.Logf("Waiting for MIG rolling update: %s (%v elapsed)",
			framework.ParseKVLines(output, progress), time.Since(start))
		if st := framework.ParseKVLines(output, status); st == done {
			return true, nil
		}
		return false, nil
	}) != nil {
		return fmt.Errorf("timeout waiting %v for MIG rolling update to complete. Last error: %v", timeout, errLast)
	}
	framework.Logf("MIG rolling update complete after %v", time.Since(start))
	return nil
}
