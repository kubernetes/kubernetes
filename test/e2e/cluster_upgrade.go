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
	"net/http"
	"os"
	"os/exec"
	"path"
	"strings"
	"time"

	"k8s.io/kubernetes/pkg/api"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/test/e2e/chaosmonkey"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

type serviceJig struct {
	f        *Framework
	svcName  string
	rcName   string
	ingress  *api.LoadBalancerIngress
	replicas int
	ip       string
	w        *ServiceTestFixture
}

// TODO use ServiceTestJig in service.go
func newServiceJig() *serviceJig {
	f := NewDefaultFramework("cluster-upgrade")
	svcName := "baz"
	return &serviceJig{
		f,
		svcName,
		"",
		nil,
		2,
		"",
		// this composition is weird/bad
		NewServerTest(f.Client, f.Namespace.Name, svcName),
	}
}

func (j *serviceJig) Setup() error {
	By("Setting up the service, RC, and pods")
	// again, weird composition
	j.rcName = j.w.CreateWebserverRC(j.replicas).ObjectMeta.Name
	svc := j.w.BuildServiceSpec()
	svc.Spec.Type = api.ServiceTypeLoadBalancer
	j.w.CreateService(svc)

	By("Waiting for the service to become reachable")
	result, err := waitForLoadBalancerIngress(j.f.Client, j.svcName, j.f.Namespace.Name)
	if err != nil {
		return err
	}
	ingresses := result.Status.LoadBalancer.Ingress
	if len(ingresses) != 1 {
		return fmt.Errorf("Was expecting only 1 ingress IP but got %d (%v): %v", len(ingresses), ingresses, result)
	}
	j.ingress = &ingresses[0]
	Logf("Got load balancer ingress point %v", j.ingress)
	j.ip = j.ingress.IP
	if j.ip == "" {
		j.ip = j.ingress.Hostname
	}
	testLoadBalancerReachable(j.ingress, 80)

	By("Validating cluster")
	return validate(j.f, j.svcName, j.rcName, j.ingress, j.replicas)
}

func (j *serviceJig) Test(done <-chan struct{}) error {
	Logf("Starting async validation")
	// this seems like something that ServiceTestFixture should do
	httpClient := http.Client{Timeout: 2 * time.Second}
	wait.Until(func() {
		if err := wait.Poll(poll, singleCallTimeout, func() (bool, error) {
			r, err := httpClient.Get("http://" + j.ip)
			if err != nil {
				Logf("Error reaching %s: %v", j.ip, err)
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
			msg := fmt.Sprintf("Failed to contact service during upgrade: %v", err)
			Logf(msg)
			Failf(msg)
		}
	}, 200*time.Millisecond, done)
	Logf("Async validation complete")
	return nil
}

func (j *serviceJig) Teardown() error {
	By("Validating cluster before teardown")
	err := validate(j.f, j.svcName, j.rcName, j.ingress, j.replicas)
	// Clean up before returning err
	j.w.Cleanup()
	return err
}

var _ = KubeDescribe("Upgrade [Feature:Upgrade]", func() {
	cm := chaosmonkey.New(func() error {
		// The version is determined once at the beginning of the test so that
		// the master and nodes won't be skewed if the value changes during the
		// test.
		By(fmt.Sprintf("Getting real version for %q", testContext.UpgradeTarget))
		v, err := realVersion(testContext.UpgradeTarget)
		if err != nil {
			return err
		}
		Logf("Version for %q is %q", testContext.UpgradeTarget, v)

		By("Performing a master upgrade")
		err = masterUpgrade(v)
		if err != nil {
			return err
		}
		By("Checking master version")
		f := NewDefaultFramework("master-version-check")
		return checkMasterVersion(f.Client, v)
	})
	// TODO(mikedanese): Add setup, validate, and teardown for:
	//  - secrets
	//  - volumes
	//  - persistent volumes
	cm.Register("maintains service reachability", newServiceJig())
	cm.Do()
})

// realVersion turns a version constant s into a version string deployable on
// GKE.  See hack/get-build.sh for more information.
func realVersion(s string) (string, error) {
	v, _, err := runCmd(path.Join(testContext.RepoRoot, "hack/get-build.sh"), "-v", s)
	if err != nil {
		return v, err
	}
	return strings.TrimPrefix(strings.TrimSpace(v), "v"), nil
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
	_, _, err := runCmd(path.Join(testContext.RepoRoot, "cluster/gce/upgrade.sh"), "-M", v)
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
		fmt.Sprintf("--cluster-version=%s", v),
		"--quiet")
	return err
}

var nodeUpgrade = func(f *Framework, replicas int, v string) error {
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
	//
	// TODO(ihmccreery) We shouldn't have to wait for nodes to be ready in
	// GKE; the operation shouldn't return until they all are.
	Logf("Waiting up to %v for all nodes to be ready after the upgrade", restartNodeReadyAgainTimeout)
	if _, err := checkNodesReady(f.Client, restartNodeReadyAgainTimeout, testContext.CloudConfig.NumNodes); err != nil {
		return err
	}
	Logf("Waiting up to %v for all pods to be running and ready after the upgrade", restartPodReadyAgainTimeout)
	return waitForPodsRunningReady(f.Namespace.Name, replicas, restartPodReadyAgainTimeout)
}

func nodeUpgradeGCE(rawV string) error {
	// TODO(ihmccreery) This code path should be identical to how a user
	// would trigger a node update; right now it's very different.
	v := "v" + rawV

	Logf("Getting the node template before the upgrade")
	tmplBefore, err := migTemplate()
	if err != nil {
		return fmt.Errorf("error getting the node template before the upgrade: %v", err)
	}

	Logf("Preparing node upgrade by creating new instance template for %q", v)
	stdout, _, err := runCmd(path.Join(testContext.RepoRoot, "cluster/gce/upgrade.sh"), "-P", v)
	if err != nil {
		cleanupNodeUpgradeGCE(tmplBefore)
		return fmt.Errorf("error preparing node upgrade: %v", err)
	}
	tmpl := strings.TrimSpace(stdout)

	Logf("Performing a node upgrade to %q; waiting at most %v per node", tmpl, restartPerNodeTimeout)
	if err := migRollingUpdate(tmpl, restartPerNodeTimeout); err != nil {
		cleanupNodeUpgradeGCE(tmplBefore)
		return fmt.Errorf("error doing node upgrade via a migRollingUpdate to %s: %v", tmpl, err)
	}
	return nil
}

func cleanupNodeUpgradeGCE(tmplBefore string) {
	Logf("Cleaning up any unused node templates")
	tmplAfter, err := migTemplate()
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

func nodeUpgradeGKE(v string) error {
	Logf("Upgrading nodes to %q", v)
	_, _, err := runCmd("gcloud", "container",
		fmt.Sprintf("--project=%s", testContext.CloudConfig.ProjectID),
		fmt.Sprintf("--zone=%s", testContext.CloudConfig.Zone),
		"clusters",
		"upgrade",
		testContext.CloudConfig.Cluster,
		fmt.Sprintf("--cluster-version=%s", v),
		"--quiet")
	return err
}

func checkMasterVersion(c *client.Client, want string) error {
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
	Logf("Master is at version %s", want)
	return nil
}

func checkNodesVersions(c *client.Client, want string) error {
	l := ListSchedulableNodesOrDie(c)
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
//
// TODO(ihmccreery) This function should either be moved into util.go or
// removed; other e2e's use bare exe.Command.
func runCmd(command string, args ...string) (string, string, error) {
	Logf("Running %s %v", command, args)
	var bout, berr bytes.Buffer
	cmd := exec.Command(command, args...)
	// We also output to the OS stdout/stderr to aid in debugging in case cmd
	// hangs and never returns before the test gets killed.
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

func validate(f *Framework, svcNameWant, rcNameWant string, ingress *api.LoadBalancerIngress, podsWant int) error {
	Logf("Beginning cluster validation")
	// Verify RC.
	rcs, err := f.Client.ReplicationControllers(f.Namespace.Name).List(api.ListOptions{})
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
// instance template named tmpl, and waits up to nt times the number of nodes
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

// migTemplate (GCE-only) returns the name of the MIG template that the
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
		_, output, err := retryCmd("gcloud", "alpha", "compute",
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
	start, timeout := time.Now(), nt*time.Duration(testContext.CloudConfig.NumNodes)
	var errLast error
	Logf("Waiting up to %v for MIG rolling update to complete.", timeout)
	if wait.Poll(restartPoll, timeout, func() (bool, error) {
		// A `rolling-updates describe` call outputs what we want to stdout.
		output, _, err := retryCmd("gcloud", "alpha", "compute",
			"rolling-updates",
			fmt.Sprintf("--project=%s", testContext.CloudConfig.ProjectID),
			fmt.Sprintf("--zone=%s", testContext.CloudConfig.Zone),
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

func testLoadBalancerReachable(ingress *api.LoadBalancerIngress, port int) bool {
	loadBalancerLagTimeout := loadBalancerLagTimeoutDefault
	if providerIs("aws") {
		loadBalancerLagTimeout = loadBalancerLagTimeoutAWS
	}
	return testLoadBalancerReachableInTime(ingress, port, loadBalancerLagTimeout)
}

func testLoadBalancerReachableInTime(ingress *api.LoadBalancerIngress, port int, timeout time.Duration) bool {
	ip := ingress.IP
	if ip == "" {
		ip = ingress.Hostname
	}

	return testReachableInTime(conditionFuncDecorator(ip, port, testReachableHTTP, "/", "test-webserver"), timeout)

}

func conditionFuncDecorator(ip string, port int, fn func(string, int, string, string) (bool, error), request string, expect string) wait.ConditionFunc {
	return func() (bool, error) {
		return fn(ip, port, request, expect)
	}
}

func testReachableInTime(testFunc wait.ConditionFunc, timeout time.Duration) bool {
	By(fmt.Sprintf("Waiting up to %v", timeout))
	err := wait.PollImmediate(poll, timeout, testFunc)
	if err != nil {
		Expect(err).NotTo(HaveOccurred(), "Error waiting")
		return false
	}
	return true
}

func waitForLoadBalancerIngress(c *client.Client, serviceName, namespace string) (*api.Service, error) {
	// TODO: once support ticket 21807001 is resolved, reduce this timeout
	// back to something reasonable
	const timeout = 20 * time.Minute
	var service *api.Service
	By(fmt.Sprintf("waiting up to %v for service %s in namespace %s to have a LoadBalancer ingress point", timeout, serviceName, namespace))
	i := 1
	for start := time.Now(); time.Since(start) < timeout; time.Sleep(3 * time.Second) {
		service, err := c.Services(namespace).Get(serviceName)
		if err != nil {
			Logf("Get service failed, ignoring for 5s: %v", err)
			continue
		}
		if len(service.Status.LoadBalancer.Ingress) > 0 {
			return service, nil
		}
		if i%5 == 0 {
			Logf("Waiting for service %s in namespace %s to have a LoadBalancer ingress point (%v)", serviceName, namespace, time.Since(start))
		}
		i++
	}
	return service, fmt.Errorf("service %s in namespace %s doesn't have a LoadBalancer ingress point after %.2f seconds", serviceName, namespace, timeout.Seconds())
}
