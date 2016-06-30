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
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	compute "google.golang.org/api/compute/v1"
	"k8s.io/kubernetes/pkg/api"
	apierrs "k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/apis/extensions"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/runtime"
	utilexec "k8s.io/kubernetes/pkg/util/exec"
	"k8s.io/kubernetes/pkg/util/sets"
	"k8s.io/kubernetes/pkg/util/wait"
	utilyaml "k8s.io/kubernetes/pkg/util/yaml"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

const (
	// parent path to yaml test manifests.
	ingressManifestPath = "test/e2e/testing-manifests/ingress"

	// timeout on a single http request.
	reqTimeout = 10 * time.Second

	// healthz port used to verify glbc restarted correctly on the master.
	glbcHealthzPort = 8086

	// On average it takes ~6 minutes for a single backend to come online in GCE.
	lbPollTimeout = 15 * time.Minute

	// General cloud resource poll timeout (eg: create static ip, firewall etc)
	cloudResourcePollTimeout = 5 * time.Minute

	// Time required by the loadbalancer to cleanup, proportional to numApps/Ing.
	lbCleanupTimeout = 5 * time.Minute
	lbPollInterval   = 30 * time.Second

	// Name of the config-map and key the ingress controller stores its uid in.
	uidConfigMap = "ingress-uid"
	uidKey       = "uid"

	// GCE only allows names < 64 characters, and the loadbalancer controller inserts
	// a single character of padding.
	nameLenLimit = 62
)

var _ = framework.KubeDescribe("Loadbalancing: L7 [Feature:Ingress]", func() {
	defer GinkgoRecover()
	f := framework.NewDefaultFramework("ingress")
	var ns string
	var jig *testJig
	manifestPath := filepath.Join(ingressManifestPath, "http")

	// These constants match the manifests used in ingressManifestPath
	tlsHost := "foo.bar.com"
	tlsSecretName := "foo"
	updatedTLSHost := "foobar.com"
	updateURLMapHost := "bar.baz.com"
	updateURLMapPath := "/testurl"

	BeforeEach(func() {
		f.BeforeEach()
		jig = newTestJig(f.Client)
		ns = f.Namespace.Name
	})

	// Platform agnostic list of tests that must be satisfied by all controllers
	conformanceTests := []struct {
		entryLog string
		execute  func()
		exitLog  string
	}{
		{
			fmt.Sprintf("should create a basic HTTP ingress"),
			func() { jig.createIngress(manifestPath, ns, map[string]string{}) },
			fmt.Sprintf("waiting for urls on basic HTTP ingress"),
		},
		{
			fmt.Sprintf("should terminate TLS for host %v", tlsHost),
			func() { jig.addHTTPS(tlsSecretName, tlsHost) },
			fmt.Sprintf("waiting for HTTPS updates to reflect in ingress"),
		},
		{
			fmt.Sprintf("should update SSL certificated with modified hostname %v", updatedTLSHost),
			func() {
				jig.update(func(ing *extensions.Ingress) {
					newRules := []extensions.IngressRule{}
					for _, rule := range ing.Spec.Rules {
						if rule.Host != tlsHost {
							newRules = append(newRules, rule)
							continue
						}
						newRules = append(newRules, extensions.IngressRule{
							Host:             updatedTLSHost,
							IngressRuleValue: rule.IngressRuleValue,
						})
					}
					ing.Spec.Rules = newRules
				})
				jig.addHTTPS(tlsSecretName, updatedTLSHost)
			},
			fmt.Sprintf("Waiting for updated certificates to accept requests for host %v", updatedTLSHost),
		},
		{
			fmt.Sprintf("should update url map for host %v to expose a single url: %v", updateURLMapHost, updateURLMapPath),
			func() {
				var pathToFail string
				jig.update(func(ing *extensions.Ingress) {
					newRules := []extensions.IngressRule{}
					for _, rule := range ing.Spec.Rules {
						if rule.Host != updateURLMapHost {
							newRules = append(newRules, rule)
							continue
						}
						existingPath := rule.IngressRuleValue.HTTP.Paths[0]
						pathToFail = existingPath.Path
						newRules = append(newRules, extensions.IngressRule{
							Host: updateURLMapHost,
							IngressRuleValue: extensions.IngressRuleValue{
								HTTP: &extensions.HTTPIngressRuleValue{
									Paths: []extensions.HTTPIngressPath{
										{
											Path:    updateURLMapPath,
											Backend: existingPath.Backend,
										},
									},
								},
							},
						})
					}
					ing.Spec.Rules = newRules
				})
				By("Checking that " + pathToFail + " is not exposed by polling for failure")
				route := fmt.Sprintf("http://%v%v", jig.address, pathToFail)
				ExpectNoError(jig.pollURL(route, updateURLMapHost, lbCleanupTimeout, &http.Client{Timeout: reqTimeout}, true))
			},
			fmt.Sprintf("Waiting for path updates to reflect in L7"),
		},
	}

	// Before enabling this loadbalancer test in any other test list you must
	// make sure the associated project has enough quota. At the time of this
	// writing a GCE project is allowed 3 backend services by default. This
	// test requires at least 5.
	//
	// Slow by design ~10m for each "It" block dominated by loadbalancer setup time
	// TODO: write similar tests for nginx, haproxy and AWS Ingress.
	framework.KubeDescribe("GCE [Feature: Ingress]", func() {
		var gceController *GCEIngressController

		// Platform specific setup
		BeforeEach(func() {
			framework.SkipUnlessProviderIs("gce", "gke")
			By("Initializing gce controller")
			gceController = &GCEIngressController{ns: ns, Project: framework.TestContext.CloudConfig.ProjectID, c: jig.client}
			gceController.init()
		})

		// Platform specific cleanup
		AfterEach(func() {
			if CurrentGinkgoTestDescription().Failed {
				describeIng(ns)
			}
			if jig.ing == nil {
				By("No ingress created, no cleanup necessary")
				return
			}
			By("Deleting ingress")
			jig.deleteIngress()

			By("Cleaning up cloud resources")
			if pollErr := wait.Poll(5*time.Second, lbCleanupTimeout, func() (bool, error) {
				if err := gceController.Cleanup(false); err != nil {
					framework.Logf("Still waiting for glbc to cleanup: %v", err)
					return false, nil
				}
				return true, nil
			}); pollErr != nil {
				if cleanupErr := gceController.Cleanup(true); cleanupErr != nil {
					framework.Logf("WARNING: Failed to cleanup resources %v", cleanupErr)
				}
				framework.Failf("Failed to cleanup GCE L7 resources.")
			}
		})

		It("should conform to Ingress spec", func() {
			for _, t := range conformanceTests {
				By(t.entryLog)
				t.execute()
				By(t.exitLog)
				jig.waitForIngress()
			}
		})

		It("shoud create ingress with given static-ip ", func() {
			ip := gceController.staticIP(ns)
			By(fmt.Sprintf("allocated static ip %v: %v through the GCE cloud provider", ns, ip))

			jig.createIngress(filepath.Join(ingressManifestPath, "static-ip"), ns, map[string]string{
				"kubernetes.io/ingress.global-static-ip-name": ns,
				"kubernetes.io/ingress.allow-http":            "false",
			})

			By("waiting for Ingress to come up with ip: " + ip)
			httpClient := buildInsecureClient(reqTimeout)
			ExpectNoError(jig.pollURL(fmt.Sprintf("https://%v/", ip), "", lbPollTimeout, httpClient, false))

			By("should reject HTTP traffic")
			ExpectNoError(jig.pollURL(fmt.Sprintf("http://%v/", ip), "", lbPollTimeout, httpClient, true))

			// TODO: uncomment the restart test once we have a way to synchronize
			// and know that the controller has resumed watching. If we delete
			// the ingress before the controller is ready we will leak.
			// By("restaring glbc")
			// restarter := NewRestartConfig(
			//	 framework.GetMasterHost(), "glbc", glbcHealthzPort, restartPollInterval, restartTimeout)
			// restarter.restart()
			// By("should continue serving on provided static-ip for 30 seconds")
			// ExpectNoError(jig.verifyURL(fmt.Sprintf("https://%v/", ip), "", 30, 1*time.Second, httpClient))
		})

		// TODO: Implement a multizone e2e that verifies traffic reaches each
		// zone based on pod labels.
	})
})

func describeIng(ns string) {
	framework.Logf("\nOutput of kubectl describe ing:\n")
	desc, _ := framework.RunKubectl(
		"describe", "ing", fmt.Sprintf("--namespace=%v", ns))
	framework.Logf(desc)
}

func exists(path string) bool {
	_, err := os.Stat(path)
	if err == nil {
		return true
	}
	if os.IsNotExist(err) {
		return false
	}
	framework.Failf("Failed to os.Stat path %v", path)
	return false
}

func newTestJig(c *client.Client) *testJig {
	return &testJig{client: c, rootCAs: map[string][]byte{}}
}

type testJig struct {
	client  *client.Client
	rootCAs map[string][]byte
	address string
	ing     *extensions.Ingress
}

// createIngress creates the Ingress and associated service/rc.
// Required: ing.yaml, rc.yaml, svc.yaml must exist in manifestPath
// Optional: secret.yaml, ingAnnotations
// If ingAnnotations is specified it will overwrite any annotations in ing.yaml
func (j *testJig) createIngress(manifestPath, ns string, ingAnnotations map[string]string) {
	mkpath := func(file string) string {
		return filepath.Join(framework.TestContext.RepoRoot, manifestPath, file)
	}

	framework.Logf("creating replication controller")
	framework.RunKubectlOrDie("create", "-f", mkpath("rc.yaml"), fmt.Sprintf("--namespace=%v", ns))

	framework.Logf("creating service")
	framework.RunKubectlOrDie("create", "-f", mkpath("svc.yaml"), fmt.Sprintf("--namespace=%v", ns))

	if exists(mkpath("secret.yaml")) {
		framework.Logf("creating secret")
		framework.RunKubectlOrDie("create", "-f", mkpath("secret.yaml"), fmt.Sprintf("--namespace=%v", ns))
	}
	j.ing = ingFromManifest(mkpath("ing.yaml"))
	j.ing.Namespace = ns
	if len(ingAnnotations) != 0 {
		j.ing.Annotations = ingAnnotations
	}
	framework.Logf(fmt.Sprintf("creating" + j.ing.Name + " ingress"))
	var err error
	j.ing, err = j.client.Extensions().Ingress(ns).Create(j.ing)
	ExpectNoError(err)
}

func (j *testJig) update(update func(ing *extensions.Ingress)) {
	var err error
	ns, name := j.ing.Namespace, j.ing.Name
	for i := 0; i < 3; i++ {
		j.ing, err = j.client.Extensions().Ingress(ns).Get(name)
		if err != nil {
			framework.Failf("failed to get ingress %q: %v", name, err)
		}
		update(j.ing)
		j.ing, err = j.client.Extensions().Ingress(ns).Update(j.ing)
		if err == nil {
			describeIng(j.ing.Namespace)
			return
		}
		if !apierrs.IsConflict(err) && !apierrs.IsServerTimeout(err) {
			framework.Failf("failed to update ingress %q: %v", name, err)
		}
	}
	framework.Failf("too many retries updating ingress %q", name)
}

func (j *testJig) addHTTPS(secretName string, hosts ...string) {
	j.ing.Spec.TLS = []extensions.IngressTLS{{Hosts: hosts, SecretName: secretName}}
	// TODO: Just create the secret in getRootCAs once we're watching secrets in
	// the ingress controller.
	_, cert, _, err := createSecret(j.client, j.ing)
	ExpectNoError(err)
	framework.Logf("Updating ingress %v to use secret %v for TLS termination", j.ing.Name, secretName)
	j.update(func(ing *extensions.Ingress) {
		ing.Spec.TLS = []extensions.IngressTLS{{Hosts: hosts, SecretName: secretName}}
	})
	j.rootCAs[secretName] = cert
}

func (j *testJig) getRootCA(secretName string) (rootCA []byte) {
	var ok bool
	rootCA, ok = j.rootCAs[secretName]
	if !ok {
		framework.Failf("Failed to retrieve rootCAs, no recorded secret by name %v", secretName)
	}
	return
}

func (j *testJig) deleteIngress() {
	ExpectNoError(j.client.Extensions().Ingress(j.ing.Namespace).Delete(j.ing.Name, nil))
}

func (j *testJig) waitForIngress() {
	// Wait for the loadbalancer IP.
	address, err := framework.WaitForIngressAddress(j.client, j.ing.Namespace, j.ing.Name, lbPollTimeout)
	if err != nil {
		framework.Failf("Ingress failed to acquire an IP address within %v", lbPollTimeout)
	}
	j.address = address
	framework.Logf("Found address %v for ingress %v", j.address, j.ing.Name)
	timeoutClient := &http.Client{Timeout: reqTimeout}

	// Check that all rules respond to a simple GET.
	for _, rules := range j.ing.Spec.Rules {
		proto := "http"
		if len(j.ing.Spec.TLS) > 0 {
			knownHosts := sets.NewString(j.ing.Spec.TLS[0].Hosts...)
			if knownHosts.Has(rules.Host) {
				timeoutClient.Transport, err = buildTransport(rules.Host, j.getRootCA(j.ing.Spec.TLS[0].SecretName))
				ExpectNoError(err)
				proto = "https"
			}
		}
		for _, p := range rules.IngressRuleValue.HTTP.Paths {
			j.curlServiceNodePort(j.ing.Namespace, p.Backend.ServiceName, int(p.Backend.ServicePort.IntVal))
			route := fmt.Sprintf("%v://%v%v", proto, address, p.Path)
			framework.Logf("Testing route %v host %v with simple GET", route, rules.Host)
			ExpectNoError(j.pollURL(route, rules.Host, lbPollTimeout, timeoutClient, false))
		}
	}
}

// verifyURL polls for the given iterations, in intervals, and fails if the
// given url returns a non-healthy http code even once.
func (j *testJig) verifyURL(route, host string, iterations int, interval time.Duration, httpClient *http.Client) error {
	for i := 0; i < iterations; i++ {
		b, err := simpleGET(httpClient, route, host)
		if err != nil {
			framework.Logf(b)
			return err
		}
		framework.Logf("Verfied %v with host %v %d times, sleeping for %v", route, host, i, interval)
		time.Sleep(interval)
	}
	return nil
}

// pollURL polls till the url responds with a healthy http code. If
// expectUnreachable is true, it breaks on first non-healthy http code instead.
func (j *testJig) pollURL(route, host string, timeout time.Duration, httpClient *http.Client, expectUnreachable bool) error {
	var lastBody string
	pollErr := wait.PollImmediate(lbPollInterval, timeout, func() (bool, error) {
		var err error
		lastBody, err = simpleGET(httpClient, route, host)
		if err != nil {
			framework.Logf("host %v path %v: %v unreachable", host, route, err)
			return expectUnreachable, nil
		}
		return !expectUnreachable, nil
	})
	if pollErr != nil {
		return fmt.Errorf("Failed to execute a successful GET within %v, Last response body for %v, host %v:\n%v\n\n%v\n",
			timeout, route, host, lastBody, pollErr)
	}
	return nil
}

func (j *testJig) curlServiceNodePort(ns, name string, port int) {
	// TODO: Curl all nodes?
	u, err := framework.GetNodePortURL(j.client, ns, name, port)
	ExpectNoError(err)
	ExpectNoError(j.pollURL(u, "", 30*time.Second, &http.Client{Timeout: reqTimeout}, false))
}

// ingFromManifest reads a .json/yaml file and returns the rc in it.
func ingFromManifest(fileName string) *extensions.Ingress {
	var ing extensions.Ingress
	framework.Logf("Parsing ingress from %v", fileName)
	data, err := ioutil.ReadFile(fileName)
	ExpectNoError(err)

	json, err := utilyaml.ToJSON(data)
	ExpectNoError(err)

	ExpectNoError(runtime.DecodeInto(api.Codecs.UniversalDecoder(), json, &ing))
	return &ing
}

// gcloudList unmarshals json output of gcloud into given out interface.
func gcloudList(resource, regex, project string, out interface{}) {
	// gcloud prints a message to stderr if it has an available update
	// so we only look at stdout.
	command := []string{
		"compute", resource, "list",
		fmt.Sprintf("--regex=%v", regex),
		fmt.Sprintf("--project=%v", project),
		"-q", "--format=json",
	}
	output, err := exec.Command("gcloud", command...).Output()
	if err != nil {
		errCode := -1
		if exitErr, ok := err.(utilexec.ExitError); ok {
			errCode = exitErr.ExitStatus()
		}
		framework.Logf("Error running gcloud command 'gcloud %s': err: %v, output: %v, status: %d", strings.Join(command, " "), err, string(output), errCode)
	}
	if err := json.Unmarshal([]byte(output), out); err != nil {
		framework.Logf("Error unmarshalling gcloud output for %v: %v, output: %v", resource, err, string(output))
	}
}

func gcloudDelete(resource, name, project string, args ...string) error {
	framework.Logf("Deleting %v: %v", resource, name)
	argList := append([]string{"compute", resource, "delete", name, fmt.Sprintf("--project=%v", project), "-q"}, args...)
	output, err := exec.Command("gcloud", argList...).CombinedOutput()
	if err != nil {
		framework.Logf("Error deleting %v, output: %v\nerror: %+v", resource, string(output), err)
	}
	return err
}

func gcloudCreate(resource, name, project string, args ...string) error {
	framework.Logf("Creating %v in project %v: %v", resource, project, name)
	argsList := append([]string{"compute", resource, "create", name, fmt.Sprintf("--project=%v", project)}, args...)
	output, err := exec.Command("gcloud", argsList...).CombinedOutput()
	if err != nil {
		framework.Logf("Error creating %v, output: %v\nerror: %+v", resource, string(output), err)
	}
	return err
}

// GCEIngressController manages implementation details of Ingress on GCE/GKE.
type GCEIngressController struct {
	ns           string
	rcPath       string
	UID          string
	Project      string
	staticIPName string
	rc           *api.ReplicationController
	svc          *api.Service
	c            *client.Client
}

func (cont *GCEIngressController) getL7AddonUID() (string, error) {
	framework.Logf("Retrieving UID from config map: %v/%v", api.NamespaceSystem, uidConfigMap)
	cm, err := cont.c.ConfigMaps(api.NamespaceSystem).Get(uidConfigMap)
	if err != nil {
		return "", err
	}
	if uid, ok := cm.Data[uidKey]; ok {
		return uid, nil
	}
	return "", fmt.Errorf("Could not find cluster UID for L7 addon pod")
}

func (cont *GCEIngressController) init() {
	uid, err := cont.getL7AddonUID()
	Expect(err).NotTo(HaveOccurred())
	cont.UID = uid
	// There's a name limit imposed by GCE. The controller will truncate.
	testName := fmt.Sprintf("k8s-fw-foo-app-X-%v--%v", cont.ns, cont.UID)
	if len(testName) > nameLenLimit {
		framework.Logf("WARNING: test name including cluster UID: %v is over the GCE limit of %v", testName, nameLenLimit)
	} else {
		framework.Logf("Deteced cluster UID %v", cont.UID)
	}
}

func (cont *GCEIngressController) staticIP(name string) string {
	ExpectNoError(gcloudCreate("addresses", name, cont.Project, "--global"))
	cont.staticIPName = name
	ipList := []compute.Address{}
	if pollErr := wait.PollImmediate(5*time.Second, cloudResourcePollTimeout, func() (bool, error) {
		gcloudList("addresses", name, cont.Project, &ipList)
		if len(ipList) != 1 {
			framework.Logf("Failed to find static ip %v even though create call succeeded, found ips %+v", name, ipList)
			return false, nil
		}
		return true, nil
	}); pollErr != nil {
		if err := gcloudDelete("addresses", name, cont.Project, "--global"); err == nil {
			framework.Logf("Failed to get AND delete address %v even though create call succeeded", name)
		}
		framework.Failf("Failed to find static ip %v even though create call succeeded, found ips %+v", name, ipList)
	}
	return ipList[0].Address
}

// Cleanup cleans up cloud resources.
// If del is false, it simply reports existing resources without deleting them.
// It always deletes resources created through it's methods, like staticIP, even
// if del is false.
func (cont *GCEIngressController) Cleanup(del bool) error {
	errMsg := ""
	// Ordering is important here because we cannot delete resources that other
	// resources hold references to.
	fwList := []compute.ForwardingRule{}
	for _, regex := range []string{fmt.Sprintf("k8s-fw-.*--%v", cont.UID), fmt.Sprintf("k8s-fws-.*--%v", cont.UID)} {
		gcloudList("forwarding-rules", regex, cont.Project, &fwList)
		if len(fwList) != 0 {
			msg := ""
			for _, f := range fwList {
				msg += fmt.Sprintf("%v\n", f.Name)
				if del {
					gcloudDelete("forwarding-rules", f.Name, cont.Project, "--global")
				}
			}
			errMsg += fmt.Sprintf("\nFound forwarding rules:\n%v", msg)
		}
	}
	// Static IPs are named after forwarding rules.
	ipList := []compute.Address{}
	gcloudList("addresses", fmt.Sprintf("k8s-fw-.*--%v", cont.UID), cont.Project, &ipList)
	if len(ipList) != 0 {
		msg := ""
		for _, ip := range ipList {
			msg += fmt.Sprintf("%v\n", ip.Name)
			if del {
				gcloudDelete("addresses", ip.Name, cont.Project)
			}
		}
		errMsg += fmt.Sprintf("Found addresses:\n%v", msg)
	}

	// If the test allocated a static ip, delete that regardless
	if cont.staticIPName != "" {
		if err := gcloudDelete("addresses", cont.staticIPName, cont.Project, "--global"); err == nil {
			cont.staticIPName = ""
		}
	}

	tpList := []compute.TargetHttpProxy{}
	gcloudList("target-http-proxies", fmt.Sprintf("k8s-tp-.*--%v", cont.UID), cont.Project, &tpList)
	if len(tpList) != 0 {
		msg := ""
		for _, t := range tpList {
			msg += fmt.Sprintf("%v\n", t.Name)
			if del {
				gcloudDelete("target-http-proxies", t.Name, cont.Project)
			}
		}
		errMsg += fmt.Sprintf("Found target proxies:\n%v", msg)
	}
	tpsList := []compute.TargetHttpsProxy{}
	gcloudList("target-https-proxies", fmt.Sprintf("k8s-tps-.*--%v", cont.UID), cont.Project, &tpsList)
	if len(tpsList) != 0 {
		msg := ""
		for _, t := range tpsList {
			msg += fmt.Sprintf("%v\n", t.Name)
			if del {
				gcloudDelete("target-https-proxies", t.Name, cont.Project)
			}
		}
		errMsg += fmt.Sprintf("Found target HTTPS proxies:\n%v", msg)
	}
	// TODO: Check for leaked ssl certs.

	umList := []compute.UrlMap{}
	gcloudList("url-maps", fmt.Sprintf("k8s-um-.*--%v", cont.UID), cont.Project, &umList)
	if len(umList) != 0 {
		msg := ""
		for _, u := range umList {
			msg += fmt.Sprintf("%v\n", u.Name)
			if del {
				gcloudDelete("url-maps", u.Name, cont.Project)
			}
		}
		errMsg += fmt.Sprintf("Found url maps:\n%v", msg)
	}

	beList := []compute.BackendService{}
	gcloudList("backend-services", fmt.Sprintf("k8s-be-[0-9]+--%v", cont.UID), cont.Project, &beList)
	if len(beList) != 0 {
		msg := ""
		for _, b := range beList {
			msg += fmt.Sprintf("%v\n", b.Name)
			if del {
				gcloudDelete("backend-services", b.Name, cont.Project)
			}
		}
		errMsg += fmt.Sprintf("Found backend services:\n%v", msg)
	}

	hcList := []compute.HttpHealthCheck{}
	gcloudList("http-health-checks", fmt.Sprintf("k8s-be-[0-9]+--%v", cont.UID), cont.Project, &hcList)
	if len(hcList) != 0 {
		msg := ""
		for _, h := range hcList {
			msg += fmt.Sprintf("%v\n", h.Name)
			if del {
				gcloudDelete("http-health-checks", h.Name, cont.Project)
			}
		}
		errMsg += fmt.Sprintf("Found health check:\n%v", msg)
	}
	// TODO: Verify instance-groups, issue #16636. Gcloud mysteriously barfs when told
	// to unmarshal instance groups into the current vendored gce-client's understanding
	// of the struct.
	if errMsg == "" {
		return nil
	}
	return fmt.Errorf(errMsg)
}
