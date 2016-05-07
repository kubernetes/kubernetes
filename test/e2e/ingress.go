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
	"encoding/json"
	"fmt"
	"net/http"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"time"

	compute "google.golang.org/api/compute/v1"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/extensions"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/labels"
	utilexec "k8s.io/kubernetes/pkg/util/exec"
	"k8s.io/kubernetes/pkg/util/intstr"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

// Before enabling this test you must make sure the associated project has
// enough quota. At the time of this writing GCE projects are allowed 3
// backend services by default. This test requires at least 5.

// This test exercises the GCE L7 loadbalancer controller cluster-addon. It
// will fail if the addon isn't running, or doesn't send traffic to the expected
// backend. Common failure modes include:
// * GCE L7 took too long to spin up
// * GCE L7 took too long to health check a backend
// * Repeated 404:
//   - L7 is sending traffic to the default backend of the addon.
//   - Backend is receiving /foo when it expects /bar.
// * Repeated 5xx:
//   - Out of quota (describe ing should show you if this is the case)
//   - Mismatched service/container port, or endpoints are dead.

var (
	appPrefix         = "app"
	pathPrefix        = "foo"
	testImage         = "gcr.io/google_containers/n-way-http:1.0"
	httpContainerPort = 8080

	expectedLBCreationTime    = 7 * time.Minute
	expectedLBHealthCheckTime = 7 * time.Minute

	// Labels applied throughout, to the RC, the default backend, as a selector etc.
	controllerLabels = map[string]string{"name": "glbc"}

	// Name of the loadbalancer controller within the cluster addon
	lbContainerName = "l7-lb-controller"

	// Labels used to identify existing loadbalancer controllers.
	// TODO: Pull this out of the RC manifest.
	clusterAddonLBLabels = map[string]string{"k8s-app": "glbc"}

	// If set, the test tries to perform an HTTP GET on each url endpoint of
	// the Ingress. Only set to false to short-circuit test runs in debugging.
	verifyHTTPGET = true

	// On average it takes ~6 minutes for a single backend to come online.
	// We *don't* expect this framework.Poll to consistently take 15 minutes for every
	// Ingress as GCE is creating/checking backends in parallel, but at the
	// same time, we're not testing GCE startup latency. So give it enough
	// time, and fail if the average is too high.
	lbPollTimeout  = 15 * time.Minute
	lbPollInterval = 30 * time.Second

	// Time required by the loadbalancer to cleanup, proportional to numApps/Ing.
	lbCleanupTimeout = 5 * time.Minute

	// One can scale this test by tweaking numApps and numIng, the former will
	// create more RCs/Services and add them to a single Ingress, while the latter
	// will create smaller, more fragmented Ingresses. The numbers 2, 1 are chosen
	// arbitrarity, we want to test more than a single endpoint.
	numApps = 2
	numIng  = 1

	// GCE only allows names < 64 characters, and the loadbalancer controller inserts
	// a single character of padding.
	nameLenLimit = 62

	// Timing out requests will lead to retries, and more importantly, the test
	// finishing in a deterministic manner.
	timeoutClient = &http.Client{Timeout: 60 * time.Second}
)

// timeSlice allows sorting of time.Duration
type timeSlice []time.Duration

func (p timeSlice) Len() int {
	return len(p)
}

func (p timeSlice) Less(i, j int) bool {
	return p[i] < p[j]
}

func (p timeSlice) Swap(i, j int) {
	p[i], p[j] = p[j], p[i]
}

// ruleByIndex returns an IngressRule for the given index.
func ruleByIndex(i int) extensions.IngressRule {
	return extensions.IngressRule{
		Host: fmt.Sprintf("foo%d.bar.com", i),
		IngressRuleValue: extensions.IngressRuleValue{
			HTTP: &extensions.HTTPIngressRuleValue{
				Paths: []extensions.HTTPIngressPath{
					{
						Path: fmt.Sprintf("/%v%d", pathPrefix, i),
						Backend: extensions.IngressBackend{
							ServiceName: fmt.Sprintf("%v%d", appPrefix, i),
							ServicePort: intstr.FromInt(httpContainerPort),
						},
					},
				},
			},
		},
	}
}

// createIngress creates an Ingress with num rules. Eg:
// start = 1 num = 2 will given you a single Ingress with 2 rules:
// Ingress {
//	 foo1.bar.com: /foo1
//	 foo2.bar.com: /foo2
// }
func generateIngressSpec(start, num int, ns string) *extensions.Ingress {
	name := fmt.Sprintf("%v%d", appPrefix, start)
	ing := extensions.Ingress{
		ObjectMeta: api.ObjectMeta{
			Name:      name,
			Namespace: ns,
		},
		Spec: extensions.IngressSpec{
			Backend: &extensions.IngressBackend{
				ServiceName: fmt.Sprintf("%v%d", appPrefix, start),
				ServicePort: intstr.FromInt(httpContainerPort),
			},
			Rules: []extensions.IngressRule{},
		},
	}
	for i := start; i < start+num; i++ {
		ing.Spec.Rules = append(ing.Spec.Rules, ruleByIndex(i))
	}
	// Create the host for the cert by appending all hosts mentioned in rules.
	hosts := []string{}
	for _, rules := range ing.Spec.Rules {
		hosts = append(hosts, rules.Host)
	}
	ing.Spec.TLS = []extensions.IngressTLS{
		{Hosts: hosts, SecretName: name},
	}
	return &ing
}

// createApp will create a single RC and Svc. The Svc will match pods of the
// RC using the selector: 'name'=<name arg>
func createApp(c *client.Client, ns string, i int) {
	name := fmt.Sprintf("%v%d", appPrefix, i)
	l := map[string]string{}

	framework.Logf("Creating svc %v", name)
	svc := svcByName(name, httpContainerPort)
	svc.Spec.Type = api.ServiceTypeNodePort
	_, err := c.Services(ns).Create(svc)
	Expect(err).NotTo(HaveOccurred())

	framework.Logf("Creating rc %v", name)
	rc := rcByNamePort(name, 1, testImage, httpContainerPort, api.ProtocolTCP, l)
	rc.Spec.Template.Spec.Containers[0].Args = []string{
		"--num=1",
		fmt.Sprintf("--start=%d", i),
		fmt.Sprintf("--prefix=%v", pathPrefix),
		fmt.Sprintf("--port=%d", httpContainerPort),
	}
	_, err = c.ReplicationControllers(ns).Create(rc)
	Expect(err).NotTo(HaveOccurred())
}

// gcloudUnmarshal unmarshals json output of gcloud into given out interface.
func gcloudUnmarshal(resource, regex, project string, out interface{}) {
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

func gcloudDelete(resource, name, project string) {
	framework.Logf("Deleting %v: %v", resource, name)
	output, err := exec.Command("gcloud", "compute", resource, "delete",
		name, fmt.Sprintf("--project=%v", project), "-q").CombinedOutput()
	if err != nil {
		framework.Logf("Error deleting %v, output: %v\nerror: %+v", resource, string(output), err)
	}
}

// kubectlLogLBController logs kubectl debug output for the L7 controller pod.
func kubectlLogLBController(c *client.Client, ns string) {
	selector := labels.SelectorFromSet(labels.Set(controllerLabels))
	options := api.ListOptions{LabelSelector: selector}
	podList, err := c.Pods(api.NamespaceAll).List(options)
	if err != nil {
		framework.Logf("Cannot log L7 controller output, error listing pods %v", err)
		return
	}
	if len(podList.Items) == 0 {
		framework.Logf("Loadbalancer controller pod not found")
		return
	}
	for _, p := range podList.Items {
		framework.Logf("\nLast 100 log lines of %v\n", p.Name)
		l, _ := framework.RunKubectl("logs", p.Name, fmt.Sprintf("--namespace=%v", ns), "-c", lbContainerName, "--tail=100")
		framework.Logf(l)
	}
}

type IngressController struct {
	ns      string
	rcPath  string
	UID     string
	Project string
	rc      *api.ReplicationController
	svc     *api.Service
	c       *client.Client
}

func (cont *IngressController) getL7AddonUID() (string, error) {
	listOpts := api.ListOptions{LabelSelector: labels.SelectorFromSet(labels.Set(clusterAddonLBLabels))}
	existingRCs, err := cont.c.ReplicationControllers(api.NamespaceSystem).List(listOpts)
	if err != nil {
		return "", err
	}
	if len(existingRCs.Items) != 1 {
		return "", fmt.Errorf("Unexpected number of lb cluster addons %v with label %v in kube-system namespace", len(existingRCs.Items), clusterAddonLBLabels)
	}
	rc := existingRCs.Items[0]
	commandPrefix := "--cluster-uid="
	for i, c := range rc.Spec.Template.Spec.Containers {
		if c.Name == lbContainerName {
			for _, arg := range rc.Spec.Template.Spec.Containers[i].Args {
				if strings.HasPrefix(arg, commandPrefix) {
					return strings.Replace(arg, commandPrefix, "", -1), nil
				}
			}
		}
	}
	return "", fmt.Errorf("Could not find cluster UID for L7 addon pod")
}

func (cont *IngressController) init() {
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

func (cont *IngressController) Cleanup(del bool) error {
	errMsg := ""
	// Ordering is important here because we cannot delete resources that other
	// resources hold references to.
	fwList := []compute.ForwardingRule{}
	for _, regex := range []string{fmt.Sprintf("k8s-fw-.*--%v", cont.UID), fmt.Sprintf("k8s-fws-.*--%v", cont.UID)} {
		gcloudUnmarshal("forwarding-rules", regex, cont.Project, &fwList)
		if len(fwList) != 0 {
			msg := ""
			for _, f := range fwList {
				msg += fmt.Sprintf("%v\n", f.Name)
				if del {
					framework.Logf("Deleting forwarding-rule: %v", f.Name)
					output, err := exec.Command("gcloud", "compute", "forwarding-rules", "delete",
						f.Name, fmt.Sprintf("--project=%v", cont.Project), "-q", "--global").CombinedOutput()
					if err != nil {
						framework.Logf("Error deleting forwarding rules, output: %v\nerror:%v", string(output), err)
					}
				}
			}
			errMsg += fmt.Sprintf("\nFound forwarding rules:\n%v", msg)
		}
	}
	// Static IPs are named after forwarding rules.
	ipList := []compute.Address{}
	gcloudUnmarshal("addresses", fmt.Sprintf("k8s-fw-.*--%v", cont.UID), cont.Project, &ipList)
	if len(ipList) != 0 {
		msg := ""
		for _, ip := range ipList {
			msg += fmt.Sprintf("%v\n", ip.Name)
			if del {
				gcloudDelete("addresses", ip.Name, cont.Project)
			}
		}
		errMsg += fmt.Sprintf("Found health check:\n%v", msg)
	}

	tpList := []compute.TargetHttpProxy{}
	gcloudUnmarshal("target-http-proxies", fmt.Sprintf("k8s-tp-.*--%v", cont.UID), cont.Project, &tpList)
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
	gcloudUnmarshal("target-https-proxies", fmt.Sprintf("k8s-tps-.*--%v", cont.UID), cont.Project, &tpsList)
	if len(tpsList) != 0 {
		msg := ""
		for _, t := range tpsList {
			msg += fmt.Sprintf("%v\n", t.Name)
			if del {
				gcloudDelete("target-http-proxies", t.Name, cont.Project)
			}
		}
		errMsg += fmt.Sprintf("Found target HTTPS proxies:\n%v", msg)
	}
	// TODO: Check for leaked ssl certs.

	umList := []compute.UrlMap{}
	gcloudUnmarshal("url-maps", fmt.Sprintf("k8s-um-.*--%v", cont.UID), cont.Project, &umList)
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
	gcloudUnmarshal("backend-services", fmt.Sprintf("k8s-be-[0-9]+--%v", cont.UID), cont.Project, &beList)
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
	gcloudUnmarshal("http-health-checks", fmt.Sprintf("k8s-be-[0-9]+--%v", cont.UID), cont.Project, &hcList)
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

// Before enabling this loadbalancer test in any other test list you must
// make sure the associated project has enough quota. At the time of this
// writing a GCE project is allowed 3 backend services by default. This
// test requires at least 5.
//
// Slow by design (10 min)
var _ = framework.KubeDescribe("GCE L7 LoadBalancer Controller [Feature:Ingress]", func() {
	// These variables are initialized after framework's beforeEach.
	var ns string
	var addonDir string
	var client *client.Client
	var responseTimes, creationTimes []time.Duration
	var ingController *IngressController

	f := framework.Framework{BaseName: "glbc"}

	BeforeEach(func() {
		// This test requires a GCE/GKE only cluster-addon
		framework.SkipUnlessProviderIs("gce", "gke")
		f.BeforeEach()
		client = f.Client
		ns = f.Namespace.Name
		addonDir = filepath.Join(
			framework.TestContext.RepoRoot, "cluster", "addons", "cluster-loadbalancing", "glbc")
		ingController = &IngressController{
			ns:      ns,
			Project: framework.TestContext.CloudConfig.ProjectID,
			c:       client,
		}
		ingController.init()
		// If we somehow get the same namespace uid as someone else in this
		// gce project, just back off.
		Expect(ingController.Cleanup(false)).NotTo(HaveOccurred())
		responseTimes = []time.Duration{}
		creationTimes = []time.Duration{}
	})

	AfterEach(func() {
		framework.Logf("Average creation time %+v, health check time %+v", creationTimes, responseTimes)
		if CurrentGinkgoTestDescription().Failed {
			kubectlLogLBController(client, ns)
			framework.Logf("\nOutput of kubectl describe ing:\n")
			desc, _ := framework.RunKubectl("describe", "ing", fmt.Sprintf("--namespace=%v", ns))
			framework.Logf(desc)
		}
		// Delete all Ingress, then wait for the controller to cleanup.
		ings, err := client.Extensions().Ingress(ns).List(api.ListOptions{})
		if err != nil {
			framework.Logf("WARNING: Failed to list ingress: %+v", err)
		} else {
			for _, ing := range ings.Items {
				framework.Logf("Deleting ingress %v/%v", ing.Namespace, ing.Name)
				if err := client.Extensions().Ingress(ns).Delete(ing.Name, nil); err != nil {
					framework.Logf("WARNING: Failed to delete ingress %v: %v", ing.Name, err)
				}
			}
		}
		pollErr := wait.Poll(5*time.Second, lbCleanupTimeout, func() (bool, error) {
			if err := ingController.Cleanup(false); err != nil {
				framework.Logf("Still waiting for glbc to cleanup: %v", err)
				return false, nil
			}
			return true, nil
		})
		// If the controller failed to cleanup the test will fail, but we want to cleanup
		// resources before that.
		if pollErr != nil {
			if cleanupErr := ingController.Cleanup(true); cleanupErr != nil {
				framework.Logf("WARNING: Failed to cleanup resources %v", cleanupErr)
			}
			framework.Failf("Failed to cleanup GCE L7 resources.")
		}
		f.AfterEach()
		framework.Logf("Successfully verified GCE L7 loadbalancer via Ingress.")
	})

	It("should create GCE L7 loadbalancers and verify Ingress", func() {
		// Create numApps apps, exposed via numIng Ingress each with 2 paths.
		// Eg with numApp=10, numIng=5:
		// apps: {foo-app-(0-10)}
		// ingress: {foo-app-(0, 2, 4, 6, 8)}
		// paths:
		//  ingress foo-app-0:
		//	  default1.bar.com
		//	  foo0.bar.com: /foo0
		//	  foo1.bar.com: /foo1
		if numApps < numIng {
			framework.Failf("Need more apps than Ingress")
		}
		framework.Logf("Starting ingress test")
		appsPerIngress := numApps / numIng
		By(fmt.Sprintf("Creating %d rcs + svc, and %d apps per Ingress", numApps, appsPerIngress))

		ingCAs := map[string][]byte{}
		for appID := 0; appID < numApps; appID = appID + appsPerIngress {
			// Creates appsPerIngress apps, then creates one Ingress with paths to all the apps.
			for j := appID; j < appID+appsPerIngress; j++ {
				createApp(client, ns, j)
			}
			var err error
			ing := generateIngressSpec(appID, appsPerIngress, ns)

			// Secrets must be created before the Ingress. The cert of each
			// Ingress contains all the hostnames of that Ingress as the subject
			// name field in the cert.
			By(fmt.Sprintf("Creating secret for ingress %v/%v", ing.Namespace, ing.Name))
			_, rootCA, _, err := createSecret(client, ing)
			Expect(err).NotTo(HaveOccurred())
			ingCAs[ing.Name] = rootCA

			By(fmt.Sprintf("Creating ingress %v/%v", ing.Namespace, ing.Name))
			ing, err = client.Extensions().Ingress(ing.Namespace).Create(ing)
			Expect(err).NotTo(HaveOccurred())
		}

		ings, err := client.Extensions().Ingress(ns).List(api.ListOptions{})
		Expect(err).NotTo(HaveOccurred())
		for _, ing := range ings.Items {
			// Wait for the loadbalancer IP.
			start := time.Now()
			address, err := framework.WaitForIngressAddress(client, ing.Namespace, ing.Name, lbPollTimeout)
			if err != nil {
				framework.Failf("Ingress failed to acquire an IP address within %v", lbPollTimeout)
			}
			Expect(err).NotTo(HaveOccurred())
			By(fmt.Sprintf("Found address %v for ingress %v, took %v to come online",
				address, ing.Name, time.Since(start)))
			creationTimes = append(creationTimes, time.Since(start))

			if !verifyHTTPGET {
				continue
			}

			// Check that all rules respond to a simple GET.
			for _, rules := range ing.Spec.Rules {
				// As of Kubernetes 1.1 we only support HTTP Ingress.
				if rules.IngressRuleValue.HTTP == nil {
					continue
				}
				timeoutClient.Transport, err = buildTransport(rules.Host, ingCAs[ing.Name])

				for _, p := range rules.IngressRuleValue.HTTP.Paths {
					route := fmt.Sprintf("https://%v%v", address, p.Path)
					framework.Logf("Testing route %v host %v with simple GET", route, rules.Host)
					if err != nil {
						framework.Failf("Unable to create transport: %v", err)
					}
					// Make sure the service node port is reachable
					Expect(curlServiceNodePort(client, ns, p.Backend.ServiceName, int(p.Backend.ServicePort.IntVal))).NotTo(HaveOccurred())

					GETStart := time.Now()
					var lastBody string
					pollErr := wait.Poll(lbPollInterval, lbPollTimeout, func() (bool, error) {
						var err error
						lastBody, err = simpleGET(timeoutClient, route, rules.Host)
						if err != nil {
							framework.Logf("host %v path %v: %v", rules.Host, route, err)
							return false, nil
						}
						return true, nil
					})
					if pollErr != nil {
						msg := fmt.Sprintf("Failed to execute a successful GET within %v, Last response body for %v, host %v:\n%v\n\n%v\n",
							lbPollTimeout, route, rules.Host, lastBody, pollErr)

						// Make sure the service node port is still reachable
						if err := curlServiceNodePort(client, ns, p.Backend.ServiceName, int(p.Backend.ServicePort.IntVal)); err != nil {
							msg += fmt.Sprintf("Also unable to curl service node port: %v", err)
						}
						framework.Failf(msg)
					}
					rt := time.Since(GETStart)
					By(fmt.Sprintf("Route %v host %v took %v to respond", route, rules.Host, rt))
					responseTimes = append(responseTimes, rt)
				}
			}
		}
		// In most cases slow loadbalancer creation/startup translates directly to
		// GCE api sluggishness. However this might be because of something the
		// controller is doing, eg: maxing out QPS by repeated polling.
		sort.Sort(timeSlice(creationTimes))
		perc50 := creationTimes[len(creationTimes)/2]
		if perc50 > expectedLBCreationTime {
			framework.Logf("WARNING: Average creation time is too high: %+v", creationTimes)
		}
		if !verifyHTTPGET {
			return
		}
		sort.Sort(timeSlice(responseTimes))
		perc50 = responseTimes[len(responseTimes)/2]
		if perc50 > expectedLBHealthCheckTime {
			framework.Logf("WARNING: Average startup time is too high: %+v", responseTimes)
		}
	})
})

func curlServiceNodePort(client *client.Client, ns, name string, port int) error {
	// TODO: Curl all nodes?
	u, err := framework.GetNodePortURL(client, ns, name, port)
	if err != nil {
		return err
	}
	var svcCurlBody string
	timeout := 30 * time.Second
	pollErr := wait.Poll(10*time.Second, timeout, func() (bool, error) {
		svcCurlBody, err = simpleGET(timeoutClient, u, "")
		if err != nil {
			framework.Logf("Failed to curl service node port, body: %v\nerror %v", svcCurlBody, err)
			return false, nil
		}
		return true, nil
	})
	if pollErr != nil {
		return fmt.Errorf("Failed to curl service node port in %v, body: %v\nerror %v", timeout, svcCurlBody, err)
	}
	framework.Logf("Successfully curled service node port, body: %v", svcCurlBody)
	return nil
}
