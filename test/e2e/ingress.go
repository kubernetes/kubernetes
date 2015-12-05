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
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/wait"

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

	// If set, the test tries to perform an HTTP GET on each url endpoint of
	// the Ingress. Only set to false to short-circuit test runs in debugging.
	verifyHTTPGET = true

	// On average it takes ~6 minutes for a single backend to come online.
	// We *don't* expect this poll to consistently take 15 minutes for every
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
							ServicePort: util.NewIntOrStringFromInt(httpContainerPort),
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
func createIngress(c *client.Client, ns string, start, num int) extensions.Ingress {
	ing := extensions.Ingress{
		ObjectMeta: api.ObjectMeta{
			Name:      fmt.Sprintf("%v%d", appPrefix, start),
			Namespace: ns,
		},
		Spec: extensions.IngressSpec{
			Backend: &extensions.IngressBackend{
				ServiceName: fmt.Sprintf("%v%d", appPrefix, start),
				ServicePort: util.NewIntOrStringFromInt(httpContainerPort),
			},
			Rules: []extensions.IngressRule{},
		},
	}
	for i := start; i < start+num; i++ {
		ing.Spec.Rules = append(ing.Spec.Rules, ruleByIndex(i))
	}
	Logf("Creating ingress %v", start)
	_, err := c.Extensions().Ingress(ns).Create(&ing)
	Expect(err).NotTo(HaveOccurred())
	return ing
}

// createApp will create a single RC and Svc. The Svc will match pods of the
// RC using the selector: 'name'=<name arg>
func createApp(c *client.Client, ns string, i int) {
	name := fmt.Sprintf("%v%d", appPrefix, i)
	l := map[string]string{}

	Logf("Creating svc %v", name)
	svc := svcByName(name, httpContainerPort)
	svc.Spec.Type = api.ServiceTypeNodePort
	_, err := c.Services(ns).Create(svc)
	Expect(err).NotTo(HaveOccurred())

	Logf("Creating rc %v", name)
	rc := rcByNamePort(name, 1, testImage, httpContainerPort, l)
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
func gcloudUnmarshal(resource, regex string, out interface{}) {
	output, err := exec.Command("gcloud", "compute", resource, "list",
		fmt.Sprintf("--regex=%v", regex),
		fmt.Sprintf("--project=%v", testContext.CloudConfig.ProjectID),
		"-q", "--format=json").CombinedOutput()
	if err != nil {
		Failf("Error unmarshalling gcloud output: %v", err)
	}
	if err := json.Unmarshal([]byte(output), out); err != nil {
		Failf("Error unmarshalling gcloud output for %v: %v", resource, err)
	}
}

func gcloudDelete(resource, name string) {
	Logf("Deleting %v: %v", resource, name)
	output, err := exec.Command("gcloud", "compute", resource, "delete",
		name, fmt.Sprintf("--project=%v", testContext.CloudConfig.ProjectID), "-q").CombinedOutput()
	if err != nil {
		Logf("Error deleting %v, output: %v\nerror: %+v", resource, string(output), err)
	}
}

// kubectlLogLBController logs kubectl debug output for the L7 controller pod.
func kubectlLogLBController(c *client.Client, ns string) {
	selector := labels.SelectorFromSet(labels.Set(controllerLabels))
	podList, err := c.Pods(api.NamespaceAll).List(selector, fields.Everything())
	if err != nil {
		Logf("Cannot log L7 controller output, error listing pods %v", err)
		return
	}
	if len(podList.Items) == 0 {
		Logf("Loadbalancer controller pod not found")
		return
	}
	for _, p := range podList.Items {
		Logf("\nLast 100 log lines of %v\n", p.Name)
		l, _ := runKubectl("logs", p.Name, fmt.Sprintf("--namespace=%v", ns), "-c", lbContainerName, "--tail=100")
		Logf(l)
	}
}

type ingressController struct {
	ns             string
	rcPath         string
	defaultSvcPath string
	UID            string
	rc             *api.ReplicationController
	svc            *api.Service
	c              *client.Client
}

func (cont *ingressController) create() {

	// TODO: This cop out is because it would be *more* brittle to duplicate all
	// the name construction logic from the controller cross-repo. We will not
	// need to be so paranoid about leaked resources once we figure out a solution
	// for issues like #16337. Currently, all names should fall within 63 chars.
	testName := fmt.Sprintf("k8s-fw-foo-app-X-%v--%v", cont.ns, cont.UID)
	if len(testName) > nameLenLimit {
		Failf("Cannot reliably test the given namespace(%v)/uid(%v), too close to GCE limit of %v",
			cont.ns, cont.UID, nameLenLimit)
	}

	if cont.defaultSvcPath != "" {
		svc := svcFromManifest(cont.defaultSvcPath)
		svc.Namespace = cont.ns
		svc.Labels = controllerLabels
		svc.Spec.Selector = controllerLabels
		cont.svc = svc
		_, err := cont.c.Services(cont.ns).Create(cont.svc)
		Expect(err).NotTo(HaveOccurred())
	}
	rc := rcFromManifest(cont.rcPath)
	existingRc, err := cont.c.ReplicationControllers(api.NamespaceSystem).Get(lbContainerName)
	Expect(err).NotTo(HaveOccurred())
	// Merge the existing spec and new spec. The modifications should not
	// manifest as functional changes to the controller. Most importantly, the
	// podTemplate shouldn't change (but for the additional test cmd line flags)
	// to ensure we test actual cluster functionality across upgrades.
	rc.Spec = existingRc.Spec
	rc.Name = "glbc"
	rc.Namespace = cont.ns
	rc.Labels = controllerLabels
	rc.Spec.Selector = controllerLabels
	rc.Spec.Template.Labels = controllerLabels
	rc.Spec.Replicas = 1

	// These command line params are only recognized by v0.51 and above.
	testArgs := []string{
		// Pass namespace uid so the controller will tag resources with it.
		fmt.Sprintf("--cluster-uid=%v", cont.UID),
		// Tell the controller to delete all resources as it quits.
		fmt.Sprintf("--delete-all-on-quit=true"),
		// Don't use the default Service from kube-system.
		fmt.Sprintf("--default-backend-service=%v/%v", cont.svc.Namespace, cont.svc.Name),
	}
	for i, c := range rc.Spec.Template.Spec.Containers {
		if c.Name == lbContainerName {
			rc.Spec.Template.Spec.Containers[i].Args = append(c.Args, testArgs...)
		}
	}
	cont.rc = rc
	_, err = cont.c.ReplicationControllers(cont.ns).Create(cont.rc)
	Expect(err).NotTo(HaveOccurred())
	Expect(waitForRCPodsRunning(cont.c, cont.ns, cont.rc.Name)).NotTo(HaveOccurred())
}

func (cont *ingressController) cleanup(del bool) error {
	errMsg := ""
	// Ordering is important here because we cannot delete resources that other
	// resources hold references to.
	fwList := []compute.ForwardingRule{}
	gcloudUnmarshal("forwarding-rules", fmt.Sprintf("k8s-fw-.*--%v", cont.UID), &fwList)
	if len(fwList) != 0 {
		msg := ""
		for _, f := range fwList {
			msg += fmt.Sprintf("%v\n", f.Name)
			if del {
				Logf("Deleting forwarding-rule: %v", f.Name)
				output, err := exec.Command("gcloud", "compute", "forwarding-rules", "delete",
					f.Name, fmt.Sprintf("--project=%v", testContext.CloudConfig.ProjectID), "-q", "--global").CombinedOutput()
				if err != nil {
					Logf("Error deleting forwarding rules, output: %v\nerror:%v", string(output), err)
				}
			}
		}
		errMsg += fmt.Sprintf("\nFound forwarding rules:\n%v", msg)
	}

	tpList := []compute.TargetHttpProxy{}
	gcloudUnmarshal("target-http-proxies", fmt.Sprintf("k8s-tp-.*--%v", cont.UID), &tpList)
	if len(tpList) != 0 {
		msg := ""
		for _, t := range tpList {
			msg += fmt.Sprintf("%v\n", t.Name)
			if del {
				gcloudDelete("target-http-proxies", t.Name)
			}
		}
		errMsg += fmt.Sprintf("Found target proxies:\n%v", msg)
	}

	umList := []compute.UrlMap{}
	gcloudUnmarshal("url-maps", fmt.Sprintf("k8s-um-.*--%v", cont.UID), &umList)
	if len(umList) != 0 {
		msg := ""
		for _, u := range umList {
			msg += fmt.Sprintf("%v\n", u.Name)
			if del {
				gcloudDelete("url-maps", u.Name)
			}
		}
		errMsg += fmt.Sprintf("Found url maps:\n%v", msg)
	}

	beList := []compute.BackendService{}
	gcloudUnmarshal("backend-services", fmt.Sprintf("k8s-be-[0-9]+--%v", cont.UID), &beList)
	if len(beList) != 0 {
		msg := ""
		for _, b := range beList {
			msg += fmt.Sprintf("%v\n", b.Name)
			if del {
				gcloudDelete("backend-services", b.Name)
			}
		}
		errMsg += fmt.Sprintf("Found backend services:\n%v", msg)
	}

	hcList := []compute.HttpHealthCheck{}
	gcloudUnmarshal("http-health-checks", fmt.Sprintf("k8s-be-[0-9]+--%v", cont.UID), &hcList)
	if len(hcList) != 0 {
		msg := ""
		for _, h := range hcList {
			msg += fmt.Sprintf("%v\n", h.Name)
			if del {
				gcloudDelete("http-health-checks", h.Name)
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

var _ = Describe("GCE L7 LoadBalancer Controller", func() {
	// These variables are initialized after framework's beforeEach.
	var ns string
	var addonDir string
	var client *client.Client
	var responseTimes, creationTimes []time.Duration
	var ingController *ingressController

	framework := Framework{BaseName: "glbc"}

	BeforeEach(func() {
		// This test requires a GCE/GKE only cluster-addon
		SkipUnlessProviderIs("gce", "gke")
		framework.beforeEach()
		client = framework.Client
		ns = framework.Namespace.Name
		// Scaled down the existing Ingress controller so it doesn't interfere with the test.
		Expect(scaleRCByName(client, api.NamespaceSystem, lbContainerName, 0)).NotTo(HaveOccurred())
		addonDir = filepath.Join(
			testContext.RepoRoot, "cluster", "addons", "cluster-loadbalancing", "glbc")

		nsParts := strings.Split(ns, "-")
		ingController = &ingressController{
			ns: ns,
			// The UID in the namespace was generated by the master, so it's
			// global to the cluster.
			UID:            nsParts[len(nsParts)-1],
			rcPath:         filepath.Join(addonDir, "glbc-controller.yaml"),
			defaultSvcPath: filepath.Join(addonDir, "default-svc.yaml"),
			c:              client,
		}
		ingController.create()
		// If we somehow get the same namespace uid as someone else in this
		// gce project, just back off.
		Expect(ingController.cleanup(false)).NotTo(HaveOccurred())
		responseTimes = []time.Duration{}
		creationTimes = []time.Duration{}
	})

	AfterEach(func() {
		Logf("Average creation time %+v, health check time %+v", creationTimes, responseTimes)
		if CurrentGinkgoTestDescription().Failed {
			kubectlLogLBController(client, ns)
			Logf("\nOutput of kubectl describe ing:\n")
			desc, _ := runKubectl("describe", "ing", fmt.Sprintf("--namespace=%v", ns))
			Logf(desc)
		}
		// Delete all Ingress, then wait for the controller to cleanup.
		ings, err := client.Extensions().Ingress(ns).List(
			labels.Everything(), fields.Everything())
		if err != nil {
			Logf("WARNING: Failed to list ingress: %+v", err)
		} else {
			for _, ing := range ings.Items {
				Logf("Deleting ingress %v/%v", ing.Namespace, ing.Name)
				if err := client.Extensions().Ingress(ns).Delete(ing.Name, nil); err != nil {
					Logf("WARNING: Failed to delete ingress %v: %v", ing.Name, err)
				}
			}
		}
		pollErr := wait.Poll(5*time.Second, lbCleanupTimeout, func() (bool, error) {
			if err := ingController.cleanup(false); err != nil {
				Logf("Still waiting for glbc to cleanup: %v", err)
				return false, nil
			}
			return true, nil
		})
		// TODO: Remove this once issue #17802 is fixed
		Expect(scaleRCByName(client, ingController.rc.Namespace, ingController.rc.Name, 0)).NotTo(HaveOccurred())

		// If the controller failed to cleanup the test will fail, but we want to cleanup
		// resources before that.
		if pollErr != nil {
			if cleanupErr := ingController.cleanup(true); cleanupErr != nil {
				Logf("WARNING: Failed to cleanup resources %v", cleanupErr)
			}
			Failf("Failed to cleanup GCE L7 resources.")
		}
		// Restore the cluster Addon.
		Expect(scaleRCByName(client, api.NamespaceSystem, lbContainerName, 1)).NotTo(HaveOccurred())
		framework.afterEach()
		Logf("Successfully verified GCE L7 loadbalancer via Ingress.")
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
			Failf("Need more apps than Ingress")
		}
		appsPerIngress := numApps / numIng
		By(fmt.Sprintf("Creating %d rcs + svc, and %d apps per Ingress", numApps, appsPerIngress))
		for appID := 0; appID < numApps; appID = appID + appsPerIngress {
			// Creates appsPerIngress apps, then creates one Ingress with paths to all the apps.
			for j := appID; j < appID+appsPerIngress; j++ {
				createApp(client, ns, j)
			}
			createIngress(client, ns, appID, appsPerIngress)
		}

		ings, err := client.Extensions().Ingress(ns).List(
			labels.Everything(), fields.Everything())
		Expect(err).NotTo(HaveOccurred())

		for _, ing := range ings.Items {
			// Wait for the loadbalancer IP.
			start := time.Now()
			address, err := waitForIngressAddress(client, ing.Namespace, ing.Name, lbPollTimeout)
			if err != nil {
				Failf("Ingress failed to acquire an IP address within %v", lbPollTimeout)
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
				for _, p := range rules.IngressRuleValue.HTTP.Paths {
					route := fmt.Sprintf("http://%v%v", address, p.Path)
					Logf("Testing route %v host %v with simple GET", route, rules.Host)

					// Make sure the service node port is reachable
					Expect(curlServiceNodePort(client, ns, p.Backend.ServiceName, int(p.Backend.ServicePort.IntVal))).NotTo(HaveOccurred())

					GETStart := time.Now()
					var lastBody string
					pollErr := wait.Poll(lbPollInterval, lbPollTimeout, func() (bool, error) {
						var err error
						lastBody, err = simpleGET(http.DefaultClient, route, rules.Host)
						if err != nil {
							Logf("host %v path %v: %v", rules.Host, route, err)
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
						Failf(msg)
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
			Logf("WARNING: Average creation time is too high: %+v", creationTimes)
		}
		if !verifyHTTPGET {
			return
		}
		sort.Sort(timeSlice(responseTimes))
		perc50 = responseTimes[len(responseTimes)/2]
		if perc50 > expectedLBHealthCheckTime {
			Logf("WARNING: Average startup time is too high: %+v", responseTimes)
		}
	})
})

func curlServiceNodePort(client *client.Client, ns, name string, port int) error {
	// TODO: Curl all nodes?
	u, err := getNodePortURL(client, ns, name, port)
	if err != nil {
		return err
	}
	svcCurlBody, err := simpleGET(http.DefaultClient, u, "")
	if err != nil {
		return fmt.Errorf("Failed to curl service node port, body: %v\nerror %v", svcCurlBody, err)
	}
	Logf("Successfully curled service node port, body: %v", svcCurlBody)
	return nil
}
