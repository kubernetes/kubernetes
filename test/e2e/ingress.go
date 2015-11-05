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
	"sort"
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
	appPrefix         = "foo-app-"
	pathPrefix        = "foo"
	testImage         = "gcr.io/google_containers/n-way-http:1.0"
	httpContainerPort = 8080

	expectedLBCreationTime    = 7 * time.Minute
	expectedLBHealthCheckTime = 7 * time.Minute

	// On average it takes ~6 minutes for a single backend to come online.
	// We *don't* expect this poll to consistently take 15 minutes for every
	// Ingress as GCE is creating/checking backends in parallel, but at the
	// same time, we're not testing GCE startup latency. So give it enough
	// time, and fail if the average is too high.
	lbPollTimeout  = 15 * time.Minute
	lbPollInterval = 30 * time.Second

	// One can scale this test by tweaking numApps and numIng, the former will
	// create more RCs/Services and add them to a single Ingress, while the latter
	// will create smaller, more fragmented Ingresses. The numbers 4, 2 are chosen
	// arbitrarity, we want to test more than a single Ingress, and it should have
	// more than 1 url endpoint going to a service.
	numApps = 4
	numIng  = 2
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
		fmt.Sprintf("--regex=%v", regex), "-q", "--format=json").CombinedOutput()
	if err != nil {
		Failf("Error unmarshalling gcloud output: %v", err)
	}
	if err := json.Unmarshal([]byte(output), out); err != nil {
		Failf("Error unmarshalling gcloud output: %v", err)
	}
}

func checkLeakedResources() error {
	msg := ""
	// Check all resources #16636.
	beList := []compute.BackendService{}
	gcloudUnmarshal("backend-services", "k8s-be-[0-9]+", &beList)
	if len(beList) != 0 {
		for _, b := range beList {
			msg += fmt.Sprintf("%v\n", b.Name)
		}
		return fmt.Errorf("Found backend services:\n%v", msg)
	}
	fwList := []compute.ForwardingRule{}
	gcloudUnmarshal("forwarding-rules", "k8s-fw-.*", &fwList)
	if len(fwList) != 0 {
		for _, f := range fwList {
			msg += fmt.Sprintf("%v\n", f.Name)
		}
		return fmt.Errorf("Found forwarding rules:\n%v", msg)
	}
	return nil
}

var _ = Describe("GCE L7 LoadBalancer Controller", func() {
	// These variables are initialized after framework's beforeEach.
	var ns string
	var client *client.Client
	var responseTimes, creationTimes []time.Duration

	framework := Framework{BaseName: "glbc"}

	BeforeEach(func() {
		// This test requires a GCE/GKE only cluster-addon
		SkipUnlessProviderIs("gce", "gke")
		framework.beforeEach()
		client = framework.Client
		ns = framework.Namespace.Name
		Expect(waitForRCPodsRunning(client, "kube-system", "glbc")).NotTo(HaveOccurred())
		Expect(checkLeakedResources()).NotTo(HaveOccurred())
		responseTimes = []time.Duration{}
		creationTimes = []time.Duration{}
	})

	AfterEach(func() {
		framework.afterEach()
		err := wait.Poll(lbPollInterval, lbPollTimeout, func() (bool, error) {
			if err := checkLeakedResources(); err != nil {
				Logf("Still waiting for glbc to cleanup: %v", err)
				return false, nil
			}
			return true, nil
		})
		Logf("Average creation time %+v, health check time %+v", creationTimes, responseTimes)
		if err != nil {
			Failf("Failed to cleanup GCE L7 resources.")
		}
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
			Expect(err).NotTo(HaveOccurred())
			By(fmt.Sprintf("Found address %v for ingress %v, took %v to come online",
				address, ing.Name, time.Since(start)))
			creationTimes = append(creationTimes, time.Since(start))

			// Check that all rules respond to a simple GET.
			for _, rules := range ing.Spec.Rules {
				// As of Kubernetes 1.1 we only support HTTP Ingress.
				if rules.IngressRuleValue.HTTP == nil {
					continue
				}
				for _, p := range rules.IngressRuleValue.HTTP.Paths {
					route := fmt.Sprintf("http://%v%v", address, p.Path)
					Logf("Testing route %v host %v with simple GET", route, rules.Host)

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
						Failf("Failed to execute a successful GET within %v, Last response body for %v, host %v:\n%v\n\n%v",
							lbPollTimeout, route, rules.Host, lastBody, pollErr)
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
			Failf("Average creation time is too high: %+v", creationTimes)
		}
		sort.Sort(timeSlice(responseTimes))
		perc50 = responseTimes[len(responseTimes)/2]
		if perc50 > expectedLBHealthCheckTime {
			Failf("Average startup time is too high: %+v", responseTimes)
		}
	})
})
