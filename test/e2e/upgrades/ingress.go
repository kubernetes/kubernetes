/*
Copyright 2017 The Kubernetes Authors.

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

package upgrades

import (
	"encoding/json"
	"fmt"
	"net/http"
	"path/filepath"
	"reflect"

	"github.com/davecgh/go-spew/spew"
	. "github.com/onsi/ginkgo"

	compute "google.golang.org/api/compute/v1"
	extensions "k8s.io/api/extensions/v1beta1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/pkg/util/version"
	"k8s.io/kubernetes/test/e2e/framework"
)

// Dependent on "static-ip-2" manifests
const path = "foo"
const host = "ingress.test.com"

// IngressUpgradeTest adapts the Ingress e2e for upgrade testing
type IngressUpgradeTest struct {
	gceController *framework.GCEIngressController
	// holds GCP resources pre-upgrade
	resourceStore *GCPResourceStore
	jig           *framework.IngressTestJig
	httpClient    *http.Client
	ip            string
	ipName        string
	skipSSLCheck  bool
}

// GCPResourceStore keeps track of the GCP resources spun up by an ingress.
// Note: Fields are exported so that we can utilize reflection.
type GCPResourceStore struct {
	Fw      *compute.Firewall
	FwdList []*compute.ForwardingRule
	UmList  []*compute.UrlMap
	TpList  []*compute.TargetHttpProxy
	TpsList []*compute.TargetHttpsProxy
	SslList []*compute.SslCertificate
	BeList  []*compute.BackendService
	Ip      *compute.Address
	IgList  []*compute.InstanceGroup
}

func (IngressUpgradeTest) Name() string { return "ingress-upgrade" }

// Setup creates a GLBC, allocates an ip, and an ingress resource,
// then waits for a successful connectivity check to the ip.
// Also keeps track of all load balancer resources for cross-checking
// during an IngressUpgrade.
func (t *IngressUpgradeTest) Setup(f *framework.Framework) {
	framework.SkipUnlessProviderIs("gce", "gke")

	// jig handles all Kubernetes testing logic
	jig := framework.NewIngressTestJig(f.ClientSet)

	ns := f.Namespace

	// gceController handles all cloud testing logic
	gceController := &framework.GCEIngressController{
		Ns:     ns.Name,
		Client: jig.Client,
		Cloud:  framework.TestContext.CloudConfig,
	}
	framework.ExpectNoError(gceController.Init())

	t.gceController = gceController
	t.jig = jig
	t.httpClient = framework.BuildInsecureClient(framework.IngressReqTimeout)

	// Allocate a static-ip for the Ingress, this IP is cleaned up via CleanupGCEIngressController
	t.ipName = fmt.Sprintf("%s-static-ip", ns.Name)
	t.ip = t.gceController.CreateStaticIP(t.ipName)

	// Create a working basic Ingress
	By(fmt.Sprintf("allocated static ip %v: %v through the GCE cloud provider", t.ipName, t.ip))
	jig.CreateIngress(filepath.Join(framework.IngressManifestPath, "static-ip-2"), ns.Name, map[string]string{
		framework.IngressStaticIPKey:  t.ipName,
		framework.IngressAllowHTTPKey: "false",
	}, map[string]string{})
	t.jig.SetHTTPS("tls-secret", "ingress.test.com")

	By("waiting for Ingress to come up with ip: " + t.ip)
	framework.ExpectNoError(framework.PollURL(fmt.Sprintf("https://%v/%v", t.ip, path), host, framework.LoadBalancerPollTimeout, t.jig.PollInterval, t.httpClient, false))

	By("keeping track of GCP resources created by Ingress")
	t.resourceStore = &GCPResourceStore{}
	t.populateGCPResourceStore(t.resourceStore)
}

// Test waits for the upgrade to complete, and then verifies
// with a connectvity check to the loadbalancer ip.
func (t *IngressUpgradeTest) Test(f *framework.Framework, done <-chan struct{}, upgrade UpgradeType) {
	switch upgrade {
	case MasterUpgrade:
		// Restarting the ingress controller shouldn't disrupt a steady state
		// Ingress. Restarting the ingress controller and deleting ingresses
		// while it's down will leak cloud resources, because the ingress
		// controller doesn't checkpoint to disk.
		t.verify(f, done, true)
	case IngressUpgrade:
		t.verify(f, done, true)
	default:
		// Currently ingress gets disrupted across node upgrade, because endpoints
		// get killed and we don't have any guarantees that 2 nodes don't overlap
		// their upgrades (even on cloud platforms like GCE, because VM level
		// rolling upgrades are not Kubernetes aware).
		t.verify(f, done, false)
	}
}

// Teardown cleans up any remaining resources.
func (t *IngressUpgradeTest) Teardown(f *framework.Framework) {
	if CurrentGinkgoTestDescription().Failed {
		framework.DescribeIng(t.gceController.Ns)
	}

	if t.jig.Ingress != nil {
		By("Deleting ingress")
		t.jig.TryDeleteIngress()
	} else {
		By("No ingress created, no cleanup necessary")
	}

	By("Cleaning up cloud resources")
	framework.ExpectNoError(t.gceController.CleanupGCEIngressController())
}

// Skip checks if the test or part of the test should be skipped.
func (t *IngressUpgradeTest) Skip(upgCtx UpgradeContext) bool {
	sslNameChangeVersion, err := version.ParseGeneric("v1.10.0")
	framework.ExpectNoError(err)
	var hasVersionBelow, hasVersionAboveOrEqual bool
	for _, v := range upgCtx.Versions {
		if v.Version.LessThan(sslNameChangeVersion) {
			hasVersionBelow = true
			continue
		}
		hasVersionAboveOrEqual = true
	}
	// Skip SSL certificates check if k8s version changes between 1.10-
	// and 1.10+ because the naming scheme has changed.
	if hasVersionBelow && hasVersionAboveOrEqual {
		t.skipSSLCheck = true
	}
	return false
}

func (t *IngressUpgradeTest) verify(f *framework.Framework, done <-chan struct{}, testDuringDisruption bool) {
	if testDuringDisruption {
		By("continuously hitting the Ingress IP")
		wait.Until(func() {
			framework.ExpectNoError(framework.PollURL(fmt.Sprintf("https://%v/%v", t.ip, path), host, framework.LoadBalancerPollTimeout, t.jig.PollInterval, t.httpClient, false))
		}, t.jig.PollInterval, done)
	} else {
		By("waiting for upgrade to finish without checking if Ingress remains up")
		<-done
	}
	By("hitting the Ingress IP " + t.ip)
	framework.ExpectNoError(framework.PollURL(fmt.Sprintf("https://%v/%v", t.ip, path), host, framework.LoadBalancerPollTimeout, t.jig.PollInterval, t.httpClient, false))

	// We want to manually trigger a sync because then we can easily verify
	// a correct sync completed after update.
	By("updating ingress spec to manually trigger a sync")
	t.jig.Update(func(ing *extensions.Ingress) {
		ing.Spec.Rules[0].IngressRuleValue.HTTP.Paths = append(
			ing.Spec.Rules[0].IngressRuleValue.HTTP.Paths,
			extensions.HTTPIngressPath{
				Path: "/test",
				// Note: Dependant on using "static-ip-2" manifest.
				Backend: ing.Spec.Rules[0].IngressRuleValue.HTTP.Paths[0].Backend,
			})
	})
	// WaitForIngress() tests that all paths are pinged, which is how we know
	// everything is synced with the cloud.
	t.jig.WaitForIngress(false)
	By("comparing GCP resources post-upgrade")
	postUpgradeResourceStore := &GCPResourceStore{}
	t.populateGCPResourceStore(postUpgradeResourceStore)

	// Stub out the number of instances as that is out of Ingress controller's control.
	for _, ig := range t.resourceStore.IgList {
		ig.Size = 0
	}
	for _, ig := range postUpgradeResourceStore.IgList {
		ig.Size = 0
	}
	// Stub out compute.SslCertificates in case we know it will change during an upgrade/downgrade.
	if t.skipSSLCheck {
		t.resourceStore.SslList = nil
		postUpgradeResourceStore.SslList = nil
	}

	// TODO(rramkumar): Remove this when GLBC v1.2.0 is released.
	t.resourceStore.BeList = nil
	postUpgradeResourceStore.BeList = nil

	framework.ExpectNoError(compareGCPResourceStores(t.resourceStore, postUpgradeResourceStore, func(v1 reflect.Value, v2 reflect.Value) error {
		i1 := v1.Interface()
		i2 := v2.Interface()
		// Skip verifying the UrlMap since we did that via WaitForIngress()
		if !reflect.DeepEqual(i1, i2) && (v1.Type() != reflect.TypeOf([]*compute.UrlMap{})) {
			return spew.Errorf("resources after ingress upgrade were different:\n Pre-Upgrade: %#v\n Post-Upgrade: %#v", i1, i2)
		}
		return nil
	}))
}

func (t *IngressUpgradeTest) populateGCPResourceStore(resourceStore *GCPResourceStore) {
	cont := t.gceController
	resourceStore.Fw = cont.GetFirewallRule()
	resourceStore.FwdList = cont.ListGlobalForwardingRules()
	resourceStore.UmList = cont.ListUrlMaps()
	resourceStore.TpList = cont.ListTargetHttpProxies()
	resourceStore.TpsList = cont.ListTargetHttpsProxies()
	resourceStore.SslList = cont.ListSslCertificates()
	resourceStore.BeList = cont.ListGlobalBackendServices()
	resourceStore.Ip = cont.GetGlobalAddress(t.ipName)
	resourceStore.IgList = cont.ListInstanceGroups()
}

func compareGCPResourceStores(rs1 *GCPResourceStore, rs2 *GCPResourceStore, compare func(v1 reflect.Value, v2 reflect.Value) error) error {
	// Before we do a comparison, remove the ServerResponse field from the
	// Compute API structs. This is needed because two objects could be the same
	// but their ServerResponse will be different if they were populated through
	// separate API calls.
	rs1Json, _ := json.Marshal(rs1)
	rs2Json, _ := json.Marshal(rs2)
	rs1New := &GCPResourceStore{}
	rs2New := &GCPResourceStore{}
	json.Unmarshal(rs1Json, rs1New)
	json.Unmarshal(rs2Json, rs2New)

	// Iterate through struct fields and perform equality checks on the fields.
	// We do this rather than performing a deep equal on the struct itself because
	// it is easier to log which field, if any, is not the same.
	rs1V := reflect.ValueOf(*rs1New)
	rs2V := reflect.ValueOf(*rs2New)
	for i := 0; i < rs1V.NumField(); i++ {
		if err := compare(rs1V.Field(i), rs2V.Field(i)); err != nil {
			return err
		}
	}
	return nil
}
