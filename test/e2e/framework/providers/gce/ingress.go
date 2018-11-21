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

package gce

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"net/http"
	"os/exec"
	"strconv"
	"strings"
	"time"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	compute "google.golang.org/api/compute/v1"
	"google.golang.org/api/googleapi"
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	utilexec "k8s.io/utils/exec"
)

const (
	// Name of the config-map and key the ingress controller stores its uid in.
	uidConfigMap = "ingress-uid"
	uidKey       = "uid"

	// all cloud resources created by the ingress controller start with this
	// prefix.
	k8sPrefix = "k8s-"

	// clusterDelimiter is the delimiter used by the ingress controller
	// to split uid from other naming/metadata.
	clusterDelimiter = "--"

	// Cloud resources created by the ingress controller older than this
	// are automatically purged to prevent running out of quota.
	// TODO(37335): write soak tests and bump this up to a week.
	maxAge = 48 * time.Hour

	// GCE only allows names < 64 characters, and the loadbalancer controller inserts
	// a single character of padding.
	nameLenLimit = 62
)

// GCEIngressController manages implementation details of Ingress on GCE/GKE.
type GCEIngressController struct {
	Ns           string
	rcPath       string
	UID          string
	staticIPName string
	rc           *v1.ReplicationController
	svc          *v1.Service
	Client       clientset.Interface
	Cloud        framework.CloudConfig
}

func (cont *GCEIngressController) CleanupGCEIngressController() error {
	return cont.CleanupGCEIngressControllerWithTimeout(framework.LoadBalancerCleanupTimeout)
}

// CleanupGCEIngressControllerWithTimeout calls the GCEIngressController.Cleanup(false)
// followed with deleting the static ip, and then a final GCEIngressController.Cleanup(true)
func (cont *GCEIngressController) CleanupGCEIngressControllerWithTimeout(timeout time.Duration) error {
	pollErr := wait.Poll(5*time.Second, timeout, func() (bool, error) {
		if err := cont.Cleanup(false); err != nil {
			framework.Logf("Monitoring glbc's cleanup of gce resources:\n%v", err)
			return false, nil
		}
		return true, nil
	})

	// Always try to cleanup even if pollErr == nil, because the cleanup
	// routine also purges old leaked resources based on creation timestamp.
	By("Performing final delete of any remaining resources")
	if cleanupErr := cont.Cleanup(true); cleanupErr != nil {
		By(fmt.Sprintf("WARNING: possibly leaked resources: %v\n", cleanupErr))
	} else {
		By("No resources leaked.")
	}

	// Static-IP allocated on behalf of the test, never deleted by the
	// controller. Delete this IP only after the controller has had a chance
	// to cleanup or it might interfere with the controller, causing it to
	// throw out confusing events.
	if ipErr := wait.Poll(5*time.Second, 1*time.Minute, func() (bool, error) {
		if err := cont.deleteStaticIPs(); err != nil {
			framework.Logf("Failed to delete static-ip: %v\n", err)
			return false, nil
		}
		return true, nil
	}); ipErr != nil {
		// If this is a persistent error, the suite will fail when we run out
		// of quota anyway.
		By(fmt.Sprintf("WARNING: possibly leaked static IP: %v\n", ipErr))
	}

	// Logging that the GLBC failed to cleanup GCE resources on ingress deletion
	// See kubernetes/ingress#431
	if pollErr != nil {
		return fmt.Errorf("error: L7 controller failed to delete all cloud resources on time. %v", pollErr)
	}
	return nil
}

func (cont *GCEIngressController) getL7AddonUID() (string, error) {
	framework.Logf("Retrieving UID from config map: %v/%v", metav1.NamespaceSystem, uidConfigMap)
	cm, err := cont.Client.CoreV1().ConfigMaps(metav1.NamespaceSystem).Get(uidConfigMap, metav1.GetOptions{})
	if err != nil {
		return "", err
	}
	if uid, ok := cm.Data[uidKey]; ok {
		return uid, nil
	}
	return "", fmt.Errorf("Could not find cluster UID for L7 addon pod")
}

func (cont *GCEIngressController) ListGlobalForwardingRules() []*compute.ForwardingRule {
	gceCloud := cont.Cloud.Provider.(*Provider).gceCloud
	fwdList := []*compute.ForwardingRule{}
	l, err := gceCloud.ListGlobalForwardingRules()
	Expect(err).NotTo(HaveOccurred())
	for _, fwd := range l {
		if cont.isOwned(fwd.Name) {
			fwdList = append(fwdList, fwd)
		}
	}
	return fwdList
}

func (cont *GCEIngressController) deleteForwardingRule(del bool) string {
	msg := ""
	fwList := []compute.ForwardingRule{}
	for _, regex := range []string{fmt.Sprintf("%vfw-.*%v.*", k8sPrefix, clusterDelimiter), fmt.Sprintf("%vfws-.*%v.*", k8sPrefix, clusterDelimiter)} {
		gcloudComputeResourceList("forwarding-rules", regex, cont.Cloud.ProjectID, &fwList)
		if len(fwList) == 0 {
			continue
		}
		for _, f := range fwList {
			if !cont.canDelete(f.Name, f.CreationTimestamp, del) {
				continue
			}
			if del {
				GcloudComputeResourceDelete("forwarding-rules", f.Name, cont.Cloud.ProjectID, "--global")
			} else {
				msg += fmt.Sprintf("%v (forwarding rule)\n", f.Name)
			}
		}
	}
	return msg
}

func (cont *GCEIngressController) GetGlobalAddress(ipName string) *compute.Address {
	gceCloud := cont.Cloud.Provider.(*Provider).gceCloud
	ip, err := gceCloud.GetGlobalAddress(ipName)
	Expect(err).NotTo(HaveOccurred())
	return ip
}

func (cont *GCEIngressController) deleteAddresses(del bool) string {
	msg := ""
	ipList := []compute.Address{}
	regex := fmt.Sprintf("%vfw-.*%v.*", k8sPrefix, clusterDelimiter)
	gcloudComputeResourceList("addresses", regex, cont.Cloud.ProjectID, &ipList)
	if len(ipList) != 0 {
		for _, ip := range ipList {
			if !cont.canDelete(ip.Name, ip.CreationTimestamp, del) {
				continue
			}
			if del {
				GcloudComputeResourceDelete("addresses", ip.Name, cont.Cloud.ProjectID, "--global")
			} else {
				msg += fmt.Sprintf("%v (static-ip)\n", ip.Name)
			}
		}
	}
	return msg
}

func (cont *GCEIngressController) ListTargetHttpProxies() []*compute.TargetHttpProxy {
	gceCloud := cont.Cloud.Provider.(*Provider).gceCloud
	tpList := []*compute.TargetHttpProxy{}
	l, err := gceCloud.ListTargetHTTPProxies()
	Expect(err).NotTo(HaveOccurred())
	for _, tp := range l {
		if cont.isOwned(tp.Name) {
			tpList = append(tpList, tp)
		}
	}
	return tpList
}

func (cont *GCEIngressController) ListTargetHttpsProxies() []*compute.TargetHttpsProxy {
	gceCloud := cont.Cloud.Provider.(*Provider).gceCloud
	tpsList := []*compute.TargetHttpsProxy{}
	l, err := gceCloud.ListTargetHTTPSProxies()
	Expect(err).NotTo(HaveOccurred())
	for _, tps := range l {
		if cont.isOwned(tps.Name) {
			tpsList = append(tpsList, tps)
		}
	}
	return tpsList
}

func (cont *GCEIngressController) deleteTargetProxy(del bool) string {
	msg := ""
	tpList := []compute.TargetHttpProxy{}
	regex := fmt.Sprintf("%vtp-.*%v.*", k8sPrefix, clusterDelimiter)
	gcloudComputeResourceList("target-http-proxies", regex, cont.Cloud.ProjectID, &tpList)
	if len(tpList) != 0 {
		for _, t := range tpList {
			if !cont.canDelete(t.Name, t.CreationTimestamp, del) {
				continue
			}
			if del {
				GcloudComputeResourceDelete("target-http-proxies", t.Name, cont.Cloud.ProjectID)
			} else {
				msg += fmt.Sprintf("%v (target-http-proxy)\n", t.Name)
			}
		}
	}
	tpsList := []compute.TargetHttpsProxy{}
	regex = fmt.Sprintf("%vtps-.*%v.*", k8sPrefix, clusterDelimiter)
	gcloudComputeResourceList("target-https-proxies", regex, cont.Cloud.ProjectID, &tpsList)
	if len(tpsList) != 0 {
		for _, t := range tpsList {
			if !cont.canDelete(t.Name, t.CreationTimestamp, del) {
				continue
			}
			if del {
				GcloudComputeResourceDelete("target-https-proxies", t.Name, cont.Cloud.ProjectID)
			} else {
				msg += fmt.Sprintf("%v (target-https-proxy)\n", t.Name)
			}
		}
	}
	return msg
}

func (cont *GCEIngressController) ListUrlMaps() []*compute.UrlMap {
	gceCloud := cont.Cloud.Provider.(*Provider).gceCloud
	umList := []*compute.UrlMap{}
	l, err := gceCloud.ListURLMaps()
	Expect(err).NotTo(HaveOccurred())
	for _, um := range l {
		if cont.isOwned(um.Name) {
			umList = append(umList, um)
		}
	}
	return umList
}

func (cont *GCEIngressController) deleteURLMap(del bool) (msg string) {
	gceCloud := cont.Cloud.Provider.(*Provider).gceCloud
	umList, err := gceCloud.ListURLMaps()
	if err != nil {
		if cont.isHTTPErrorCode(err, http.StatusNotFound) {
			return msg
		}
		return fmt.Sprintf("Failed to list url maps: %v", err)
	}
	if len(umList) == 0 {
		return msg
	}
	for _, um := range umList {
		if !cont.canDelete(um.Name, um.CreationTimestamp, del) {
			continue
		}
		if del {
			framework.Logf("Deleting url-map: %s", um.Name)
			if err := gceCloud.DeleteURLMap(um.Name); err != nil &&
				!cont.isHTTPErrorCode(err, http.StatusNotFound) {
				msg += fmt.Sprintf("Failed to delete url map %v\n", um.Name)
			}
		} else {
			msg += fmt.Sprintf("%v (url-map)\n", um.Name)
		}
	}
	return msg
}

func (cont *GCEIngressController) ListGlobalBackendServices() []*compute.BackendService {
	gceCloud := cont.Cloud.Provider.(*Provider).gceCloud
	beList := []*compute.BackendService{}
	l, err := gceCloud.ListGlobalBackendServices()
	Expect(err).NotTo(HaveOccurred())
	for _, be := range l {
		if cont.isOwned(be.Name) {
			beList = append(beList, be)
		}
	}
	return beList
}

func (cont *GCEIngressController) deleteBackendService(del bool) (msg string) {
	gceCloud := cont.Cloud.Provider.(*Provider).gceCloud
	beList, err := gceCloud.ListGlobalBackendServices()
	if err != nil {
		if cont.isHTTPErrorCode(err, http.StatusNotFound) {
			return msg
		}
		return fmt.Sprintf("Failed to list backend services: %v", err)
	}
	if len(beList) == 0 {
		framework.Logf("No backend services found")
		return msg
	}
	for _, be := range beList {
		if !cont.canDelete(be.Name, be.CreationTimestamp, del) {
			continue
		}
		if del {
			framework.Logf("Deleting backed-service: %s", be.Name)
			if err := gceCloud.DeleteGlobalBackendService(be.Name); err != nil &&
				!cont.isHTTPErrorCode(err, http.StatusNotFound) {
				msg += fmt.Sprintf("Failed to delete backend service %v: %v\n", be.Name, err)
			}
		} else {
			msg += fmt.Sprintf("%v (backend-service)\n", be.Name)
		}
	}
	return msg
}

func (cont *GCEIngressController) deleteHTTPHealthCheck(del bool) (msg string) {
	gceCloud := cont.Cloud.Provider.(*Provider).gceCloud
	hcList, err := gceCloud.ListHTTPHealthChecks()
	if err != nil {
		if cont.isHTTPErrorCode(err, http.StatusNotFound) {
			return msg
		}
		return fmt.Sprintf("Failed to list HTTP health checks: %v", err)
	}
	if len(hcList) == 0 {
		return msg
	}
	for _, hc := range hcList {
		if !cont.canDelete(hc.Name, hc.CreationTimestamp, del) {
			continue
		}
		if del {
			framework.Logf("Deleting http-health-check: %s", hc.Name)
			if err := gceCloud.DeleteHTTPHealthCheck(hc.Name); err != nil &&
				!cont.isHTTPErrorCode(err, http.StatusNotFound) {
				msg += fmt.Sprintf("Failed to delete HTTP health check %v\n", hc.Name)
			}
		} else {
			msg += fmt.Sprintf("%v (http-health-check)\n", hc.Name)
		}
	}
	return msg
}

func (cont *GCEIngressController) ListSslCertificates() []*compute.SslCertificate {
	gceCloud := cont.Cloud.Provider.(*Provider).gceCloud
	sslList := []*compute.SslCertificate{}
	l, err := gceCloud.ListSslCertificates()
	Expect(err).NotTo(HaveOccurred())
	for _, ssl := range l {
		if cont.isOwned(ssl.Name) {
			sslList = append(sslList, ssl)
		}
	}
	return sslList
}

func (cont *GCEIngressController) deleteSSLCertificate(del bool) (msg string) {
	gceCloud := cont.Cloud.Provider.(*Provider).gceCloud
	sslList, err := gceCloud.ListSslCertificates()
	if err != nil {
		if cont.isHTTPErrorCode(err, http.StatusNotFound) {
			return msg
		}
		return fmt.Sprintf("Failed to list ssl certificates: %v", err)
	}
	if len(sslList) != 0 {
		for _, s := range sslList {
			if !cont.canDelete(s.Name, s.CreationTimestamp, del) {
				continue
			}
			if del {
				framework.Logf("Deleting ssl-certificate: %s", s.Name)
				if err := gceCloud.DeleteSslCertificate(s.Name); err != nil &&
					!cont.isHTTPErrorCode(err, http.StatusNotFound) {
					msg += fmt.Sprintf("Failed to delete ssl certificates: %v\n", s.Name)
				}
			} else {
				msg += fmt.Sprintf("%v (ssl-certificate)\n", s.Name)
			}
		}
	}
	return msg
}

func (cont *GCEIngressController) ListInstanceGroups() []*compute.InstanceGroup {
	gceCloud := cont.Cloud.Provider.(*Provider).gceCloud
	igList := []*compute.InstanceGroup{}
	l, err := gceCloud.ListInstanceGroups(cont.Cloud.Zone)
	Expect(err).NotTo(HaveOccurred())
	for _, ig := range l {
		if cont.isOwned(ig.Name) {
			igList = append(igList, ig)
		}
	}
	return igList
}

func (cont *GCEIngressController) deleteInstanceGroup(del bool) (msg string) {
	gceCloud := cont.Cloud.Provider.(*Provider).gceCloud
	// TODO: E2E cloudprovider has only 1 zone, but the cluster can have many.
	// We need to poll on all IGs across all zones.
	igList, err := gceCloud.ListInstanceGroups(cont.Cloud.Zone)
	if err != nil {
		if cont.isHTTPErrorCode(err, http.StatusNotFound) {
			return msg
		}
		return fmt.Sprintf("Failed to list instance groups: %v", err)
	}
	if len(igList) == 0 {
		return msg
	}
	for _, ig := range igList {
		if !cont.canDelete(ig.Name, ig.CreationTimestamp, del) {
			continue
		}
		if del {
			framework.Logf("Deleting instance-group: %s", ig.Name)
			if err := gceCloud.DeleteInstanceGroup(ig.Name, cont.Cloud.Zone); err != nil &&
				!cont.isHTTPErrorCode(err, http.StatusNotFound) {
				msg += fmt.Sprintf("Failed to delete instance group %v\n", ig.Name)
			}
		} else {
			msg += fmt.Sprintf("%v (instance-group)\n", ig.Name)
		}
	}
	return msg
}

func (cont *GCEIngressController) deleteNetworkEndpointGroup(del bool) (msg string) {
	gceCloud := cont.Cloud.Provider.(*Provider).gceCloud
	// TODO: E2E cloudprovider has only 1 zone, but the cluster can have many.
	// We need to poll on all NEGs across all zones.
	negList, err := gceCloud.ListNetworkEndpointGroup(cont.Cloud.Zone)
	if err != nil {
		if cont.isHTTPErrorCode(err, http.StatusNotFound) {
			return msg
		}
		// Do not return error as NEG is still alpha.
		framework.Logf("Failed to list network endpoint group: %v", err)
		return msg
	}
	if len(negList) == 0 {
		return msg
	}
	for _, neg := range negList {
		if !cont.canDeleteNEG(neg.Name, neg.CreationTimestamp, del) {
			continue
		}
		if del {
			framework.Logf("Deleting network-endpoint-group: %s", neg.Name)
			if err := gceCloud.DeleteNetworkEndpointGroup(neg.Name, cont.Cloud.Zone); err != nil &&
				!cont.isHTTPErrorCode(err, http.StatusNotFound) {
				msg += fmt.Sprintf("Failed to delete network endpoint group %v\n", neg.Name)
			}
		} else {
			msg += fmt.Sprintf("%v (network-endpoint-group)\n", neg.Name)
		}
	}
	return msg
}

// canDelete returns true if either the name ends in a suffix matching this
// controller's UID, or the creationTimestamp exceeds the maxAge and del is set
// to true. Always returns false if the name doesn't match that we expect for
// Ingress cloud resources.
func (cont *GCEIngressController) canDelete(resourceName, creationTimestamp string, delOldResources bool) bool {
	// ignore everything not created by an ingress controller.
	splitName := strings.Split(resourceName, clusterDelimiter)
	if !strings.HasPrefix(resourceName, k8sPrefix) || len(splitName) != 2 {
		return false
	}

	// Resources created by the GLBC have a "0"" appended to the end if truncation
	// occurred. Removing the zero allows the following match.
	truncatedClusterUID := splitName[1]
	if len(truncatedClusterUID) >= 1 && strings.HasSuffix(truncatedClusterUID, "0") {
		truncatedClusterUID = truncatedClusterUID[:len(truncatedClusterUID)-1]
	}

	// always delete things that are created by the current ingress controller.
	// Because of resource name truncation, this looks for a common prefix
	if strings.HasPrefix(cont.UID, truncatedClusterUID) {
		return true
	}
	if !delOldResources {
		return false
	}
	return canDeleteWithTimestamp(resourceName, creationTimestamp)
}

// isOwned returns true if the resourceName ends in a suffix matching this
// controller UID.
func (cont *GCEIngressController) isOwned(resourceName string) bool {
	return cont.canDelete(resourceName, "", false)
}

// canDeleteNEG returns true if either the name contains this controller's UID,
// or the creationTimestamp exceeds the maxAge and del is set to true.
func (cont *GCEIngressController) canDeleteNEG(resourceName, creationTimestamp string, delOldResources bool) bool {
	if !strings.HasPrefix(resourceName, "k8s") {
		return false
	}

	if strings.Contains(resourceName, cont.UID) {
		return true
	}

	if !delOldResources {
		return false
	}

	return canDeleteWithTimestamp(resourceName, creationTimestamp)
}

func canDeleteWithTimestamp(resourceName, creationTimestamp string) bool {
	createdTime, err := time.Parse(time.RFC3339, creationTimestamp)
	if err != nil {
		framework.Logf("WARNING: Failed to parse creation timestamp %v for %v: %v", creationTimestamp, resourceName, err)
		return false
	}
	if time.Since(createdTime) > maxAge {
		framework.Logf("%v created on %v IS too old", resourceName, creationTimestamp)
		return true
	}
	return false
}

// GetFirewallRuleName returns the name of the firewall used for the GCEIngressController.
func (cont *GCEIngressController) GetFirewallRuleName() string {
	return fmt.Sprintf("%vfw-l7%v%v", k8sPrefix, clusterDelimiter, cont.UID)
}

// GetFirewallRule returns the firewall used by the GCEIngressController.
// Causes a fatal error incase of an error.
// TODO: Rename this to GetFirewallRuleOrDie and similarly rename all other
// methods here to be consistent with rest of the code in this repo.
func (cont *GCEIngressController) GetFirewallRule() *compute.Firewall {
	fw, err := cont.GetFirewallRuleOrError()
	Expect(err).NotTo(HaveOccurred())
	return fw
}

// GetFirewallRule returns the firewall used by the GCEIngressController.
// Returns an error if that fails.
// TODO: Rename this to GetFirewallRule when the above method with that name is renamed.
func (cont *GCEIngressController) GetFirewallRuleOrError() (*compute.Firewall, error) {
	gceCloud := cont.Cloud.Provider.(*Provider).gceCloud
	fwName := cont.GetFirewallRuleName()
	return gceCloud.GetFirewall(fwName)
}

func (cont *GCEIngressController) deleteFirewallRule(del bool) (msg string) {
	fwList := []compute.Firewall{}
	regex := fmt.Sprintf("%vfw-l7%v.*", k8sPrefix, clusterDelimiter)
	gcloudComputeResourceList("firewall-rules", regex, cont.Cloud.ProjectID, &fwList)
	if len(fwList) != 0 {
		for _, f := range fwList {
			if !cont.canDelete(f.Name, f.CreationTimestamp, del) {
				continue
			}
			if del {
				GcloudComputeResourceDelete("firewall-rules", f.Name, cont.Cloud.ProjectID)
			} else {
				msg += fmt.Sprintf("%v (firewall rule)\n", f.Name)
			}
		}
	}
	return msg
}

func (cont *GCEIngressController) isHTTPErrorCode(err error, code int) bool {
	apiErr, ok := err.(*googleapi.Error)
	return ok && apiErr.Code == code
}

// BackendServiceUsingNEG returns true only if all global backend service with matching nodeports pointing to NEG as backend
func (cont *GCEIngressController) BackendServiceUsingNEG(svcPorts map[string]v1.ServicePort) (bool, error) {
	return cont.backendMode(svcPorts, "networkEndpointGroups")
}

// BackendServiceUsingIG returns true only if all global backend service with matching svcPorts pointing to IG as backend
func (cont *GCEIngressController) BackendServiceUsingIG(svcPorts map[string]v1.ServicePort) (bool, error) {
	return cont.backendMode(svcPorts, "instanceGroups")
}

func (cont *GCEIngressController) backendMode(svcPorts map[string]v1.ServicePort, keyword string) (bool, error) {
	gceCloud := cont.Cloud.Provider.(*Provider).gceCloud
	beList, err := gceCloud.ListGlobalBackendServices()
	if err != nil {
		return false, fmt.Errorf("failed to list backend services: %v", err)
	}

	hcList, err := gceCloud.ListHealthChecks()
	if err != nil {
		return false, fmt.Errorf("failed to list health checks: %v", err)
	}

	uid := cont.UID
	if len(uid) > 8 {
		uid = uid[:8]
	}

	matchingBackendService := 0
	for svcName, sp := range svcPorts {
		match := false
		bsMatch := &compute.BackendService{}
		// Non-NEG BackendServices are named with the Nodeport in the name.
		// NEG BackendServices' names contain the a sha256 hash of a string.
		negString := strings.Join([]string{uid, cont.Ns, svcName, fmt.Sprintf("%v", sp.Port)}, ";")
		negHash := fmt.Sprintf("%x", sha256.Sum256([]byte(negString)))[:8]
		for _, bs := range beList {
			if strings.Contains(bs.Name, strconv.Itoa(int(sp.NodePort))) ||
				strings.Contains(bs.Name, negHash) {
				match = true
				bsMatch = bs
				matchingBackendService += 1
				break
			}
		}

		if match {
			for _, be := range bsMatch.Backends {
				if !strings.Contains(be.Group, keyword) {
					return false, nil
				}
			}

			// Check that the correct HealthCheck exists for the BackendService
			hcMatch := false
			for _, hc := range hcList {
				if hc.Name == bsMatch.Name {
					hcMatch = true
					break
				}
			}

			if !hcMatch {
				return false, fmt.Errorf("missing healthcheck for backendservice: %v", bsMatch.Name)
			}
		}
	}
	return matchingBackendService == len(svcPorts), nil
}

// Cleanup cleans up cloud resources.
// If del is false, it simply reports existing resources without deleting them.
// If dle is true, it deletes resources it finds acceptable (see canDelete func).
func (cont *GCEIngressController) Cleanup(del bool) error {
	// Ordering is important here because we cannot delete resources that other
	// resources hold references to.
	errMsg := cont.deleteForwardingRule(del)
	// Static IPs are named after forwarding rules.
	errMsg += cont.deleteAddresses(del)

	errMsg += cont.deleteTargetProxy(del)
	errMsg += cont.deleteURLMap(del)
	errMsg += cont.deleteBackendService(del)
	errMsg += cont.deleteHTTPHealthCheck(del)

	errMsg += cont.deleteInstanceGroup(del)
	errMsg += cont.deleteNetworkEndpointGroup(del)
	errMsg += cont.deleteFirewallRule(del)
	errMsg += cont.deleteSSLCertificate(del)

	// TODO: Verify instance-groups, issue #16636. Gcloud mysteriously barfs when told
	// to unmarshal instance groups into the current vendored gce-client's understanding
	// of the struct.
	if errMsg == "" {
		return nil
	}
	return fmt.Errorf(errMsg)
}

// Init initializes the GCEIngressController with an UID
func (cont *GCEIngressController) Init() error {
	uid, err := cont.getL7AddonUID()
	if err != nil {
		return err
	}
	cont.UID = uid
	// There's a name limit imposed by GCE. The controller will truncate.
	testName := fmt.Sprintf("k8s-fw-foo-app-X-%v--%v", cont.Ns, cont.UID)
	if len(testName) > nameLenLimit {
		framework.Logf("WARNING: test name including cluster UID: %v is over the GCE limit of %v", testName, nameLenLimit)
	} else {
		framework.Logf("Detected cluster UID %v", cont.UID)
	}
	return nil
}

// CreateStaticIP allocates a random static ip with the given name. Returns a string
// representation of the ip. Caller is expected to manage cleanup of the ip by
// invoking deleteStaticIPs.
func (cont *GCEIngressController) CreateStaticIP(name string) string {
	gceCloud := cont.Cloud.Provider.(*Provider).gceCloud
	addr := &compute.Address{Name: name}
	if err := gceCloud.ReserveGlobalAddress(addr); err != nil {
		if delErr := gceCloud.DeleteGlobalAddress(name); delErr != nil {
			if cont.isHTTPErrorCode(delErr, http.StatusNotFound) {
				framework.Logf("Static ip with name %v was not allocated, nothing to delete", name)
			} else {
				framework.Logf("Failed to delete static ip %v: %v", name, delErr)
			}
		}
		framework.Failf("Failed to allocate static ip %v: %v", name, err)
	}

	ip, err := gceCloud.GetGlobalAddress(name)
	if err != nil {
		framework.Failf("Failed to get newly created static ip %v: %v", name, err)
	}

	cont.staticIPName = ip.Name
	framework.Logf("Reserved static ip %v: %v", cont.staticIPName, ip.Address)
	return ip.Address
}

// deleteStaticIPs delets all static-ips allocated through calls to
// CreateStaticIP.
func (cont *GCEIngressController) deleteStaticIPs() error {
	if cont.staticIPName != "" {
		if err := GcloudComputeResourceDelete("addresses", cont.staticIPName, cont.Cloud.ProjectID, "--global"); err == nil {
			cont.staticIPName = ""
		} else {
			return err
		}
	} else {
		e2eIPs := []compute.Address{}
		gcloudComputeResourceList("addresses", "e2e-.*", cont.Cloud.ProjectID, &e2eIPs)
		ips := []string{}
		for _, ip := range e2eIPs {
			ips = append(ips, ip.Name)
		}
		framework.Logf("None of the remaining %d static-ips were created by this e2e: %v", len(ips), strings.Join(ips, ", "))
	}
	return nil
}

// gcloudComputeResourceList unmarshals json output of gcloud into given out interface.
func gcloudComputeResourceList(resource, regex, project string, out interface{}) {
	// gcloud prints a message to stderr if it has an available update
	// so we only look at stdout.
	command := []string{
		"compute", resource, "list",
		fmt.Sprintf("--filter='name ~ \"%q\"'", regex),
		fmt.Sprintf("--project=%v", project),
		"-q", "--format=json",
	}
	output, err := exec.Command("gcloud", command...).Output()
	if err != nil {
		errCode := -1
		errMsg := ""
		if exitErr, ok := err.(utilexec.ExitError); ok {
			errCode = exitErr.ExitStatus()
			errMsg = exitErr.Error()
			if osExitErr, ok := err.(*exec.ExitError); ok {
				errMsg = fmt.Sprintf("%v, stderr %v", errMsg, string(osExitErr.Stderr))
			}
		}
		framework.Logf("Error running gcloud command 'gcloud %s': err: %v, output: %v, status: %d, msg: %v", strings.Join(command, " "), err, string(output), errCode, errMsg)
	}
	if err := json.Unmarshal([]byte(output), out); err != nil {
		framework.Logf("Error unmarshalling gcloud output for %v: %v, output: %v", resource, err, string(output))
	}
}

// GcloudComputeResourceDelete deletes the specified compute resource by name and project.
func GcloudComputeResourceDelete(resource, name, project string, args ...string) error {
	framework.Logf("Deleting %v: %v", resource, name)
	argList := append([]string{"compute", resource, "delete", name, fmt.Sprintf("--project=%v", project), "-q"}, args...)
	output, err := exec.Command("gcloud", argList...).CombinedOutput()
	if err != nil {
		framework.Logf("Error deleting %v, output: %v\nerror: %+v", resource, string(output), err)
	}
	return err
}

// GcloudComputeResourceCreate creates a compute resource with a name and arguments.
func GcloudComputeResourceCreate(resource, name, project string, args ...string) error {
	framework.Logf("Creating %v in project %v: %v", resource, project, name)
	argsList := append([]string{"compute", resource, "create", name, fmt.Sprintf("--project=%v", project)}, args...)
	framework.Logf("Running command: gcloud %+v", strings.Join(argsList, " "))
	output, err := exec.Command("gcloud", argsList...).CombinedOutput()
	if err != nil {
		framework.Logf("Error creating %v, output: %v\nerror: %+v", resource, string(output), err)
	}
	return err
}
