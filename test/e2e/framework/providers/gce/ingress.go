//go:build !providerless
// +build !providerless

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
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"net/http"
	"os/exec"
	"strconv"
	"strings"
	"time"

	"github.com/onsi/ginkgo/v2"
	compute "google.golang.org/api/compute/v1"
	"google.golang.org/api/googleapi"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2eservice "k8s.io/kubernetes/test/e2e/framework/service"
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

	negBackend = backendType("networkEndpointGroup")
	igBackend  = backendType("instanceGroup")
)

type backendType string

// IngressController manages implementation details of Ingress on GCE/GKE.
type IngressController struct {
	Ns           string
	UID          string
	staticIPName string
	Client       clientset.Interface
	Cloud        framework.CloudConfig
}

// CleanupIngressController calls cont.CleanupIngressControllerWithTimeout with hard-coded timeout
func (cont *IngressController) CleanupIngressController(ctx context.Context) error {
	return cont.CleanupIngressControllerWithTimeout(ctx, e2eservice.LoadBalancerCleanupTimeout)
}

// CleanupIngressControllerWithTimeout calls the IngressController.Cleanup(false)
// followed with deleting the static ip, and then a final IngressController.Cleanup(true)
func (cont *IngressController) CleanupIngressControllerWithTimeout(ctx context.Context, timeout time.Duration) error {
	pollErr := wait.PollWithContext(ctx, 5*time.Second, timeout, func(ctx context.Context) (bool, error) {
		if err := cont.Cleanup(false); err != nil {
			framework.Logf("Monitoring glbc's cleanup of gce resources:\n%v", err)
			return false, nil
		}
		return true, nil
	})

	// Always try to cleanup even if pollErr == nil, because the cleanup
	// routine also purges old leaked resources based on creation timestamp.
	ginkgo.By("Performing final delete of any remaining resources")
	if cleanupErr := cont.Cleanup(true); cleanupErr != nil {
		ginkgo.By(fmt.Sprintf("WARNING: possibly leaked resources: %v\n", cleanupErr))
	} else {
		ginkgo.By("No resources leaked.")
	}

	// Static-IP allocated on behalf of the test, never deleted by the
	// controller. Delete this IP only after the controller has had a chance
	// to cleanup or it might interfere with the controller, causing it to
	// throw out confusing events.
	if ipErr := wait.PollWithContext(ctx, 5*time.Second, 1*time.Minute, func(ctx context.Context) (bool, error) {
		if err := cont.deleteStaticIPs(); err != nil {
			framework.Logf("Failed to delete static-ip: %v\n", err)
			return false, nil
		}
		return true, nil
	}); ipErr != nil {
		// If this is a persistent error, the suite will fail when we run out
		// of quota anyway.
		ginkgo.By(fmt.Sprintf("WARNING: possibly leaked static IP: %v\n", ipErr))
	}

	// Logging that the GLBC failed to cleanup GCE resources on ingress deletion
	// See kubernetes/ingress#431
	if pollErr != nil {
		return fmt.Errorf("error: L7 controller failed to delete all cloud resources on time. %v", pollErr)
	}
	return nil
}

func (cont *IngressController) getL7AddonUID(ctx context.Context) (string, error) {
	framework.Logf("Retrieving UID from config map: %v/%v", metav1.NamespaceSystem, uidConfigMap)
	cm, err := cont.Client.CoreV1().ConfigMaps(metav1.NamespaceSystem).Get(ctx, uidConfigMap, metav1.GetOptions{})
	if err != nil {
		return "", err
	}
	if uid, ok := cm.Data[uidKey]; ok {
		return uid, nil
	}
	return "", fmt.Errorf("Could not find cluster UID for L7 addon pod")
}

func (cont *IngressController) deleteForwardingRule(del bool) string {
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

func (cont *IngressController) deleteAddresses(del bool) string {
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

func (cont *IngressController) deleteTargetProxy(del bool) string {
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

func (cont *IngressController) deleteURLMap(del bool) (msg string) {
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

func (cont *IngressController) deleteBackendService(del bool) (msg string) {
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

func (cont *IngressController) deleteHTTPHealthCheck(del bool) (msg string) {
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

func (cont *IngressController) deleteSSLCertificate(del bool) (msg string) {
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

func (cont *IngressController) deleteInstanceGroup(del bool) (msg string) {
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

func (cont *IngressController) deleteNetworkEndpointGroup(del bool) (msg string) {
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
func (cont *IngressController) canDelete(resourceName, creationTimestamp string, delOldResources bool) bool {
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

// canDeleteNEG returns true if either the name contains this controller's UID,
// or the creationTimestamp exceeds the maxAge and del is set to true.
func (cont *IngressController) canDeleteNEG(resourceName, creationTimestamp string, delOldResources bool) bool {
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

func (cont *IngressController) deleteFirewallRule(del bool) (msg string) {
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

func (cont *IngressController) isHTTPErrorCode(err error, code int) bool {
	apiErr, ok := err.(*googleapi.Error)
	return ok && apiErr.Code == code
}

// WaitForNegBackendService waits for the expected backend service to become
func (cont *IngressController) WaitForNegBackendService(ctx context.Context, svcPorts map[string]v1.ServicePort) error {
	return wait.PollWithContext(ctx, 5*time.Second, 1*time.Minute, func(ctx context.Context) (bool, error) {
		err := cont.verifyBackendMode(svcPorts, negBackend)
		if err != nil {
			framework.Logf("Err while checking if backend service is using NEG: %v", err)
			return false, nil
		}
		return true, nil
	})
}

// BackendServiceUsingNEG returns true only if all global backend service with matching svcPorts pointing to NEG as backend
func (cont *IngressController) BackendServiceUsingNEG(svcPorts map[string]v1.ServicePort) error {
	return cont.verifyBackendMode(svcPorts, negBackend)
}

// BackendServiceUsingIG returns true only if all global backend service with matching svcPorts pointing to IG as backend
func (cont *IngressController) BackendServiceUsingIG(svcPorts map[string]v1.ServicePort) error {
	return cont.verifyBackendMode(svcPorts, igBackend)
}

func (cont *IngressController) verifyBackendMode(svcPorts map[string]v1.ServicePort, backendType backendType) error {
	gceCloud := cont.Cloud.Provider.(*Provider).gceCloud
	beList, err := gceCloud.ListGlobalBackendServices()
	if err != nil {
		return fmt.Errorf("failed to list backend services: %w", err)
	}

	hcList, err := gceCloud.ListHealthChecks()
	if err != nil {
		return fmt.Errorf("failed to list health checks: %w", err)
	}

	// Generate short UID
	uid := cont.UID
	if len(uid) > 8 {
		uid = uid[:8]
	}

	matchingBackendService := 0
	for svcName, sp := range svcPorts {
		match := false
		bsMatch := &compute.BackendService{}
		// NEG BackendServices' names contain the a sha256 hash of a string.
		// This logic is copied from the ingress-gce namer.
		// WARNING: This needs to adapt if the naming convention changed.
		negString := strings.Join([]string{uid, cont.Ns, svcName, fmt.Sprintf("%v", sp.Port)}, ";")
		negHash := fmt.Sprintf("%x", sha256.Sum256([]byte(negString)))[:8]
		for _, bs := range beList {
			// Non-NEG BackendServices are named with the Nodeport in the name.
			if backendType == igBackend && strings.Contains(bs.Name, strconv.Itoa(int(sp.NodePort))) {
				match = true
				bsMatch = bs
				matchingBackendService++
				break
			}

			// NEG BackendServices' names contain the a sha256 hash of a string.
			if backendType == negBackend && strings.Contains(bs.Name, negHash) {
				match = true
				bsMatch = bs
				matchingBackendService++
				break
			}
		}

		if match {
			for _, be := range bsMatch.Backends {
				if !strings.Contains(be.Group, string(backendType)) {
					return fmt.Errorf("expect to find backends with type %q, but got backend group: %v", backendType, be.Group)
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
				return fmt.Errorf("missing healthcheck for backendservice: %v", bsMatch.Name)
			}
		}
	}

	if matchingBackendService != len(svcPorts) {
		beNames := []string{}
		for _, be := range beList {
			beNames = append(beNames, be.Name)
		}
		return fmt.Errorf("expect %d backend service with backend type: %v, but got %d matching backend service. Expect backend services for service ports: %v, but got backend services: %v", len(svcPorts), backendType, matchingBackendService, svcPorts, beNames)
	}

	return nil
}

// Cleanup cleans up cloud resources.
// If del is false, it simply reports existing resources without deleting them.
// If dle is true, it deletes resources it finds acceptable (see canDelete func).
func (cont *IngressController) Cleanup(del bool) error {
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

// Init initializes the IngressController with an UID
func (cont *IngressController) Init(ctx context.Context) error {
	uid, err := cont.getL7AddonUID(ctx)
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

// deleteStaticIPs deletes all static-ips allocated through calls to
// CreateStaticIP.
func (cont *IngressController) deleteStaticIPs() error {
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
