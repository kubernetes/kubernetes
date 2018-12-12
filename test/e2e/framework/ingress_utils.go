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

package framework

import (
	"bytes"
	"crypto/rand"
	"crypto/rsa"
	"crypto/sha256"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/json"
	"encoding/pem"
	"fmt"
	"math/big"
	"net"
	"net/http"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/golang/glog"

	compute "google.golang.org/api/compute/v1"
	"google.golang.org/api/googleapi"
	"k8s.io/api/core/v1"
	extensions "k8s.io/api/extensions/v1beta1"
	apierrs "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/intstr"
	utilnet "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	gcecloud "k8s.io/kubernetes/pkg/cloudprovider/providers/gce"
	utilfile "k8s.io/kubernetes/pkg/util/file"
	"k8s.io/kubernetes/test/e2e/manifest"
	testutils "k8s.io/kubernetes/test/utils"
	utilexec "k8s.io/utils/exec"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

const (
	rsaBits  = 2048
	validFor = 365 * 24 * time.Hour

	// Ingress class annotation defined in ingress repository.
	// TODO: All these annotations should be reused from
	// ingress-gce/pkg/annotations instead of duplicating them here.
	IngressClassKey = "kubernetes.io/ingress.class"

	// Ingress class annotation value for multi cluster ingress.
	MulticlusterIngressClassValue = "gce-multi-cluster"

	// Static IP annotation defined in ingress repository.
	IngressStaticIPKey = "kubernetes.io/ingress.global-static-ip-name"

	// Allow HTTP annotation defined in ingress repository.
	IngressAllowHTTPKey = "kubernetes.io/ingress.allow-http"

	// Pre-shared-cert annotation defined in ingress repository.
	IngressPreSharedCertKey = "ingress.gcp.kubernetes.io/pre-shared-cert"

	// ServiceApplicationProtocolKey annotation defined in ingress repository.
	ServiceApplicationProtocolKey = "service.alpha.kubernetes.io/app-protocols"

	// all cloud resources created by the ingress controller start with this
	// prefix.
	k8sPrefix = "k8s-"

	// clusterDelimiter is the delimiter used by the ingress controller
	// to split uid from other naming/metadata.
	clusterDelimiter = "--"

	// Name of the default http backend service
	defaultBackendName = "default-http-backend"

	// Cloud resources created by the ingress controller older than this
	// are automatically purged to prevent running out of quota.
	// TODO(37335): write soak tests and bump this up to a week.
	maxAge = 48 * time.Hour

	// IngressManifestPath is the parent path to yaml test manifests.
	IngressManifestPath = "test/e2e/testing-manifests/ingress"

	// IngressReqTimeout is the timeout on a single http request.
	IngressReqTimeout = 10 * time.Second

	// healthz port used to verify glbc restarted correctly on the master.
	glbcHealthzPort = 8086

	// General cloud resource poll timeout (eg: create static ip, firewall etc)
	cloudResourcePollTimeout = 5 * time.Minute

	// Name of the config-map and key the ingress controller stores its uid in.
	uidConfigMap = "ingress-uid"
	uidKey       = "uid"

	// GCE only allows names < 64 characters, and the loadbalancer controller inserts
	// a single character of padding.
	nameLenLimit = 62

	NEGAnnotation       = "cloud.google.com/neg"
	NEGStatusAnnotation = "cloud.google.com/neg-status"
	NEGUpdateTimeout    = 2 * time.Minute

	InstanceGroupAnnotation = "ingress.gcp.kubernetes.io/instance-groups"

	// Prefix for annotation keys used by the ingress controller to specify the
	// names of GCP resources such as forwarding rules, url maps, target proxies, etc
	// that it created for the corresponding ingress.
	StatusPrefix = "ingress.kubernetes.io"
)

type TestLogger interface {
	Infof(format string, args ...interface{})
	Errorf(format string, args ...interface{})
}

type GLogger struct{}

func (l *GLogger) Infof(format string, args ...interface{}) {
	glog.Infof(format, args...)
}

func (l *GLogger) Errorf(format string, args ...interface{}) {
	glog.Errorf(format, args...)
}

type E2ELogger struct{}

func (l *E2ELogger) Infof(format string, args ...interface{}) {
	Logf(format, args...)
}

func (l *E2ELogger) Errorf(format string, args ...interface{}) {
	Logf(format, args...)
}

// IngressConformanceTests contains a closure with an entry and exit log line.
type IngressConformanceTests struct {
	EntryLog string
	Execute  func()
	ExitLog  string
}

// NegStatus contains name and zone of the Network Endpoint Group
// resources associated with this service.
// Needs to be consistent with the NEG internal structs in ingress-gce.
type NegStatus struct {
	// NetworkEndpointGroups returns the mapping between service port and NEG
	// resource. key is service port, value is the name of the NEG resource.
	NetworkEndpointGroups map[int32]string `json:"network_endpoint_groups,omitempty"`
	Zones                 []string         `json:"zones,omitempty"`
}

// CreateIngressComformanceTests generates an slice of sequential test cases:
// a simple http ingress, ingress with HTTPS, ingress HTTPS with a modified hostname,
// ingress https with a modified URLMap
func CreateIngressComformanceTests(jig *IngressTestJig, ns string, annotations map[string]string) []IngressConformanceTests {
	manifestPath := filepath.Join(IngressManifestPath, "http")
	// These constants match the manifests used in IngressManifestPath
	tlsHost := "foo.bar.com"
	tlsSecretName := "foo"
	updatedTLSHost := "foobar.com"
	updateURLMapHost := "bar.baz.com"
	updateURLMapPath := "/testurl"
	// Platform agnostic list of tests that must be satisfied by all controllers
	tests := []IngressConformanceTests{
		{
			fmt.Sprintf("should create a basic HTTP ingress"),
			func() { jig.CreateIngress(manifestPath, ns, annotations, annotations) },
			fmt.Sprintf("waiting for urls on basic HTTP ingress"),
		},
		{
			fmt.Sprintf("should terminate TLS for host %v", tlsHost),
			func() { jig.SetHTTPS(tlsSecretName, tlsHost) },
			fmt.Sprintf("waiting for HTTPS updates to reflect in ingress"),
		},
		{
			fmt.Sprintf("should update url map for host %v to expose a single url: %v", updateURLMapHost, updateURLMapPath),
			func() {
				var pathToFail string
				jig.Update(func(ing *extensions.Ingress) {
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
				route := fmt.Sprintf("http://%v%v", jig.Address, pathToFail)
				ExpectNoError(PollURL(route, updateURLMapHost, LoadBalancerCleanupTimeout, jig.PollInterval, &http.Client{Timeout: IngressReqTimeout}, true))
			},
			fmt.Sprintf("Waiting for path updates to reflect in L7"),
		},
	}
	// Skip the Update TLS cert test for kubemci: https://github.com/GoogleCloudPlatform/k8s-multicluster-ingress/issues/141.
	if jig.Class != MulticlusterIngressClassValue {
		tests = append(tests, IngressConformanceTests{
			fmt.Sprintf("should update SSL certificate with modified hostname %v", updatedTLSHost),
			func() {
				jig.Update(func(ing *extensions.Ingress) {
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
				jig.SetHTTPS(tlsSecretName, updatedTLSHost)
			},
			fmt.Sprintf("Waiting for updated certificates to accept requests for host %v", updatedTLSHost),
		})
	}
	return tests
}

// GenerateRSACerts generates a basic self signed certificate using a key length
// of rsaBits, valid for validFor time.
func GenerateRSACerts(host string, isCA bool) ([]byte, []byte, error) {
	if len(host) == 0 {
		return nil, nil, fmt.Errorf("Require a non-empty host for client hello")
	}
	priv, err := rsa.GenerateKey(rand.Reader, rsaBits)
	if err != nil {
		return nil, nil, fmt.Errorf("Failed to generate key: %v", err)
	}
	notBefore := time.Now()
	notAfter := notBefore.Add(validFor)

	serialNumberLimit := new(big.Int).Lsh(big.NewInt(1), 128)
	serialNumber, err := rand.Int(rand.Reader, serialNumberLimit)

	if err != nil {
		return nil, nil, fmt.Errorf("failed to generate serial number: %s", err)
	}
	template := x509.Certificate{
		SerialNumber: serialNumber,
		Subject: pkix.Name{
			CommonName:   "default",
			Organization: []string{"Acme Co"},
		},
		NotBefore: notBefore,
		NotAfter:  notAfter,

		KeyUsage:              x509.KeyUsageKeyEncipherment | x509.KeyUsageDigitalSignature,
		ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
		BasicConstraintsValid: true,
	}

	hosts := strings.Split(host, ",")
	for _, h := range hosts {
		if ip := net.ParseIP(h); ip != nil {
			template.IPAddresses = append(template.IPAddresses, ip)
		} else {
			template.DNSNames = append(template.DNSNames, h)
		}
	}

	if isCA {
		template.IsCA = true
		template.KeyUsage |= x509.KeyUsageCertSign
	}

	var keyOut, certOut bytes.Buffer
	derBytes, err := x509.CreateCertificate(rand.Reader, &template, &template, &priv.PublicKey, priv)
	if err != nil {
		return nil, nil, fmt.Errorf("Failed to create certificate: %s", err)
	}
	if err := pem.Encode(&certOut, &pem.Block{Type: "CERTIFICATE", Bytes: derBytes}); err != nil {
		return nil, nil, fmt.Errorf("Failed creating cert: %v", err)
	}
	if err := pem.Encode(&keyOut, &pem.Block{Type: "RSA PRIVATE KEY", Bytes: x509.MarshalPKCS1PrivateKey(priv)}); err != nil {
		return nil, nil, fmt.Errorf("Failed creating keay: %v", err)
	}
	return certOut.Bytes(), keyOut.Bytes(), nil
}

// buildTransportWithCA creates a transport for use in executing HTTPS requests with
// the given certs. Note that the given rootCA must be configured with isCA=true.
func buildTransportWithCA(serverName string, rootCA []byte) (*http.Transport, error) {
	pool := x509.NewCertPool()
	ok := pool.AppendCertsFromPEM(rootCA)
	if !ok {
		return nil, fmt.Errorf("Unable to load serverCA")
	}
	return utilnet.SetTransportDefaults(&http.Transport{
		TLSClientConfig: &tls.Config{
			InsecureSkipVerify: false,
			ServerName:         serverName,
			RootCAs:            pool,
		},
	}), nil
}

// BuildInsecureClient returns an insecure http client. Can be used for "curl -k".
func BuildInsecureClient(timeout time.Duration) *http.Client {
	t := &http.Transport{TLSClientConfig: &tls.Config{InsecureSkipVerify: true}}
	return &http.Client{Timeout: timeout, Transport: utilnet.SetTransportDefaults(t)}
}

// createTLSSecret creates a secret containing TLS certificates.
// If a secret with the same name already pathExists in the namespace of the
// Ingress, it's updated.
func createTLSSecret(kubeClient clientset.Interface, namespace, secretName string, hosts ...string) (host string, rootCA, privKey []byte, err error) {
	host = strings.Join(hosts, ",")
	Logf("Generating RSA cert for host %v", host)
	cert, key, err := GenerateRSACerts(host, true)
	if err != nil {
		return
	}
	secret := &v1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name: secretName,
		},
		Data: map[string][]byte{
			v1.TLSCertKey:       cert,
			v1.TLSPrivateKeyKey: key,
		},
	}
	var s *v1.Secret
	if s, err = kubeClient.CoreV1().Secrets(namespace).Get(secretName, metav1.GetOptions{}); err == nil {
		// TODO: Retry the update. We don't really expect anything to conflict though.
		Logf("Updating secret %v in ns %v with hosts %v", secret.Name, namespace, host)
		s.Data = secret.Data
		_, err = kubeClient.CoreV1().Secrets(namespace).Update(s)
	} else {
		Logf("Creating secret %v in ns %v with hosts %v", secret.Name, namespace, host)
		_, err = kubeClient.CoreV1().Secrets(namespace).Create(secret)
	}
	return host, cert, key, err
}

// GCEIngressController manages implementation details of Ingress on GCE/GKE.
type GCEIngressController struct {
	Ns           string
	rcPath       string
	UID          string
	staticIPName string
	rc           *v1.ReplicationController
	svc          *v1.Service
	Client       clientset.Interface
	Cloud        CloudConfig
}

func (cont *GCEIngressController) CleanupGCEIngressController() error {
	return cont.CleanupGCEIngressControllerWithTimeout(LoadBalancerCleanupTimeout)
}

// CleanupGCEIngressControllerWithTimeout calls the GCEIngressController.Cleanup(false)
// followed with deleting the static ip, and then a final GCEIngressController.Cleanup(true)
func (cont *GCEIngressController) CleanupGCEIngressControllerWithTimeout(timeout time.Duration) error {
	pollErr := wait.Poll(5*time.Second, timeout, func() (bool, error) {
		if err := cont.Cleanup(false); err != nil {
			Logf("Monitoring glbc's cleanup of gce resources:\n%v", err)
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
			Logf("Failed to delete static-ip: %v\n", err)
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
	Logf("Retrieving UID from config map: %v/%v", metav1.NamespaceSystem, uidConfigMap)
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
	gceCloud := cont.Cloud.Provider.(*gcecloud.GCECloud)
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
	gceCloud := cont.Cloud.Provider.(*gcecloud.GCECloud)
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
	gceCloud := cont.Cloud.Provider.(*gcecloud.GCECloud)
	tpList := []*compute.TargetHttpProxy{}
	l, err := gceCloud.ListTargetHttpProxies()
	Expect(err).NotTo(HaveOccurred())
	for _, tp := range l {
		if cont.isOwned(tp.Name) {
			tpList = append(tpList, tp)
		}
	}
	return tpList
}

func (cont *GCEIngressController) ListTargetHttpsProxies() []*compute.TargetHttpsProxy {
	gceCloud := cont.Cloud.Provider.(*gcecloud.GCECloud)
	tpsList := []*compute.TargetHttpsProxy{}
	l, err := gceCloud.ListTargetHttpsProxies()
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
	gceCloud := cont.Cloud.Provider.(*gcecloud.GCECloud)
	umList := []*compute.UrlMap{}
	l, err := gceCloud.ListUrlMaps()
	Expect(err).NotTo(HaveOccurred())
	for _, um := range l {
		if cont.isOwned(um.Name) {
			umList = append(umList, um)
		}
	}
	return umList
}

func (cont *GCEIngressController) deleteURLMap(del bool) (msg string) {
	gceCloud := cont.Cloud.Provider.(*gcecloud.GCECloud)
	umList, err := gceCloud.ListUrlMaps()
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
			Logf("Deleting url-map: %s", um.Name)
			if err := gceCloud.DeleteUrlMap(um.Name); err != nil &&
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
	gceCloud := cont.Cloud.Provider.(*gcecloud.GCECloud)
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
	gceCloud := cont.Cloud.Provider.(*gcecloud.GCECloud)
	beList, err := gceCloud.ListGlobalBackendServices()
	if err != nil {
		if cont.isHTTPErrorCode(err, http.StatusNotFound) {
			return msg
		}
		return fmt.Sprintf("Failed to list backend services: %v", err)
	}
	if len(beList) == 0 {
		Logf("No backend services found")
		return msg
	}
	for _, be := range beList {
		if !cont.canDelete(be.Name, be.CreationTimestamp, del) {
			continue
		}
		if del {
			Logf("Deleting backed-service: %s", be.Name)
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
	gceCloud := cont.Cloud.Provider.(*gcecloud.GCECloud)
	hcList, err := gceCloud.ListHttpHealthChecks()
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
			Logf("Deleting http-health-check: %s", hc.Name)
			if err := gceCloud.DeleteHttpHealthCheck(hc.Name); err != nil &&
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
	gceCloud := cont.Cloud.Provider.(*gcecloud.GCECloud)
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
	gceCloud := cont.Cloud.Provider.(*gcecloud.GCECloud)
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
				Logf("Deleting ssl-certificate: %s", s.Name)
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
	gceCloud := cont.Cloud.Provider.(*gcecloud.GCECloud)
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
	gceCloud := cont.Cloud.Provider.(*gcecloud.GCECloud)
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
			Logf("Deleting instance-group: %s", ig.Name)
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
	gceCloud := cont.Cloud.Provider.(*gcecloud.GCECloud)
	// TODO: E2E cloudprovider has only 1 zone, but the cluster can have many.
	// We need to poll on all NEGs across all zones.
	negList, err := gceCloud.ListNetworkEndpointGroup(cont.Cloud.Zone)
	if err != nil {
		if cont.isHTTPErrorCode(err, http.StatusNotFound) {
			return msg
		}
		// Do not return error as NEG is still alpha.
		Logf("Failed to list network endpoint group: %v", err)
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
			Logf("Deleting network-endpoint-group: %s", neg.Name)
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
		Logf("WARNING: Failed to parse creation timestamp %v for %v: %v", creationTimestamp, resourceName, err)
		return false
	}
	if time.Since(createdTime) > maxAge {
		Logf("%v created on %v IS too old", resourceName, creationTimestamp)
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
	gceCloud := cont.Cloud.Provider.(*gcecloud.GCECloud)
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
	gceCloud := cont.Cloud.Provider.(*gcecloud.GCECloud)
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
		Logf("WARNING: test name including cluster UID: %v is over the GCE limit of %v", testName, nameLenLimit)
	} else {
		Logf("Detected cluster UID %v", cont.UID)
	}
	return nil
}

// CreateStaticIP allocates a random static ip with the given name. Returns a string
// representation of the ip. Caller is expected to manage cleanup of the ip by
// invoking deleteStaticIPs.
func (cont *GCEIngressController) CreateStaticIP(name string) string {
	gceCloud := cont.Cloud.Provider.(*gcecloud.GCECloud)
	addr := &compute.Address{Name: name}
	if err := gceCloud.ReserveGlobalAddress(addr); err != nil {
		if delErr := gceCloud.DeleteGlobalAddress(name); delErr != nil {
			if cont.isHTTPErrorCode(delErr, http.StatusNotFound) {
				Logf("Static ip with name %v was not allocated, nothing to delete", name)
			} else {
				Logf("Failed to delete static ip %v: %v", name, delErr)
			}
		}
		Failf("Failed to allocate static ip %v: %v", name, err)
	}

	ip, err := gceCloud.GetGlobalAddress(name)
	if err != nil {
		Failf("Failed to get newly created static ip %v: %v", name, err)
	}

	cont.staticIPName = ip.Name
	Logf("Reserved static ip %v: %v", cont.staticIPName, ip.Address)
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
		Logf("None of the remaining %d static-ips were created by this e2e: %v", len(ips), strings.Join(ips, ", "))
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
		Logf("Error running gcloud command 'gcloud %s': err: %v, output: %v, status: %d, msg: %v", strings.Join(command, " "), err, string(output), errCode, errMsg)
	}
	if err := json.Unmarshal([]byte(output), out); err != nil {
		Logf("Error unmarshalling gcloud output for %v: %v, output: %v", resource, err, string(output))
	}
}

// GcloudComputeResourceDelete deletes the specified compute resource by name and project.
func GcloudComputeResourceDelete(resource, name, project string, args ...string) error {
	Logf("Deleting %v: %v", resource, name)
	argList := append([]string{"compute", resource, "delete", name, fmt.Sprintf("--project=%v", project), "-q"}, args...)
	output, err := exec.Command("gcloud", argList...).CombinedOutput()
	if err != nil {
		Logf("Error deleting %v, output: %v\nerror: %+v", resource, string(output), err)
	}
	return err
}

// GcloudComputeResourceCreate creates a compute resource with a name and arguments.
func GcloudComputeResourceCreate(resource, name, project string, args ...string) error {
	Logf("Creating %v in project %v: %v", resource, project, name)
	argsList := append([]string{"compute", resource, "create", name, fmt.Sprintf("--project=%v", project)}, args...)
	Logf("Running command: gcloud %+v", strings.Join(argsList, " "))
	output, err := exec.Command("gcloud", argsList...).CombinedOutput()
	if err != nil {
		Logf("Error creating %v, output: %v\nerror: %+v", resource, string(output), err)
	}
	return err
}

// IngressTestJig holds the relevant state and parameters of the ingress test.
type IngressTestJig struct {
	Client clientset.Interface
	Logger TestLogger

	RootCAs map[string][]byte
	Address string
	Ingress *extensions.Ingress
	// class is the value of the annotation keyed under
	// `kubernetes.io/ingress.class`. It's added to all ingresses created by
	// this jig.
	Class string

	// The interval used to poll urls
	PollInterval time.Duration
}

// NewIngressTestJig instantiates struct with client
func NewIngressTestJig(c clientset.Interface) *IngressTestJig {
	return &IngressTestJig{
		Client:       c,
		RootCAs:      map[string][]byte{},
		PollInterval: LoadBalancerPollInterval,
		Logger:       &E2ELogger{},
	}
}

// CreateIngress creates the Ingress and associated service/rc.
// Required: ing.yaml, rc.yaml, svc.yaml must exist in manifestPath
// Optional: secret.yaml, ingAnnotations
// If ingAnnotations is specified it will overwrite any annotations in ing.yaml
// If svcAnnotations is specified it will overwrite any annotations in svc.yaml
func (j *IngressTestJig) CreateIngress(manifestPath, ns string, ingAnnotations map[string]string, svcAnnotations map[string]string) {
	var err error
	mkpath := func(file string) string {
		return filepath.Join(TestContext.RepoRoot, manifestPath, file)
	}

	j.Logger.Infof("creating replication controller")
	RunKubectlOrDie("create", "-f", mkpath("rc.yaml"), fmt.Sprintf("--namespace=%v", ns))

	j.Logger.Infof("creating service")
	RunKubectlOrDie("create", "-f", mkpath("svc.yaml"), fmt.Sprintf("--namespace=%v", ns))
	if len(svcAnnotations) > 0 {
		svcList, err := j.Client.CoreV1().Services(ns).List(metav1.ListOptions{})
		ExpectNoError(err)
		for _, svc := range svcList.Items {
			svc.Annotations = svcAnnotations
			_, err = j.Client.CoreV1().Services(ns).Update(&svc)
			ExpectNoError(err)
		}
	}

	if exists, _ := utilfile.FileExists(mkpath("secret.yaml")); exists {
		j.Logger.Infof("creating secret")
		RunKubectlOrDie("create", "-f", mkpath("secret.yaml"), fmt.Sprintf("--namespace=%v", ns))
	}
	j.Logger.Infof("Parsing ingress from %v", filepath.Join(manifestPath, "ing.yaml"))

	j.Ingress, err = manifest.IngressFromManifest(filepath.Join(manifestPath, "ing.yaml"))
	ExpectNoError(err)
	j.Ingress.Namespace = ns
	j.Ingress.Annotations = map[string]string{IngressClassKey: j.Class}
	for k, v := range ingAnnotations {
		j.Ingress.Annotations[k] = v
	}
	j.Logger.Infof(fmt.Sprintf("creating " + j.Ingress.Name + " ingress"))
	j.Ingress, err = j.runCreate(j.Ingress)
	ExpectNoError(err)
}

// runCreate runs the required command to create the given ingress.
func (j *IngressTestJig) runCreate(ing *extensions.Ingress) (*extensions.Ingress, error) {
	if j.Class != MulticlusterIngressClassValue {
		return j.Client.ExtensionsV1beta1().Ingresses(ing.Namespace).Create(ing)
	}
	// Use kubemci to create a multicluster ingress.
	filePath := TestContext.OutputDir + "/mci.yaml"
	if err := manifest.IngressToManifest(ing, filePath); err != nil {
		return nil, err
	}
	_, err := RunKubemciWithKubeconfig("create", ing.Name, fmt.Sprintf("--ingress=%s", filePath))
	return ing, err
}

// runUpdate runs the required command to update the given ingress.
func (j *IngressTestJig) runUpdate(ing *extensions.Ingress) (*extensions.Ingress, error) {
	if j.Class != MulticlusterIngressClassValue {
		return j.Client.ExtensionsV1beta1().Ingresses(ing.Namespace).Update(ing)
	}
	// Use kubemci to update a multicluster ingress.
	// kubemci does not have an update command. We use "create --force" to update an existing ingress.
	filePath := TestContext.OutputDir + "/mci.yaml"
	if err := manifest.IngressToManifest(ing, filePath); err != nil {
		return nil, err
	}
	_, err := RunKubemciWithKubeconfig("create", ing.Name, fmt.Sprintf("--ingress=%s", filePath), "--force")
	return ing, err
}

// Update retrieves the ingress, performs the passed function, and then updates it.
func (j *IngressTestJig) Update(update func(ing *extensions.Ingress)) {
	var err error
	ns, name := j.Ingress.Namespace, j.Ingress.Name
	for i := 0; i < 3; i++ {
		j.Ingress, err = j.Client.ExtensionsV1beta1().Ingresses(ns).Get(name, metav1.GetOptions{})
		if err != nil {
			Failf("failed to get ingress %s/%s: %v", ns, name, err)
		}
		update(j.Ingress)
		j.Ingress, err = j.runUpdate(j.Ingress)
		if err == nil {
			DescribeIng(j.Ingress.Namespace)
			return
		}
		if !apierrs.IsConflict(err) && !apierrs.IsServerTimeout(err) {
			Failf("failed to update ingress %s/%s: %v", ns, name, err)
		}
	}
	Failf("too many retries updating ingress %s/%s", ns, name)
}

// AddHTTPS updates the ingress to add this secret for these hosts.
func (j *IngressTestJig) AddHTTPS(secretName string, hosts ...string) {
	// TODO: Just create the secret in GetRootCAs once we're watching secrets in
	// the ingress controller.
	_, cert, _, err := createTLSSecret(j.Client, j.Ingress.Namespace, secretName, hosts...)
	ExpectNoError(err)
	j.Logger.Infof("Updating ingress %v to also use secret %v for TLS termination", j.Ingress.Name, secretName)
	j.Update(func(ing *extensions.Ingress) {
		ing.Spec.TLS = append(ing.Spec.TLS, extensions.IngressTLS{Hosts: hosts, SecretName: secretName})
	})
	j.RootCAs[secretName] = cert
}

// SetHTTPS updates the ingress to use only this secret for these hosts.
func (j *IngressTestJig) SetHTTPS(secretName string, hosts ...string) {
	_, cert, _, err := createTLSSecret(j.Client, j.Ingress.Namespace, secretName, hosts...)
	ExpectNoError(err)
	j.Logger.Infof("Updating ingress %v to only use secret %v for TLS termination", j.Ingress.Name, secretName)
	j.Update(func(ing *extensions.Ingress) {
		ing.Spec.TLS = []extensions.IngressTLS{{Hosts: hosts, SecretName: secretName}}
	})
	j.RootCAs = map[string][]byte{secretName: cert}
}

// RemoveHTTPS updates the ingress to not use this secret for TLS.
// Note: Does not delete the secret.
func (j *IngressTestJig) RemoveHTTPS(secretName string) {
	newTLS := []extensions.IngressTLS{}
	for _, ingressTLS := range j.Ingress.Spec.TLS {
		if secretName != ingressTLS.SecretName {
			newTLS = append(newTLS, ingressTLS)
		}
	}
	j.Logger.Infof("Updating ingress %v to not use secret %v for TLS termination", j.Ingress.Name, secretName)
	j.Update(func(ing *extensions.Ingress) {
		ing.Spec.TLS = newTLS
	})
	delete(j.RootCAs, secretName)
}

// PrepareTLSSecret creates a TLS secret and caches the cert.
func (j *IngressTestJig) PrepareTLSSecret(namespace, secretName string, hosts ...string) error {
	_, cert, _, err := createTLSSecret(j.Client, namespace, secretName, hosts...)
	if err != nil {
		return err
	}
	j.RootCAs[secretName] = cert
	return nil
}

// GetRootCA returns a rootCA from the ingress test jig.
func (j *IngressTestJig) GetRootCA(secretName string) (rootCA []byte) {
	var ok bool
	rootCA, ok = j.RootCAs[secretName]
	if !ok {
		Failf("Failed to retrieve rootCAs, no recorded secret by name %v", secretName)
	}
	return
}

// TryDeleteIngress attempts to delete the ingress resource and logs errors if they occur.
func (j *IngressTestJig) TryDeleteIngress() {
	j.TryDeleteGivenIngress(j.Ingress)
}

func (j *IngressTestJig) TryDeleteGivenIngress(ing *extensions.Ingress) {
	if err := j.runDelete(ing); err != nil {
		j.Logger.Infof("Error while deleting the ingress %v/%v with class %s: %v", ing.Namespace, ing.Name, j.Class, err)
	}
}

func (j *IngressTestJig) TryDeleteGivenService(svc *v1.Service) {
	err := j.Client.CoreV1().Services(svc.Namespace).Delete(svc.Name, nil)
	if err != nil {
		j.Logger.Infof("Error while deleting the service %v/%v: %v", svc.Namespace, svc.Name, err)
	}
}

// runDelete runs the required command to delete the given ingress.
func (j *IngressTestJig) runDelete(ing *extensions.Ingress) error {
	if j.Class != MulticlusterIngressClassValue {
		return j.Client.ExtensionsV1beta1().Ingresses(ing.Namespace).Delete(ing.Name, nil)
	}
	// Use kubemci to delete a multicluster ingress.
	filePath := TestContext.OutputDir + "/mci.yaml"
	if err := manifest.IngressToManifest(ing, filePath); err != nil {
		return err
	}
	_, err := RunKubemciWithKubeconfig("delete", ing.Name, fmt.Sprintf("--ingress=%s", filePath))
	return err
}

// getIngressAddressFromKubemci returns the IP address of the given multicluster ingress using kubemci.
// TODO(nikhiljindal): Update this to be able to return hostname as well.
func getIngressAddressFromKubemci(name string) ([]string, error) {
	var addresses []string
	out, err := RunKubemciCmd("get-status", name)
	if err != nil {
		return addresses, err
	}
	ip := findIPv4(out)
	if ip != "" {
		addresses = append(addresses, ip)
	}
	return addresses, err
}

// findIPv4 returns the first IPv4 address found in the given string.
func findIPv4(input string) string {
	numBlock := "(25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9]?[0-9])"
	regexPattern := numBlock + "\\." + numBlock + "\\." + numBlock + "\\." + numBlock

	regEx := regexp.MustCompile(regexPattern)
	return regEx.FindString(input)
}

// getIngressAddress returns the ips/hostnames associated with the Ingress.
func getIngressAddress(client clientset.Interface, ns, name, class string) ([]string, error) {
	if class == MulticlusterIngressClassValue {
		return getIngressAddressFromKubemci(name)
	}
	ing, err := client.ExtensionsV1beta1().Ingresses(ns).Get(name, metav1.GetOptions{})
	if err != nil {
		return nil, err
	}
	var addresses []string
	for _, a := range ing.Status.LoadBalancer.Ingress {
		if a.IP != "" {
			addresses = append(addresses, a.IP)
		}
		if a.Hostname != "" {
			addresses = append(addresses, a.Hostname)
		}
	}
	return addresses, nil
}

// WaitForIngressAddress waits for the Ingress to acquire an address.
func (j *IngressTestJig) WaitForIngressAddress(c clientset.Interface, ns, ingName string, timeout time.Duration) (string, error) {
	var address string
	err := wait.PollImmediate(10*time.Second, timeout, func() (bool, error) {
		ipOrNameList, err := getIngressAddress(c, ns, ingName, j.Class)
		if err != nil || len(ipOrNameList) == 0 {
			j.Logger.Errorf("Waiting for Ingress %s/%s to acquire IP, error: %v, ipOrNameList: %v", ns, ingName, err, ipOrNameList)
			if testutils.IsRetryableAPIError(err) {
				return false, nil
			}
			return false, err
		}
		address = ipOrNameList[0]
		j.Logger.Infof("Found address %s for ingress %s/%s", address, ns, ingName)
		return true, nil
	})
	return address, err
}

func (j *IngressTestJig) pollIngressWithCert(ing *extensions.Ingress, address string, knownHosts []string, cert []byte, waitForNodePort bool, timeout time.Duration) error {
	// Check that all rules respond to a simple GET.
	knownHostsSet := sets.NewString(knownHosts...)
	for _, rules := range ing.Spec.Rules {
		timeoutClient := &http.Client{Timeout: IngressReqTimeout}
		proto := "http"
		if knownHostsSet.Has(rules.Host) {
			var err error
			// Create transport with cert to verify if the server uses the correct one.
			timeoutClient.Transport, err = buildTransportWithCA(rules.Host, cert)
			if err != nil {
				return err
			}
			proto = "https"
		}
		for _, p := range rules.IngressRuleValue.HTTP.Paths {
			if waitForNodePort {
				nodePort := int(p.Backend.ServicePort.IntVal)
				if err := j.pollServiceNodePort(ing.Namespace, p.Backend.ServiceName, nodePort); err != nil {
					j.Logger.Infof("Error in waiting for nodeport %d on service %v/%v: %s", nodePort, ing.Namespace, p.Backend.ServiceName, err)
					return err
				}
			}
			route := fmt.Sprintf("%v://%v%v", proto, address, p.Path)
			j.Logger.Infof("Testing route %v host %v with simple GET", route, rules.Host)
			if err := PollURL(route, rules.Host, timeout, j.PollInterval, timeoutClient, false); err != nil {
				return err
			}
		}
	}
	j.Logger.Infof("Finished polling on all rules on ingress %q", ing.Name)
	return nil
}

func (j *IngressTestJig) WaitForIngress(waitForNodePort bool) {
	if err := j.WaitForGivenIngressWithTimeout(j.Ingress, waitForNodePort, LoadBalancerPollTimeout); err != nil {
		Failf("error in waiting for ingress to get an address: %s", err)
	}
}

// WaitForGivenIngressWithTimeout waits till the ingress acquires an IP,
// then waits for its hosts/urls to respond to a protocol check (either
// http or https). If waitForNodePort is true, the NodePort of the Service
// is verified before verifying the Ingress. NodePort is currently a
// requirement for cloudprovider Ingress.
func (j *IngressTestJig) WaitForGivenIngressWithTimeout(ing *extensions.Ingress, waitForNodePort bool, timeout time.Duration) error {
	// Wait for the loadbalancer IP.
	address, err := j.WaitForIngressAddress(j.Client, ing.Namespace, ing.Name, timeout)
	if err != nil {
		return fmt.Errorf("Ingress failed to acquire an IP address within %v", timeout)
	}

	var knownHosts []string
	var cert []byte
	if len(ing.Spec.TLS) > 0 {
		knownHosts = ing.Spec.TLS[0].Hosts
		cert = j.GetRootCA(ing.Spec.TLS[0].SecretName)
	}
	return j.pollIngressWithCert(ing, address, knownHosts, cert, waitForNodePort, timeout)
}

// WaitForIngress waits till the ingress acquires an IP, then waits for its
// hosts/urls to respond to a protocol check (either http or https). If
// waitForNodePort is true, the NodePort of the Service is verified before
// verifying the Ingress. NodePort is currently a requirement for cloudprovider
// Ingress. Hostnames and certificate need to be explicitly passed in.
func (j *IngressTestJig) WaitForIngressWithCert(waitForNodePort bool, knownHosts []string, cert []byte) error {
	// Wait for the loadbalancer IP.
	address, err := j.WaitForIngressAddress(j.Client, j.Ingress.Namespace, j.Ingress.Name, LoadBalancerPollTimeout)
	if err != nil {
		return fmt.Errorf("Ingress failed to acquire an IP address within %v", LoadBalancerPollTimeout)
	}

	return j.pollIngressWithCert(j.Ingress, address, knownHosts, cert, waitForNodePort, LoadBalancerPollTimeout)
}

// VerifyURL polls for the given iterations, in intervals, and fails if the
// given url returns a non-healthy http code even once.
func (j *IngressTestJig) VerifyURL(route, host string, iterations int, interval time.Duration, httpClient *http.Client) error {
	for i := 0; i < iterations; i++ {
		b, err := SimpleGET(httpClient, route, host)
		if err != nil {
			Logf(b)
			return err
		}
		j.Logger.Infof("Verfied %v with host %v %d times, sleeping for %v", route, host, i, interval)
		time.Sleep(interval)
	}
	return nil
}

func (j *IngressTestJig) pollServiceNodePort(ns, name string, port int) error {
	// TODO: Curl all nodes?
	u, err := GetNodePortURL(j.Client, ns, name, port)
	if err != nil {
		return err
	}
	return PollURL(u, "", 30*time.Second, j.PollInterval, &http.Client{Timeout: IngressReqTimeout}, false)
}

func (j *IngressTestJig) GetDefaultBackendNodePort() (int32, error) {
	defaultSvc, err := j.Client.CoreV1().Services(metav1.NamespaceSystem).Get(defaultBackendName, metav1.GetOptions{})
	if err != nil {
		return 0, err
	}
	return defaultSvc.Spec.Ports[0].NodePort, nil
}

// GetIngressNodePorts returns related backend services' nodePorts.
// Current GCE ingress controller allows traffic to the default HTTP backend
// by default, so retrieve its nodePort if includeDefaultBackend is true.
func (j *IngressTestJig) GetIngressNodePorts(includeDefaultBackend bool) []string {
	nodePorts := []string{}
	svcPorts := j.GetServicePorts(includeDefaultBackend)
	for _, svcPort := range svcPorts {
		nodePorts = append(nodePorts, strconv.Itoa(int(svcPort.NodePort)))
	}
	return nodePorts
}

// GetIngressNodePorts returns related backend services' svcPorts.
// Current GCE ingress controller allows traffic to the default HTTP backend
// by default, so retrieve its nodePort if includeDefaultBackend is true.
func (j *IngressTestJig) GetServicePorts(includeDefaultBackend bool) map[string]v1.ServicePort {
	svcPorts := make(map[string]v1.ServicePort)
	if includeDefaultBackend {
		defaultSvc, err := j.Client.CoreV1().Services(metav1.NamespaceSystem).Get(defaultBackendName, metav1.GetOptions{})
		Expect(err).NotTo(HaveOccurred())
		svcPorts[defaultBackendName] = defaultSvc.Spec.Ports[0]
	}

	backendSvcs := []string{}
	if j.Ingress.Spec.Backend != nil {
		backendSvcs = append(backendSvcs, j.Ingress.Spec.Backend.ServiceName)
	}
	for _, rule := range j.Ingress.Spec.Rules {
		for _, ingPath := range rule.HTTP.Paths {
			backendSvcs = append(backendSvcs, ingPath.Backend.ServiceName)
		}
	}
	for _, svcName := range backendSvcs {
		svc, err := j.Client.CoreV1().Services(j.Ingress.Namespace).Get(svcName, metav1.GetOptions{})
		Expect(err).NotTo(HaveOccurred())
		svcPorts[svcName] = svc.Spec.Ports[0]
	}
	return svcPorts
}

// ConstructFirewallForIngress returns the expected GCE firewall rule for the ingress resource
func (j *IngressTestJig) ConstructFirewallForIngress(gceController *GCEIngressController, nodeTags []string) *compute.Firewall {
	nodePorts := j.GetIngressNodePorts(true)

	fw := compute.Firewall{}
	fw.Name = gceController.GetFirewallRuleName()
	fw.SourceRanges = gcecloud.LoadBalancerSrcRanges()
	fw.TargetTags = nodeTags
	fw.Allowed = []*compute.FirewallAllowed{
		{
			IPProtocol: "tcp",
			Ports:      nodePorts,
		},
	}
	return &fw
}

// GetDistinctResponseFromIngress tries GET call to the ingress VIP and return all distinct responses.
func (j *IngressTestJig) GetDistinctResponseFromIngress() (sets.String, error) {
	// Wait for the loadbalancer IP.
	address, err := j.WaitForIngressAddress(j.Client, j.Ingress.Namespace, j.Ingress.Name, LoadBalancerPollTimeout)
	if err != nil {
		Failf("Ingress failed to acquire an IP address within %v", LoadBalancerPollTimeout)
	}
	responses := sets.NewString()
	timeoutClient := &http.Client{Timeout: IngressReqTimeout}

	for i := 0; i < 100; i++ {
		url := fmt.Sprintf("http://%v", address)
		res, err := SimpleGET(timeoutClient, url, "")
		if err != nil {
			j.Logger.Errorf("Failed to GET %q. Got responses: %q: %v", url, res, err)
			return responses, err
		}
		responses.Insert(res)
	}
	return responses, nil
}

// NginxIngressController manages implementation details of Ingress on Nginx.
type NginxIngressController struct {
	Ns         string
	rc         *v1.ReplicationController
	pod        *v1.Pod
	Client     clientset.Interface
	externalIP string
}

// Init initializes the NginxIngressController
func (cont *NginxIngressController) Init() {
	mkpath := func(file string) string {
		return filepath.Join(TestContext.RepoRoot, IngressManifestPath, "nginx", file)
	}
	Logf("initializing nginx ingress controller")
	RunKubectlOrDie("create", "-f", mkpath("rc.yaml"), fmt.Sprintf("--namespace=%v", cont.Ns))

	rc, err := cont.Client.CoreV1().ReplicationControllers(cont.Ns).Get("nginx-ingress-controller", metav1.GetOptions{})
	ExpectNoError(err)
	cont.rc = rc

	Logf("waiting for pods with label %v", rc.Spec.Selector)
	sel := labels.SelectorFromSet(labels.Set(rc.Spec.Selector))
	ExpectNoError(testutils.WaitForPodsWithLabelRunning(cont.Client, cont.Ns, sel))
	pods, err := cont.Client.CoreV1().Pods(cont.Ns).List(metav1.ListOptions{LabelSelector: sel.String()})
	ExpectNoError(err)
	if len(pods.Items) == 0 {
		Failf("Failed to find nginx ingress controller pods with selector %v", sel)
	}
	cont.pod = &pods.Items[0]
	cont.externalIP, err = GetHostExternalAddress(cont.Client, cont.pod)
	ExpectNoError(err)
	Logf("ingress controller running in pod %v on ip %v", cont.pod.Name, cont.externalIP)
}

func generateBacksideHTTPSIngressSpec(ns string) *extensions.Ingress {
	return &extensions.Ingress{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "echoheaders-https",
			Namespace: ns,
		},
		Spec: extensions.IngressSpec{
			// Note kubemci requres a default backend.
			Backend: &extensions.IngressBackend{
				ServiceName: "echoheaders-https",
				ServicePort: intstr.IntOrString{
					Type:   intstr.Int,
					IntVal: 443,
				},
			},
		},
	}
}

func generateBacksideHTTPSServiceSpec() *v1.Service {
	return &v1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name: "echoheaders-https",
			Annotations: map[string]string{
				ServiceApplicationProtocolKey: `{"my-https-port":"HTTPS"}`,
			},
		},
		Spec: v1.ServiceSpec{
			Ports: []v1.ServicePort{{
				Name:       "my-https-port",
				Protocol:   v1.ProtocolTCP,
				Port:       443,
				TargetPort: intstr.FromString("echo-443"),
			}},
			Selector: map[string]string{
				"app": "echoheaders-https",
			},
			Type: v1.ServiceTypeNodePort,
		},
	}
}

func generateBacksideHTTPSDeploymentSpec() *extensions.Deployment {
	return &extensions.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name: "echoheaders-https",
		},
		Spec: extensions.DeploymentSpec{
			Selector: &metav1.LabelSelector{MatchLabels: map[string]string{
				"app": "echoheaders-https",
			}},
			Template: v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"app": "echoheaders-https",
					},
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:  "echoheaders-https",
							Image: "k8s.gcr.io/echoserver:1.10",
							Ports: []v1.ContainerPort{{
								ContainerPort: 8443,
								Name:          "echo-443",
							}},
						},
					},
				},
			},
		},
	}
}

// SetUpBacksideHTTPSIngress sets up deployment, service and ingress with backside HTTPS configured.
func (j *IngressTestJig) SetUpBacksideHTTPSIngress(cs clientset.Interface, namespace string, staticIPName string) (*extensions.Deployment, *v1.Service, *extensions.Ingress, error) {
	deployCreated, err := cs.ExtensionsV1beta1().Deployments(namespace).Create(generateBacksideHTTPSDeploymentSpec())
	if err != nil {
		return nil, nil, nil, err
	}
	svcCreated, err := cs.CoreV1().Services(namespace).Create(generateBacksideHTTPSServiceSpec())
	if err != nil {
		return nil, nil, nil, err
	}
	ingToCreate := generateBacksideHTTPSIngressSpec(namespace)
	if staticIPName != "" {
		if ingToCreate.Annotations == nil {
			ingToCreate.Annotations = map[string]string{}
		}
		ingToCreate.Annotations[IngressStaticIPKey] = staticIPName
	}
	ingCreated, err := j.runCreate(ingToCreate)
	if err != nil {
		return nil, nil, nil, err
	}
	return deployCreated, svcCreated, ingCreated, nil
}

// DeleteTestResource deletes given deployment, service and ingress.
func (j *IngressTestJig) DeleteTestResource(cs clientset.Interface, deploy *extensions.Deployment, svc *v1.Service, ing *extensions.Ingress) []error {
	var errs []error
	if ing != nil {
		if err := j.runDelete(ing); err != nil {
			errs = append(errs, fmt.Errorf("error while deleting ingress %s/%s: %v", ing.Namespace, ing.Name, err))
		}
	}
	if svc != nil {
		if err := cs.CoreV1().Services(svc.Namespace).Delete(svc.Name, nil); err != nil {
			errs = append(errs, fmt.Errorf("error while deleting service %s/%s: %v", svc.Namespace, svc.Name, err))
		}
	}
	if deploy != nil {
		if err := cs.ExtensionsV1beta1().Deployments(deploy.Namespace).Delete(deploy.Name, nil); err != nil {
			errs = append(errs, fmt.Errorf("error while deleting deployment %s/%s: %v", deploy.Namespace, deploy.Name, err))
		}
	}
	return errs
}
