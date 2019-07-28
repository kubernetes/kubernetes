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

package ingress

import (
	"bytes"
	"crypto/rand"
	"crypto/rsa"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"fmt"
	"math/big"
	"net"
	"net/http"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"

	compute "google.golang.org/api/compute/v1"
	"k8s.io/klog"

	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	networkingv1beta1 "k8s.io/api/networking/v1beta1"
	apierrs "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/intstr"
	utilnet "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2elog "k8s.io/kubernetes/test/e2e/framework/log"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2eservice "k8s.io/kubernetes/test/e2e/framework/service"
	"k8s.io/kubernetes/test/e2e/framework/testfiles"
	"k8s.io/kubernetes/test/e2e/manifest"
	testutils "k8s.io/kubernetes/test/utils"

	"github.com/onsi/ginkgo"
)

const (
	rsaBits  = 2048
	validFor = 365 * 24 * time.Hour

	// IngressClassKey is ingress class annotation defined in ingress repository.
	// TODO: All these annotations should be reused from
	// ingress-gce/pkg/annotations instead of duplicating them here.
	IngressClassKey = "kubernetes.io/ingress.class"

	// MulticlusterIngressClassValue is ingress class annotation value for multi cluster ingress.
	MulticlusterIngressClassValue = "gce-multi-cluster"

	// IngressStaticIPKey is static IP annotation defined in ingress repository.
	IngressStaticIPKey = "kubernetes.io/ingress.global-static-ip-name"

	// IngressAllowHTTPKey is Allow HTTP annotation defined in ingress repository.
	IngressAllowHTTPKey = "kubernetes.io/ingress.allow-http"

	// IngressPreSharedCertKey is Pre-shared-cert annotation defined in ingress repository.
	IngressPreSharedCertKey = "ingress.gcp.kubernetes.io/pre-shared-cert"

	// ServiceApplicationProtocolKey annotation defined in ingress repository.
	ServiceApplicationProtocolKey = "service.alpha.kubernetes.io/app-protocols"

	// Name of the default http backend service
	defaultBackendName = "default-http-backend"

	// IngressManifestPath is the parent path to yaml test manifests.
	IngressManifestPath = "test/e2e/testing-manifests/ingress"

	// GCEIngressManifestPath is the parent path to GCE-specific yaml test manifests.
	GCEIngressManifestPath = IngressManifestPath + "/gce"

	// IngressReqTimeout is the timeout on a single http request.
	IngressReqTimeout = 10 * time.Second

	// healthz port used to verify glbc restarted correctly on the master.
	glbcHealthzPort = 8086

	// General cloud resource poll timeout (eg: create static ip, firewall etc)
	cloudResourcePollTimeout = 5 * time.Minute

	// NEGAnnotation is NEG annotation.
	NEGAnnotation = "cloud.google.com/neg"

	// NEGStatusAnnotation is NEG status annotation.
	NEGStatusAnnotation = "cloud.google.com/neg-status"

	// StatusPrefix is prefix for annotation keys used by the ingress controller to specify the
	// names of GCP resources such as forwarding rules, url maps, target proxies, etc
	// that it created for the corresponding ingress.
	StatusPrefix = "ingress.kubernetes.io"
)

// TestLogger is an interface for log.
type TestLogger interface {
	Infof(format string, args ...interface{})
	Errorf(format string, args ...interface{})
}

// GLogger is test logger.
type GLogger struct{}

// Infof outputs log with info level.
func (l *GLogger) Infof(format string, args ...interface{}) {
	klog.Infof(format, args...)
}

// Errorf outputs log with error level.
func (l *GLogger) Errorf(format string, args ...interface{}) {
	klog.Errorf(format, args...)
}

// E2ELogger is test logger.
type E2ELogger struct{}

// Infof outputs log.
func (l *E2ELogger) Infof(format string, args ...interface{}) {
	e2elog.Logf(format, args...)
}

// Errorf outputs log.
func (l *E2ELogger) Errorf(format string, args ...interface{}) {
	e2elog.Logf(format, args...)
}

// ConformanceTests contains a closure with an entry and exit log line.
type ConformanceTests struct {
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
func CreateIngressComformanceTests(jig *TestJig, ns string, annotations map[string]string) []ConformanceTests {
	manifestPath := filepath.Join(IngressManifestPath, "http")
	// These constants match the manifests used in IngressManifestPath
	tlsHost := "foo.bar.com"
	tlsSecretName := "foo"
	updatedTLSHost := "foobar.com"
	updateURLMapHost := "bar.baz.com"
	updateURLMapPath := "/testurl"
	// Platform agnostic list of tests that must be satisfied by all controllers
	tests := []ConformanceTests{
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
				jig.Update(func(ing *networkingv1beta1.Ingress) {
					newRules := []networkingv1beta1.IngressRule{}
					for _, rule := range ing.Spec.Rules {
						if rule.Host != updateURLMapHost {
							newRules = append(newRules, rule)
							continue
						}
						existingPath := rule.IngressRuleValue.HTTP.Paths[0]
						pathToFail = existingPath.Path
						newRules = append(newRules, networkingv1beta1.IngressRule{
							Host: updateURLMapHost,
							IngressRuleValue: networkingv1beta1.IngressRuleValue{
								HTTP: &networkingv1beta1.HTTPIngressRuleValue{
									Paths: []networkingv1beta1.HTTPIngressPath{
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
				ginkgo.By("Checking that " + pathToFail + " is not exposed by polling for failure")
				route := fmt.Sprintf("http://%v%v", jig.Address, pathToFail)
				framework.ExpectNoError(framework.PollURL(route, updateURLMapHost, e2eservice.LoadBalancerCleanupTimeout, jig.PollInterval, &http.Client{Timeout: IngressReqTimeout}, true))
			},
			fmt.Sprintf("Waiting for path updates to reflect in L7"),
		},
	}
	// Skip the Update TLS cert test for kubemci: https://github.com/GoogleCloudPlatform/k8s-multicluster-ingress/issues/141.
	if jig.Class != MulticlusterIngressClassValue {
		tests = append(tests, ConformanceTests{
			fmt.Sprintf("should update SSL certificate with modified hostname %v", updatedTLSHost),
			func() {
				jig.Update(func(ing *networkingv1beta1.Ingress) {
					newRules := []networkingv1beta1.IngressRule{}
					for _, rule := range ing.Spec.Rules {
						if rule.Host != tlsHost {
							newRules = append(newRules, rule)
							continue
						}
						newRules = append(newRules, networkingv1beta1.IngressRule{
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
		return nil, nil, fmt.Errorf("Failed creating key: %v", err)
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
	e2elog.Logf("Generating RSA cert for host %v", host)
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
		e2elog.Logf("Updating secret %v in ns %v with hosts %v", secret.Name, namespace, host)
		s.Data = secret.Data
		_, err = kubeClient.CoreV1().Secrets(namespace).Update(s)
	} else {
		e2elog.Logf("Creating secret %v in ns %v with hosts %v", secret.Name, namespace, host)
		_, err = kubeClient.CoreV1().Secrets(namespace).Create(secret)
	}
	return host, cert, key, err
}

// TestJig holds the relevant state and parameters of the ingress test.
type TestJig struct {
	Client clientset.Interface
	Logger TestLogger

	RootCAs map[string][]byte
	Address string
	Ingress *networkingv1beta1.Ingress
	// class is the value of the annotation keyed under
	// `kubernetes.io/ingress.class`. It's added to all ingresses created by
	// this jig.
	Class string

	// The interval used to poll urls
	PollInterval time.Duration
}

// NewIngressTestJig instantiates struct with client
func NewIngressTestJig(c clientset.Interface) *TestJig {
	return &TestJig{
		Client:       c,
		RootCAs:      map[string][]byte{},
		PollInterval: e2eservice.LoadBalancerPollInterval,
		Logger:       &E2ELogger{},
	}
}

// CreateIngress creates the Ingress and associated service/rc.
// Required: ing.yaml, rc.yaml, svc.yaml must exist in manifestPath
// Optional: secret.yaml, ingAnnotations
// If ingAnnotations is specified it will overwrite any annotations in ing.yaml
// If svcAnnotations is specified it will overwrite any annotations in svc.yaml
func (j *TestJig) CreateIngress(manifestPath, ns string, ingAnnotations map[string]string, svcAnnotations map[string]string) {
	var err error
	read := func(file string) string {
		return string(testfiles.ReadOrDie(filepath.Join(manifestPath, file), ginkgo.Fail))
	}
	exists := func(file string) bool {
		return testfiles.Exists(filepath.Join(manifestPath, file), ginkgo.Fail)
	}

	j.Logger.Infof("creating replication controller")
	framework.RunKubectlOrDieInput(read("rc.yaml"), "create", "-f", "-", fmt.Sprintf("--namespace=%v", ns))

	j.Logger.Infof("creating service")
	framework.RunKubectlOrDieInput(read("svc.yaml"), "create", "-f", "-", fmt.Sprintf("--namespace=%v", ns))
	if len(svcAnnotations) > 0 {
		svcList, err := j.Client.CoreV1().Services(ns).List(metav1.ListOptions{})
		framework.ExpectNoError(err)
		for _, svc := range svcList.Items {
			svc.Annotations = svcAnnotations
			_, err = j.Client.CoreV1().Services(ns).Update(&svc)
			framework.ExpectNoError(err)
		}
	}

	if exists("secret.yaml") {
		j.Logger.Infof("creating secret")
		framework.RunKubectlOrDieInput(read("secret.yaml"), "create", "-f", "-", fmt.Sprintf("--namespace=%v", ns))
	}
	j.Logger.Infof("Parsing ingress from %v", filepath.Join(manifestPath, "ing.yaml"))

	j.Ingress, err = manifest.IngressFromManifest(filepath.Join(manifestPath, "ing.yaml"))
	framework.ExpectNoError(err)
	j.Ingress.Namespace = ns
	j.Ingress.Annotations = map[string]string{IngressClassKey: j.Class}
	for k, v := range ingAnnotations {
		j.Ingress.Annotations[k] = v
	}
	j.Logger.Infof(fmt.Sprintf("creating " + j.Ingress.Name + " ingress"))
	j.Ingress, err = j.runCreate(j.Ingress)
	framework.ExpectNoError(err)
}

// runCreate runs the required command to create the given ingress.
func (j *TestJig) runCreate(ing *networkingv1beta1.Ingress) (*networkingv1beta1.Ingress, error) {
	if j.Class != MulticlusterIngressClassValue {
		return j.Client.NetworkingV1beta1().Ingresses(ing.Namespace).Create(ing)
	}
	// Use kubemci to create a multicluster ingress.
	filePath := framework.TestContext.OutputDir + "/mci.yaml"
	if err := manifest.IngressToManifest(ing, filePath); err != nil {
		return nil, err
	}
	_, err := framework.RunKubemciWithKubeconfig("create", ing.Name, fmt.Sprintf("--ingress=%s", filePath))
	return ing, err
}

// runUpdate runs the required command to update the given ingress.
func (j *TestJig) runUpdate(ing *networkingv1beta1.Ingress) (*networkingv1beta1.Ingress, error) {
	if j.Class != MulticlusterIngressClassValue {
		return j.Client.NetworkingV1beta1().Ingresses(ing.Namespace).Update(ing)
	}
	// Use kubemci to update a multicluster ingress.
	// kubemci does not have an update command. We use "create --force" to update an existing ingress.
	filePath := framework.TestContext.OutputDir + "/mci.yaml"
	if err := manifest.IngressToManifest(ing, filePath); err != nil {
		return nil, err
	}
	_, err := framework.RunKubemciWithKubeconfig("create", ing.Name, fmt.Sprintf("--ingress=%s", filePath), "--force")
	return ing, err
}

// Update retrieves the ingress, performs the passed function, and then updates it.
func (j *TestJig) Update(update func(ing *networkingv1beta1.Ingress)) {
	var err error
	ns, name := j.Ingress.Namespace, j.Ingress.Name
	for i := 0; i < 3; i++ {
		j.Ingress, err = j.Client.NetworkingV1beta1().Ingresses(ns).Get(name, metav1.GetOptions{})
		if err != nil {
			e2elog.Failf("failed to get ingress %s/%s: %v", ns, name, err)
		}
		update(j.Ingress)
		j.Ingress, err = j.runUpdate(j.Ingress)
		if err == nil {
			framework.DescribeIng(j.Ingress.Namespace)
			return
		}
		if !apierrs.IsConflict(err) && !apierrs.IsServerTimeout(err) {
			e2elog.Failf("failed to update ingress %s/%s: %v", ns, name, err)
		}
	}
	e2elog.Failf("too many retries updating ingress %s/%s", ns, name)
}

// AddHTTPS updates the ingress to add this secret for these hosts.
func (j *TestJig) AddHTTPS(secretName string, hosts ...string) {
	// TODO: Just create the secret in GetRootCAs once we're watching secrets in
	// the ingress controller.
	_, cert, _, err := createTLSSecret(j.Client, j.Ingress.Namespace, secretName, hosts...)
	framework.ExpectNoError(err)
	j.Logger.Infof("Updating ingress %v to also use secret %v for TLS termination", j.Ingress.Name, secretName)
	j.Update(func(ing *networkingv1beta1.Ingress) {
		ing.Spec.TLS = append(ing.Spec.TLS, networkingv1beta1.IngressTLS{Hosts: hosts, SecretName: secretName})
	})
	j.RootCAs[secretName] = cert
}

// SetHTTPS updates the ingress to use only this secret for these hosts.
func (j *TestJig) SetHTTPS(secretName string, hosts ...string) {
	_, cert, _, err := createTLSSecret(j.Client, j.Ingress.Namespace, secretName, hosts...)
	framework.ExpectNoError(err)
	j.Logger.Infof("Updating ingress %v to only use secret %v for TLS termination", j.Ingress.Name, secretName)
	j.Update(func(ing *networkingv1beta1.Ingress) {
		ing.Spec.TLS = []networkingv1beta1.IngressTLS{{Hosts: hosts, SecretName: secretName}}
	})
	j.RootCAs = map[string][]byte{secretName: cert}
}

// RemoveHTTPS updates the ingress to not use this secret for TLS.
// Note: Does not delete the secret.
func (j *TestJig) RemoveHTTPS(secretName string) {
	newTLS := []networkingv1beta1.IngressTLS{}
	for _, ingressTLS := range j.Ingress.Spec.TLS {
		if secretName != ingressTLS.SecretName {
			newTLS = append(newTLS, ingressTLS)
		}
	}
	j.Logger.Infof("Updating ingress %v to not use secret %v for TLS termination", j.Ingress.Name, secretName)
	j.Update(func(ing *networkingv1beta1.Ingress) {
		ing.Spec.TLS = newTLS
	})
	delete(j.RootCAs, secretName)
}

// PrepareTLSSecret creates a TLS secret and caches the cert.
func (j *TestJig) PrepareTLSSecret(namespace, secretName string, hosts ...string) error {
	_, cert, _, err := createTLSSecret(j.Client, namespace, secretName, hosts...)
	if err != nil {
		return err
	}
	j.RootCAs[secretName] = cert
	return nil
}

// GetRootCA returns a rootCA from the ingress test jig.
func (j *TestJig) GetRootCA(secretName string) (rootCA []byte) {
	var ok bool
	rootCA, ok = j.RootCAs[secretName]
	if !ok {
		e2elog.Failf("Failed to retrieve rootCAs, no recorded secret by name %v", secretName)
	}
	return
}

// TryDeleteIngress attempts to delete the ingress resource and logs errors if they occur.
func (j *TestJig) TryDeleteIngress() {
	j.tryDeleteGivenIngress(j.Ingress)
}

func (j *TestJig) tryDeleteGivenIngress(ing *networkingv1beta1.Ingress) {
	if err := j.runDelete(ing); err != nil {
		j.Logger.Infof("Error while deleting the ingress %v/%v with class %s: %v", ing.Namespace, ing.Name, j.Class, err)
	}
}

// runDelete runs the required command to delete the given ingress.
func (j *TestJig) runDelete(ing *networkingv1beta1.Ingress) error {
	if j.Class != MulticlusterIngressClassValue {
		return j.Client.NetworkingV1beta1().Ingresses(ing.Namespace).Delete(ing.Name, nil)
	}
	// Use kubemci to delete a multicluster ingress.
	filePath := framework.TestContext.OutputDir + "/mci.yaml"
	if err := manifest.IngressToManifest(ing, filePath); err != nil {
		return err
	}
	_, err := framework.RunKubemciWithKubeconfig("delete", ing.Name, fmt.Sprintf("--ingress=%s", filePath))
	return err
}

// getIngressAddressFromKubemci returns the IP address of the given multicluster ingress using kubemci.
// TODO(nikhiljindal): Update this to be able to return hostname as well.
func getIngressAddressFromKubemci(name string) ([]string, error) {
	var addresses []string
	out, err := framework.RunKubemciCmd("get-status", name)
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
	ing, err := client.NetworkingV1beta1().Ingresses(ns).Get(name, metav1.GetOptions{})
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
func (j *TestJig) WaitForIngressAddress(c clientset.Interface, ns, ingName string, timeout time.Duration) (string, error) {
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

func (j *TestJig) pollIngressWithCert(ing *networkingv1beta1.Ingress, address string, knownHosts []string, cert []byte, waitForNodePort bool, timeout time.Duration) error {
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
			if err := framework.PollURL(route, rules.Host, timeout, j.PollInterval, timeoutClient, false); err != nil {
				return err
			}
		}
	}
	j.Logger.Infof("Finished polling on all rules on ingress %q", ing.Name)
	return nil
}

// WaitForIngress waits for the Ingress to get an address.
// WaitForIngress returns when it gets the first 200 response
func (j *TestJig) WaitForIngress(waitForNodePort bool) {
	if err := j.WaitForGivenIngressWithTimeout(j.Ingress, waitForNodePort, e2eservice.LoadBalancerPollTimeout); err != nil {
		e2elog.Failf("error in waiting for ingress to get an address: %s", err)
	}
}

// WaitForIngressToStable waits for the LB return 100 consecutive 200 responses.
func (j *TestJig) WaitForIngressToStable() {
	if err := wait.Poll(10*time.Second, e2eservice.LoadBalancerCreateTimeoutDefault, func() (bool, error) {
		_, err := j.GetDistinctResponseFromIngress()
		if err != nil {
			return false, nil
		}
		return true, nil
	}); err != nil {
		e2elog.Failf("error in waiting for ingress to stablize: %v", err)
	}
}

// WaitForGivenIngressWithTimeout waits till the ingress acquires an IP,
// then waits for its hosts/urls to respond to a protocol check (either
// http or https). If waitForNodePort is true, the NodePort of the Service
// is verified before verifying the Ingress. NodePort is currently a
// requirement for cloudprovider Ingress.
func (j *TestJig) WaitForGivenIngressWithTimeout(ing *networkingv1beta1.Ingress, waitForNodePort bool, timeout time.Duration) error {
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

// WaitForIngressWithCert waits till the ingress acquires an IP, then waits for its
// hosts/urls to respond to a protocol check (either http or https). If
// waitForNodePort is true, the NodePort of the Service is verified before
// verifying the Ingress. NodePort is currently a requirement for cloudprovider
// Ingress. Hostnames and certificate need to be explicitly passed in.
func (j *TestJig) WaitForIngressWithCert(waitForNodePort bool, knownHosts []string, cert []byte) error {
	// Wait for the loadbalancer IP.
	address, err := j.WaitForIngressAddress(j.Client, j.Ingress.Namespace, j.Ingress.Name, e2eservice.LoadBalancerPollTimeout)
	if err != nil {
		return fmt.Errorf("Ingress failed to acquire an IP address within %v", e2eservice.LoadBalancerPollTimeout)
	}

	return j.pollIngressWithCert(j.Ingress, address, knownHosts, cert, waitForNodePort, e2eservice.LoadBalancerPollTimeout)
}

// VerifyURL polls for the given iterations, in intervals, and fails if the
// given url returns a non-healthy http code even once.
func (j *TestJig) VerifyURL(route, host string, iterations int, interval time.Duration, httpClient *http.Client) error {
	for i := 0; i < iterations; i++ {
		b, err := framework.SimpleGET(httpClient, route, host)
		if err != nil {
			e2elog.Logf(b)
			return err
		}
		j.Logger.Infof("Verified %v with host %v %d times, sleeping for %v", route, host, i, interval)
		time.Sleep(interval)
	}
	return nil
}

func (j *TestJig) pollServiceNodePort(ns, name string, port int) error {
	// TODO: Curl all nodes?
	u, err := e2enode.GetPortURL(j.Client, ns, name, port)
	if err != nil {
		return err
	}
	return framework.PollURL(u, "", 30*time.Second, j.PollInterval, &http.Client{Timeout: IngressReqTimeout}, false)
}

// GetIngressNodePorts returns related backend services' nodePorts.
// Current GCE ingress controller allows traffic to the default HTTP backend
// by default, so retrieve its nodePort if includeDefaultBackend is true.
func (j *TestJig) GetIngressNodePorts(includeDefaultBackend bool) []string {
	nodePorts := []string{}
	svcPorts := j.GetServicePorts(includeDefaultBackend)
	for _, svcPort := range svcPorts {
		nodePorts = append(nodePorts, strconv.Itoa(int(svcPort.NodePort)))
	}
	return nodePorts
}

// GetServicePorts returns related backend services' svcPorts.
// Current GCE ingress controller allows traffic to the default HTTP backend
// by default, so retrieve its nodePort if includeDefaultBackend is true.
func (j *TestJig) GetServicePorts(includeDefaultBackend bool) map[string]v1.ServicePort {
	svcPorts := make(map[string]v1.ServicePort)
	if includeDefaultBackend {
		defaultSvc, err := j.Client.CoreV1().Services(metav1.NamespaceSystem).Get(defaultBackendName, metav1.GetOptions{})
		framework.ExpectNoError(err)
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
		framework.ExpectNoError(err)
		svcPorts[svcName] = svc.Spec.Ports[0]
	}
	return svcPorts
}

// ConstructFirewallForIngress returns the expected GCE firewall rule for the ingress resource
func (j *TestJig) ConstructFirewallForIngress(firewallRuleName string, nodeTags []string) *compute.Firewall {
	nodePorts := j.GetIngressNodePorts(true)

	fw := compute.Firewall{}
	fw.Name = firewallRuleName
	fw.SourceRanges = framework.TestContext.CloudConfig.Provider.LoadBalancerSrcRanges()
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
func (j *TestJig) GetDistinctResponseFromIngress() (sets.String, error) {
	// Wait for the loadbalancer IP.
	address, err := j.WaitForIngressAddress(j.Client, j.Ingress.Namespace, j.Ingress.Name, e2eservice.LoadBalancerPollTimeout)
	if err != nil {
		e2elog.Failf("Ingress failed to acquire an IP address within %v", e2eservice.LoadBalancerPollTimeout)
	}
	responses := sets.NewString()
	timeoutClient := &http.Client{Timeout: IngressReqTimeout}

	for i := 0; i < 100; i++ {
		url := fmt.Sprintf("http://%v", address)
		res, err := framework.SimpleGET(timeoutClient, url, "")
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
	read := func(file string) string {
		return string(testfiles.ReadOrDie(filepath.Join(IngressManifestPath, "nginx", file), ginkgo.Fail))
	}
	e2elog.Logf("initializing nginx ingress controller")
	framework.RunKubectlOrDieInput(read("rc.yaml"), "create", "-f", "-", fmt.Sprintf("--namespace=%v", cont.Ns))

	rc, err := cont.Client.CoreV1().ReplicationControllers(cont.Ns).Get("nginx-ingress-controller", metav1.GetOptions{})
	framework.ExpectNoError(err)
	cont.rc = rc

	e2elog.Logf("waiting for pods with label %v", rc.Spec.Selector)
	sel := labels.SelectorFromSet(labels.Set(rc.Spec.Selector))
	framework.ExpectNoError(testutils.WaitForPodsWithLabelRunning(cont.Client, cont.Ns, sel))
	pods, err := cont.Client.CoreV1().Pods(cont.Ns).List(metav1.ListOptions{LabelSelector: sel.String()})
	framework.ExpectNoError(err)
	if len(pods.Items) == 0 {
		e2elog.Failf("Failed to find nginx ingress controller pods with selector %v", sel)
	}
	cont.pod = &pods.Items[0]
	cont.externalIP, err = framework.GetHostExternalAddress(cont.Client, cont.pod)
	framework.ExpectNoError(err)
	e2elog.Logf("ingress controller running in pod %v on ip %v", cont.pod.Name, cont.externalIP)
}

func generateBacksideHTTPSIngressSpec(ns string) *networkingv1beta1.Ingress {
	return &networkingv1beta1.Ingress{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "echoheaders-https",
			Namespace: ns,
		},
		Spec: networkingv1beta1.IngressSpec{
			// Note kubemci requires a default backend.
			Backend: &networkingv1beta1.IngressBackend{
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

func generateBacksideHTTPSDeploymentSpec() *appsv1.Deployment {
	return &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name: "echoheaders-https",
		},
		Spec: appsv1.DeploymentSpec{
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
func (j *TestJig) SetUpBacksideHTTPSIngress(cs clientset.Interface, namespace string, staticIPName string) (*appsv1.Deployment, *v1.Service, *networkingv1beta1.Ingress, error) {
	deployCreated, err := cs.AppsV1().Deployments(namespace).Create(generateBacksideHTTPSDeploymentSpec())
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
func (j *TestJig) DeleteTestResource(cs clientset.Interface, deploy *appsv1.Deployment, svc *v1.Service, ing *networkingv1beta1.Ingress) []error {
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
		if err := cs.AppsV1().Deployments(deploy.Namespace).Delete(deploy.Name, nil); err != nil {
			errs = append(errs, fmt.Errorf("error while deleting deployment %s/%s: %v", deploy.Namespace, deploy.Name, err))
		}
	}
	return errs
}
