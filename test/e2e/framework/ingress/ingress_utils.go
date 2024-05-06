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
	"context"
	"crypto/rand"
	"crypto/rsa"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"fmt"
	"io"
	"math/big"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"

	compute "google.golang.org/api/compute/v1"
	netutils "k8s.io/utils/net"

	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	networkingv1 "k8s.io/api/networking/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/intstr"
	utilnet "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	utilyaml "k8s.io/apimachinery/pkg/util/yaml"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/kubernetes/test/e2e/framework"
	e2edeployment "k8s.io/kubernetes/test/e2e/framework/deployment"
	e2ekubectl "k8s.io/kubernetes/test/e2e/framework/kubectl"
	e2eservice "k8s.io/kubernetes/test/e2e/framework/service"
	e2etestfiles "k8s.io/kubernetes/test/e2e/framework/testfiles"
	testutils "k8s.io/kubernetes/test/utils"
	imageutils "k8s.io/kubernetes/test/utils/image"

	"github.com/onsi/ginkgo/v2"
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

	// NEGAnnotation is NEG annotation.
	NEGAnnotation = "cloud.google.com/neg"

	// NEGStatusAnnotation is NEG status annotation.
	NEGStatusAnnotation = "cloud.google.com/neg-status"

	// StatusPrefix is prefix for annotation keys used by the ingress controller to specify the
	// names of GCP resources such as forwarding rules, url maps, target proxies, etc
	// that it created for the corresponding ingress.
	StatusPrefix = "ingress.kubernetes.io"

	// poll is how often to Poll pods, nodes and claims.
	poll = 2 * time.Second
)

// TestLogger is an interface for log.
type TestLogger interface {
	Infof(format string, args ...interface{})
	Errorf(format string, args ...interface{})
}

// E2ELogger is test logger.
type E2ELogger struct{}

// Infof outputs log.
func (l *E2ELogger) Infof(format string, args ...interface{}) {
	framework.Logf(format, args...)
}

// Errorf outputs log.
func (l *E2ELogger) Errorf(format string, args ...interface{}) {
	framework.Logf(format, args...)
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

// SimpleGET executes a get on the given url, returns error if non-200 returned.
func SimpleGET(ctx context.Context, c *http.Client, url, host string) (string, error) {
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return "", err
	}
	req.Host = host
	res, err := c.Do(req)
	if err != nil {
		return "", err
	}
	defer res.Body.Close()
	rawBody, err := io.ReadAll(res.Body)
	if err != nil {
		return "", err
	}
	body := string(rawBody)
	if res.StatusCode != http.StatusOK {
		err = fmt.Errorf(
			"GET returned http error %v", res.StatusCode)
	}
	return body, err
}

// PollURL polls till the url responds with a healthy http code. If
// expectUnreachable is true, it breaks on first non-healthy http code instead.
func PollURL(ctx context.Context, route, host string, timeout time.Duration, interval time.Duration, httpClient *http.Client, expectUnreachable bool) error {
	var lastBody string
	pollErr := wait.PollUntilContextTimeout(ctx, interval, timeout, true, func(ctx context.Context) (bool, error) {
		var err error
		lastBody, err = SimpleGET(ctx, httpClient, route, host)
		if err != nil {
			framework.Logf("host %v path %v: %v unreachable", host, route, err)
			return expectUnreachable, nil
		}
		framework.Logf("host %v path %v: reached", host, route)
		return !expectUnreachable, nil
	})
	if pollErr != nil {
		return fmt.Errorf("Failed to execute a successful GET within %v, Last response body for %v, host %v:\n%v\n\n%v",
			timeout, route, host, lastBody, pollErr)
	}
	return nil
}

// CreateIngressComformanceTests generates an slice of sequential test cases:
// a simple http ingress, ingress with HTTPS, ingress HTTPS with a modified hostname,
// ingress https with a modified URLMap
func CreateIngressComformanceTests(ctx context.Context, jig *TestJig, ns string, annotations map[string]string) []ConformanceTests {
	manifestPath := filepath.Join(IngressManifestPath, "http")
	// These constants match the manifests used in IngressManifestPath
	tlsHost := "foo.bar.com"
	tlsSecretName := "foo"
	updatedTLSHost := "foobar.com"
	updateURLMapHost := "bar.baz.com"
	updateURLMapPath := "/testurl"
	prefixPathType := networkingv1.PathTypeImplementationSpecific
	// Platform agnostic list of tests that must be satisfied by all controllers
	tests := []ConformanceTests{
		{
			fmt.Sprintf("should create a basic HTTP ingress"),
			func() { jig.CreateIngress(ctx, manifestPath, ns, annotations, annotations) },
			fmt.Sprintf("waiting for urls on basic HTTP ingress"),
		},
		{
			fmt.Sprintf("should terminate TLS for host %v", tlsHost),
			func() { jig.SetHTTPS(ctx, tlsSecretName, tlsHost) },
			fmt.Sprintf("waiting for HTTPS updates to reflect in ingress"),
		},
		{
			fmt.Sprintf("should update url map for host %v to expose a single url: %v", updateURLMapHost, updateURLMapPath),
			func() {
				var pathToFail string
				jig.Update(ctx, func(ing *networkingv1.Ingress) {
					newRules := []networkingv1.IngressRule{}
					for _, rule := range ing.Spec.Rules {
						if rule.Host != updateURLMapHost {
							newRules = append(newRules, rule)
							continue
						}
						existingPath := rule.IngressRuleValue.HTTP.Paths[0]
						pathToFail = existingPath.Path
						newRules = append(newRules, networkingv1.IngressRule{
							Host: updateURLMapHost,
							IngressRuleValue: networkingv1.IngressRuleValue{
								HTTP: &networkingv1.HTTPIngressRuleValue{
									Paths: []networkingv1.HTTPIngressPath{
										{
											Path:     updateURLMapPath,
											PathType: &prefixPathType,
											Backend:  existingPath.Backend,
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
				framework.ExpectNoError(PollURL(ctx, route, updateURLMapHost, e2eservice.LoadBalancerCleanupTimeout, jig.PollInterval, &http.Client{Timeout: IngressReqTimeout}, true))
			},
			fmt.Sprintf("Waiting for path updates to reflect in L7"),
		},
	}
	// Skip the Update TLS cert test for kubemci: https://github.com/GoogleCloudPlatform/k8s-multicluster-ingress/issues/141.
	if jig.Class != MulticlusterIngressClassValue {
		tests = append(tests, ConformanceTests{
			fmt.Sprintf("should update SSL certificate with modified hostname %v", updatedTLSHost),
			func() {
				jig.Update(ctx, func(ing *networkingv1.Ingress) {
					newRules := []networkingv1.IngressRule{}
					for _, rule := range ing.Spec.Rules {
						if rule.Host != tlsHost {
							newRules = append(newRules, rule)
							continue
						}
						newRules = append(newRules, networkingv1.IngressRule{
							Host:             updatedTLSHost,
							IngressRuleValue: rule.IngressRuleValue,
						})
					}
					ing.Spec.Rules = newRules
				})
				jig.SetHTTPS(ctx, tlsSecretName, updatedTLSHost)
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
		return nil, nil, fmt.Errorf("Failed to generate key: %w", err)
	}
	notBefore := time.Now()
	notAfter := notBefore.Add(validFor)

	serialNumberLimit := new(big.Int).Lsh(big.NewInt(1), 128)
	serialNumber, err := rand.Int(rand.Reader, serialNumberLimit)

	if err != nil {
		return nil, nil, fmt.Errorf("failed to generate serial number: %w", err)
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
		if ip := netutils.ParseIPSloppy(h); ip != nil {
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
		return nil, nil, fmt.Errorf("Failed to create certificate: %w", err)
	}
	if err := pem.Encode(&certOut, &pem.Block{Type: "CERTIFICATE", Bytes: derBytes}); err != nil {
		return nil, nil, fmt.Errorf("Failed creating cert: %w", err)
	}
	if err := pem.Encode(&keyOut, &pem.Block{Type: "RSA PRIVATE KEY", Bytes: x509.MarshalPKCS1PrivateKey(priv)}); err != nil {
		return nil, nil, fmt.Errorf("Failed creating key: %w", err)
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
func createTLSSecret(ctx context.Context, kubeClient clientset.Interface, namespace, secretName string, hosts ...string) (host string, rootCA, privKey []byte, err error) {
	host = strings.Join(hosts, ",")
	framework.Logf("Generating RSA cert for host %v", host)
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
	if s, err = kubeClient.CoreV1().Secrets(namespace).Get(ctx, secretName, metav1.GetOptions{}); err == nil {
		framework.Logf("Updating secret %v in ns %v with hosts %v", secret.Name, namespace, host)
		s.Data = secret.Data
		_, err = kubeClient.CoreV1().Secrets(namespace).Update(ctx, s, metav1.UpdateOptions{})
	} else {
		framework.Logf("Creating secret %v in ns %v with hosts %v", secret.Name, namespace, host)
		_, err = kubeClient.CoreV1().Secrets(namespace).Create(ctx, secret, metav1.CreateOptions{})
	}
	return host, cert, key, err
}

// TestJig holds the relevant state and parameters of the ingress test.
type TestJig struct {
	Client clientset.Interface
	Logger TestLogger

	RootCAs map[string][]byte
	Address string
	Ingress *networkingv1.Ingress
	// class was the value of the annotation keyed under `kubernetes.io/ingress.class`.
	// A new ingressClassName field has been added that is used to reference the IngressClass.
	// It's added to all ingresses created by this jig.
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
func (j *TestJig) CreateIngress(ctx context.Context, manifestPath, ns string, ingAnnotations map[string]string, svcAnnotations map[string]string) {
	var err error
	read := func(file string) string {
		data, err := e2etestfiles.Read(filepath.Join(manifestPath, file))
		if err != nil {
			framework.Fail(err.Error())
		}
		return string(data)
	}
	exists := func(file string) bool {
		found, err := e2etestfiles.Exists(filepath.Join(manifestPath, file))
		if err != nil {
			framework.Fail(fmt.Sprintf("fatal error looking for test file %s: %s", file, err))
		}
		return found
	}

	j.Logger.Infof("creating replication controller")
	e2ekubectl.RunKubectlOrDieInput(ns, read("rc.yaml"), "create", "-f", "-")

	j.Logger.Infof("creating service")
	e2ekubectl.RunKubectlOrDieInput(ns, read("svc.yaml"), "create", "-f", "-")
	if len(svcAnnotations) > 0 {
		svcList, err := j.Client.CoreV1().Services(ns).List(ctx, metav1.ListOptions{})
		framework.ExpectNoError(err)
		for _, svc := range svcList.Items {
			svc.Annotations = svcAnnotations
			_, err = j.Client.CoreV1().Services(ns).Update(ctx, &svc, metav1.UpdateOptions{})
			framework.ExpectNoError(err)
		}
	}

	if exists("secret.yaml") {
		j.Logger.Infof("creating secret")
		e2ekubectl.RunKubectlOrDieInput(ns, read("secret.yaml"), "create", "-f", "-")
	}
	j.Logger.Infof("Parsing ingress from %v", filepath.Join(manifestPath, "ing.yaml"))

	j.Ingress, err = ingressFromManifest(filepath.Join(manifestPath, "ing.yaml"))
	framework.ExpectNoError(err)
	j.Ingress.Namespace = ns
	if j.Class != "" {
		j.Ingress.Spec.IngressClassName = &j.Class
	}
	j.Logger.Infof("creating %v ingress", j.Ingress.Name)
	j.Ingress, err = j.runCreate(ctx, j.Ingress)
	framework.ExpectNoError(err)
}

// marshalToYaml marshals an object into YAML for a given GroupVersion.
// The object must be known in SupportedMediaTypes() for the Codecs under "client-go/kubernetes/scheme".
func marshalToYaml(obj runtime.Object, gv schema.GroupVersion) ([]byte, error) {
	mediaType := "application/yaml"
	info, ok := runtime.SerializerInfoForMediaType(scheme.Codecs.SupportedMediaTypes(), mediaType)
	if !ok {
		return []byte{}, fmt.Errorf("unsupported media type %q", mediaType)
	}
	encoder := scheme.Codecs.EncoderForVersion(info.Serializer, gv)
	return runtime.Encode(encoder, obj)
}

// ingressFromManifest reads a .json/yaml file and returns the ingress in it.
func ingressFromManifest(fileName string) (*networkingv1.Ingress, error) {
	var ing networkingv1.Ingress
	data, err := e2etestfiles.Read(fileName)
	if err != nil {
		return nil, err
	}

	json, err := utilyaml.ToJSON(data)
	if err != nil {
		return nil, err
	}
	if err := runtime.DecodeInto(scheme.Codecs.UniversalDecoder(), json, &ing); err != nil {
		return nil, err
	}
	return &ing, nil
}

// ingressToManifest generates a yaml file in the given path with the given ingress.
// Assumes that a directory exists at the given path.
func ingressToManifest(ing *networkingv1.Ingress, path string) error {
	serialized, err := marshalToYaml(ing, networkingv1.SchemeGroupVersion)
	if err != nil {
		return fmt.Errorf("failed to marshal ingress %v to YAML: %w", ing, err)
	}

	if err := os.WriteFile(path, serialized, 0600); err != nil {
		return fmt.Errorf("error in writing ingress to file: %w", err)
	}
	return nil
}

// runCreate runs the required command to create the given ingress.
func (j *TestJig) runCreate(ctx context.Context, ing *networkingv1.Ingress) (*networkingv1.Ingress, error) {
	if j.Class != MulticlusterIngressClassValue {
		return j.Client.NetworkingV1().Ingresses(ing.Namespace).Create(ctx, ing, metav1.CreateOptions{})
	}
	// Use kubemci to create a multicluster ingress.
	filePath := framework.TestContext.OutputDir + "/mci.yaml"
	if err := ingressToManifest(ing, filePath); err != nil {
		return nil, err
	}
	_, err := e2ekubectl.RunKubemciWithKubeconfig("create", ing.Name, fmt.Sprintf("--ingress=%s", filePath))
	return ing, err
}

// runUpdate runs the required command to update the given ingress.
func (j *TestJig) runUpdate(ctx context.Context, ing *networkingv1.Ingress) (*networkingv1.Ingress, error) {
	if j.Class != MulticlusterIngressClassValue {
		return j.Client.NetworkingV1().Ingresses(ing.Namespace).Update(ctx, ing, metav1.UpdateOptions{})
	}
	// Use kubemci to update a multicluster ingress.
	// kubemci does not have an update command. We use "create --force" to update an existing ingress.
	filePath := framework.TestContext.OutputDir + "/mci.yaml"
	if err := ingressToManifest(ing, filePath); err != nil {
		return nil, err
	}
	_, err := e2ekubectl.RunKubemciWithKubeconfig("create", ing.Name, fmt.Sprintf("--ingress=%s", filePath), "--force")
	return ing, err
}

// DescribeIng describes information of ingress by running kubectl describe ing.
func DescribeIng(ns string) {
	framework.Logf("\nOutput of kubectl describe ing:\n")
	desc, _ := e2ekubectl.RunKubectl(
		ns, "describe", "ing")
	framework.Logf(desc)
}

// Update retrieves the ingress, performs the passed function, and then updates it.
func (j *TestJig) Update(ctx context.Context, update func(ing *networkingv1.Ingress)) {
	var err error
	ns, name := j.Ingress.Namespace, j.Ingress.Name
	for i := 0; i < 3; i++ {
		j.Ingress, err = j.Client.NetworkingV1().Ingresses(ns).Get(ctx, name, metav1.GetOptions{})
		if err != nil {
			framework.Failf("failed to get ingress %s/%s: %v", ns, name, err)
		}
		update(j.Ingress)
		j.Ingress, err = j.runUpdate(ctx, j.Ingress)
		if err == nil {
			DescribeIng(j.Ingress.Namespace)
			return
		}
		if !apierrors.IsConflict(err) && !apierrors.IsServerTimeout(err) {
			framework.Failf("failed to update ingress %s/%s: %v", ns, name, err)
		}
	}
	framework.Failf("too many retries updating ingress %s/%s", ns, name)
}

// AddHTTPS updates the ingress to add this secret for these hosts.
func (j *TestJig) AddHTTPS(ctx context.Context, secretName string, hosts ...string) {
	// TODO: Just create the secret in GetRootCAs once we're watching secrets in
	// the ingress controller.
	_, cert, _, err := createTLSSecret(ctx, j.Client, j.Ingress.Namespace, secretName, hosts...)
	framework.ExpectNoError(err)
	j.Logger.Infof("Updating ingress %v to also use secret %v for TLS termination", j.Ingress.Name, secretName)
	j.Update(ctx, func(ing *networkingv1.Ingress) {
		ing.Spec.TLS = append(ing.Spec.TLS, networkingv1.IngressTLS{Hosts: hosts, SecretName: secretName})
	})
	j.RootCAs[secretName] = cert
}

// SetHTTPS updates the ingress to use only this secret for these hosts.
func (j *TestJig) SetHTTPS(ctx context.Context, secretName string, hosts ...string) {
	_, cert, _, err := createTLSSecret(ctx, j.Client, j.Ingress.Namespace, secretName, hosts...)
	framework.ExpectNoError(err)
	j.Logger.Infof("Updating ingress %v to only use secret %v for TLS termination", j.Ingress.Name, secretName)
	j.Update(ctx, func(ing *networkingv1.Ingress) {
		ing.Spec.TLS = []networkingv1.IngressTLS{{Hosts: hosts, SecretName: secretName}}
	})
	j.RootCAs = map[string][]byte{secretName: cert}
}

// RemoveHTTPS updates the ingress to not use this secret for TLS.
// Note: Does not delete the secret.
func (j *TestJig) RemoveHTTPS(ctx context.Context, secretName string) {
	newTLS := []networkingv1.IngressTLS{}
	for _, ingressTLS := range j.Ingress.Spec.TLS {
		if secretName != ingressTLS.SecretName {
			newTLS = append(newTLS, ingressTLS)
		}
	}
	j.Logger.Infof("Updating ingress %v to not use secret %v for TLS termination", j.Ingress.Name, secretName)
	j.Update(ctx, func(ing *networkingv1.Ingress) {
		ing.Spec.TLS = newTLS
	})
	delete(j.RootCAs, secretName)
}

// PrepareTLSSecret creates a TLS secret and caches the cert.
func (j *TestJig) PrepareTLSSecret(ctx context.Context, namespace, secretName string, hosts ...string) error {
	_, cert, _, err := createTLSSecret(ctx, j.Client, namespace, secretName, hosts...)
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
		framework.Failf("Failed to retrieve rootCAs, no recorded secret by name %v", secretName)
	}
	return
}

// TryDeleteIngress attempts to delete the ingress resource and logs errors if they occur.
func (j *TestJig) TryDeleteIngress(ctx context.Context) {
	j.tryDeleteGivenIngress(ctx, j.Ingress)
}

func (j *TestJig) tryDeleteGivenIngress(ctx context.Context, ing *networkingv1.Ingress) {
	if err := j.runDelete(ctx, ing); err != nil {
		j.Logger.Infof("Error while deleting the ingress %v/%v with class %s: %v", ing.Namespace, ing.Name, j.Class, err)
	}
}

// runDelete runs the required command to delete the given ingress.
func (j *TestJig) runDelete(ctx context.Context, ing *networkingv1.Ingress) error {
	if j.Class != MulticlusterIngressClassValue {
		return j.Client.NetworkingV1().Ingresses(ing.Namespace).Delete(ctx, ing.Name, metav1.DeleteOptions{})
	}
	// Use kubemci to delete a multicluster ingress.
	filePath := framework.TestContext.OutputDir + "/mci.yaml"
	if err := ingressToManifest(ing, filePath); err != nil {
		return err
	}
	_, err := e2ekubectl.RunKubemciWithKubeconfig("delete", ing.Name, fmt.Sprintf("--ingress=%s", filePath))
	return err
}

// getIngressAddressFromKubemci returns the IP address of the given multicluster ingress using kubemci.
// TODO(nikhiljindal): Update this to be able to return hostname as well.
func getIngressAddressFromKubemci(name string) ([]string, error) {
	var addresses []string
	out, err := e2ekubectl.RunKubemciCmd("get-status", name)
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
func getIngressAddress(ctx context.Context, client clientset.Interface, ns, name, class string) ([]string, error) {
	if class == MulticlusterIngressClassValue {
		return getIngressAddressFromKubemci(name)
	}
	ing, err := client.NetworkingV1().Ingresses(ns).Get(ctx, name, metav1.GetOptions{})
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
func (j *TestJig) WaitForIngressAddress(ctx context.Context, c clientset.Interface, ns, ingName string, timeout time.Duration) (string, error) {
	var address string
	err := wait.PollUntilContextTimeout(ctx, 10*time.Second, timeout, true, func(ctx context.Context) (bool, error) {
		ipOrNameList, err := getIngressAddress(ctx, c, ns, ingName, j.Class)
		if err != nil || len(ipOrNameList) == 0 {
			j.Logger.Errorf("Waiting for Ingress %s/%s to acquire IP, error: %v, ipOrNameList: %v", ns, ingName, err, ipOrNameList)
			return false, err
		}
		address = ipOrNameList[0]
		j.Logger.Infof("Found address %s for ingress %s/%s", address, ns, ingName)
		return true, nil
	})
	return address, err
}

func (j *TestJig) pollIngressWithCert(ctx context.Context, ing *networkingv1.Ingress, address string, knownHosts []string, cert []byte, waitForNodePort bool, timeout time.Duration) error {
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
				nodePort := int(p.Backend.Service.Port.Number)
				if err := j.pollServiceNodePort(ctx, ing.Namespace, p.Backend.Service.Name, nodePort); err != nil {
					j.Logger.Infof("Error in waiting for nodeport %d on service %v/%v: %s", nodePort, ing.Namespace, p.Backend.Service.Name, err)
					return err
				}
			}
			route := fmt.Sprintf("%v://%v%v", proto, address, p.Path)
			j.Logger.Infof("Testing route %v host %v with simple GET", route, rules.Host)
			if err := PollURL(ctx, route, rules.Host, timeout, j.PollInterval, timeoutClient, false); err != nil {
				return err
			}
		}
	}
	j.Logger.Infof("Finished polling on all rules on ingress %q", ing.Name)
	return nil
}

// WaitForIngress waits for the Ingress to get an address.
// WaitForIngress returns when it gets the first 200 response
func (j *TestJig) WaitForIngress(ctx context.Context, waitForNodePort bool) {
	if err := j.WaitForGivenIngressWithTimeout(ctx, j.Ingress, waitForNodePort, e2eservice.GetServiceLoadBalancerPropagationTimeout(ctx, j.Client)); err != nil {
		framework.Failf("error in waiting for ingress to get an address: %s", err)
	}
}

// WaitForIngressToStable waits for the LB return 100 consecutive 200 responses.
func (j *TestJig) WaitForIngressToStable(ctx context.Context) {
	if err := wait.PollWithContext(ctx, 10*time.Second, e2eservice.GetServiceLoadBalancerPropagationTimeout(ctx, j.Client), func(ctx context.Context) (bool, error) {
		_, err := j.GetDistinctResponseFromIngress(ctx)
		if err != nil {
			return false, nil
		}
		return true, nil
	}); err != nil {
		framework.Failf("error in waiting for ingress to stabilize: %v", err)
	}
}

// WaitForGivenIngressWithTimeout waits till the ingress acquires an IP,
// then waits for its hosts/urls to respond to a protocol check (either
// http or https). If waitForNodePort is true, the NodePort of the Service
// is verified before verifying the Ingress. NodePort is currently a
// requirement for cloudprovider Ingress.
func (j *TestJig) WaitForGivenIngressWithTimeout(ctx context.Context, ing *networkingv1.Ingress, waitForNodePort bool, timeout time.Duration) error {
	// Wait for the loadbalancer IP.
	address, err := j.WaitForIngressAddress(ctx, j.Client, ing.Namespace, ing.Name, timeout)
	if err != nil {
		return fmt.Errorf("Ingress failed to acquire an IP address within %v", timeout)
	}

	var knownHosts []string
	var cert []byte
	if len(ing.Spec.TLS) > 0 {
		knownHosts = ing.Spec.TLS[0].Hosts
		cert = j.GetRootCA(ing.Spec.TLS[0].SecretName)
	}
	return j.pollIngressWithCert(ctx, ing, address, knownHosts, cert, waitForNodePort, timeout)
}

// WaitForIngressWithCert waits till the ingress acquires an IP, then waits for its
// hosts/urls to respond to a protocol check (either http or https). If
// waitForNodePort is true, the NodePort of the Service is verified before
// verifying the Ingress. NodePort is currently a requirement for cloudprovider
// Ingress. Hostnames and certificate need to be explicitly passed in.
func (j *TestJig) WaitForIngressWithCert(ctx context.Context, waitForNodePort bool, knownHosts []string, cert []byte) error {
	// Wait for the loadbalancer IP.
	propagationTimeout := e2eservice.GetServiceLoadBalancerPropagationTimeout(ctx, j.Client)
	address, err := j.WaitForIngressAddress(ctx, j.Client, j.Ingress.Namespace, j.Ingress.Name, propagationTimeout)
	if err != nil {
		return fmt.Errorf("Ingress failed to acquire an IP address within %v", propagationTimeout)
	}

	return j.pollIngressWithCert(ctx, j.Ingress, address, knownHosts, cert, waitForNodePort, propagationTimeout)
}

// VerifyURL polls for the given iterations, in intervals, and fails if the
// given url returns a non-healthy http code even once.
func (j *TestJig) VerifyURL(ctx context.Context, route, host string, iterations int, interval time.Duration, httpClient *http.Client) error {
	for i := 0; i < iterations; i++ {
		b, err := SimpleGET(ctx, httpClient, route, host)
		if err != nil {
			framework.Logf(b)
			return err
		}
		j.Logger.Infof("Verified %v with host %v %d times, sleeping for %v", route, host, i, interval)
		time.Sleep(interval)
	}
	return nil
}

func (j *TestJig) pollServiceNodePort(ctx context.Context, ns, name string, port int) error {
	// TODO: Curl all nodes?
	u, err := getPortURL(ctx, j.Client, ns, name, port)
	if err != nil {
		return err
	}
	return PollURL(ctx, u, "", 30*time.Second, j.PollInterval, &http.Client{Timeout: IngressReqTimeout}, false)
}

// getSvcNodePort returns the node port for the given service:port.
func getSvcNodePort(ctx context.Context, client clientset.Interface, ns, name string, svcPort int) (int, error) {
	svc, err := client.CoreV1().Services(ns).Get(ctx, name, metav1.GetOptions{})
	if err != nil {
		return 0, err
	}
	for _, p := range svc.Spec.Ports {
		if p.Port == int32(svcPort) {
			if p.NodePort != 0 {
				return int(p.NodePort), nil
			}
		}
	}
	return 0, fmt.Errorf(
		"no node port found for service %v, port %v", name, svcPort)
}

// getPortURL returns the url to a nodeport Service.
func getPortURL(ctx context.Context, client clientset.Interface, ns, name string, svcPort int) (string, error) {
	nodePort, err := getSvcNodePort(ctx, client, ns, name, svcPort)
	if err != nil {
		return "", err
	}
	// This list of nodes must not include the any control plane nodes, which are marked
	// unschedulable, since control plane nodes don't run kube-proxy. Without
	// kube-proxy NodePorts won't work.
	var nodes *v1.NodeList
	if wait.PollUntilContextTimeout(ctx, poll, framework.SingleCallTimeout, true, func(ctx context.Context) (bool, error) {
		nodes, err = client.CoreV1().Nodes().List(ctx, metav1.ListOptions{FieldSelector: fields.Set{
			"spec.unschedulable": "false",
		}.AsSelector().String()})
		if err != nil {
			return false, err
		}
		return true, nil
	}) != nil {
		return "", err
	}
	if len(nodes.Items) == 0 {
		return "", fmt.Errorf("Unable to list nodes in cluster")
	}
	for _, node := range nodes.Items {
		for _, address := range node.Status.Addresses {
			if address.Type == v1.NodeExternalIP {
				if address.Address != "" {
					host := net.JoinHostPort(address.Address, fmt.Sprint(nodePort))
					return fmt.Sprintf("http://%s", host), nil
				}
			}
		}
	}
	return "", fmt.Errorf("failed to find external address for service %v", name)
}

// GetIngressNodePorts returns related backend services' nodePorts.
// Current GCE ingress controller allows traffic to the default HTTP backend
// by default, so retrieve its nodePort if includeDefaultBackend is true.
func (j *TestJig) GetIngressNodePorts(ctx context.Context, includeDefaultBackend bool) []string {
	nodePorts := []string{}
	svcPorts := j.GetServicePorts(ctx, includeDefaultBackend)
	for _, svcPort := range svcPorts {
		nodePorts = append(nodePorts, strconv.Itoa(int(svcPort.NodePort)))
	}
	return nodePorts
}

// GetServicePorts returns related backend services' svcPorts.
// Current GCE ingress controller allows traffic to the default HTTP backend
// by default, so retrieve its nodePort if includeDefaultBackend is true.
func (j *TestJig) GetServicePorts(ctx context.Context, includeDefaultBackend bool) map[string]v1.ServicePort {
	svcPorts := make(map[string]v1.ServicePort)
	if includeDefaultBackend {
		defaultSvc, err := j.Client.CoreV1().Services(metav1.NamespaceSystem).Get(ctx, defaultBackendName, metav1.GetOptions{})
		framework.ExpectNoError(err)
		svcPorts[defaultBackendName] = defaultSvc.Spec.Ports[0]
	}

	backendSvcs := []string{}
	if j.Ingress.Spec.DefaultBackend != nil {
		backendSvcs = append(backendSvcs, j.Ingress.Spec.DefaultBackend.Service.Name)
	}
	for _, rule := range j.Ingress.Spec.Rules {
		for _, ingPath := range rule.HTTP.Paths {
			backendSvcs = append(backendSvcs, ingPath.Backend.Service.Name)
		}
	}
	for _, svcName := range backendSvcs {
		svc, err := j.Client.CoreV1().Services(j.Ingress.Namespace).Get(ctx, svcName, metav1.GetOptions{})
		framework.ExpectNoError(err)
		svcPorts[svcName] = svc.Spec.Ports[0]
	}
	return svcPorts
}

// ConstructFirewallForIngress returns the expected GCE firewall rule for the ingress resource
func (j *TestJig) ConstructFirewallForIngress(ctx context.Context, firewallRuleName string, nodeTags []string) *compute.Firewall {
	nodePorts := j.GetIngressNodePorts(ctx, true)

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
func (j *TestJig) GetDistinctResponseFromIngress(ctx context.Context) (sets.String, error) {
	// Wait for the loadbalancer IP.
	propagationTimeout := e2eservice.GetServiceLoadBalancerPropagationTimeout(ctx, j.Client)
	address, err := j.WaitForIngressAddress(ctx, j.Client, j.Ingress.Namespace, j.Ingress.Name, propagationTimeout)
	if err != nil {
		framework.Failf("Ingress failed to acquire an IP address within %v", propagationTimeout)
	}
	responses := sets.NewString()
	timeoutClient := &http.Client{Timeout: IngressReqTimeout}

	for i := 0; i < 100; i++ {
		url := fmt.Sprintf("http://%v", address)
		res, err := SimpleGET(ctx, timeoutClient, url, "")
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
	Ns     string
	rc     *v1.ReplicationController
	pod    *v1.Pod
	Client clientset.Interface
	lbSvc  *v1.Service
}

// Init initializes the NginxIngressController
func (cont *NginxIngressController) Init(ctx context.Context) {
	// Set up a LoadBalancer service in front of nginx ingress controller and pass it via
	// --publish-service flag (see <IngressManifestPath>/nginx/rc.yaml) to make it work in private
	// clusters, i.e. clusters where nodes don't have public IPs.
	framework.Logf("Creating load balancer service for nginx ingress controller")
	serviceJig := e2eservice.NewTestJig(cont.Client, cont.Ns, "nginx-ingress-lb")
	_, err := serviceJig.CreateTCPService(ctx, func(svc *v1.Service) {
		svc.Spec.Type = v1.ServiceTypeLoadBalancer
		svc.Spec.Selector = map[string]string{"k8s-app": "nginx-ingress-lb"}
		svc.Spec.Ports = []v1.ServicePort{
			{Name: "http", Port: 80},
			{Name: "https", Port: 443},
			{Name: "stats", Port: 18080}}
	})
	framework.ExpectNoError(err)
	cont.lbSvc, err = serviceJig.WaitForLoadBalancer(ctx, e2eservice.GetServiceLoadBalancerCreationTimeout(ctx, cont.Client))
	framework.ExpectNoError(err)

	read := func(file string) string {
		data, err := e2etestfiles.Read(filepath.Join(IngressManifestPath, "nginx", file))
		if err != nil {
			framework.Fail(err.Error())
		}
		return string(data)
	}

	framework.Logf("initializing nginx ingress controller")
	e2ekubectl.RunKubectlOrDieInput(cont.Ns, read("rc.yaml"), "create", "-f", "-")

	rc, err := cont.Client.CoreV1().ReplicationControllers(cont.Ns).Get(ctx, "nginx-ingress-controller", metav1.GetOptions{})
	framework.ExpectNoError(err)
	cont.rc = rc

	framework.Logf("waiting for pods with label %v", rc.Spec.Selector)
	sel := labels.SelectorFromSet(labels.Set(rc.Spec.Selector))
	framework.ExpectNoError(testutils.WaitForPodsWithLabelRunning(cont.Client, cont.Ns, sel))
	pods, err := cont.Client.CoreV1().Pods(cont.Ns).List(ctx, metav1.ListOptions{LabelSelector: sel.String()})
	framework.ExpectNoError(err)
	if len(pods.Items) == 0 {
		framework.Failf("Failed to find nginx ingress controller pods with selector %v", sel)
	}
	cont.pod = &pods.Items[0]
	framework.Logf("ingress controller running in pod %v", cont.pod.Name)
}

// TearDown cleans up the NginxIngressController.
func (cont *NginxIngressController) TearDown(ctx context.Context) {
	if cont.lbSvc == nil {
		framework.Logf("No LoadBalancer service created, no cleanup necessary")
		return
	}
	e2eservice.WaitForServiceDeletedWithFinalizer(ctx, cont.Client, cont.Ns, cont.lbSvc.Name)
}

func generateBacksideHTTPSIngressSpec(ns string) *networkingv1.Ingress {
	return &networkingv1.Ingress{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "echoheaders-https",
			Namespace: ns,
		},
		Spec: networkingv1.IngressSpec{
			// Note kubemci requires a default backend.
			DefaultBackend: &networkingv1.IngressBackend{
				Service: &networkingv1.IngressServiceBackend{
					Name: "echoheaders-https",
					Port: networkingv1.ServiceBackendPort{
						Number: 443,
					},
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
	labels := map[string]string{"app": "echoheaders-https"}
	d := e2edeployment.NewDeployment("echoheaders-https", 0, labels, "echoheaders-https", imageutils.GetE2EImage(imageutils.Agnhost), appsv1.RollingUpdateDeploymentStrategyType)
	d.Spec.Replicas = nil
	d.Spec.Template.Spec.Containers[0].Command = []string{
		"/agnhost",
		"netexec",
		"--http-port=8443",
		"--tls-cert-file=/localhost.crt",
		"--tls-private-key-file=/localhost.key",
	}
	d.Spec.Template.Spec.Containers[0].Ports = []v1.ContainerPort{{
		ContainerPort: 8443,
		Name:          "echo-443",
	}}
	return d
}

// SetUpBacksideHTTPSIngress sets up deployment, service and ingress with backside HTTPS configured.
func (j *TestJig) SetUpBacksideHTTPSIngress(ctx context.Context, cs clientset.Interface, namespace string, staticIPName string) (*appsv1.Deployment, *v1.Service, *networkingv1.Ingress, error) {
	deployCreated, err := cs.AppsV1().Deployments(namespace).Create(ctx, generateBacksideHTTPSDeploymentSpec(), metav1.CreateOptions{})
	if err != nil {
		return nil, nil, nil, err
	}
	svcCreated, err := cs.CoreV1().Services(namespace).Create(ctx, generateBacksideHTTPSServiceSpec(), metav1.CreateOptions{})
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
	ingCreated, err := j.runCreate(ctx, ingToCreate)
	if err != nil {
		return nil, nil, nil, err
	}
	return deployCreated, svcCreated, ingCreated, nil
}

// DeleteTestResource deletes given deployment, service and ingress.
func (j *TestJig) DeleteTestResource(ctx context.Context, cs clientset.Interface, deploy *appsv1.Deployment, svc *v1.Service, ing *networkingv1.Ingress) []error {
	var errs []error
	if ing != nil {
		if err := j.runDelete(ctx, ing); err != nil {
			errs = append(errs, fmt.Errorf("error while deleting ingress %s/%s: %w", ing.Namespace, ing.Name, err))
		}
	}
	if svc != nil {
		if err := cs.CoreV1().Services(svc.Namespace).Delete(ctx, svc.Name, metav1.DeleteOptions{}); err != nil {
			errs = append(errs, fmt.Errorf("error while deleting service %s/%s: %w", svc.Namespace, svc.Name, err))
		}
	}
	if deploy != nil {
		if err := cs.AppsV1().Deployments(deploy.Namespace).Delete(ctx, deploy.Name, metav1.DeleteOptions{}); err != nil {
			errs = append(errs, fmt.Errorf("error while deleting deployment %s/%s: %w", deploy.Namespace, deploy.Name, err))
		}
	}
	return errs
}
