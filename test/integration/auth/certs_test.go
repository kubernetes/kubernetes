package auth

import (
	"context"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"fmt"
	"math/big"
	"os"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	auditinternal "k8s.io/apiserver/pkg/apis/audit"
	auditv1 "k8s.io/apiserver/pkg/apis/audit/v1"
	kubex509 "k8s.io/apiserver/pkg/authentication/request/x509"

	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	utiltesting "k8s.io/client-go/util/testing"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/kubernetes/test/utils"
)

var (
	auditPolicy = `
apiVersion: audit.k8s.io/v1
kind: Policy
rules:
  - level: Request
    resources:
    - group: ""
      resources: ["pods"]
    verbs: ["get"]
`
)

func TestCerts(t *testing.T) {
	logFile, err := os.CreateTemp("", "audit.log")
	if err != nil {
		t.Fatalf("Failed to create audit log file: %v", err)
	}
	defer utiltesting.CloseAndRemove(t, logFile)

	policyFile, err := os.CreateTemp("", "audit-policy.yaml")
	if err != nil {
		t.Fatalf("Failed to create audit policy file: %v", err)
	}
	if _, err := policyFile.Write([]byte(auditPolicy)); err != nil {
		t.Fatalf("Failed to write audit policy file: %v", err)
	}

	s := kubeapiservertesting.StartTestServerOrDie(t, kubeapiservertesting.NewDefaultTestServerOptions(), []string{
		"--audit-policy-file", policyFile.Name(),
		"--audit-log-version", "audit.k8s.io/v1",
		"--audit-log-mode", "blocking",
		"--audit-log-path", logFile.Name(),
	}, framework.SharedEtcd())
	defer s.TearDownFn()

	// Generate self-signed certificate
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	keyRaw, err := x509.MarshalECPrivateKey(key)
	if err != nil {
		t.Fatal(err)
	}
	serialNumberLimit := new(big.Int).Lsh(big.NewInt(1), 128)
	serialNumber, err := rand.Int(rand.Reader, serialNumberLimit)
	if err != nil {
		t.Fatal(err)
	}
	commonName := "test-cn"
	notBefore := time.Now().Truncate(time.Second)
	notAfter := notBefore.Add(time.Hour)
	cert := &x509.Certificate{
		SerialNumber: serialNumber,
		Subject:      pkix.Name{CommonName: commonName},
		NotBefore:    notBefore,
		NotAfter:     notAfter,

		KeyUsage:              x509.KeyUsageKeyEncipherment | x509.KeyUsageDigitalSignature,
		ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth},
		BasicConstraintsValid: true,
	}
	certRaw, err := x509.CreateCertificate(rand.Reader, cert, cert, key.Public(), key)
	if err != nil {
		t.Fatal(err)
	}

	// Use self-signed certificate in client config
	clientConfig := rest.CopyConfig(s.ClientConfig)
	clientConfig.BearerToken = ""
	clientConfig.CertData = pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: certRaw})
	clientConfig.KeyData = pem.EncodeToMemory(&pem.Block{Type: "PRIVATE KEY", Bytes: keyRaw})
	client, err := clientset.NewForConfig(clientConfig)
	if err != nil {
		t.Fatalf("error occurred: %v", err)
	}

	// Make a request using the client
	podName := "foobar"
	currentTime := time.Now().Truncate(time.Second)
	_, err = client.CoreV1().Pods("default").Get(context.TODO(), podName, metav1.GetOptions{})
	if err == nil {
		t.Fatal("expected error, but it was nil")
	}

	// Verify that audit log has the expected entry
	stream, err := os.OpenFile(logFile.Name(), os.O_RDWR, 0600)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	defer stream.Close()

	annotationDetails := fmt.Sprintf(`certificate "%s" [client] issuer="<self>" (%s to %s (now=%s)) failed: x509: certificate signed by unknown authority`, commonName, notBefore.UTC(), notAfter.UTC(), currentTime.UTC())
	expectedEvents := []utils.AuditEvent{
		{
			Level:      auditinternal.LevelRequest,
			Stage:      auditinternal.StageResponseStarted,
			RequestURI: fmt.Sprintf("/api/v1/namespaces/default/pods/%s", podName),
			Verb:       "get",
			Resource:   "pods",
			Namespace:  "default",
			Code:       401,
			CustomAuditAnnotations: map[string]string{
				kubex509.CertificateErrorAuditAnnotation: annotationDetails,
			},
		},
	}

	auditAnnotationFilter := func(key, val string) bool {
		return key == kubex509.CertificateErrorAuditAnnotation
	}

	missing, err := utils.CheckAuditLinesFiltered(stream, expectedEvents, auditv1.SchemeGroupVersion, auditAnnotationFilter)
	if err != nil {
		t.Errorf("unexpected error checking audit lines: %v", err)
	}
	if len(missing.MissingEvents) > 0 {
		t.Errorf("failed to get expected events -- missing: %s", missing)
	}
}
