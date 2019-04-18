package kubeletserver

import (
	"bytes"
	"crypto/x509"
	"encoding/pem"
	"fmt"
	"net"
	"time"

	"github.com/pkg/errors"
	certificates "k8s.io/api/certificates/v1beta1"
	certs "k8s.io/api/certificates/v1beta1"
	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	certutil "k8s.io/client-go/util/cert"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/pkiutil"
	"k8s.io/kubernetes/staging/src/k8s.io/client-go/util/keyutil"
)

const (
	CertificateRequestBlockType = "CERTIFICATE REQUEST"
)

var (
	approveTimeout = time.Second * 30
)

func KubeletServerCert(kubeClient kubernetes.Interface, localEndpoint kubeadmapi.APIEndpoint, nodeRegistration kubeadm.NodeRegistrationOptions) ([]byte, []byte, error) {
	altNames, err := getAltNames(localEndpoint, nodeRegistration)

	if err != nil {
		return nil, nil, err
	}

	csr, key, err := pkiutil.NewCSRAndKey(
		&certutil.Config{
			CommonName: nodeRegistration.Name,
			Usages:     []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
			AltNames:   *altNames,
		})

	block := &pem.Block{
		Type:  CertificateRequestBlockType,
		Bytes: csr.Raw,
	}

	buffer := &bytes.Buffer{}
	if err := pem.Encode(buffer, block); err != nil {
		return nil, nil, errors.Wrap(err, "error encoding certificate signing request")
	}

	certReq := &certs.CertificateSigningRequest{
		Spec: certs.CertificateSigningRequestSpec{
			Request: buffer.Bytes(),
			Usages:  []certs.KeyUsage{certs.UsageServerAuth},
		},
	}

	certReq.Name = nodeRegistration.Name
	req, err := kubeClient.CertificatesV1beta1().CertificateSigningRequests().Create(certReq)

	if err != nil {
		return nil, nil, errors.Wrap(err, "error creating certificate signing request")
	}

	req.Status.Conditions = append(req.Status.Conditions, certificates.CertificateSigningRequestCondition{
		Type: certificates.CertificateApproved,
	})

	req, err = kubeClient.CertificatesV1beta1().CertificateSigningRequests().UpdateApproval(req)

	if err != nil {
		return nil, nil, errors.Wrap(err, "error getting approving certificate signing request")
	}

	// We need to wait some time before certificate request gets approved
	fmt.Printf("[kubelet-server] Wait for certificate approve %s\n", approveTimeout)
	time.Sleep(approveTimeout)

	req, err = kubeClient.CertificatesV1beta1().CertificateSigningRequests().Get(req.Name, v1.GetOptions{})

	if err != nil {
		return nil, nil, errors.Wrap(err, "error getting certificate signing request")
	}

	certPem := req.Status.Certificate
	keyPem, err := keyutil.MarshalPrivateKeyToPEM(key)

	if err != nil {
		return nil, nil, errors.Wrap(err, "error marshalling private key to pem")
	}

	return certPem, keyPem, err
}

// getAltNames builds an AltNames object with the cfg and certName.
func getAltNames(endpoint kubeadmapi.APIEndpoint, nodeRegistration kubeadmapi.NodeRegistrationOptions) (*certutil.AltNames, error) {
	// advertise address
	advertiseAddress := net.ParseIP(endpoint.AdvertiseAddress)
	if advertiseAddress == nil {
		return nil, errors.Errorf("error parsing LocalAPIEndpoint AdvertiseAddress %v: is not a valid textual representation of an IP address",
			endpoint.AdvertiseAddress)
	}

	// create AltNames with defaults DNSNames/IPs
	altNames := &certutil.AltNames{
		DNSNames: []string{nodeRegistration.Name, "localhost"},
		IPs:      []net.IP{advertiseAddress, net.IPv4(127, 0, 0, 1), net.IPv6loopback},
	}

	return altNames, nil
}
