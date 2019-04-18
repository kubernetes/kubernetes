package kubeletserver

import (
	"bytes"
	"encoding/pem"
	"github.com/pkg/errors"
	certificates "k8s.io/api/certificates/v1beta1"
	certs "k8s.io/api/certificates/v1beta1"
	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	certutil "k8s.io/client-go/util/cert"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/pkiutil"
	"k8s.io/kubernetes/staging/src/k8s.io/client-go/util/keyutil"
)

const (
	CertificateRequestBlockType = "CERTIFICATE REQUEST"
)

func KubeletServerCert(kubeClient kubernetes.Interface, nodeRegistration kubeadm.NodeRegistrationOptions) ([]byte, []byte, error) {
	csr, key, err := pkiutil.NewCSRAndKey(
		&certutil.Config{
			CommonName: "kubelet-worker",
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
