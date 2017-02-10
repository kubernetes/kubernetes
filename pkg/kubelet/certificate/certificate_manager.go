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

package certificate

import (
	"crypto/tls"
	"fmt"
	"sync"
	"time"

	"github.com/golang/glog"

	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	certutil "k8s.io/client-go/util/cert"
	clientcertificates "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/typed/certificates/v1beta1"
	"k8s.io/kubernetes/pkg/kubelet/util/csr"
)

const (
	syncPeriod = 1 * time.Hour
)

// Manager maintains and updates the certificates in use by this kubelet. In
// the background it communicates with the API server to get new certificates
// for certificates about to expire.
type Manager interface {
	// Start the API server status sync loop.
	Start()
	// GetCertificate gets the current certificate from the certificate
	// manager. This function matches the signature required by
	// tls.Config.GetCertificate so it can be passed as TLS configuration. A
	// TLS server will automatically call back here to get the correct
	// certificate when establishing each new connection.
	GetCertificate(clientHello *tls.ClientHelloInfo) (*tls.Certificate, error)
}

// Store is responsible for getting and updating the current certificate.
// Depending on the concrete implementation, the backing store for this
// behavior may vary.
type Store interface {
	Current() (*tls.Certificate, error)
	// Accepts the PEM data for the cert/key pair and makes the new cert/key
	// pair the 'current' pair, that will be returned by future calls to
	// Current().
	Update(cert, key []byte) (*tls.Certificate, error)
}

type manager struct {
	certSigningRequestClient clientcertificates.CertificateSigningRequestInterface
	nodeName                 types.NodeName
	certStore                Store
	certAccessLock           sync.RWMutex
	cert                     *tls.Certificate
	shouldRotatePercent      int32
}

// NewManager returns a new certificate manager. A certificate manager is
// responsible for being the authoritative source of certificates in the
// Kubelet and handling updates due to rotation.
func NewManager(
	certSigningRequestClient clientcertificates.CertificateSigningRequestInterface,
	nodeName types.NodeName,
	certificateStore Store,
	certRotationPercent int32) (Manager, error) {

	cert, err := certificateStore.Current()
	if err != nil {
		return nil, err
	}

	if certRotationPercent > 100 {
		certRotationPercent = 100
	}

	m := manager{
		certSigningRequestClient: certSigningRequestClient,
		nodeName:                 nodeName,
		certStore:                certificateStore,
		cert:                     cert,
		shouldRotatePercent:      certRotationPercent,
	}

	return &m, nil
}

// GetCertificate returns the certificate that should be used TLS connections.
// The value returned by this function will change over time as the certificate
// is rotated. If a reference to this method is passed directly into the TLS
// options for a connection, certificate rotation will be handled correctly by
// the underlying go libraries.
//
//    tlsOptions := &server.TLSOptions{
//        ...
//        GetCertificate: certificateManager.GetCertificate
//        ...
//    }
//
func (m *manager) GetCertificate(clientHello *tls.ClientHelloInfo) (*tls.Certificate, error) {
	m.certAccessLock.RLock()
	defer m.certAccessLock.RUnlock()
	return m.cert, nil
}

// Start will start the background work of rotating the certificates.
func (m *manager) Start() {
	if m.shouldRotatePercent < 1 {
		glog.V(2).Infof("Certificate rotation is not enabled.")
		return
	}

	// Certificate rotation depends on access to the API server certificate
	// signing API, so don't start the certificate manager if we don't have a
	// client. This will happen on the master, where the kubelet is responsible
	// for bootstrapping the pods of the master components.
	if m.certSigningRequestClient == nil {
		glog.V(2).Infof("Certificate rotation is not enabled, no connection to the apiserver.")
		return
	}

	glog.V(2).Infof("Certificate rotation is enabled.")
	go wait.Forever(func() {
		for range time.Tick(syncPeriod) {
			err := m.rotateCerts()
			if err != nil {
				glog.Errorf("Could not rotate certificates: %v", err)
			}
		}
	}, 0)
}

// shouldRotate looks at how close the current certificate is to expiring and
// decides if it is time to rotate or not.
func (m *manager) shouldRotate() bool {
	m.certAccessLock.RLock()
	defer m.certAccessLock.RUnlock()
	notAfter := m.cert.Leaf.NotAfter
	total := notAfter.Sub(m.cert.Leaf.NotBefore)
	remaining := notAfter.Sub(time.Now())
	return int32(remaining*100/total) < m.shouldRotatePercent
}

func (m *manager) rotateCerts() error {
	if !m.shouldRotate() {
		return nil
	}

	// Generate a new private key.
	keyData, err := certutil.MakeEllipticPrivateKeyPEM()
	if err != nil {
		return fmt.Errorf("unable to generate a new private key: %v", err)
	}

	// Call the Certificate Signing Request API to get a certificate for the
	// new private key.
	certData, err := csr.RequestNodeCertificate(m.certSigningRequestClient, keyData, m.nodeName)
	if err != nil {
		return fmt.Errorf("unable to get a new key signed for %q: %v", m.nodeName, err)
	}

	cert, err := m.certStore.Update(certData, keyData)
	if err != nil {
		return fmt.Errorf("unable to store the new cert/key pair: %v", err)
	}

	m.certAccessLock.Lock()
	defer m.certAccessLock.Unlock()
	m.cert = cert
	return nil
}
