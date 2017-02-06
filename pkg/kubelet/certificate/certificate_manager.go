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
	"crypto/x509"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"
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
	syncPeriod     = 1 * time.Hour
	keyExtension   = ".key"
	certExtension  = ".crt"
	pairNamePrefix = "kubelet"
	// shouldRotatePercent is the time remaining threshold for when the
	// certificate should be rotated. It is calculated by using the difference
	// of the certificate end date and the certificate start date and checking
	// what percentage of that interval is left.
	shouldRotatePercent = 10
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

type manager struct {
	certSigningRequestClient clientcertificates.CertificateSigningRequestInterface
	nodeName                 types.NodeName
	certAccessLock           sync.Mutex
	certDirectory            string
	keyDirectory             string
	cert                     *tls.Certificate
	certRotationEnabled      bool
}

// NewManager returns a new certificate manager. A certificate manager is
// responsible for being the authoritative source of certificates in the
// Kubelet and handling updates due to rotation.
func NewManager(
	certSigningRequestClient clientcertificates.CertificateSigningRequestInterface,
	nodeName types.NodeName,
	certDirectory string,
	keyDirectory string,
	certFile string,
	keyFile string,
	certRotationEnabled bool) (Manager, error) {

	var selectedCertFile, selectedKeyFile string
	if fileExists(certFile) && fileExists(keyFile) {
		selectedCertFile = certFile
		selectedKeyFile = keyFile
	} else if fileExists(filepath.Join(certDirectory, pairNamePrefix+certExtension)) && fileExists(filepath.Join(keyDirectory, pairNamePrefix+keyExtension)) {
		selectedCertFile = filepath.Join(certDirectory, pairNamePrefix+certExtension)
		selectedKeyFile = filepath.Join(keyDirectory, pairNamePrefix+keyExtension)
	} else {
		return nil, fmt.Errorf("no cert/key files read at (%q, %q) or in (%q, %q)",
			certFile,
			keyFile,
			certDirectory,
			keyDirectory)
	}

	glog.Infof("Loading cert/key data from (%q, %q)", selectedCertFile, selectedKeyFile)
	cert, err := loadX509KeyPair(selectedCertFile, selectedKeyFile)
	if err != nil {
		return nil, err
	}

	m := manager{
		certSigningRequestClient: certSigningRequestClient,
		nodeName:                 nodeName,
		certDirectory:            filepath.Dir(selectedCertFile),
		keyDirectory:             filepath.Dir(selectedKeyFile),
		certRotationEnabled:      certRotationEnabled,
		cert:                     &cert,
	}

	if m.certRotationEnabled {
		if err := m.cleanupPairs(); err == nil {
			return &m, nil
		} else {
			return nil, err
		}
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
	m.certAccessLock.Lock()
	defer m.certAccessLock.Unlock()
	return m.cert, nil
}

// Start will start the background work of rotating the certificates.
func (m *manager) Start() {
	if !m.certRotationEnabled {
		glog.Infof("Certificate rotation is not enabled.")
		return
	}

	// Certificate rotation depends on access to the API server certificate
	// signing API, so don't start the certificate manager if we don't have a
	// client. This will happen on the master, where the kubelet is responsible
	// for bootstrapping the pods of the master components.
	if m.certSigningRequestClient == nil {
		glog.Infof("Kubernetes client is nil, not starting certificate manager.")
		return
	}

	glog.Info("Starting to rotate expiring certificates.")
	go wait.Forever(func() {
		for range time.Tick(syncPeriod) {
			err := m.rotateCerts()
			if err != nil {
				glog.Errorf("Could not rotate certificates: %v", err)
			}
		}
	}, 0)
}

// cleanupPairs is responsible for recovering from any partial certificate
// rotations rolling back to the most recent matching pair, and deleting any
// keys that don't have a matching certificate.
func (m *manager) cleanupPairs() error {
	if err := m.removeUnmatchedPairs(); err != nil {
		return err
	}

	keyPath := filepath.Join(m.keyDirectory, pairNamePrefix+keyExtension)
	certPath := filepath.Join(m.certDirectory, pairNamePrefix+certExtension)
	if fileExists(keyPath) && fileExists(certPath) {
		// Make sure the selected key/cert pair are symlinks. This facilitates
		// key rotation.
		if err := fixSymlink(keyPath); err != nil {
			return err
		}
		if err := fixSymlink(certPath); err != nil {
			return err
		}
	}

	return nil
}

// removeUnmatchedPairs will remove key files that don't have a matching cert
// file, and cert files that don't have a matching key file.
func (m *manager) removeUnmatchedPairs() error {
	keyfiles, err := ioutil.ReadDir(m.keyDirectory)
	if err != nil {
		return err
	}
	keys := map[string]bool{}
	for _, keyfile := range keyfiles {
		if filepath.Ext(keyfile.Name()) == keyExtension {
			keys[withoutExt(keyfile.Name())] = false
		}
	}
	certfiles, err := ioutil.ReadDir(m.certDirectory)
	if err != nil {
		return err
	}
	for _, certfile := range certfiles {
		if filepath.Ext(certfile.Name()) == certExtension {
			if _, ok := keys[withoutExt(certfile.Name())]; ok {
				keys[withoutExt(certfile.Name())] = true
			} else {
				certPath := filepath.Join(m.certDirectory, certfile.Name())
				if err := os.Remove(certPath); err != nil {
					glog.Errorf("The certificate %q doesn't have a matching key but can't be deleted: %v", certPath, err)
				}
			}
		}
	}
	for filename, certFound := range keys {
		if !certFound {
			keyPath := filepath.Join(m.keyDirectory, filename+keyExtension)
			if err := os.Remove(keyPath); err != nil {
				glog.Errorf("The certificate %q doesn't have a matching key but can't be deleted: %v", keyPath, err)
			}
		}
	}
	return nil
}

func fileExists(file string) bool {
	_, err := os.Stat(file)
	return err == nil || os.IsExist(err)
}

// withoutExt returns the given filename after removing the extension. The
// extension to remove will be the result of filepath.Ext().
func withoutExt(filename string) string {
	return strings.TrimSuffix(filename, filepath.Ext(filename))
}

// fixSymLink moves the given file to another location and replaces it with a
// symlink to the new location.
func fixSymlink(filename string) error {
	// Check if the given file is already a symlink.
	if fi, err := os.Lstat(filename); err != nil {
		return err
	} else if fi.Mode()&os.ModeSymlink == os.ModeSymlink {
		return nil
	}

	newFilename := withoutExt(filename) + ".orig" + filepath.Ext(filename)

	// Move the file to the backup location.
	if err := os.Rename(filename, newFilename); err != nil {
		return fmt.Errorf("unable to rename %q to %q: %v", filename, newFilename, err)
	}
	if err := os.Symlink(newFilename, filename); err != nil {
		return fmt.Errorf("unable to create a symlink from %q to %q: %v", newFilename, filename, err)
	}
	return nil
}

func filename(qualifier, extension string) string {
	return pairNamePrefix + "-" + qualifier + extension
}

// shouldRotate looks at how close the current certificate is to expiring and
// decides if it is time to rotate or not.
func (m *manager) shouldRotate() bool {
	notAfter := m.cert.Leaf.NotAfter
	total := notAfter.Sub(m.cert.Leaf.NotBefore)
	remaining := notAfter.Sub(time.Now())
	return (remaining * 100 / total) < shouldRotatePercent
}

func (m *manager) rotateCerts() error {
	if !m.shouldRotate() {
		return nil
	}

	ts := time.Now().Format("2006-01-02-15-04-05")
	keyFilename := filename(ts, keyExtension)
	certFilename := filename(ts, certExtension)

	success := false

	// Get the private key.
	keyPath := filepath.Join(m.keyDirectory, keyFilename)
	keyData, generatedKeyFile, err := certutil.LoadOrGenerateKeyFile(keyPath)
	if err != nil {
		return err
	}
	if generatedKeyFile {
		defer func() {
			if !success {
				if err := os.Remove(keyPath); err != nil {
					glog.Warningf("Cannot clean up the key file %q: %v", keyPath, err)
				}
			}
		}()
	}

	// Call the Certificate Signing Request API to get a certificate for the
	// new private key.
	certPath := filepath.Join(m.certDirectory, certFilename)
	certData, err := csr.RequestNodeCertificate(m.certSigningRequestClient, keyData, m.nodeName)
	if err != nil {
		return err
	}
	if err := certutil.WriteCert(certPath, certData); err != nil {
		return err
	}
	defer func() {
		if !success {
			if err := os.Remove(certPath); err != nil {
				glog.Warningf("Cannot clean up the cert file %q: %v", certPath, err)
			}
		}
	}()

	// Reload the cert/key data from disk so if there are any parsing or
	// persisting problems they are discovered now before rotating to use the
	// new certificate.
	cert, err := loadX509KeyPair(certPath, keyPath)
	if err != nil {
		return err
	}

	designatedKeyPath := filepath.Join(m.keyDirectory, pairNamePrefix+keyExtension)
	updateSymlink(keyPath, designatedKeyPath)
	designatedCertPath := filepath.Join(m.certDirectory, pairNamePrefix+certExtension)
	updateSymlink(certPath, designatedCertPath)

	success = true
	m.certAccessLock.Lock()
	defer m.certAccessLock.Unlock()
	m.cert = &cert
	return nil
}

func loadX509KeyPair(certFile, keyFile string) (tls.Certificate, error) {
	cert, err := tls.LoadX509KeyPair(certFile, keyFile)
	if err != nil {
		return cert, err
	}
	certs, err := x509.ParseCertificates(cert.Certificate[0])
	if err != nil {
		return cert, fmt.Errorf("unable to parse certificate data: %v", err)
	}
	cert.Leaf = certs[0]
	return cert, nil
}

func updateSymlink(oldname, newname string) error {
	if _, err := os.Stat(newname); err == nil {
		if err := os.Remove(newname); err != nil {
			return fmt.Errorf("unable to remove %q: %v", newname, err)
		}
	}
	if err := os.Symlink(oldname, newname); err != nil {
		glog.Errorf("Error while symlinking: %v", err)
		return fmt.Errorf("unable to symlink %q to %q: %v", newname, oldname, err)
	}
	return nil
}
