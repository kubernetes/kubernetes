/*
Copyright The Kubernetes Authors.

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

package transport

import (
	"bytes"
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"os"
	"sync"
	"time"

	utilnet "k8s.io/apimachinery/pkg/util/net"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/util/connrotation"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog/v2"
)

const caWorkItemKey = "ca-key"

// CACallbackRefreshDuration is exposed so that integration tests can crank up the reload speed.
var CACallbackRefreshDuration = 5 * time.Minute

// dynamicCALoader handles hot-reload of CA certificates using polling.
// When the CA changes, it closes existing connections to force new TLS handshakes.
type dynamicCALoader struct {
	logger     klog.Logger
	caFile     string
	serverName string

	caData   []byte
	caMtx    sync.RWMutex
	certPool *x509.CertPool

	connDialer *connrotation.Dialer
	queue      workqueue.TypedRateLimitingInterface[string]
}

func caRotatingDialer(logger klog.Logger, caFile string, serverName string, initialPool *x509.CertPool, initialCAData []byte, dial utilnet.DialFunc) *dynamicCALoader {
	return &dynamicCALoader{
		logger:     logger,
		caFile:     caFile,
		serverName: serverName,
		caData:     initialCAData,
		certPool:   initialPool,
		connDialer: connrotation.NewDialer(connrotation.DialFunc(dial)),
		queue: workqueue.NewTypedRateLimitingQueueWithConfig(
			workqueue.DefaultTypedControllerRateLimiter[string](),
			workqueue.TypedRateLimitingQueueConfig[string]{Name: "DynamicCACertificate"},
		),
	}
}

// CertPool returns the current CA certificate pool. Safe for concurrent use.
func (c *dynamicCALoader) CertPool() *x509.CertPool {
	c.caMtx.RLock()
	defer c.caMtx.RUnlock()
	return c.certPool
}

func (c *dynamicCALoader) loadCA() error {
	caData, err := os.ReadFile(c.caFile)
	if err != nil {
		return fmt.Errorf("failed to read CA file %s: %w", c.caFile, err)
	}

	c.caMtx.RLock()
	unchanged := bytes.Equal(c.caData, caData)
	c.caMtx.RUnlock()
	if unchanged {
		return nil
	}

	certPool := x509.NewCertPool()
	if ok := certPool.AppendCertsFromPEM(caData); !ok {
		return fmt.Errorf("failed to parse CA certificates from %s", c.caFile)
	}

	c.caMtx.Lock()
	hadCA := c.caData != nil
	c.caData = caData
	c.certPool = certPool
	c.caMtx.Unlock()

	if !hadCA {
		return nil
	}

	c.logger.V(1).Info("CA certificate rotation detected, shutting down client connections to start using new CA")
	c.connDialer.CloseAll()

	return nil
}

func (c *dynamicCALoader) run(stopCh <-chan struct{}) {
	defer utilruntime.HandleCrashWithLogger(c.logger)
	defer c.queue.ShutDown()

	c.logger.V(3).Info("Starting CA certificate rotation controller")
	defer c.logger.V(3).Info("Shutting down CA certificate rotation controller")

	go wait.Until(c.runWorker, time.Second, stopCh)

	go wait.PollImmediateUntil(CACallbackRefreshDuration, func() (bool, error) {
		c.queue.Add(caWorkItemKey)
		return false, nil
	}, stopCh)

	<-stopCh
}

func (c *dynamicCALoader) runWorker() {
	for c.processNextWorkItem() {
	}
}

func (c *dynamicCALoader) processNextWorkItem() bool {
	dsKey, quit := c.queue.Get()
	if quit {
		return false
	}
	defer c.queue.Done(dsKey)

	if err := c.loadCA(); err != nil {
		utilruntime.HandleErrorWithLogger(c.logger, err, "Loading CA cert failed", "key", dsKey)
		c.queue.AddRateLimited(dsKey)
		return true
	}

	c.queue.Forget(dsKey)
	return true
}

// VerifyConnection verifies server certificates against the dynamically loaded CA pool.
func (c *dynamicCALoader) VerifyConnection(cs tls.ConnectionState) error {
	if len(cs.PeerCertificates) == 0 {
		return fmt.Errorf("no peer certificates presented")
	}

	certPool := c.CertPool()
	if certPool == nil {
		return fmt.Errorf("no CA certificates loaded")
	}

	intermediates := x509.NewCertPool()
	for _, cert := range cs.PeerCertificates[1:] {
		intermediates.AddCert(cert)
	}

	dnsName := cs.ServerName
	if dnsName == "" && c.serverName != "" {
		dnsName = c.serverName
	}
	if dnsName == "" {
		c.logger.V(1).Info("Warning: no server name provided for certificate verification, hostname verification will be skipped")
	}

	opts := x509.VerifyOptions{
		Roots:         certPool,
		Intermediates: intermediates,
		DNSName:       dnsName,
	}

	if _, err := cs.PeerCertificates[0].Verify(opts); err != nil {
		return fmt.Errorf("failed to verify server certificate: %w", err)
	}

	return nil
}
