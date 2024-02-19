/*
Copyright 2020 The Kubernetes Authors.

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
	"fmt"
	"reflect"
	"sync"
	"time"

	utilnet "k8s.io/apimachinery/pkg/util/net"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/util/connrotation"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog/v2"
)

const workItemKey = "key"

// CertCallbackRefreshDuration is exposed so that integration tests can crank up the reload speed.
var CertCallbackRefreshDuration = 5 * time.Minute

type reloadFunc func(*tls.CertificateRequestInfo) (*tls.Certificate, error)

type dynamicClientCert struct {
	clientCert *tls.Certificate
	certMtx    sync.RWMutex

	reload     reloadFunc
	connDialer *connrotation.Dialer

	// queue only ever has one item, but it has nice error handling backoff/retry semantics
	queue workqueue.RateLimitingInterface
}

func certRotatingDialer(reload reloadFunc, dial utilnet.DialFunc) *dynamicClientCert {
	d := &dynamicClientCert{
		reload:     reload,
		connDialer: connrotation.NewDialer(connrotation.DialFunc(dial)),
		queue:      workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "DynamicClientCertificate"),
	}

	return d
}

// loadClientCert calls the callback and rotates connections if needed
func (c *dynamicClientCert) loadClientCert() (*tls.Certificate, error) {
	cert, err := c.reload(nil)
	if err != nil {
		return nil, err
	}

	// check to see if we have a change. If the values are the same, do nothing.
	c.certMtx.RLock()
	haveCert := c.clientCert != nil
	if certsEqual(c.clientCert, cert) {
		c.certMtx.RUnlock()
		return c.clientCert, nil
	}
	c.certMtx.RUnlock()

	c.certMtx.Lock()
	c.clientCert = cert
	c.certMtx.Unlock()

	// The first certificate requested is not a rotation that is worth closing connections for
	if !haveCert {
		return cert, nil
	}

	klog.V(1).Infof("certificate rotation detected, shutting down client connections to start using new credentials")
	c.connDialer.CloseAll()

	return cert, nil
}

// certsEqual compares tls Certificates, ignoring the Leaf which may get filled in dynamically
func certsEqual(left, right *tls.Certificate) bool {
	if left == nil || right == nil {
		return left == right
	}

	if !byteMatrixEqual(left.Certificate, right.Certificate) {
		return false
	}

	if !reflect.DeepEqual(left.PrivateKey, right.PrivateKey) {
		return false
	}

	if !byteMatrixEqual(left.SignedCertificateTimestamps, right.SignedCertificateTimestamps) {
		return false
	}

	if !bytes.Equal(left.OCSPStaple, right.OCSPStaple) {
		return false
	}

	return true
}

func byteMatrixEqual(left, right [][]byte) bool {
	if len(left) != len(right) {
		return false
	}

	for i := range left {
		if !bytes.Equal(left[i], right[i]) {
			return false
		}
	}
	return true
}

// run starts the controller and blocks until stopCh is closed.
func (c *dynamicClientCert) Run(stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()
	defer c.queue.ShutDown()

	klog.V(3).Infof("Starting client certificate rotation controller")
	defer klog.V(3).Infof("Shutting down client certificate rotation controller")

	go wait.Until(c.runWorker, time.Second, stopCh)

	go wait.PollImmediateUntil(CertCallbackRefreshDuration, func() (bool, error) {
		c.queue.Add(workItemKey)
		return false, nil
	}, stopCh)

	<-stopCh
}

func (c *dynamicClientCert) runWorker() {
	for c.processNextWorkItem() {
	}
}

func (c *dynamicClientCert) processNextWorkItem() bool {
	dsKey, quit := c.queue.Get()
	if quit {
		return false
	}
	defer c.queue.Done(dsKey)

	_, err := c.loadClientCert()
	if err == nil {
		c.queue.Forget(dsKey)
		return true
	}

	utilruntime.HandleError(fmt.Errorf("%v failed with : %v", dsKey, err))
	c.queue.AddRateLimited(dsKey)

	return true
}

func (c *dynamicClientCert) GetClientCertificate(*tls.CertificateRequestInfo) (*tls.Certificate, error) {
	return c.loadClientCert()
}
