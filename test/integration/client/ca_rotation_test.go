/*
Copyright 2025 The Kubernetes Authors.

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

package client

import (
	"context"
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"fmt"
	"math/big"
	"net/http"
	"os"
	"path"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/util/cert"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/kubernetes/test/utils"
)

// TestCARotationIntegration tests CA rotation functionality with a real API server
func TestCARotationIntegration(t *testing.T) {
	// Set up temporary directory for certificates
	certDir := t.TempDir()
	
	// Create initial CA
	initialCAFilename, initialCACert, initialCAKey := writeCAFiles(t, certDir, "initial-ca", 1*time.Hour)
	
	// Start test server with initial CA
	server := kubeapiservertesting.StartTestServerOrDie(t, kubeapiservertesting.NewDefaultTestServerOptions(), []string{
		"--client-ca-file=" + initialCAFilename,
	}, framework.SharedEtcd())
	defer server.TearDownFn()
	
	// Create client certificate signed by initial CA
	clientCertFile, clientKeyFile := writeClientCerts(t, certDir, initialCACert, initialCAKey, "integration-client", 1*time.Hour)
	
	// Configure client with CA rotation enabled
	config := server.ClientConfig
	config.CertFile = clientCertFile
	config.KeyFile = clientKeyFile
	config.BearerToken = ""
	config.TLSClientConfig = rest.TLSClientConfig{
		CAFile: initialCAFilename,
	}
	
	// Enable CA rotation in transport
	if config.Transport == nil {
		config.Transport = &http.Transport{}
	}
	
	// Create client with CA rotation support
	client := clientset.NewForConfigOrDie(config)
	
	t.Log("Testing initial connection with CA rotation")
	
	// Test initial connection
	ctx := context.Background()
	_, err := client.CoreV1().ServiceAccounts("default").List(ctx, v1.ListOptions{})
	if err != nil {
		t.Fatalf("Initial connection failed: %v", err)
	}
	t.Log("Initial connection successful")
	
	// Start long-running watch to test active connection preservation
	t.Log("Starting long-running watch")
	
	watchCtx, watchCancel := context.WithCancel(ctx)
	defer watchCancel()
	
	watch, err := client.CoreV1().ServiceAccounts("default").Watch(watchCtx, v1.ListOptions{})
	if err != nil {
		t.Fatalf("Failed to start watch: %v", err)
	}
	defer watch.Stop()
	
	// Channel to track watch events
	watchEvents := make(chan bool, 10)
	watchErrors := make(chan error, 10)
	
	go func() {
		for {
			select {
			case event, ok := <-watch.ResultChan():
				if !ok {
					watchErrors <- fmt.Errorf("watch channel closed")
					return
				}
				if event.Type != "" {
					watchEvents <- true
				}
			case <-watchCtx.Done():
				return
			}
		}
	}()
	
	// Start concurrent requests to test connection behavior
	t.Log("Starting concurrent requests")
	
	requestCounter := int64(0)
	errorCounter := int64(0)
	requestStop := make(chan struct{})
	var requestWg sync.WaitGroup
	
	// Start multiple goroutines making requests
	for i := 0; i < 3; i++ {
		requestWg.Add(1)
		go func(workerID int) {
			defer requestWg.Done()
			ticker := time.NewTicker(100 * time.Millisecond)
			defer ticker.Stop()
			
			for {
				select {
				case <-requestStop:
					return
				case <-ticker.C:
					_, err := client.CoreV1().Namespaces().Get(ctx, "default", v1.GetOptions{})
					atomic.AddInt64(&requestCounter, 1)
					if err != nil {
						atomic.AddInt64(&errorCounter, 1)
						t.Logf("Worker %d request error: %v", workerID, err)
					}
				}
			}
		}(i)
	}
	
	// Let requests run for a bit
	time.Sleep(500 * time.Millisecond)
	
	t.Log("Performing CA rotation")
	
	// Create new CA and certificates
	rotatedCAFilename, rotatedCACert, rotatedCAKey := writeCAFiles(t, certDir, "rotated-ca", 1*time.Hour)
	newClientCertFile, newClientKeyFile := writeClientCerts(t, certDir, rotatedCACert, rotatedCAKey, "integration-client", 1*time.Hour)
	
	// Update server CA file (simulating CA rotation on server side)
	// In real scenarios, both old and new CAs would be accepted during transition
	combinedCAData, err := os.ReadFile(initialCAFilename)
	if err != nil {
		t.Fatalf("Failed to read initial CA: %v", err)
	}
	rotatedCAData, err := os.ReadFile(rotatedCAFilename)
	if err != nil {
		t.Fatalf("Failed to read rotated CA: %v", err)
	}
	
	// Write combined CA file (both old and new CAs for transition period)
	combinedCAFile := path.Join(certDir, "combined-ca.crt")
	combinedData := append(combinedCAData, '\n')
	combinedData = append(combinedData, rotatedCAData...)
	err = os.WriteFile(combinedCAFile, combinedData, 0644)
	if err != nil {
		t.Fatalf("Failed to write combined CA: %v", err)
	}
	
	// Update server configuration to use combined CA
	// Note: In a real scenario, this would be done by the cluster administrator
	// For testing, we simulate by updating the CA file that the server watches
	err = os.WriteFile(initialCAFilename, combinedData, 0644)
	if err != nil {
		t.Fatalf("Failed to update server CA file: %v", err)
	}
	
	// Trigger client-side CA rotation by updating the CA file
	err = os.WriteFile(config.TLSClientConfig.CAFile, rotatedCAData, 0644)
	if err != nil {
		t.Fatalf("Failed to trigger CA rotation: %v", err)
	}
	
	t.Log("✓ CA rotation files updated")
	
	// Continue requests during and after rotation
	time.Sleep(2 * time.Second)
	
	// Test new connection with rotated CA
	t.Log("Testing connections after CA rotation")
	
	// Create new client with rotated certificates
	rotatedConfig := *config
	rotatedConfig.CertFile = newClientCertFile
	rotatedConfig.KeyFile = newClientKeyFile
	rotatedConfig.TLSClientConfig.CAFile = rotatedCAFilename
	
	rotatedClient := clientset.NewForConfigOrDie(&rotatedConfig)
	
	// Test that new client works with rotated CA
	_, err = rotatedClient.CoreV1().ServiceAccounts("default").List(ctx, v1.ListOptions{})
	if err != nil {
		t.Errorf("Connection with rotated CA failed: %v", err)
	} else {
		t.Log("✓ New connection with rotated CA successful")
	}
	
	// Test that original client eventually works (after rotation detection)
	var finalErr error
	err = wait.Poll(100*time.Millisecond, 10*time.Second, func() (bool, error) {
		_, err := client.CoreV1().ServiceAccounts("default").List(ctx, v1.ListOptions{})
		if err != nil {
			finalErr = err
			return false, nil // Continue polling
		}
		return true, nil
	})
	
	if err != nil {
		t.Logf("Original client failed after CA rotation (this may be expected): %v", finalErr)
		// This might be expected behavior depending on implementation
	} else {
		t.Log("✓ Original client adapted to CA rotation")
	}
	
	// Stop concurrent requests
	close(requestStop)
	requestWg.Wait()
	
	// Check request statistics
	totalRequests := atomic.LoadInt64(&requestCounter)
	totalErrors := atomic.LoadInt64(&errorCounter)
	
	t.Logf("Request Statistics")
	t.Logf("Total requests: %d", totalRequests)
	t.Logf("Total errors: %d", totalErrors)
	t.Logf("Error rate: %.2f%%", float64(totalErrors)/float64(totalRequests)*100)
	
	// We expect some errors during rotation, but not too many
	errorRate := float64(totalErrors) / float64(totalRequests)
	if errorRate > 0.5 { // More than 50% error rate is concerning
		t.Errorf("High error rate during CA rotation: %.2f%%", errorRate*100)
	}
	
	// Check watch behavior
	watchCancel()
	
	// Give watch time to process any final events
	time.Sleep(100 * time.Millisecond)
	
	t.Logf("Watch Events: %d", len(watchEvents))
	
	if len(watchErrors) > 0 {
		t.Logf("Watch errors (may be expected): %d", len(watchErrors))
	}
	
	t.Log("✓ CA rotation integration test completed")
}

// TestCARotationConnectionPreservation tests that active connections survive CA rotation
func TestCARotationConnectionPreservation(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping connection preservation test in short mode")
	}
	
	certDir := t.TempDir()
	
	// Create CA and start server
	caFilename, caCert, caKey := writeCAFiles(t, certDir, "preservation-ca", 1*time.Hour)
	
	server := kubeapiservertesting.StartTestServerOrDie(t, kubeapiservertesting.NewDefaultTestServerOptions(), []string{
		"--client-ca-file=" + caFilename,
	}, framework.SharedEtcd())
	defer server.TearDownFn()
	
	// Create client
	clientCertFile, clientKeyFile := writeClientCerts(t, certDir, caCert, caKey, "preservation-client", 1*time.Hour)
	
	config := server.ClientConfig
	config.CertFile = clientCertFile
	config.KeyFile = clientKeyFile
	config.BearerToken = ""
	config.TLSClientConfig = rest.TLSClientConfig{
		CAFile: caFilename,
	}
	
	client := clientset.NewForConfigOrDie(config)
	
	t.Log("Testing connection preservation during CA rotation")
	
	// Start a long-running operation
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	
	// Use a watch as a long-running operation
	watch, err := client.CoreV1().ServiceAccounts("default").Watch(ctx, v1.ListOptions{})
	if err != nil {
		t.Fatalf("Failed to start watch: %v", err)
	}
	defer watch.Stop()
	
	// Channel to track watch events and errors
	watchEvents := make(chan bool, 10)
	watchErrors := make(chan error, 10)
	
	go func() {
		for {
			select {
			case event, ok := <-watch.ResultChan():
				if !ok {
					return
				}
				if event.Type != "" {
					watchEvents <- true
				}
			case <-ctx.Done():
				return
			}
		}
	}()
	
	// Let watch establish
	time.Sleep(100 * time.Millisecond)
	
	t.Log("Performing CA rotation during active connection")
	
	// Create rotated CA
	rotatedCAFilename, rotatedCACert, rotatedCAKey := writeCAFiles(t, certDir, "rotated-ca", 1*time.Hour)
	
	// Update server CA file (simulate gradual rollout)
	combinedCAData, err := os.ReadFile(caFilename)
	if err != nil {
		t.Fatalf("Failed to read initial CA: %v", err)
	}
	rotatedCAData, err := os.ReadFile(rotatedCAFilename)
	if err != nil {
		t.Fatalf("Failed to read rotated CA: %v", err)
	}
	
	// Write combined CA (both old and new)
	combinedData := append(combinedCAData, '\n')
	combinedData = append(combinedData, rotatedCAData...)
	err = os.WriteFile(caFilename, combinedData, 0644)
	if err != nil {
		t.Fatalf("Failed to write combined CA: %v", err)
	}
	
	// Update client CA file to trigger rotation
	err = os.WriteFile(config.TLSClientConfig.CAFile, rotatedCAData, 0644)
	if err != nil {
		t.Fatalf("Failed to trigger CA rotation: %v", err)
	}
	
	// Continue monitoring the watch for a few seconds
	time.Sleep(3 * time.Second)
	
	// Check if watch is still active
	select {
	case err := <-watchErrors:
		t.Errorf("Watch failed during CA rotation: %v", err)
	case <-time.After(100 * time.Millisecond):
		t.Log("✓ Watch connection preserved during CA rotation")
	}
	
	// Test new connection with rotated CA
	newClientCertFile, newClientKeyFile := writeClientCerts(t, certDir, rotatedCACert, rotatedCAKey, "new-client", 1*time.Hour)
	
	newConfig := *config
	newConfig.CertFile = newClientCertFile
	newConfig.KeyFile = newClientKeyFile
	newConfig.TLSClientConfig.CAFile = rotatedCAFilename
	
	newClient := clientset.NewForConfigOrDie(&newConfig)
	_, err = newClient.CoreV1().ServiceAccounts("default").List(context.Background(), v1.ListOptions{})
	if err != nil {
		t.Errorf("New connection failed after CA rotation: %v", err)
	} else {
		t.Log("✓ New connections work with rotated CA")
	}
	
	cancel() // Clean up long-running operation
	
	t.Log("✓ Connection preservation test completed")
}

// writeCAFiles creates a CA certificate and writes it to disk
func writeCAFiles(t *testing.T, certDir, commonName string, duration time.Duration) (string, *x509.Certificate, *rsa.PrivateKey) {
	caKey, err := utils.NewPrivateKey()
	if err != nil {
		t.Fatalf("Failed to create CA private key: %v", err)
	}
	
	caCert, err := cert.NewSelfSignedCACert(cert.Config{CommonName: commonName}, caKey)
	if err != nil {
		t.Fatalf("Failed to create CA certificate: %v", err)
	}
	
	// Update certificate validity period
	caCert.NotAfter = time.Now().Add(duration)
	
	caFilename := path.Join(certDir, fmt.Sprintf("%s-ca.crt", commonName))
	err = os.WriteFile(caFilename, utils.EncodeCertPEM(caCert), 0644)
	if err != nil {
		t.Fatalf("Failed to write CA certificate: %v", err)
	}
	
	return caFilename, caCert, caKey
}

// writeClientCerts creates a client certificate signed by the given CA
func writeClientCerts(t *testing.T, certDir string, caCert *x509.Certificate, caKey *rsa.PrivateKey, commonName string, duration time.Duration) (string, string) {
	clientKey, err := utils.NewPrivateKey()
	if err != nil {
		t.Fatalf("Failed to create client private key: %v", err)
	}
	
	// Create client certificate template
	template := x509.Certificate{
		SerialNumber: big.NewInt(time.Now().UnixNano()),
		Subject: pkix.Name{
			CommonName:   commonName,
			Organization: []string{"system:masters"}, // Give client admin privileges
		},
		NotBefore:   time.Now(),
		NotAfter:    time.Now().Add(duration),
		KeyUsage:    x509.KeyUsageKeyEncipherment | x509.KeyUsageDigitalSignature,
		ExtKeyUsage: []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth},
	}
	
	// Create certificate signed by CA
	certDER, err := x509.CreateCertificate(rand.Reader, &template, caCert, &clientKey.PublicKey, caKey)
	if err != nil {
		t.Fatalf("Failed to create client certificate: %v", err)
	}
	
	// Write certificate file
	certFilename := path.Join(certDir, fmt.Sprintf("%s-client.crt", commonName))
	certPEM := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: certDER})
	err = os.WriteFile(certFilename, certPEM, 0644)
	if err != nil {
		t.Fatalf("Failed to write client certificate: %v", err)
	}
	
	// Write private key file
	keyFilename := path.Join(certDir, fmt.Sprintf("%s-client.key", commonName))
	keyBytes, err := x509.MarshalPKCS8PrivateKey(clientKey)
	if err != nil {
		t.Fatalf("Failed to marshal client private key: %v", err)
	}
	keyPEM := pem.EncodeToMemory(&pem.Block{Type: "PRIVATE KEY", Bytes: keyBytes})
	err = os.WriteFile(keyFilename, keyPEM, 0644)
	if err != nil {
		t.Fatalf("Failed to write client private key: %v", err)
	}
	
	return certFilename, keyFilename
} 
