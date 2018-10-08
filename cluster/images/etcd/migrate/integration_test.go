// +build integration

/*
Copyright 2018 The Kubernetes Authors.

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

package main

import (
	"bytes"
	cryptorand "crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"fmt"
	"io/ioutil"
	"math/big"
	"net"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/blang/semver"
)

var (
	testSupportedVersions = MustParseSupportedVersions("2.2.1, 2.3.7, 3.0.17, 3.1.12")
	testVersionOldest     = &EtcdVersion{semver.MustParse("2.2.1")}
	testVersionPrevious   = &EtcdVersion{semver.MustParse("3.0.17")}
	testVersionLatest     = &EtcdVersion{semver.MustParse("3.1.12")}
)

func TestMigrate(t *testing.T) {
	migrations := []struct {
		title        string
		memberCount  int
		startVersion string
		endVersion   string
		protocol     string
	}{
		// upgrades
		{"v2-v3-up", 1, "2.2.1/etcd2", "3.0.17/etcd3", "https"},
		{"v3-v3-up", 1, "3.0.17/etcd3", "3.1.12/etcd3", "https"},
		{"oldest-newest-up", 1, "2.2.1/etcd2", "3.1.12/etcd3", "https"},

		// warning: v2->v3 ha upgrades not currently supported.
		{"ha-v3-v3-up", 3, "3.0.17/etcd3", "3.1.12/etcd3", "https"},

		// downgrades
		{"v3-v2-down", 1, "3.0.17/etcd3", "2.2.1/etcd2", "https"},
		{"v3-v3-down", 1, "3.1.12/etcd3", "3.0.17/etcd3", "https"},

		// warning: ha downgrades not yet supported.
	}

	for _, m := range migrations {
		t.Run(m.title, func(t *testing.T) {
			start := MustParseEtcdVersionPair(m.startVersion)
			end := MustParseEtcdVersionPair(m.endVersion)

			testCfgs := clusterConfig(t, m.title, m.memberCount, m.protocol)

			servers := []*EtcdMigrateServer{}
			for _, cfg := range testCfgs {
				client, err := NewEtcdMigrateClient(cfg)
				if err != nil {
					t.Fatalf("Failed to create client: %v", err)
				}
				server := NewEtcdMigrateServer(cfg, client)
				servers = append(servers, server)
			}

			// Start the servers.
			parallel(servers, func(server *EtcdMigrateServer) {
				dataDir, err := OpenOrCreateDataDirectory(server.cfg.dataDirectory)
				if err != nil {
					t.Fatalf("Error opening or creating data directory %s: %v", server.cfg.dataDirectory, err)
				}
				migrator := &Migrator{server.cfg, dataDir, server.client}
				err = migrator.MigrateIfNeeded(start)
				if err != nil {
					t.Fatalf("Migration failed: %v", err)
				}
				err = server.Start(start.version)
				if err != nil {
					t.Fatalf("Failed to start server: %v", err)
				}
			})

			// Write a value to each server, read it back.
			parallel(servers, func(server *EtcdMigrateServer) {
				key := fmt.Sprintf("/registry/%s", server.cfg.name)
				value := fmt.Sprintf("value-%s", server.cfg.name)
				err := server.client.Put(start.version, key, value)
				if err != nil {
					t.Fatalf("failed to write text value: %v", err)
				}

				checkVal, err := server.client.Get(start.version, key)
				if err != nil {
					t.Errorf("Error getting %s for validation: %v", key, err)
				}
				if checkVal != value {
					t.Errorf("Expected %s from %s but got %s", value, key, checkVal)
				}
			})

			// Migrate the servers in series.
			serial(servers, func(server *EtcdMigrateServer) {
				err := server.Stop()
				if err != nil {
					t.Fatalf("Stop server failed: %v", err)
				}
				dataDir, err := OpenOrCreateDataDirectory(server.cfg.dataDirectory)
				if err != nil {
					t.Fatalf("Error opening or creating data directory %s: %v", server.cfg.dataDirectory, err)
				}
				migrator := &Migrator{server.cfg, dataDir, server.client}
				err = migrator.MigrateIfNeeded(end)
				if err != nil {
					t.Fatalf("Migration failed: %v", err)
				}
				err = server.Start(end.version)
				if err != nil {
					t.Fatalf("Start server failed: %v", err)
				}
			})

			// Check that all test values can be read back from all the servers.
			parallel(servers, func(server *EtcdMigrateServer) {
				for _, s := range servers {
					key := fmt.Sprintf("/registry/%s", s.cfg.name)
					value := fmt.Sprintf("value-%s", s.cfg.name)
					checkVal, err := server.client.Get(end.version, key)
					if err != nil {
						t.Errorf("Error getting %s from etcd 2.x after rollback from 3.x: %v", key, err)
					}
					if checkVal != value {
						t.Errorf("Expected %s from %s but got %s when reading after rollback from %s to %s", value, key, checkVal, start, end)
					}
				}
			})

			// Stop the servers.
			parallel(servers, func(server *EtcdMigrateServer) {
				err := server.Stop()
				if err != nil {
					t.Fatalf("Failed to stop server: %v", err)
				}
			})

			// Check that version.txt contains the correct end version.
			parallel(servers, func(server *EtcdMigrateServer) {
				dataDir, err := OpenOrCreateDataDirectory(server.cfg.dataDirectory)
				v, err := dataDir.versionFile.Read()
				if err != nil {
					t.Fatalf("Failed to read version.txt file: %v", err)
				}
				if !v.Equals(end) {
					t.Errorf("Expected version.txt to contain %s but got %s", end, v)
				}
				// Integration tests are run in a docker container with umask of 0022.
				checkPermissions(t, server.cfg.dataDirectory, 0755|os.ModeDir)
				checkPermissions(t, dataDir.versionFile.path, 0644)
			})
		})
	}
}

func parallel(servers []*EtcdMigrateServer, fn func(server *EtcdMigrateServer)) {
	var wg sync.WaitGroup
	wg.Add(len(servers))
	for _, server := range servers {
		go func(s *EtcdMigrateServer) {
			defer wg.Done()
			fn(s)
		}(server)
	}
	wg.Wait()
}

func serial(servers []*EtcdMigrateServer, fn func(server *EtcdMigrateServer)) {
	for _, server := range servers {
		fn(server)
	}
}

func checkPermissions(t *testing.T, path string, expected os.FileMode) {
	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("Failed to stat file %s: %v", path, err)
	}
	if info.Mode() != expected {
		t.Errorf("Expected permissions for file %s of %s, but got %s", path, expected, info.Mode())
	}
}

func clusterConfig(t *testing.T, name string, memberCount int, protocol string) []*EtcdMigrateCfg {
	peers := []string{}
	for i := 0; i < memberCount; i++ {
		memberName := fmt.Sprintf("%s-%d", name, i)
		peerPort := uint64(2380 + i*10000)
		peer := fmt.Sprintf("%s=%s://127.0.0.1:%d", memberName, protocol, peerPort)
		peers = append(peers, peer)
	}
	initialCluster := strings.Join(peers, ",")

	extraArgs := ""
	if protocol == "https" {
		extraArgs = getOrCreateTLSPeerCertArgs(t)
	}

	cfgs := []*EtcdMigrateCfg{}
	for i := 0; i < memberCount; i++ {
		memberName := fmt.Sprintf("%s-%d", name, i)
		peerURL := fmt.Sprintf("%s://127.0.0.1:%d", protocol, uint64(2380+i*10000))
		cfg := &EtcdMigrateCfg{
			binPath:           "/usr/local/bin",
			name:              memberName,
			initialCluster:    initialCluster,
			port:              uint64(2379 + i*10000),
			peerListenUrls:    peerURL,
			peerAdvertiseUrls: peerURL,
			etcdDataPrefix:    "/registry",
			ttlKeysDirectory:  "/registry/events",
			supportedVersions: testSupportedVersions,
			dataDirectory:     fmt.Sprintf("/tmp/etcd-data-dir-%s", memberName),
			etcdServerArgs:    extraArgs,
		}
		cfgs = append(cfgs, cfg)
	}
	return cfgs
}

func getOrCreateTLSPeerCertArgs(t *testing.T) string {
	spec := TestCertSpec{
		host: "localhost",
		ips:  []string{"127.0.0.1"},
	}
	certDir := "/tmp/certs"
	certFile := filepath.Join(certDir, "test.crt")
	keyFile := filepath.Join(certDir, "test.key")
	err := getOrCreateTestCertFiles(certFile, keyFile, spec)
	if err != nil {
		t.Fatalf("failed to create server cert: %v", err)
	}
	return fmt.Sprintf("--peer-client-cert-auth --peer-trusted-ca-file=%s --peer-cert-file=%s --peer-key-file=%s", certFile, certFile, keyFile)
}

type TestCertSpec struct {
	host       string
	names, ips []string // in certificate
}

func getOrCreateTestCertFiles(certFileName, keyFileName string, spec TestCertSpec) (err error) {
	if _, err := os.Stat(certFileName); err == nil {
		if _, err := os.Stat(keyFileName); err == nil {
			return nil
		}
	}

	certPem, keyPem, err := generateSelfSignedCertKey(spec.host, parseIPList(spec.ips), spec.names)
	if err != nil {
		return err
	}

	os.MkdirAll(filepath.Dir(certFileName), os.FileMode(0777))
	err = ioutil.WriteFile(certFileName, certPem, os.FileMode(0777))
	if err != nil {
		return err
	}

	os.MkdirAll(filepath.Dir(keyFileName), os.FileMode(0777))
	err = ioutil.WriteFile(keyFileName, keyPem, os.FileMode(0777))
	if err != nil {
		return err
	}

	return nil
}

func parseIPList(ips []string) []net.IP {
	var netIPs []net.IP
	for _, ip := range ips {
		netIPs = append(netIPs, net.ParseIP(ip))
	}
	return netIPs
}

// generateSelfSignedCertKey creates a self-signed certificate and key for the given host.
// Host may be an IP or a DNS name
// You may also specify additional subject alt names (either ip or dns names) for the certificate
func generateSelfSignedCertKey(host string, alternateIPs []net.IP, alternateDNS []string) ([]byte, []byte, error) {
	priv, err := rsa.GenerateKey(cryptorand.Reader, 2048)
	if err != nil {
		return nil, nil, err
	}

	template := x509.Certificate{
		SerialNumber: big.NewInt(1),
		Subject: pkix.Name{
			CommonName: fmt.Sprintf("%s@%d", host, time.Now().Unix()),
		},
		NotBefore: time.Unix(0, 0),
		NotAfter:  time.Now().Add(time.Hour * 24 * 365 * 100),

		KeyUsage:              x509.KeyUsageKeyEncipherment | x509.KeyUsageDigitalSignature | x509.KeyUsageCertSign,
		ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth, x509.ExtKeyUsageClientAuth},
		BasicConstraintsValid: true,
		IsCA:                  true,
	}

	if ip := net.ParseIP(host); ip != nil {
		template.IPAddresses = append(template.IPAddresses, ip)
	} else {
		template.DNSNames = append(template.DNSNames, host)
	}

	template.IPAddresses = append(template.IPAddresses, alternateIPs...)
	template.DNSNames = append(template.DNSNames, alternateDNS...)

	derBytes, err := x509.CreateCertificate(cryptorand.Reader, &template, &template, &priv.PublicKey, priv)
	if err != nil {
		return nil, nil, err
	}

	// Generate cert
	certBuffer := bytes.Buffer{}
	if err := pem.Encode(&certBuffer, &pem.Block{Type: "CERTIFICATE", Bytes: derBytes}); err != nil {
		return nil, nil, err
	}

	// Generate key
	keyBuffer := bytes.Buffer{}
	if err := pem.Encode(&keyBuffer, &pem.Block{Type: "RSA PRIVATE KEY", Bytes: x509.MarshalPKCS1PrivateKey(priv)}); err != nil {
		return nil, nil, err
	}

	return certBuffer.Bytes(), keyBuffer.Bytes(), nil
}
