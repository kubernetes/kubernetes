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

package main

import (
	"crypto"
	"crypto/mldsa"
	"crypto/rand"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"fmt"
	"math/big"
	"net"
	"net/netip"
	"os"
	"path/filepath"
	"strings"
	"time"
)

type caKeyPair struct {
	cert *x509.Certificate
	key  crypto.Signer
}

func main() {
	if len(os.Args) < 3 {
		fatal("usage: %s <variant> <output-dir> [key=value ...]", os.Args[0])
	}
	variant, outDir := os.Args[1], os.Args[2]
	args := parseKV(os.Args[3:])
	genKey := makeKeyGen(variant)

	serverCA := genCA(genKey, "server-ca")
	clientCA := genCA(genKey, "client-ca")
	reqHeaderCA := genCA(genKey, "request-header-ca")

	writeCA(outDir, "server-ca", serverCA)
	writeCA(outDir, "client-ca", clientCA)
	writeCA(outDir, "request-header-ca", reqHeaderCA)

	genServing(genKey, serverCA, outDir, "serving-kube-apiserver",
		csv(args["api-dns"]), parseIPs(args["api-ips"]))
	genServing(genKey, serverCA, outDir, "serving-kube-aggregator",
		csv(args["agg-dns"]), parseIPs(args["agg-ips"]))

	genClient(genKey, clientCA, outDir, "client-controller", "system:kube-controller-manager", nil)
	genClient(genKey, clientCA, outDir, "client-scheduler", "system:kube-scheduler", nil)
	genClient(genKey, clientCA, outDir, "client-admin", "system:admin", []string{"system:masters"})
	genClient(genKey, clientCA, outDir, "client-kube-apiserver", "kube-apiserver", nil)
	genClient(genKey, clientCA, outDir, "client-kube-aggregator", "system:kube-aggregator", []string{"system:masters"})
	genClient(genKey, clientCA, outDir, "client-kube-proxy", "system:kube-proxy", []string{"system:nodes"})
	genClient(genKey, reqHeaderCA, outDir, "client-auth-proxy", "system:auth-proxy", nil)

	if n := args["node-name"]; n != "" {
		genClient(genKey, clientCA, outDir, "client-kubelet", "system:node:"+n, []string{"system:nodes"})
	}
}

func makeKeyGen(variant string) func() crypto.Signer {
	return func() crypto.Signer {
		var key crypto.Signer
		var err error
		switch variant {
		case "MLDSA44":
			key, err = mldsa.GenerateKey(mldsa.MLDSA44())
		case "MLDSA65":
			key, err = mldsa.GenerateKey(mldsa.MLDSA65())
		case "MLDSA87":
			key, err = mldsa.GenerateKey(mldsa.MLDSA87())
		default:
			fatal("unsupported variant: %s", variant)
		}
		if err != nil {
			fatal("generating key: %v", err)
		}
		return key
	}
}

func genCA(genKey func() crypto.Signer, cn string) *caKeyPair {
	key := genKey()
	tmpl := &x509.Certificate{
		SerialNumber:          mustSerial(),
		Subject:               pkix.Name{CommonName: cn},
		NotBefore:             time.Now(),
		NotAfter:              time.Now().Add(10 * 365 * 24 * time.Hour),
		IsCA:                  true,
		BasicConstraintsValid: true,
		KeyUsage:              x509.KeyUsageCertSign | x509.KeyUsageCRLSign,
	}
	der, err := x509.CreateCertificate(rand.Reader, tmpl, tmpl, key.Public(), key)
	if err != nil {
		fatal("creating CA %s: %v", cn, err)
	}
	cert, err := x509.ParseCertificate(der)
	if err != nil {
		fatal("parsing CA %s: %v", cn, err)
	}
	return &caKeyPair{cert: cert, key: key}
}

func genServing(genKey func() crypto.Signer, ca *caKeyPair, outDir, name string, dns []string, ips []net.IP) {
	key := genKey()
	tmpl := &x509.Certificate{
		SerialNumber: mustSerial(),
		Subject:      pkix.Name{CommonName: name},
		NotBefore:    time.Now(),
		NotAfter:     time.Now().Add(5 * 365 * 24 * time.Hour),
		KeyUsage:     x509.KeyUsageDigitalSignature,
		ExtKeyUsage:  []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
		DNSNames:     dns,
		IPAddresses:  ips,
	}
	writeSigned(outDir, name, tmpl, key, ca)
}

func genClient(genKey func() crypto.Signer, ca *caKeyPair, outDir, name, cn string, orgs []string) {
	key := genKey()
	tmpl := &x509.Certificate{
		SerialNumber: mustSerial(),
		Subject:      pkix.Name{CommonName: cn, Organization: orgs},
		NotBefore:    time.Now(),
		NotAfter:     time.Now().Add(5 * 365 * 24 * time.Hour),
		KeyUsage:     x509.KeyUsageDigitalSignature,
		ExtKeyUsage:  []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth},
	}
	writeSigned(outDir, name, tmpl, key, ca)
}

func writeSigned(outDir, name string, tmpl *x509.Certificate, key crypto.Signer, ca *caKeyPair) {
	der, err := x509.CreateCertificate(rand.Reader, tmpl, ca.cert, key.Public(), ca.key)
	if err != nil {
		fatal("signing %s: %v", name, err)
	}
	mustWritePEM(filepath.Join(outDir, name+".crt"), "CERTIFICATE", der)
	keyDER, err := x509.MarshalPKCS8PrivateKey(key)
	if err != nil {
		fatal("marshaling key for %s: %v", name, err)
	}
	mustWritePEM(filepath.Join(outDir, name+".key"), "PRIVATE KEY", keyDER)
}

func writeCA(outDir, name string, ca *caKeyPair) {
	mustWritePEM(filepath.Join(outDir, name+".crt"), "CERTIFICATE", ca.cert.Raw)
	keyDER, err := x509.MarshalPKCS8PrivateKey(ca.key)
	if err != nil {
		fatal("marshaling CA key %s: %v", name, err)
	}
	mustWritePEM(filepath.Join(outDir, name+".key"), "PRIVATE KEY", keyDER)
}

func mustWritePEM(path, typ string, data []byte) {
	f, err := os.Create(path)
	if err != nil {
		fatal("creating %s: %v", path, err)
	}
	defer f.Close()
	if err := pem.Encode(f, &pem.Block{Type: typ, Bytes: data}); err != nil {
		fatal("writing %s: %v", path, err)
	}
}

func mustSerial() *big.Int {
	s, err := rand.Int(rand.Reader, new(big.Int).Lsh(big.NewInt(1), 128))
	if err != nil {
		fatal("generating serial: %v", err)
	}
	return s
}

func parseKV(args []string) map[string]string {
	m := make(map[string]string)
	for _, a := range args {
		k, v, _ := strings.Cut(a, "=")
		m[k] = v
	}
	return m
}

func csv(s string) []string {
	if s == "" {
		return nil
	}
	return strings.Split(s, ",")
}

func parseIPs(s string) []net.IP {
	if s == "" {
		return nil
	}
	var ips []net.IP
	for _, p := range strings.Split(s, ",") {
		if addr, err := netip.ParseAddr(strings.TrimSpace(p)); err == nil {
			ips = append(ips, addr.AsSlice())
		}
	}
	return ips
}

func fatal(format string, args ...any) {
	fmt.Fprintf(os.Stderr, format+"\n", args...)
	os.Exit(1)
}
