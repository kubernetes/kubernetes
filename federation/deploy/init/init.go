/*
Copyright 2016 The Kubernetes Authors.

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
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/base64"
	"encoding/pem"
	"fmt"
	"math/big"
	"net"
	"time"

	log "github.com/Sirupsen/logrus"
	flag "github.com/spf13/pflag"
	release_1_4 "k8s.io/client-go/1.4/kubernetes"
	api "k8s.io/client-go/1.4/pkg/api"
	"k8s.io/client-go/1.4/pkg/api/errors"
	corev1 "k8s.io/client-go/1.4/pkg/api/v1"
	"k8s.io/client-go/1.4/pkg/watch"
	"k8s.io/client-go/1.4/rest"
)

const (
	orgName        = "Kubernetes Cluster Federation"
	cmName         = "federation-controller-manager"
	kubeconfigName = "kubeconfig"
)

var (
	namespace    = flag.String("namespace", "federation", "namespace of the federation control plane components")
	secretName   = flag.String("secret", "federation-apiserver-secrets", "name of the federation ")
	svcName      = flag.String("service", "federation-apiserver", "namespace of the federation control plane components")
	timeout      = flag.Duration("timeout", 5*time.Minute, "duration to wait to obtain the federation API server's loadbalancer name/address before timing out")
	certValidity = flag.Duration("cert-validity", 365*24*time.Hour, "Certificate validity duration")
)

type certKey struct {
	cert []byte
	key  []byte
}

func main() {
	flag.Parse()

	ccfg, err := rest.InClusterConfig()
	if err != nil {
		log.Fatal("Failed to obtain in-cluster client config")
	}
	clientset := release_1_4.NewForConfigOrDie(ccfg)

	exists, err := secretExists(clientset, *namespace, *secretName)
	if err != nil {
		log.WithFields(log.Fields{
			"namespace": *namespace,
			"error":     err,
		}).Fatal("Failed to check whether federation certificates secret already exists")
	}
	if exists {
		log.WithFields(log.Fields{
			"namespace": *namespace,
			"secret":    *secretName,
		}).Info("Secret already exists, nothing to do here. Exiting")
		return
	}

	ips, hostnames, err := lbTargets(clientset, *namespace, *svcName, *timeout)
	if err != nil {
		log.WithFields(log.Fields{
			"namespace": *namespace,
			"error":     err,
		}).Fatal("Failed to retrieve federation API server load balancer IP addresses and/or hostnames")
	}

	cks, err := certs(*svcName, *certValidity, ips, hostnames)
	if err != nil {
		log.WithField("error", err).
			Fatal("Failed to generate certificates")
	}

	if err := createSecret(clientset, *namespace, *secretName, cks); err != nil {
		log.WithField("error", err).
			Fatal("Failed to create a secret")
	}

	// DEBUG: Remove before submission
	time.Sleep(5 * time.Minute)
}

func secretExists(clientset *release_1_4.Clientset, namespace, secretName string) (bool, error) {
	secret, err := clientset.Core().Secrets(namespace).Get(secretName)
	if errors.IsNotFound(err) {
		return false, nil
	}
	if err != nil {
		return false, fmt.Errorf("failed to get secret: %v", err)
	}
	return secret != nil && secret.Name == secretName, nil
}

func lbTargets(clientset *release_1_4.Clientset, namespace, svcName string, timeout time.Duration) ([]string, []string, error) {
	ips := []string{}
	hostnames := []string{}

	listOptions := api.SingleObject(api.ObjectMeta{
		Name: svcName,
	})
	w, err := clientset.Core().Services(namespace).Watch(listOptions)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to set a watch on federation control plane services")
	}

	_, err = watch.Until(timeout, w, func(e watch.Event) (bool, error) {
		switch e.Type {
		case watch.Added, watch.Modified:
			svc, ok := e.Object.(*corev1.Service)
			if !ok {
				log.WithFields(log.Fields{
					"namespace": namespace,
					"type":      fmt.Sprintf("%T", svc),
				}).Warn("Expected events for Services")
				return false, nil
			}

			// This has a potential problem when a service loadbalancer cloud
			// provider implementation populates both IPs and Hostnames, and
			// particularly when it populates multiple of such entries. We get
			// notified when the very first address/name is added and we
			// process it and return. If a new address/name is added later, we
			// can't process them.
			// TODO(madhusudancs): Provide a way to specify which of the IPs
			// or hostnames to wait for and for how long to wait.
			if svc.Spec.Type == corev1.ServiceTypeLoadBalancer {
				for _, ing := range svc.Status.LoadBalancer.Ingress {
					if len(ing.IP) > 0 {
						ips = append(ips, ing.IP)
					}
					if len(ing.Hostname) > 0 {
						hostnames = append(hostnames, ing.Hostname)
					}
				}
			}

			log.WithFields(log.Fields{
				"namespace": namespace,
				"service":   svc.Name,
				"IPs":       ips,
				"Hostnames": hostnames,
			}).Info("Added/Modified")

			if len(ips)+len(hostnames) > 0 {
				return true, nil
			}

			return false, nil
		case watch.Deleted:
			svc, ok := e.Object.(*corev1.Service)
			if !ok {
				log.WithFields(log.Fields{
					"namespace": namespace,
					"type":      fmt.Sprintf("%T", svc),
				}).Warn("Expected events for Services")
				return false, nil
			}
			log.WithFields(log.Fields{
				"namespace": namespace,
				"service":   svc.Name,
			}).Info("Deleted")
			return false, nil
		case watch.Error:
			// Handle error and return with an error.
			log.WithFields(log.Fields{
				"namespace": namespace,
			}).Info("Error event")
			return false, fmt.Errorf("received watch error: %+v", e.Object)
		default:
			return false, nil
		}
	})
	if err != nil {
		return nil, nil, fmt.Errorf("failed to watch federation control plane services")
	}

	return ips, hostnames, nil
}

func certs(svcName string, certValidity time.Duration, ips, hostnames []string) (map[string]*certKey, error) {
	template := x509.Certificate{
		Subject: pkix.Name{
			Organization: []string{orgName},
		},
		NotBefore: time.Now(),
		NotAfter:  time.Now().Add(certValidity),

		KeyUsage:              x509.KeyUsageKeyEncipherment | x509.KeyUsageDigitalSignature,
		ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
		BasicConstraintsValid: true,
	}

	for _, ip := range ips {
		addr := net.ParseIP(ip)
		if addr == nil {
			log.WithField("ip", ip).
				Warn("Failed to parse IP")
			continue
		}
		template.IPAddresses = append(template.IPAddresses, addr)
	}
	for _, hn := range hostnames {
		template.DNSNames = append(template.DNSNames, hn)
	}

	// Arbitrarily choose the first federation API server IP address or
	// the hostname as the CA common name.
	caCN := ""
	if len(ips) > 0 {
		caCN = ips[0]
	} else if len(hostnames) > 0 {
		caCN = hostnames[0]
	} else {
		return nil, fmt.Errorf("at least one IP address or hostname must be specified")
	}
	caCN = fmt.Sprintf("%s@%d", template.NotBefore.Unix())

	// CA
	caCertKey, err := cert(caCN, template, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to generate a CA certificate: %v", err)
	}

	caCertObj, err := x509.ParseCertificate(caCertKey.cert)
	if err != nil {
		return nil, fmt.Errorf("failed to parse the CA certificate: %v", err)
	}

	// Server
	serverCertKey, err := cert(svcName, template, caCertObj)
	if err != nil {
		return nil, fmt.Errorf("failed to generate a server certificate: %v", err)
	}

	// TODO(madhusudancs): Ideally these clients (Controller Manager and Admin
	// kubeconfig) should generate their own private keys and a CSR, submit
	// that CSR to the server for signing and retrieve the signed certificate
	// for their use. This process is being automated/simplified by the TLS
	// bootstrap proposal. Once that proposal is implemented in federation API
	// Server, we should just switch to that instead of generating the client
	// keys and certificates in this federation API server's init process and
	// passing it down to the clients.

	// Controller Manager
	cmCertKey, err := cert(cmName, template, caCertObj)
	if err != nil {
		return nil, fmt.Errorf("failed to generate a client certificate for controller manager: %v", err)
	}

	// Admin kubeconfig
	kubeconfigCertKey, err := cert(kubeconfigName, template, caCertObj)
	if err != nil {
		return nil, fmt.Errorf("failed to generate a client certificate for admin kubeconfig: %v", err)
	}

	return map[string]*certKey{
		"ca":                 caCertKey,
		"server":             serverCertKey,
		"controller-manager": cmCertKey,
		"kubeconfig":         kubeconfigCertKey,
	}, nil
}

// cert generates a certificate/key pair for a given template.
// Args:
//   name: the common name of the entity for which the certificate/key pair
//         must be generated.
//   template: a template for which a certificate/key pair must be produced. It
//             must be passed by-value because it is modified in this function.
//   caCert: An optional CA certificate. If it is supplied, then that
//           certificate will be used to sign the generated certificate. If it
//           is nil, then it will be assumed that this certificate is for a new
//           CA and a self-signed CA certificate will be generated.
func cert(name string, template x509.Certificate, caCert *x509.Certificate) (*certKey, error) {
	var parent *x509.Certificate
	if caCert == nil {
		parent = &template
		template.IsCA = true
		template.KeyUsage |= x509.KeyUsageCertSign
	} else {
		parent = caCert
	}

	serialNumberLimit := new(big.Int).Lsh(big.NewInt(1), 128)
	serialNumber, err := rand.Int(rand.Reader, serialNumberLimit)
	if err != nil {
		return nil, fmt.Errorf("failed to generate serial number: %v", err)
	}
	template.SerialNumber = serialNumber

	template.Subject.CommonName = name

	privKey, err := ecdsa.GenerateKey(elliptic.P384(), rand.Reader)
	if err != nil {
		return nil, fmt.Errorf("failed to generate private key: %v", err)
	}

	cert, err := x509.CreateCertificate(rand.Reader, &template, parent, &privKey.PublicKey, privKey)
	if err != nil {
		return nil, fmt.Errorf("failed to create certificate: %v", err)
	}
	priv, err := x509.MarshalECPrivateKey(privKey)
	if err != nil {
		return nil, fmt.Errorf("unable to marshal ECDSA private key: %v", err)
	}
	return &certKey{cert, priv}, nil
}

func createSecret(clientset *release_1_4.Clientset, namespace, secretName string, cks map[string]*certKey) error {
	buf := &bytes.Buffer{}

	cksPemBase64 := make(map[string][]byte)
	for name, ck := range cks {
		ckPemBase64, err := encodeCertKey(buf, ck)
		if err != nil {
			return fmt.Errorf("%s pair failed to encode: %v", name, err)
		}
		cksPemBase64[fmt.Sprintf("%s.crt", name)] = ckPemBase64.cert
		cksPemBase64[fmt.Sprintf("%s.key", name)] = ckPemBase64.key
	}

	// Intentionally using the default secret type - `SecretTypeOpaque` (not
	// set, but it will be defaulted) instead of `SecretTypeTLS` or other
	// types because this secret will hold tokens and other credentials along
	// with certificate/key pairs.
	secret := &corev1.Secret{
		ObjectMeta: corev1.ObjectMeta{
			Name: secretName,
		},
		Data: cksPemBase64,
	}

	_, err := clientset.Core().Secrets(namespace).Create(secret)
	if errors.IsAlreadyExists(err) {
		log.WithFields(log.Fields{
			"namespace": namespace,
			"secret":    secretName,
		}).Info("Secret already exists, shouldn't be overwritten")
		return nil
	}
	return err
}

func encodeCertKey(buf *bytes.Buffer, ck *certKey) (*certKey, error) {
	certPemBase64, err := encodeToPemBase64(buf, ck.cert, "CERTIFICATE")
	if err != nil {
		return nil, fmt.Errorf("failed to encode the certificate: %v", err)
	}
	keyPemBase64, err := encodeToPemBase64(buf, ck.key, "EC PRIVATE KEY")
	if err != nil {
		return nil, fmt.Errorf("failed to encode the key: %v", err)
	}
	return &certKey{certPemBase64, keyPemBase64}, nil
}

func encodeToPemBase64(buf *bytes.Buffer, derBytes []byte, btype string) ([]byte, error) {
	buf.Reset()
	enc := base64.NewEncoder(base64.StdEncoding, buf)

	_, err := enc.Write(pem.EncodeToMemory(&pem.Block{Type: btype, Bytes: derBytes}))
	if err != nil {
		return nil, fmt.Errorf("failed to encode: %v", err)
	}
	if err := enc.Close(); err != nil {
		return nil, fmt.Errorf("failed to close and flush the base64 encoder: %v", err)
	}
	return buf.Bytes(), nil
}
