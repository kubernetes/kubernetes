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

// This program implements the initialization code required to start a
// federation API server. This code is intended to be run inside an init
// container (http://kubernetes.io/docs/user-guide/production-pods/#handling-initialization)
// in the federation API server pod.
//
// Federation API server is deployed as a Kubernetes Deployment resource and
// hence the number of Deployment pod replicas can be scaled up and down. This
// means that there can be multiple copies of the federation API server pod
// running. Since init containers are run by every pod, we need to ensure that
// only one of these initializers can create the necessary secrets that store
// the credentials and other configuration. Allowing multiple initializers to
// create these secrets might mix up values from different initializers
// leading to invalid credentials. We could elect a leader to decide which of
// these initializers are allowed to create those credentials, but that isn't
// sufficient since we also need guards against pod restarts. So instead, we
// use the secret whose name is provided by the `--secret` flag as a lock to
// ensure that the credentials are created only once.

// At the very beginning, this code checks if the secret indicated by the flag
// already exists. If the secret exists, then the init container just
// successfully exits. Otherwise, it continues to run the initialization code.
// After creating all the credentials in memory, it attempts to write them to
// the secret indicated by the flag. If that fails it successfully exits.
// Otherwise it continues to write all the other secrets. Effectively, the
// first secret, i.e. the secret indicated by the `--secret` flag acts as
// lock here.
package main

import (
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/x509"
	"crypto/x509/pkix"
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
	"k8s.io/client-go/1.4/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/1.4/tools/clientcmd/api"
)

const (
	orgName        = "Kubernetes Cluster Federation"
	cmName         = "federation-controller-manager"
	kubeconfigName = "kubeconfig"

	// credCharTable provides a source of valid characters to credentials
	// generator.
	//
	// NOTE: Do *NOT* modify this table. In particular, do not insert
	// non-ASCII characters. The credentials generator indexes into this
	// string assuming that the bytes of this string are also valid ASCII
	// characters, which isn't true for non-ASCII characters. See
	// https://blog.golang.org/strings for a detailed discussion on this
	// topic.
	credCharTable = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

	passwdLen = 16
	tokenLen  = 32
	adminUser = "admin"
	adminUID  = "admin"
)

var (
	namespace    = flag.String("namespace", "federation", "namespace of the federation control plane components")
	secretName   = flag.String("secret", "federation-apiserver-credentials", "name of the secret where the federation API server user credentials are stored")
	cmSecretName = flag.String("controllermanager-kubeconfig-secret", "federation-controller-manager-kubeconfig", "name of the secret where the federation controller manager kubeconfig is stored")
	svcName      = flag.String("service", "federation-apiserver", "name of the federation API server service")
	timeout      = flag.Duration("timeout", 5*time.Minute, "duration to wait to obtain the federation API server's loadbalancer name/address before timing out")
	certValidity = flag.Duration("cert-validity", 365*24*time.Hour, "certificate validity duration")
)

type certKey struct {
	cert []byte
	key  []byte
	priv *ecdsa.PrivateKey
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

	cAddr, err := canonicalAddress(ips, hostnames)
	if err != nil {
		log.WithFields(log.Fields{
			"namespace": *namespace,
			"ips":       ips,
			"hostnames": hostnames,
			"error":     err,
		}).Fatal("Failed to find the federation API server's canonical address")
	}

	cks, err := certs(*svcName, *certValidity, cAddr, ips, hostnames)
	if err != nil {
		log.WithField("error", err).
			Fatal("Failed to generate certificates")
	}

	password, err := credGen(passwdLen)
	if err != nil {
		log.WithField("error", err).
			Fatal("Failed to generate pseudorandom password")
	}

	token, err := credGen(tokenLen)
	if err != nil {
		log.WithField("error", err).
			Fatal("Failed to generate pseudorandom token")
	}

	cmKubeconfig, err := controllerManagerKubeconfig(*namespace, *svcName, cks["ca"].cert, cks["controller-manager"])
	if err != nil {
		log.WithField("error", err).
			Fatal("Failed to controller manager kubeconfig")
	}

	if err := createSecrets(clientset, *namespace, *svcName, cAddr, *secretName, *cmSecretName, password, token, cks, cmKubeconfig); err != nil {
		log.WithField("error", err).
			Fatal("Failed to create secrets")
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

func lbTargets(clientset *release_1_4.Clientset, namespace, svcName string, timeout time.Duration) ([]net.IP, []string, error) {
	ips := []net.IP{}
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
						ip := net.ParseIP(ing.IP)
						if ip == nil {
							log.WithField("IP", ing.IP).
								Warn("Failed to parse IP")
						} else {
							ips = append(ips, ip)
						}
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

	log.WithFields(log.Fields{
		"namespace": namespace,
		"service":   svcName,
		"IPs":       ips,
		"Hostnames": hostnames,
	}).Info("Load balancer targets detected")

	return ips, hostnames, nil
}

// canonicalAddress returns the authoritative address of the federation API
// server endpoint.
// The current implementation arbitrarily chooses either the first IP address
// if it exists, or the first hostname as the canonical address.
func canonicalAddress(ips []net.IP, hostnames []string) (string, error) {
	if len(ips) > 0 {
		return ips[0].String(), nil
	} else if len(hostnames) > 0 {
		return hostnames[0], nil
	}
	return "", fmt.Errorf("at least one IP address or hostname must be supplied")
}

func certs(svcName string, certValidity time.Duration, cAddr string, ips []net.IP, hostnames []string) (map[string]*certKey, error) {
	template := x509.Certificate{
		Subject: pkix.Name{
			Organization: []string{orgName},
		},
		NotBefore: time.Now(),
		NotAfter:  time.Now().Add(certValidity),

		KeyUsage:              x509.KeyUsageKeyEncipherment | x509.KeyUsageDigitalSignature,
		ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth, x509.ExtKeyUsageClientAuth},
		BasicConstraintsValid: true,
	}

	caCN := fmt.Sprintf("%s@%d", cAddr, template.NotBefore.Unix())

	zeroIPs := []net.IP{}
	zeroHostnames := []string{}

	// CA
	caCertKey, err := cert(caCN, zeroIPs, zeroHostnames, template, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to generate a CA certificate: %v", err)
	}

	// Server
	serverCertKey, err := cert(svcName, ips, hostnames, template, caCertKey)
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
	cmCertKey, err := cert(cmName, zeroIPs, zeroHostnames, template, caCertKey)
	if err != nil {
		return nil, fmt.Errorf("failed to generate a client certificate for controller manager: %v", err)
	}

	// Admin kubeconfig
	kubeconfigCertKey, err := cert(kubeconfigName, zeroIPs, zeroHostnames, template, caCertKey)
	if err != nil {
		return nil, fmt.Errorf("failed to generate a client certificate for admin kubeconfig: %v", err)
	}

	log.WithFields(log.Fields{
		"ServerName": svcName,
		"IPs":        ips,
		"Hostnames":  hostnames,
	}).Info("Certificates generated")

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
//   ips: IP addresses for subject alternative names.
//   hostnames: DNS names for subject alternative names.
//   template: a template for which a certificate/key pair must be produced. It
//             must be passed by-value because it is modified in this function.
//   caCertKey: An optional CA certificate/key pair. If it is supplied, then
//              that certificate will be used to sign the generated
//              certificate. If it is nil, then it will be assumed that this
//              certificate is for a new CA and a self-signed CA certificate
//              will be generated.
func cert(name string, ips []net.IP, hostnames []string, template x509.Certificate, caCertKey *certKey) (*certKey, error) {
	privKey, err := ecdsa.GenerateKey(elliptic.P384(), rand.Reader)
	if err != nil {
		return nil, fmt.Errorf("failed to generate private key: %v", err)
	}

	var signCert *x509.Certificate
	var signKey *ecdsa.PrivateKey
	if caCertKey == nil {
		signCert = &template
		signKey = privKey
		template.IsCA = true
		template.KeyUsage |= x509.KeyUsageCertSign
	} else {
		caCertObj, err := x509.ParseCertificate(caCertKey.cert)
		if err != nil {
			return nil, fmt.Errorf("failed to parse the CA certificate: %v", err)
		}
		signCert = caCertObj
		signKey = caCertKey.priv
	}

	template.IPAddresses = append(template.IPAddresses, ips...)
	template.DNSNames = append(template.DNSNames, hostnames...)

	serialNumberLimit := new(big.Int).Lsh(big.NewInt(1), 128)
	serialNumber, err := rand.Int(rand.Reader, serialNumberLimit)
	if err != nil {
		return nil, fmt.Errorf("failed to generate serial number: %v", err)
	}
	template.SerialNumber = serialNumber

	template.Subject.CommonName = name

	cert, err := x509.CreateCertificate(rand.Reader, &template, signCert, &privKey.PublicKey, signKey)
	if err != nil {
		return nil, fmt.Errorf("failed to create certificate: %v", err)
	}
	priv, err := x509.MarshalECPrivateKey(privKey)
	if err != nil {
		return nil, fmt.Errorf("unable to marshal ECDSA private key: %v", err)
	}
	return &certKey{cert, priv, privKey}, nil
}

func createSecrets(clientset *release_1_4.Clientset, namespace, svcName, cAddr, secretName, cmSecretName, password, token string, cks map[string]*certKey, cmKubeconfig []byte) error {
	data := map[string][]byte{
		fmt.Sprintf("%s-advertise-address", svcName): []byte(cAddr),
		"basic_auth.csv":                             []byte(fmt.Sprintf("%s,%s,%s", password, adminUser, adminUID)),
		"known_tokens.csv":                           []byte(fmt.Sprintf("%s,%s,%s", token, adminUser, adminUID)),
	}

	for name, ck := range cks {
		ckPem := encodeCertKey(ck)
		data[fmt.Sprintf("%s.crt", name)] = ckPem.cert
		data[fmt.Sprintf("%s.key", name)] = ckPem.key
	}

	// Intentionally using the default secret type - `SecretTypeOpaque` (not
	// set, but it will be defaulted) instead of `SecretTypeTLS` or other
	// types because this secret will hold tokens and other credentials along
	// with certificate/key pairs.
	secret := &corev1.Secret{
		ObjectMeta: corev1.ObjectMeta{
			Name: secretName,
		},
		Data: data,
	}

	secret, err := clientset.Core().Secrets(namespace).Create(secret)
	if errors.IsAlreadyExists(err) {
		log.WithFields(log.Fields{
			"namespace": namespace,
			"secret":    secretName,
		}).Info("Secret already exists, shouldn't be overwritten. Not attempting to write any more secrets")
		return nil
	}
	if err != nil {
		return err
	}
	log.WithFields(log.Fields{
		"namespace": namespace,
		"secret":    secret.Name,
	}).Info("Secret created")

	cmKubeconfigSecret := &corev1.Secret{
		ObjectMeta: corev1.ObjectMeta{
			Name: cmSecretName,
		},
		Data: map[string][]byte{
			"kubeconfig": cmKubeconfig,
		},
	}

	// Any form of error at this point is fatal. But it is not just fatal,
	// we also need to rollback all the secrets we have created so far, so
	// that the next time this initializer is run, it creates a consistent
	// set of secrets. See the package doc for more details.
	cmKubeconfigSecret, err = clientset.Core().Secrets(namespace).Create(cmKubeconfigSecret)
	if err != nil {
		gps := int64(0)
		dOpts := &api.DeleteOptions{
			GracePeriodSeconds: &gps,
		}

		dErr := clientset.Core().Secrets(namespace).Delete(secretName, dOpts)
		if dErr != nil {
			log.WithFields(log.Fields{
				"namespace": namespace,
				"secret":    secretName,
				"error":     dErr,
			}).Error("Failed to delete secret during rollback")
		} else {
			log.WithFields(log.Fields{
				"namespace": namespace,
				"secret":    secretName,
			}).Info("Secret deleted")
		}

		dErr = clientset.Core().Secrets(namespace).Delete(cmSecretName, dOpts)
		if dErr != nil {
			log.WithFields(log.Fields{
				"namespace": namespace,
				"secret":    cmSecretName,
				"error":     dErr,
			}).Error("Failed to delete secret during rollback")
		} else {
			log.WithFields(log.Fields{
				"namespace": namespace,
				"secret":    cmSecretName,
			}).Info("Secret deleted")
		}

		return err
	}
	log.WithFields(log.Fields{
		"namespace": namespace,
		"secret":    cmKubeconfigSecret.Name,
	}).Info("Secret created")

	return err
}

func encodeCertKey(ck *certKey) *certKey {
	certPem := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: ck.cert})
	keyPem := pem.EncodeToMemory(&pem.Block{Type: "EC PRIVATE KEY", Bytes: ck.key})
	return &certKey{certPem, keyPem, nil}
}

func controllerManagerKubeconfig(namespace, svcName string, caCert []byte, cmCertKey *certKey) ([]byte, error) {
	config := clientcmdapi.Config{
		Kind:       "Config",
		APIVersion: "v1",
		Clusters: map[string]*clientcmdapi.Cluster{
			svcName: {
				Server: "https://" + svcName,
				CertificateAuthorityData: caCert,
			},
		},
		AuthInfos: map[string]*clientcmdapi.AuthInfo{
			svcName: {
				ClientCertificateData: cmCertKey.cert,
				ClientKeyData:         cmCertKey.key,
			},
		},
		Contexts: map[string]*clientcmdapi.Context{
			svcName: {
				Cluster:   svcName,
				AuthInfo:  svcName,
				Namespace: namespace,
			},
		},
		CurrentContext: svcName,
	}
	return clientcmd.Write(config)
}

func credGen(n uint) (string, error) {
	max := big.NewInt(int64(len(credCharTable)))

	cred := ""
	for i := uint(0); i < n; i++ {
		rn, err := rand.Int(rand.Reader, max)
		if err != nil {
			return "", fmt.Errorf("failed to generate a random number(%d): %v", i, err)
		}
		// This type of string concatenation, i.e. using arithmetic plus
		// operator is not necessarily efficient but is more readable.
		// Since it is not necessary to be fast here, readability wins.
		cred += string(credCharTable[rn.Int64()])
	}
	return cred, nil
}
