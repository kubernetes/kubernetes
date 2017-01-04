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

package localkube

import (
	"crypto/x509"
	"encoding/pem"
	"fmt"
	"io/ioutil"
	"net"
	"path"

	"github.com/golang/glog"

	"k8s.io/kubernetes/pkg/util/config"
	utilnet "k8s.io/kubernetes/pkg/util/net"
)

const serverInterval = 200

// LocalkubeServer provides a fully functional Kubernetes cluster running entirely through goroutines
type LocalkubeServer struct {
	// Inherits Servers
	Servers

	// Options
	Containerized            bool
	EnableDNS                bool
	DNSDomain                string
	DNSIP                    net.IP
	LocalkubeDirectory       string
	ServiceClusterIPRange    net.IPNet
	APIServerAddress         net.IP
	APIServerPort            int
	APIServerInsecureAddress net.IP
	APIServerInsecurePort    int
	ShouldGenerateCerts      bool
	ShowVersion              bool
	RuntimeConfig            config.ConfigurationMap
	NodeIP                   net.IP
	ContainerRuntime         string
	NetworkPlugin            string
	ExtraConfig              ExtraOptionSlice
}

func (lk *LocalkubeServer) AddServer(server Server) {
	lk.Servers = append(lk.Servers, server)
}

func (lk LocalkubeServer) GetEtcdDataDirectory() string {
	return path.Join(lk.LocalkubeDirectory, "etcd")
}

func (lk LocalkubeServer) GetDNSDataDirectory() string {
	return path.Join(lk.LocalkubeDirectory, "dns")
}

func (lk LocalkubeServer) GetCertificateDirectory() string {
	return path.Join(lk.LocalkubeDirectory, "certs")
}
func (lk LocalkubeServer) GetPrivateKeyCertPath() string {
	return path.Join(lk.GetCertificateDirectory(), "apiserver.key")
}
func (lk LocalkubeServer) GetPublicKeyCertPath() string {
	return path.Join(lk.GetCertificateDirectory(), "apiserver.crt")
}
func (lk LocalkubeServer) GetCAPrivateKeyCertPath() string {
	return path.Join(lk.GetCertificateDirectory(), "ca.key")
}
func (lk LocalkubeServer) GetCAPublicKeyCertPath() string {
	return path.Join(lk.GetCertificateDirectory(), "ca.crt")
}

func (lk LocalkubeServer) GetAPIServerSecureURL() string {
	return fmt.Sprintf("https://%s:%d", lk.APIServerAddress.String(), lk.APIServerPort)
}

func (lk LocalkubeServer) GetAPIServerInsecureURL() string {
	return fmt.Sprintf("http://%s:%d", lk.APIServerInsecureAddress.String(), lk.APIServerInsecurePort)
}

// Get the host's public IP address
func (lk LocalkubeServer) GetHostIP() (net.IP, error) {
	return utilnet.ChooseBindAddress(net.ParseIP("0.0.0.0"))
}

func (lk LocalkubeServer) getExtraConfigForComponent(component string) []ExtraOption {
	e := []ExtraOption{}
	for _, c := range lk.ExtraConfig {
		if c.Component == component {
			e = append(e, c)
		}
	}
	return e
}

func (lk LocalkubeServer) SetExtraConfigForComponent(component string, config interface{}) {
	extra := lk.getExtraConfigForComponent(component)
	for _, e := range extra {
		glog.Infof("Setting %s to %s on %s.\n", e.Key, e.Value, component)
		if err := FindAndSet(e.Key, config, e.Value); err != nil {
			glog.Warningf("Unable to set %s to %s. Error: %s", e.Key, e.Value, err)
		}
	}
}

func (lk LocalkubeServer) loadCert(path string) (*x509.Certificate, error) {
	contents, err := ioutil.ReadFile(path)
	if err != nil {
		return nil, err
	}
	decoded, _ := pem.Decode(contents)
	if decoded == nil {
		return nil, fmt.Errorf("Unable to decode certificate.")
	}

	return x509.ParseCertificate(decoded.Bytes)
}

func (lk LocalkubeServer) shouldGenerateCerts(ips []net.IP) bool {
	if !(CanReadFile(lk.GetPublicKeyCertPath()) &&
		CanReadFile(lk.GetPrivateKeyCertPath())) {
		fmt.Println("Regenerating certs because the files aren't readable")
		return true
	}

	cert, err := lk.loadCert(lk.GetPublicKeyCertPath())
	if err != nil {
		fmt.Println("Regenerating certs because there was an error loading the certificate: ", err)
		return true
	}

	certIPs := map[string]bool{}
	for _, certIP := range cert.IPAddresses {
		certIPs[certIP.String()] = true
	}

	for _, ip := range ips {
		if _, ok := certIPs[ip.String()]; !ok {
			fmt.Println("Regenerating certs becase an IP is missing: ", ip)
			return true
		}
	}
	return false
}

func (lk LocalkubeServer) shouldGenerateCACerts() bool {
	if !(CanReadFile(lk.GetCAPublicKeyCertPath()) &&
		CanReadFile(lk.GetCAPrivateKeyCertPath())) {
		fmt.Println("Regenerating CA certs because the files aren't readable")
		return true
	}

	_, err := lk.loadCert(lk.GetCAPublicKeyCertPath())
	if err != nil {
		fmt.Println("Regenerating CA certs because there was an error loading the certificate: ", err)
		return true
	}

	return false
}

func (lk LocalkubeServer) getAllIPs() ([]net.IP, error) {
	ips := []net.IP{lk.ServiceClusterIPRange.IP}
	addrs, err := net.InterfaceAddrs()
	if err != nil {
		return nil, err
	}
	for _, addr := range addrs {
		ipnet, ok := addr.(*net.IPNet)
		if !ok {
			fmt.Println("Skipping: ", addr)
			continue
		}
		ips = append(ips, ipnet.IP)
	}
	return ips, nil
}

func (lk LocalkubeServer) GenerateCerts() error {
	if !lk.shouldGenerateCACerts() {
		fmt.Println("Using these existing CA certs: ", lk.GetCAPublicKeyCertPath(), lk.GetCAPrivateKeyCertPath())
	} else {
		fmt.Println("Creating CA cert")
		if err := GenerateCACert(lk.GetCAPublicKeyCertPath(), lk.GetCAPrivateKeyCertPath()); err != nil {
			fmt.Println("Failed to create CA certs: ", err)
			return err
		}
	}

	ips, err := lk.getAllIPs()
	if err != nil {
		return err
	}

	if !lk.shouldGenerateCerts(ips) {
		fmt.Println("Using these existing certs: ", lk.GetPublicKeyCertPath(), lk.GetPrivateKeyCertPath())
		return nil
	}
	fmt.Println("Creating cert with IPs: ", ips)

	if err := GenerateSignedCert(lk.GetPublicKeyCertPath(), lk.GetPrivateKeyCertPath(), ips, GetAlternateDNS(lk.DNSDomain), lk.GetCAPublicKeyCertPath(), lk.GetCAPrivateKeyCertPath()); err != nil {
		fmt.Println("Failed to create certs: ", err)
		return err
	}

	return nil
}
