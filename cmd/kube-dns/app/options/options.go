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

// Package options contains flags for initializing a proxy.
package options

import (
	"net/url"
	"os"

	"fmt"
	_ "net/http/pprof"
	"strings"

	"github.com/spf13/pflag"
	"k8s.io/kubernetes/pkg/util/validation"
)

type KubeDNSConfig struct {
	ClusterDomain  string
	KubeConfigFile string
	KubeMasterURL  string
	HealthzPort    int
	DNSPort        int
	// Federations maps federation names to their registered domain names.
	Federations map[string]string
}

func NewKubeDNSConfig() *KubeDNSConfig {
	return &KubeDNSConfig{
		ClusterDomain:  "cluster.local.",
		KubeConfigFile: "",
		KubeMasterURL:  "",
		HealthzPort:    8081,
		DNSPort:        53,
		Federations:    make(map[string]string),
	}
}

type clusterDomainVar struct {
	val *string
}

func (m clusterDomainVar) Set(v string) error {
	v = strings.TrimSuffix(v, ".")
	segments := strings.Split(v, ".")
	for _, segment := range segments {
		if errs := validation.IsDNS1123Label(segment); len(errs) > 0 {
			return fmt.Errorf("Not a valid DNS label. %v", errs)
		}
	}
	if !strings.HasSuffix(v, ".") {
		v = fmt.Sprintf("%s.", v)
	}
	*m.val = v
	return nil
}

func (m clusterDomainVar) String() string {
	return *m.val
}

func (m clusterDomainVar) Type() string {
	return "string"
}

type kubeMasterURLVar struct {
	val *string
}

func (m kubeMasterURLVar) Set(v string) error {
	parsedURL, err := url.Parse(os.ExpandEnv(v))
	if err != nil {
		return fmt.Errorf("failed to parse kube-master-url")
	}
	if parsedURL.Scheme == "" || parsedURL.Host == "" || parsedURL.Host == ":" {
		return fmt.Errorf("invalid kube-master-url specified")
	}
	*m.val = v
	return nil
}

func (m kubeMasterURLVar) String() string {
	return *m.val
}

func (m kubeMasterURLVar) Type() string {
	return "string"
}

type federationsVar struct {
	nameDomainMap map[string]string
}

// Set deserializes the input string in the format
// "myfederation1=example.com,myfederation2=second.example.com,myfederation3=example.com"
// into a map of key-value pairs of federation names to domain names.
func (fv federationsVar) Set(keyVal string) error {
	for _, val := range strings.Split(keyVal, ",") {
		splits := strings.SplitN(strings.TrimSpace(val), "=", 2)
		name := strings.TrimSpace(splits[0])
		domain := strings.TrimSpace(splits[1])
		if errs := validation.IsDNS1123Label(name); len(errs) != 0 {
			return fmt.Errorf("%q not a valid federation name: %q", name, errs)
		}
		// The federation domain name need not strictly be domain names, we
		// accept valid dns names with subdomain components.
		if errs := validation.IsDNS1123Subdomain(domain); len(errs) != 0 {
			return fmt.Errorf("%q not a valid domain name: %q", domain, errs)
		}
		fv.nameDomainMap[name] = domain
	}
	return nil
}

func (fv federationsVar) String() string {
	var splits []string
	for name, domain := range fv.nameDomainMap {
		splits = append(splits, fmt.Sprintf("%s=%s", name, domain))
	}
	return strings.Join(splits, ",")
}

func (fv federationsVar) Type() string {
	return "[]string"
}

func (s *KubeDNSConfig) AddFlags(fs *pflag.FlagSet) {
	fs.Var(clusterDomainVar{&s.ClusterDomain}, "domain", "domain under which to create names")
	fs.StringVar(&s.KubeConfigFile, "kubecfg-file", s.KubeConfigFile, "Location of kubecfg file for access to kubernetes master service; --kube-master-url overrides the URL part of this; if neither this nor --kube-master-url are provided, defaults to service account tokens")
	fs.Var(kubeMasterURLVar{&s.KubeMasterURL}, "kube-master-url", "URL to reach kubernetes master. Env variables in this flag will be expanded.")
	fs.IntVar(&s.HealthzPort, "healthz-port", s.HealthzPort, "port on which to serve a kube-dns HTTP readiness probe.")
	fs.IntVar(&s.DNSPort, "dns-port", s.DNSPort, "port on which to serve DNS requests.")
	fs.Var(federationsVar{s.Federations}, "federations", "a comma separated list of the federation names and their corresponding domain names to which this cluster belongs. Example: \"myfederation1=example.com,myfederation2=example2.com,myfederation3=example.com\"")
}
