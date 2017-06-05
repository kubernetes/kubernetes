/*
Copyright 2015 The Kubernetes Authors.

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

package initialresources

import (
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"io/ioutil"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	"github.com/golang/glog"
	"github.com/hawkular/hawkular-client-go/metrics"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/api"

	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
)

type hawkularSource struct {
	client       *metrics.Client
	uri          *url.URL
	useNamespace bool
	modifiers    []metrics.Modifier
}

const (
	containerImageTag string = "container_base_image"
	descriptorTag     string = "descriptor_name"
	separator         string = "/"

	defaultServiceAccountFile = "/var/run/secrets/kubernetes.io/serviceaccount/token"
)

// heapsterName gets the equivalent MetricDescriptor.Name used in the Heapster
func heapsterName(kind api.ResourceName) string {
	switch kind {
	case api.ResourceCPU:
		return "cpu/usage"
	case api.ResourceMemory:
		return "memory/usage"
	default:
		return ""
	}
}

// tagQuery creates tagFilter query for Hawkular
func tagQuery(kind api.ResourceName, image string, exactMatch bool) map[string]string {
	q := make(map[string]string)

	// Add here the descriptor_tag..
	q[descriptorTag] = heapsterName(kind)

	if exactMatch {
		q[containerImageTag] = image
	} else {
		split := strings.Index(image, "@")
		if split < 0 {
			split = strings.Index(image, ":")
		}
		q[containerImageTag] = fmt.Sprintf("%s:*", image[:split])
	}

	return q
}

// dataSource API

func (hs *hawkularSource) GetUsagePercentile(kind api.ResourceName, perc int64, image, namespace string, exactMatch bool, start, end time.Time) (int64, int64, error) {
	q := tagQuery(kind, image, exactMatch)

	m := make([]metrics.Modifier, len(hs.modifiers), 2+len(hs.modifiers))
	copy(m, hs.modifiers)

	if namespace != metav1.NamespaceAll {
		m = append(m, metrics.Tenant(namespace))
	}

	p := float64(perc)
	m = append(m, metrics.Filters(metrics.TagsFilter(q), metrics.BucketsFilter(1), metrics.StartTimeFilter(start), metrics.EndTimeFilter(end), metrics.PercentilesFilter([]float64{p})))

	bp, err := hs.client.ReadBuckets(metrics.Counter, m...)
	if err != nil {
		return 0, 0, err
	}

	if len(bp) > 0 && len(bp[0].Percentiles) > 0 {
		return int64(bp[0].Percentiles[0].Value), int64(bp[0].Samples), nil
	}
	return 0, 0, nil
}

// newHawkularSource creates a new Hawkular Source. The uri follows the scheme from Heapster
func newHawkularSource(uri string) (dataSource, error) {
	u, err := url.Parse(uri)
	if err != nil {
		return nil, err
	}

	d := &hawkularSource{
		uri: u,
	}
	if err = d.init(); err != nil {
		return nil, err
	}
	return d, nil
}

// init initializes the Hawkular dataSource. Almost equal to the Heapster initialization
func (hs *hawkularSource) init() error {
	hs.modifiers = make([]metrics.Modifier, 0)
	p := metrics.Parameters{
		Tenant: "heapster", // This data is stored by the heapster - for no-namespace hits
		Url:    hs.uri.String(),
	}

	opts := hs.uri.Query()

	if v, found := opts["tenant"]; found {
		p.Tenant = v[0]
	}

	if v, found := opts["useServiceAccount"]; found {
		if b, _ := strconv.ParseBool(v[0]); b {
			accountFile := defaultServiceAccountFile
			if file, f := opts["serviceAccountFile"]; f {
				accountFile = file[0]
			}

			// If a readable service account token exists, then use it
			if contents, err := ioutil.ReadFile(accountFile); err == nil {
				p.Token = string(contents)
			} else {
				glog.Errorf("Could not read contents of %s, no token authentication is used\n", defaultServiceAccountFile)
			}
		}
	}

	// Authentication / Authorization parameters
	tC := &tls.Config{}

	if v, found := opts["auth"]; found {
		if _, f := opts["caCert"]; f {
			return fmt.Errorf("both auth and caCert files provided, combination is not supported")
		}
		if len(v[0]) > 0 {
			// Authfile
			kubeConfig, err := clientcmd.NewNonInteractiveDeferredLoadingClientConfig(&clientcmd.ClientConfigLoadingRules{
				ExplicitPath: v[0]},
				&clientcmd.ConfigOverrides{}).ClientConfig()
			if err != nil {
				return err
			}
			tC, err = restclient.TLSConfigFor(kubeConfig)
			if err != nil {
				return err
			}
		}
	}

	if u, found := opts["user"]; found {
		if _, wrong := opts["useServiceAccount"]; wrong {
			return fmt.Errorf("if user and password are used, serviceAccount cannot be used")
		}
		if p, f := opts["pass"]; f {
			hs.modifiers = append(hs.modifiers, func(req *http.Request) error {
				req.SetBasicAuth(u[0], p[0])
				return nil
			})
		}
	}

	if v, found := opts["caCert"]; found {
		caCert, err := ioutil.ReadFile(v[0])
		if err != nil {
			return err
		}

		caCertPool := x509.NewCertPool()
		caCertPool.AppendCertsFromPEM(caCert)

		tC.RootCAs = caCertPool
	}

	if v, found := opts["insecure"]; found {
		insecure, err := strconv.ParseBool(v[0])
		if err != nil {
			return err
		}
		tC.InsecureSkipVerify = insecure
	}

	p.TLSConfig = tC

	c, err := metrics.NewHawkularClient(p)
	if err != nil {
		return err
	}

	hs.client = c

	glog.Infof("Initialised Hawkular Source with parameters %v", p)
	return nil
}
