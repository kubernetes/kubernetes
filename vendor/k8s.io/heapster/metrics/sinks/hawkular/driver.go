// Copyright 2015 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package hawkular

import (
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"io/ioutil"
	"net/http"
	"net/url"
	"strconv"
	"sync"

	"github.com/golang/glog"
	"github.com/hawkular/hawkular-client-go/metrics"

	"k8s.io/heapster/metrics/core"
	kube_client "k8s.io/kubernetes/pkg/client/restclient"
	kubeClientCmd "k8s.io/kubernetes/pkg/client/unversioned/clientcmd"
)

const (
	unitsTag           = "units"
	descriptionTag     = "_description"
	descriptorTag      = "descriptor_name"
	groupTag           = "group_id"
	separator          = "/"
	batchSizeDefault   = 1000
	concurrencyDefault = 5

	nodeId string = "labelNodeId"

	defaultServiceAccountFile = "/var/run/secrets/kubernetes.io/serviceaccount/token"
)

// START: ExternalSink interface implementations

func (h *hawkularSink) Register(mds []core.MetricDescriptor) error {
	// Create model definitions based on the MetricDescriptors
	for _, md := range mds {
		hmd := h.descriptorToDefinition(&md)
		h.models[md.Name] = &hmd
	}

	// Fetch currently known metrics from Hawkular-Metrics and cache them
	types := []metrics.MetricType{metrics.Gauge, metrics.Counter}
	for _, t := range types {
		err := h.updateDefinitions(t)
		if err != nil {
			return err
		}
	}

	return nil
}

func (h *hawkularSink) Stop() {
	h.regLock.Lock()
	defer h.regLock.Unlock()
	h.init()
}

func (h *hawkularSink) ExportData(db *core.DataBatch) {
	totalCount := 0
	for _, ms := range db.MetricSets {
		totalCount += len(ms.MetricValues)
		totalCount += len(ms.LabeledMetrics)
	}

	if len(db.MetricSets) > 0 {
		tmhs := make(map[string][]metrics.MetricHeader)

		if len(h.labelTenant) == 0 {
			tmhs[h.client.Tenant] = make([]metrics.MetricHeader, 0, totalCount)
		}

		wg := &sync.WaitGroup{}

		for _, ms := range db.MetricSets {

			// // Transform ms.MetricValues to LabeledMetrics first
			lms := metricValueToLabeledMetric(ms.MetricValues)
			ms.LabeledMetrics = append(ms.LabeledMetrics, lms...)

		Store:
			for _, labeledMetric := range ms.LabeledMetrics {

				for _, filter := range h.filters {
					if !filter(ms, labeledMetric.Name) {
						continue Store
					}
				}

				tenant := h.client.Tenant

				if len(h.labelTenant) > 0 {
					if v, found := ms.Labels[h.labelTenant]; found {
						tenant = v
					}
				}

				wg.Add(1)
				go func(ms *core.MetricSet, labeledMetric core.LabeledMetric, tenant string) {
					defer wg.Done()
					h.registerLabeledIfNecessary(ms, labeledMetric, metrics.Tenant(tenant))
				}(ms, labeledMetric, tenant)

				mH, err := h.pointToLabeledMetricHeader(ms, labeledMetric, db.Timestamp)
				if err != nil {
					// One transformation error should not prevent the whole process
					glog.Errorf(err.Error())
					continue
				}

				if _, found := tmhs[tenant]; !found {
					tmhs[tenant] = make([]metrics.MetricHeader, 0)
				}

				tmhs[tenant] = append(tmhs[tenant], *mH)
			}
		}
		h.sendData(tmhs, wg) // Send to a limited channel? Only batches.. egg.
		wg.Wait()
	}
}

func metricValueToLabeledMetric(msValues map[string]core.MetricValue) []core.LabeledMetric {
	lms := make([]core.LabeledMetric, 0, len(msValues))
	for metricName, metricValue := range msValues {
		lm := core.LabeledMetric{
			Name:        metricName,
			MetricValue: metricValue,
			Labels:      make(map[string]string, 0),
		}
		lms = append(lms, lm)
	}
	return lms
}

func (h *hawkularSink) DebugInfo() string {
	info := fmt.Sprintf("%s\n", h.Name())

	h.regLock.Lock()
	defer h.regLock.Unlock()
	info += fmt.Sprintf("Known metrics: %d\n", len(h.reg))
	if len(h.labelTenant) > 0 {
		info += fmt.Sprintf("Using label '%s' as tenant information\n", h.labelTenant)
	}
	if len(h.labelNodeId) > 0 {
		info += fmt.Sprintf("Using label '%s' as node identified in resourceid\n", h.labelNodeId)
	}

	// TODO Add here statistics from the Hawkular-Metrics client instance
	return info
}

func (h *hawkularSink) Name() string {
	return "Hawkular-Metrics Sink"
}

// NewHawkularSink Creates and returns a new hawkularSink instance
func NewHawkularSink(u *url.URL) (core.DataSink, error) {
	sink := &hawkularSink{
		uri:       u,
		batchSize: 1000,
	}
	if err := sink.init(); err != nil {
		return nil, err
	}

	metrics := make([]core.MetricDescriptor, 0, len(core.AllMetrics))
	for _, metric := range core.AllMetrics {
		metrics = append(metrics, metric.MetricDescriptor)
	}
	sink.Register(metrics)
	return sink, nil
}

func (h *hawkularSink) init() error {
	h.reg = make(map[string]*metrics.MetricDefinition)
	h.models = make(map[string]*metrics.MetricDefinition)
	h.modifiers = make([]metrics.Modifier, 0)
	h.filters = make([]Filter, 0)
	h.batchSize = batchSizeDefault

	p := metrics.Parameters{
		Tenant:      "heapster",
		Url:         h.uri.String(),
		Concurrency: concurrencyDefault,
	}

	opts := h.uri.Query()

	if v, found := opts["tenant"]; found {
		p.Tenant = v[0]
	}

	if v, found := opts["labelToTenant"]; found {
		h.labelTenant = v[0]
	}

	if v, found := opts[nodeId]; found {
		h.labelNodeId = v[0]
	}

	if v, found := opts["useServiceAccount"]; found {
		if b, _ := strconv.ParseBool(v[0]); b {
			// If a readable service account token exists, then use it
			if contents, err := ioutil.ReadFile(defaultServiceAccountFile); err == nil {
				p.Token = string(contents)
			}
		}
	}

	// Authentication / Authorization parameters
	tC := &tls.Config{}

	if v, found := opts["auth"]; found {
		if _, f := opts["caCert"]; f {
			return fmt.Errorf("Both auth and caCert files provided, combination is not supported")
		}
		if len(v[0]) > 0 {
			// Authfile
			kubeConfig, err := kubeClientCmd.NewNonInteractiveDeferredLoadingClientConfig(&kubeClientCmd.ClientConfigLoadingRules{
				ExplicitPath: v[0]},
				&kubeClientCmd.ConfigOverrides{}).ClientConfig()
			if err != nil {
				return err
			}
			tC, err = kube_client.TLSConfigFor(kubeConfig)
			if err != nil {
				return err
			}
		}
	}

	if u, found := opts["user"]; found {
		if _, wrong := opts["useServiceAccount"]; wrong {
			return fmt.Errorf("If user and password are used, serviceAccount cannot be used")
		}
		if p, f := opts["pass"]; f {
			h.modifiers = append(h.modifiers, func(req *http.Request) error {
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
		_, f := opts["caCert"]
		_, f2 := opts["auth"]
		if f || f2 {
			return fmt.Errorf("Insecure can't be defined with auth or caCert")
		}
		insecure, err := strconv.ParseBool(v[0])
		if err != nil {
			return err
		}
		tC.InsecureSkipVerify = insecure
	}

	p.TLSConfig = tC

	// Filters
	if v, found := opts["filter"]; found {
		filters, err := parseFilters(v)
		if err != nil {
			return err
		}
		h.filters = filters
	}

	// Concurrency limitations
	if v, found := opts["concurrencyLimit"]; found {
		cs, err := strconv.Atoi(v[0])
		if err != nil || cs < 0 {
			return fmt.Errorf("Supplied concurrency value of %s is invalid", v[0])
		}
		p.Concurrency = cs
	}

	if v, found := opts["batchSize"]; found {
		bs, err := strconv.Atoi(v[0])
		if err != nil || bs < 0 {
			return fmt.Errorf("Supplied batchSize value of %s is invalid", v[0])
		}
		h.batchSize = bs
	}

	c, err := metrics.NewHawkularClient(p)
	if err != nil {
		return err
	}

	h.client = c

	glog.Infof("Initialised Hawkular Sink with parameters %v", p)
	return nil
}
