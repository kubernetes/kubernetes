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
	"regexp"
	"strconv"
	"strings"
	"sync"

	"github.com/golang/glog"
	"github.com/hawkular/hawkular-client-go/metrics"
	"k8s.io/heapster/extpoints"

	sink_api "k8s.io/heapster/sinks/api"
	kube_api "k8s.io/kubernetes/pkg/api"
	kube_client "k8s.io/kubernetes/pkg/client/unversioned"
	kubeClientCmd "k8s.io/kubernetes/pkg/client/unversioned/clientcmd"
)

const (
	unitsTag       = "units"
	descriptionTag = "_description"
	descriptorTag  = "descriptor_name"
	groupTag       = "group_id"
	separator      = "/"

	defaultServiceAccountFile = "/var/run/secrets/kubernetes.io/serviceaccount/token"
)

// Filter which Timeseries are stored. Store if returns true, filter if returns false
type Filter func(*sink_api.Timeseries) bool
type FilterType int

const (
	// Filter by label's value
	Label FilterType = iota
	// Filter by metric name
	Name
	// Unknown filter type
	Unknown
)

func (f FilterType) From(s string) FilterType {
	switch s {
	case "label":
		return Label
	case "name":
		return Name
	default:
		return Unknown
	}
}

type hawkularSink struct {
	client  *metrics.Client
	models  map[string]*metrics.MetricDefinition // Model definitions
	regLock sync.Mutex
	reg     map[string]*metrics.MetricDefinition // Real definitions

	uri *url.URL

	labelTenant string
	modifiers   []metrics.Modifier
	filters     []Filter
}

// START: ExternalSink interface implementations

func (self *hawkularSink) Register(mds []sink_api.MetricDescriptor) error {
	// Create model definitions based on the MetricDescriptors
	for _, md := range mds {
		hmd := self.descriptorToDefinition(&md)
		self.models[md.Name] = &hmd
	}

	// Fetch currently known metrics from Hawkular-Metrics and cache them
	types := []metrics.MetricType{metrics.Gauge, metrics.Counter}
	for _, t := range types {
		err := self.updateDefinitions(t)
		if err != nil {
			return err
		}
	}

	return nil
}

// Fetches definitions from the server and checks that they're matching the descriptors
func (self *hawkularSink) updateDefinitions(mt metrics.MetricType) error {
	m := make([]metrics.Modifier, len(self.modifiers), len(self.modifiers)+1)
	copy(m, self.modifiers)
	m = append(m, metrics.Filters(metrics.TypeFilter(mt)))

	mds, err := self.client.Definitions(m...)
	if err != nil {
		return err
	}

	self.regLock.Lock()
	defer self.regLock.Unlock()

	for _, p := range mds {
		// If no descriptorTag is found, this metric does not belong to Heapster
		if mk, found := p.Tags[descriptorTag]; found {
			if model, f := self.models[mk]; f {
				if !self.recent(p, model) {
					if err := self.client.UpdateTags(mt, p.Id, p.Tags, self.modifiers...); err != nil {
						return err
					}
				}
			}
			self.reg[p.Id] = p
		}
	}
	return nil
}

func (self *hawkularSink) Unregister(mds []sink_api.MetricDescriptor) error {
	self.regLock.Lock()
	defer self.regLock.Unlock()
	return self.init()
}

// Checks that stored definition is up to date with the model
func (self *hawkularSink) recent(live *metrics.MetricDefinition, model *metrics.MetricDefinition) bool {
	recent := true
	for k := range model.Tags {
		if v, found := live.Tags[k]; !found {
			// There's a label that wasn't in our stored definition
			live.Tags[k] = v
			recent = false
		}
	}

	return recent
}

// Transform the MetricDescriptor to a format used by Hawkular-Metrics
func (self *hawkularSink) descriptorToDefinition(md *sink_api.MetricDescriptor) metrics.MetricDefinition {
	tags := make(map[string]string)
	// Postfix description tags with _description
	for _, l := range md.Labels {
		if len(l.Description) > 0 {
			tags[l.Key+descriptionTag] = l.Description
		}
	}

	if len(md.Units.String()) > 0 {
		tags[unitsTag] = md.Units.String()
	}

	tags[descriptorTag] = md.Name

	hmd := metrics.MetricDefinition{
		Id:   md.Name,
		Tags: tags,
		Type: heapsterTypeToHawkularType(md.Type),
	}

	return hmd
}

func (self *hawkularSink) groupName(p *sink_api.Point) string {
	n := []string{p.Labels[sink_api.LabelContainerName.Key], p.Name}
	return strings.Join(n, separator)
}

func (self *hawkularSink) idName(p *sink_api.Point) string {
	n := make([]string, 0, 3)
	n = append(n, p.Labels[sink_api.LabelContainerName.Key])
	if p.Labels[sink_api.LabelPodId.Key] != "" {
		n = append(n, p.Labels[sink_api.LabelPodId.Key])
	} else {
		n = append(n, p.Labels[sink_api.LabelHostID.Key])
	}
	n = append(n, p.Name)

	return strings.Join(n, separator)
}

// Check that metrics tags are defined on the Hawkular server and if not,
// register the metric definition.
func (self *hawkularSink) registerIfNecessary(t *sink_api.Timeseries, m ...metrics.Modifier) error {
	key := self.idName(t.Point)

	self.regLock.Lock()
	defer self.regLock.Unlock()

	// If found, check it matches the current stored definition (could be old info from
	// the stored metrics cache for example)
	if _, found := self.reg[key]; !found {
		// Register the metric descriptor here..
		if md, f := self.models[t.MetricDescriptor.Name]; f {
			// Copy the original map
			mdd := *md
			tags := make(map[string]string)
			for k, v := range mdd.Tags {
				tags[k] = v
			}
			mdd.Tags = tags

			// Set tag values
			for k, v := range t.Point.Labels {
				mdd.Tags[k] = v
			}

			mdd.Tags[groupTag] = self.groupName(t.Point)
			mdd.Tags[descriptorTag] = t.MetricDescriptor.Name

			m = append(m, self.modifiers...)

			// Create metric, use updateTags instead of Create because we know it is unique
			if err := self.client.UpdateTags(mdd.Type, key, mdd.Tags, m...); err != nil {
				// Log error and don't add this key to the lookup table
				glog.Errorf("Could not update tags: %s", err)
				return err
			}

			// Add to the lookup table
			self.reg[key] = &mdd
		} else {
			return fmt.Errorf("Could not find definition model with name %s", t.MetricDescriptor.Name)
		}
	}
	// TODO Compare the definition tags and update if necessary? Quite expensive operation..

	return nil
}

func (self *hawkularSink) StoreTimeseries(ts []sink_api.Timeseries) error {
	if len(ts) > 0 {
		tmhs := make(map[string][]metrics.MetricHeader)

		if &self.labelTenant == nil {
			tmhs[self.client.Tenant] = make([]metrics.MetricHeader, 0, len(ts))
		}

		wg := &sync.WaitGroup{}

	Store:
		for _, t := range ts {
			t := t

			for _, filter := range self.filters {
				if !filter(&t) {
					continue Store
				}
			}

			tenant := self.client.Tenant

			if &self.labelTenant != nil {
				if v, found := t.Point.Labels[self.labelTenant]; found {
					tenant = v
				}
			}

			// Registering should not block the processing
			wg.Add(1)
			go func(t *sink_api.Timeseries, tenant string) {
				defer wg.Done()
				self.registerIfNecessary(t, metrics.Tenant(tenant))
			}(&t, tenant)

			if t.MetricDescriptor.ValueType == sink_api.ValueBool {
				// TODO: Model to availability type once we see some real world examples
				break
			}

			mH, err := self.pointToMetricHeader(&t)
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

		for k, v := range tmhs {
			wg.Add(1)
			go func(v []metrics.MetricHeader, k string) {
				defer wg.Done()
				m := make([]metrics.Modifier, len(self.modifiers), len(self.modifiers)+1)
				copy(m, self.modifiers)
				m = append(m, metrics.Tenant(k))
				if err := self.client.Write(v, m...); err != nil {
					glog.Errorf(err.Error())
				}
			}(v, k)
		}
		wg.Wait()
	}
	return nil
}

// Converts Timeseries to metric structure used by the Hawkular
func (self *hawkularSink) pointToMetricHeader(t *sink_api.Timeseries) (*metrics.MetricHeader, error) {

	p := t.Point
	name := self.idName(p)

	value, err := metrics.ConvertToFloat64(p.Value)
	if err != nil {
		return nil, err
	}

	m := metrics.Datapoint{
		Value:     value,
		Timestamp: metrics.UnixMilli(p.End),
	}

	mh := &metrics.MetricHeader{
		Id:   name,
		Data: []metrics.Datapoint{m},
		Type: heapsterTypeToHawkularType(t.MetricDescriptor.Type),
	}

	return mh, nil
}

func heapsterTypeToHawkularType(t sink_api.MetricType) metrics.MetricType {
	switch t {
	case sink_api.MetricCumulative:
		return metrics.Counter
	case sink_api.MetricGauge:
		return metrics.Gauge
	default:
		return metrics.Gauge
	}
}

func (self *hawkularSink) DebugInfo() string {
	info := fmt.Sprintf("%s\n", self.Name())

	self.regLock.Lock()
	defer self.regLock.Unlock()
	info += fmt.Sprintf("Known metrics: %d\n", len(self.reg))
	if &self.labelTenant != nil {
		info += fmt.Sprintf("Using label '%s' as tenant information\n", self.labelTenant)
	}

	// TODO Add here statistics from the Hawkular-Metrics client instance
	return info
}

func (self *hawkularSink) StoreEvents(events []kube_api.Event) error {
	return nil
}

func (self *hawkularSink) Name() string {
	return "Hawkular-Metrics Sink"
}

// END: ExternalSink

func init() {
	extpoints.SinkFactories.Register(NewHawkularSink, "hawkular")
}

func (self *hawkularSink) init() error {
	self.reg = make(map[string]*metrics.MetricDefinition)
	self.models = make(map[string]*metrics.MetricDefinition)
	self.modifiers = make([]metrics.Modifier, 0)
	self.filters = make([]Filter, 0)

	p := metrics.Parameters{
		Tenant: "heapster",
		Url:    self.uri.String(),
	}

	opts := self.uri.Query()

	if v, found := opts["tenant"]; found {
		p.Tenant = v[0]
	}

	if v, found := opts["labelToTenant"]; found {
		self.labelTenant = v[0]
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
			self.modifiers = append(self.modifiers, func(req *http.Request) error {
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
		self.filters = filters
	}

	c, err := metrics.NewHawkularClient(p)
	if err != nil {
		return err
	}

	self.client = c

	glog.Infof("Initialised Hawkular Sink with parameters %v", p)
	return nil
}

// If Heapster gets filters, remove these..
func parseFilters(v []string) ([]Filter, error) {
	fs := make([]Filter, 0, len(v))
	for _, s := range v {
		p := strings.Index(s, "(")
		if p < 0 {
			return nil, fmt.Errorf("Incorrect syntax in filter parameters, missing (")
		}

		if strings.Index(s, ")") != len(s)-1 {
			return nil, fmt.Errorf("Incorrect syntax in filter parameters, missing )")
		}

		t := Unknown.From(s[:p])
		if t == Unknown {
			return nil, fmt.Errorf("Unknown filter type")
		}

		command := s[p+1 : len(s)-1]

		switch t {
		case Label:
			proto := strings.SplitN(command, ":", 2)
			if len(proto) < 2 {
				return nil, fmt.Errorf("Missing : from label filter")
			}
			r, err := regexp.Compile(proto[1])
			if err != nil {
				return nil, err
			}
			fs = append(fs, labelFilter(proto[0], r))
			break
		case Name:
			r, err := regexp.Compile(command)
			if err != nil {
				return nil, err
			}
			fs = append(fs, nameFilter(r))
			break
		}
	}
	return fs, nil
}

func labelFilter(label string, r *regexp.Regexp) Filter {
	return func(t *sink_api.Timeseries) bool {
		for k, v := range t.Point.Labels {
			if k == label {
				if r.MatchString(v) {
					return false
				}
			}
		}
		return true
	}
}

func nameFilter(r *regexp.Regexp) Filter {
	return func(t *sink_api.Timeseries) bool {
		return !r.MatchString(t.Point.Name)
	}
}

func NewHawkularSink(u *url.URL, _ extpoints.HeapsterConf) ([]sink_api.ExternalSink, error) {
	sink := &hawkularSink{
		uri: u,
	}
	if err := sink.init(); err != nil {
		return nil, err
	}
	return []sink_api.ExternalSink{sink}, nil
}
