// Copyright 2016 Google Inc. All Rights Reserved.
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
	"fmt"
	"math"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/golang/glog"
	"github.com/hawkular/hawkular-client-go/metrics"
	"k8s.io/heapster/metrics/core"
)

// Fetches definitions from the server and checks that they're matching the descriptors
func (h *hawkularSink) updateDefinitions(mt metrics.MetricType) error {
	m := make([]metrics.Modifier, len(h.modifiers), len(h.modifiers)+1)
	copy(m, h.modifiers)
	m = append(m, metrics.Filters(metrics.TypeFilter(mt)))

	mds, err := h.client.Definitions(m...)
	if err != nil {
		return err
	}

	h.regLock.Lock()
	defer h.regLock.Unlock()

	for _, p := range mds {
		// If no descriptorTag is found, this metric does not belong to Heapster
		if mk, found := p.Tags[descriptorTag]; found {
			if model, f := h.models[mk]; f && !h.recent(p, model) {
				if err := h.client.UpdateTags(mt, p.Id, p.Tags, h.modifiers...); err != nil {
					return err
				}
			}
			h.reg[p.Id] = p
		}
	}
	return nil
}

// Checks that stored definition is up to date with the model
func (h *hawkularSink) recent(live *metrics.MetricDefinition, model *metrics.MetricDefinition) bool {
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
func (h *hawkularSink) descriptorToDefinition(md *core.MetricDescriptor) metrics.MetricDefinition {
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

func (h *hawkularSink) groupName(ms *core.MetricSet, metricName string) string {
	n := []string{ms.Labels[core.LabelContainerName.Key], metricName}
	return strings.Join(n, separator)
}

func (h *hawkularSink) idName(ms *core.MetricSet, metricName string) string {
	n := make([]string, 0, 3)

	metricType := ms.Labels[core.LabelMetricSetType.Key]
	switch metricType {
	case core.MetricSetTypeNode:
		n = append(n, "machine")
		n = append(n, h.nodeName(ms))
	case core.MetricSetTypeSystemContainer:
		n = append(n, core.MetricSetTypeSystemContainer)
		n = append(n, ms.Labels[core.LabelContainerName.Key])
		n = append(n, ms.Labels[core.LabelPodId.Key])
	case core.MetricSetTypeCluster:
		n = append(n, core.MetricSetTypeCluster)
	case core.MetricSetTypeNamespace:
		n = append(n, core.MetricSetTypeNamespace)
		n = append(n, ms.Labels[core.LabelNamespaceName.Key])
	case core.MetricSetTypePod:
		n = append(n, core.MetricSetTypePod)
		n = append(n, ms.Labels[core.LabelPodId.Key])
	case core.MetricSetTypePodContainer:
		n = append(n, ms.Labels[core.LabelContainerName.Key])
		n = append(n, ms.Labels[core.LabelPodId.Key])
	default:
		n = append(n, ms.Labels[core.LabelContainerName.Key])
		if ms.Labels[core.LabelPodId.Key] != "" {
			n = append(n, ms.Labels[core.LabelPodId.Key])
		} else {
			n = append(n, h.nodeName(ms))
		}
	}

	n = append(n, metricName)

	return strings.Join(n, separator)
}

func (h *hawkularSink) nodeName(ms *core.MetricSet) string {
	if len(h.labelNodeId) > 0 {
		if v, found := ms.Labels[h.labelNodeId]; found {
			return v
		}
		glog.V(4).Infof("The labelNodeId was set to %s but there is no label with this value."+
			"Using the default 'nodename' label instead.", h.labelNodeId)
	}

	return ms.Labels[core.LabelNodename.Key]
}

func (h *hawkularSink) registerLabeledIfNecessary(ms *core.MetricSet, metric core.LabeledMetric, m ...metrics.Modifier) error {
	key := h.idName(ms, metric.Name)

	if resourceID, found := metric.Labels[core.LabelResourceID.Key]; found {
		key = h.idName(ms, metric.Name+separator+resourceID)
	}

	h.regLock.Lock()
	defer h.regLock.Unlock()

	// If found, check it matches the current stored definition (could be old info from
	// the stored metrics cache for example)
	if _, found := h.reg[key]; !found {
		// Register the metric descriptor here..
		if md, f := h.models[metric.Name]; f {
			// Copy the original map
			mdd := *md
			tags := make(map[string]string)
			for k, v := range mdd.Tags {
				tags[k] = v
			}
			mdd.Tags = tags

			// Set tag values
			for k, v := range ms.Labels {
				mdd.Tags[k] = v
			}

			// Set the labelled values
			for k, v := range metric.Labels {
				mdd.Tags[k] = v
			}

			mdd.Tags[groupTag] = h.groupName(ms, metric.Name)
			mdd.Tags[descriptorTag] = metric.Name

			m = append(m, h.modifiers...)

			// Create metric, use updateTags instead of Create because we know it is unique
			if err := h.client.UpdateTags(heapsterTypeToHawkularType(metric.MetricType), key, mdd.Tags, m...); err != nil {
				// Log error and don't add this key to the lookup table
				glog.Errorf("Could not update tags: %s", err)
				return err
			}

			// Add to the lookup table
			h.reg[key] = &mdd
		} else {
			return fmt.Errorf("Could not find definition model with name %s", metric.Name)
		}
	}
	// TODO Compare the definition tags and update if necessary? Quite expensive operation..

	return nil
}

func toBatches(m []metrics.MetricHeader, batchSize int) chan []metrics.MetricHeader {
	if batchSize == 0 {
		c := make(chan []metrics.MetricHeader, 1)
		c <- m
		return c
	}

	size := int(math.Ceil(float64(len(m)) / float64(batchSize)))
	c := make(chan []metrics.MetricHeader, size)

	for i := 0; i < len(m); i += batchSize {
		n := i + batchSize
		if len(m) < n {
			n = len(m)
		}
		part := m[i:n]
		c <- part
	}

	return c
}

func (h *hawkularSink) sendData(tmhs map[string][]metrics.MetricHeader, wg *sync.WaitGroup) {
	for k, v := range tmhs {
		parts := toBatches(v, h.batchSize)
		close(parts)

		for p := range parts {
			wg.Add(1)
			go func(batch []metrics.MetricHeader, tenant string) {
				defer wg.Done()

				m := make([]metrics.Modifier, len(h.modifiers), len(h.modifiers)+1)
				copy(m, h.modifiers)
				m = append(m, metrics.Tenant(tenant))
				if err := h.client.Write(batch, m...); err != nil {
					glog.Errorf(err.Error())
				}
			}(p, k)
		}
	}
}

// Converts Timeseries to metric structure used by the Hawkular
func (h *hawkularSink) pointToLabeledMetricHeader(ms *core.MetricSet, metric core.LabeledMetric, timestamp time.Time) (*metrics.MetricHeader, error) {

	name := h.idName(ms, metric.Name)
	if resourceID, found := metric.Labels[core.LabelResourceID.Key]; found {
		name = h.idName(ms, metric.Name+separator+resourceID)
	}

	var value float64
	if metric.ValueType == core.ValueInt64 {
		value = float64(metric.IntValue)
	} else {
		value = float64(metric.FloatValue)
	}

	m := metrics.Datapoint{
		Value:     value,
		Timestamp: metrics.UnixMilli(timestamp),
	}

	mh := &metrics.MetricHeader{
		Id:   name,
		Data: []metrics.Datapoint{m},
		Type: heapsterTypeToHawkularType(metric.MetricType),
	}

	return mh, nil
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
	return func(ms *core.MetricSet, metricName string) bool {
		for k, v := range ms.Labels {
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
	return func(ms *core.MetricSet, metricName string) bool {
		return !r.MatchString(metricName)
	}
}
