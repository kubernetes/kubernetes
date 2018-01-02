// Copyright 2015 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Copyright (c) 2013, The Prometheus Authors
// All rights reserved.
//
// Use of this source code is governed by a BSD-style license that can be found
// in the LICENSE file.

// Package push provides functions to push metrics to a Pushgateway. The metrics
// to push are either collected from a provided registry, or from explicitly
// listed collectors.
//
// See the documentation of the Pushgateway to understand the meaning of the
// grouping parameters and the differences between push.Registry and
// push.Collectors on the one hand and push.AddRegistry and push.AddCollectors
// on the other hand: https://github.com/prometheus/pushgateway
package push

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"net/http"
	"net/url"
	"os"
	"strings"

	"github.com/prometheus/common/expfmt"
	"github.com/prometheus/common/model"

	"github.com/prometheus/client_golang/prometheus"
)

const contentTypeHeader = "Content-Type"

// FromGatherer triggers a metric collection by the provided Gatherer (which is
// usually implemented by a prometheus.Registry) and pushes all gathered metrics
// to the Pushgateway specified by url, using the provided job name and the
// (optional) further grouping labels (the grouping map may be nil). See the
// Pushgateway documentation for detailed implications of the job and other
// grouping labels. Neither the job name nor any grouping label value may
// contain a "/". The metrics pushed must not contain a job label of their own
// nor any of the grouping labels.
//
// You can use just host:port or ip:port as url, in which case 'http://' is
// added automatically. You can also include the schema in the URL. However, do
// not include the '/metrics/jobs/...' part.
//
// Note that all previously pushed metrics with the same job and other grouping
// labels will be replaced with the metrics pushed by this call. (It uses HTTP
// method 'PUT' to push to the Pushgateway.)
func FromGatherer(job string, grouping map[string]string, url string, g prometheus.Gatherer) error {
	return push(job, grouping, url, g, "PUT")
}

// AddFromGatherer works like FromGatherer, but only previously pushed metrics
// with the same name (and the same job and other grouping labels) will be
// replaced. (It uses HTTP method 'POST' to push to the Pushgateway.)
func AddFromGatherer(job string, grouping map[string]string, url string, g prometheus.Gatherer) error {
	return push(job, grouping, url, g, "POST")
}

func push(job string, grouping map[string]string, pushURL string, g prometheus.Gatherer, method string) error {
	if !strings.Contains(pushURL, "://") {
		pushURL = "http://" + pushURL
	}
	if strings.HasSuffix(pushURL, "/") {
		pushURL = pushURL[:len(pushURL)-1]
	}

	if strings.Contains(job, "/") {
		return fmt.Errorf("job contains '/': %s", job)
	}
	urlComponents := []string{url.QueryEscape(job)}
	for ln, lv := range grouping {
		if !model.LabelName(ln).IsValid() {
			return fmt.Errorf("grouping label has invalid name: %s", ln)
		}
		if strings.Contains(lv, "/") {
			return fmt.Errorf("value of grouping label %s contains '/': %s", ln, lv)
		}
		urlComponents = append(urlComponents, ln, lv)
	}
	pushURL = fmt.Sprintf("%s/metrics/job/%s", pushURL, strings.Join(urlComponents, "/"))

	mfs, err := g.Gather()
	if err != nil {
		return err
	}
	buf := &bytes.Buffer{}
	enc := expfmt.NewEncoder(buf, expfmt.FmtProtoDelim)
	// Check for pre-existing grouping labels:
	for _, mf := range mfs {
		for _, m := range mf.GetMetric() {
			for _, l := range m.GetLabel() {
				if l.GetName() == "job" {
					return fmt.Errorf("pushed metric %s (%s) already contains a job label", mf.GetName(), m)
				}
				if _, ok := grouping[l.GetName()]; ok {
					return fmt.Errorf(
						"pushed metric %s (%s) already contains grouping label %s",
						mf.GetName(), m, l.GetName(),
					)
				}
			}
		}
		enc.Encode(mf)
	}
	req, err := http.NewRequest(method, pushURL, buf)
	if err != nil {
		return err
	}
	req.Header.Set(contentTypeHeader, string(expfmt.FmtProtoDelim))
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 202 {
		body, _ := ioutil.ReadAll(resp.Body) // Ignore any further error as this is for an error message only.
		return fmt.Errorf("unexpected status code %d while pushing to %s: %s", resp.StatusCode, pushURL, body)
	}
	return nil
}

// Collectors works like FromGatherer, but it does not use a Gatherer. Instead,
// it collects from the provided collectors directly. It is a convenient way to
// push only a few metrics.
func Collectors(job string, grouping map[string]string, url string, collectors ...prometheus.Collector) error {
	return pushCollectors(job, grouping, url, "PUT", collectors...)
}

// AddCollectors works like AddFromGatherer, but it does not use a Gatherer.
// Instead, it collects from the provided collectors directly. It is a
// convenient way to push only a few metrics.
func AddCollectors(job string, grouping map[string]string, url string, collectors ...prometheus.Collector) error {
	return pushCollectors(job, grouping, url, "POST", collectors...)
}

func pushCollectors(job string, grouping map[string]string, url, method string, collectors ...prometheus.Collector) error {
	r := prometheus.NewRegistry()
	for _, collector := range collectors {
		if err := r.Register(collector); err != nil {
			return err
		}
	}
	return push(job, grouping, url, r, method)
}

// HostnameGroupingKey returns a label map with the only entry
// {instance="<hostname>"}. This can be conveniently used as the grouping
// parameter if metrics should be pushed with the hostname as label. The
// returned map is created upon each call so that the caller is free to add more
// labels to the map.
func HostnameGroupingKey() map[string]string {
	hostname, err := os.Hostname()
	if err != nil {
		return map[string]string{"instance": "unknown"}
	}
	return map[string]string{"instance": hostname}
}
