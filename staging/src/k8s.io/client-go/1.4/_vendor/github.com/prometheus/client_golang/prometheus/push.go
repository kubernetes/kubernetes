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

package prometheus

// Push triggers a metric collection by the default registry and pushes all
// collected metrics to the Pushgateway specified by addr. See the Pushgateway
// documentation for detailed implications of the job and instance
// parameter. instance can be left empty. You can use just host:port or ip:port
// as url, in which case 'http://' is added automatically. You can also include
// the schema in the URL. However, do not include the '/metrics/jobs/...' part.
//
// Note that all previously pushed metrics with the same job and instance will
// be replaced with the metrics pushed by this call. (It uses HTTP method 'PUT'
// to push to the Pushgateway.)
func Push(job, instance, url string) error {
	return defRegistry.Push(job, instance, url, "PUT")
}

// PushAdd works like Push, but only previously pushed metrics with the same
// name (and the same job and instance) will be replaced. (It uses HTTP method
// 'POST' to push to the Pushgateway.)
func PushAdd(job, instance, url string) error {
	return defRegistry.Push(job, instance, url, "POST")
}

// PushCollectors works like Push, but it does not collect from the default
// registry. Instead, it collects from the provided collectors. It is a
// convenient way to push only a few metrics.
func PushCollectors(job, instance, url string, collectors ...Collector) error {
	return pushCollectors(job, instance, url, "PUT", collectors...)
}

// PushAddCollectors works like PushAdd, but it does not collect from the
// default registry. Instead, it collects from the provided collectors. It is a
// convenient way to push only a few metrics.
func PushAddCollectors(job, instance, url string, collectors ...Collector) error {
	return pushCollectors(job, instance, url, "POST", collectors...)
}

func pushCollectors(job, instance, url, method string, collectors ...Collector) error {
	r := newRegistry()
	for _, collector := range collectors {
		if _, err := r.Register(collector); err != nil {
			return err
		}
	}
	return r.Push(job, instance, url, method)
}
