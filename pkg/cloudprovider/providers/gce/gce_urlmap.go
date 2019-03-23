/*
Copyright 2017 The Kubernetes Authors.

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

package gce

import (
	compute "google.golang.org/api/compute/v1"

	"github.com/GoogleCloudPlatform/k8s-cloud-provider/pkg/cloud"
	"github.com/GoogleCloudPlatform/k8s-cloud-provider/pkg/cloud/filter"
	"github.com/GoogleCloudPlatform/k8s-cloud-provider/pkg/cloud/meta"
)

func newURLMapMetricContext(request string) *metricContext {
	return newGenericMetricContext("urlmap", request, unusedMetricLabel, unusedMetricLabel, computeV1Version)
}

// GetURLMap returns the UrlMap by name.
func (g *Cloud) GetURLMap(name string) (*compute.UrlMap, error) {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newURLMapMetricContext("get")
	v, err := g.c.UrlMaps().Get(ctx, meta.GlobalKey(name))
	return v, mc.Observe(err)
}

// CreateURLMap creates a url map
func (g *Cloud) CreateURLMap(urlMap *compute.UrlMap) error {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newURLMapMetricContext("create")
	return mc.Observe(g.c.UrlMaps().Insert(ctx, meta.GlobalKey(urlMap.Name), urlMap))
}

// UpdateURLMap applies the given UrlMap as an update
func (g *Cloud) UpdateURLMap(urlMap *compute.UrlMap) error {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newURLMapMetricContext("update")
	return mc.Observe(g.c.UrlMaps().Update(ctx, meta.GlobalKey(urlMap.Name), urlMap))
}

// DeleteURLMap deletes a url map by name.
func (g *Cloud) DeleteURLMap(name string) error {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newURLMapMetricContext("delete")
	return mc.Observe(g.c.UrlMaps().Delete(ctx, meta.GlobalKey(name)))
}

// ListURLMaps lists all UrlMaps in the project.
func (g *Cloud) ListURLMaps() ([]*compute.UrlMap, error) {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newURLMapMetricContext("list")
	v, err := g.c.UrlMaps().List(ctx, filter.None)
	return v, mc.Observe(err)
}
