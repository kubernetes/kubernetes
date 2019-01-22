/*
Copyright 2018 The Kubernetes Authors.

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
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"google.golang.org/api/googleapi"
	tpuapi "google.golang.org/api/tpu/v1"
	"k8s.io/klog"

	"k8s.io/apimachinery/pkg/util/wait"
)

// newTPUService returns a new tpuService using the client to communicate with
// the Cloud TPU APIs.
func newTPUService(client *http.Client) (*tpuService, error) {
	s, err := tpuapi.New(client)
	if err != nil {
		return nil, err
	}
	return &tpuService{
		projects: tpuapi.NewProjectsService(s),
	}, nil
}

// tpuService encapsulates the TPU services on nodes and the operations on the
// nodes.
type tpuService struct {
	projects *tpuapi.ProjectsService
}

// CreateTPU creates the Cloud TPU node with the specified name in the
// specified zone.
func (g *Cloud) CreateTPU(ctx context.Context, name, zone string, node *tpuapi.Node) (*tpuapi.Node, error) {
	var err error
	mc := newTPUMetricContext("create", zone)
	defer mc.Observe(err)

	var op *tpuapi.Operation
	parent := getTPUParentName(g.projectID, zone)
	op, err = g.tpuService.projects.Locations.Nodes.Create(parent, node).NodeId(name).Do()
	if err != nil {
		return nil, err
	}
	klog.V(2).Infof("Creating Cloud TPU %q in zone %q with operation %q", name, zone, op.Name)

	op, err = g.waitForTPUOp(ctx, op)
	if err != nil {
		return nil, err
	}
	err = getErrorFromTPUOp(op)
	if err != nil {
		return nil, err
	}

	output := new(tpuapi.Node)
	err = json.Unmarshal(op.Response, output)
	if err != nil {
		err = fmt.Errorf("failed to unmarshal response from operation %q: response = %v, err = %v", op.Name, op.Response, err)
		return nil, err
	}
	return output, nil
}

// DeleteTPU deletes the Cloud TPU with the specified name in the specified
// zone.
func (g *Cloud) DeleteTPU(ctx context.Context, name, zone string) error {
	var err error
	mc := newTPUMetricContext("delete", zone)
	defer mc.Observe(err)

	var op *tpuapi.Operation
	name = getTPUName(g.projectID, zone, name)
	op, err = g.tpuService.projects.Locations.Nodes.Delete(name).Do()
	if err != nil {
		return err
	}
	klog.V(2).Infof("Deleting Cloud TPU %q in zone %q with operation %q", name, zone, op.Name)

	op, err = g.waitForTPUOp(ctx, op)
	if err != nil {
		return err
	}
	err = getErrorFromTPUOp(op)
	if err != nil {
		return err
	}
	return nil
}

// GetTPU returns the Cloud TPU with the specified name in the specified zone.
func (g *Cloud) GetTPU(ctx context.Context, name, zone string) (*tpuapi.Node, error) {
	mc := newTPUMetricContext("get", zone)

	name = getTPUName(g.projectID, zone, name)
	node, err := g.tpuService.projects.Locations.Nodes.Get(name).Do()
	if err != nil {
		return nil, mc.Observe(err)
	}
	return node, mc.Observe(nil)
}

// ListTPUs returns Cloud TPUs in the specified zone.
func (g *Cloud) ListTPUs(ctx context.Context, zone string) ([]*tpuapi.Node, error) {
	mc := newTPUMetricContext("list", zone)

	parent := getTPUParentName(g.projectID, zone)
	response, err := g.tpuService.projects.Locations.Nodes.List(parent).Do()
	if err != nil {
		return nil, mc.Observe(err)
	}
	return response.Nodes, mc.Observe(nil)
}

// ListLocations returns the zones where Cloud TPUs are available.
func (g *Cloud) ListLocations(ctx context.Context) ([]*tpuapi.Location, error) {
	mc := newTPUMetricContext("list_locations", "")
	parent := getTPUProjectURL(g.projectID)
	response, err := g.tpuService.projects.Locations.List(parent).Do()
	if err != nil {
		return nil, mc.Observe(err)
	}
	return response.Locations, mc.Observe(nil)
}

// waitForTPUOp checks whether the op is done every 30 seconds before the ctx
// is cancelled.
func (g *Cloud) waitForTPUOp(ctx context.Context, op *tpuapi.Operation) (*tpuapi.Operation, error) {
	if err := wait.PollInfinite(30*time.Second, func() (bool, error) {
		// Check if context has been cancelled.
		select {
		case <-ctx.Done():
			klog.V(3).Infof("Context for operation %q has been cancelled: %s", op.Name, ctx.Err())
			return true, ctx.Err()
		default:
		}

		klog.V(3).Infof("Waiting for operation %q to complete...", op.Name)

		start := time.Now()
		g.operationPollRateLimiter.Accept()
		duration := time.Now().Sub(start)
		if duration > 5*time.Second {
			klog.V(2).Infof("Getting operation %q throttled for %v", op.Name, duration)
		}

		var err error
		op, err = g.tpuService.projects.Locations.Operations.Get(op.Name).Do()
		if err != nil {
			return true, err
		}
		if op.Done {
			klog.V(3).Infof("Operation %q has completed", op.Name)
			return true, nil
		}
		return false, nil
	}); err != nil {
		return nil, fmt.Errorf("failed to wait for operation %q: %s", op.Name, err)
	}
	return op, nil
}

// newTPUMetricContext returns a new metricContext used for recording metrics
// of Cloud TPU API calls.
func newTPUMetricContext(request, zone string) *metricContext {
	return newGenericMetricContext("tpus", request, unusedMetricLabel, zone, "v1")
}

// getErrorFromTPUOp returns the error in the failed op, or nil if the op
// succeed.
func getErrorFromTPUOp(op *tpuapi.Operation) error {
	if op != nil && op.Error != nil {
		return &googleapi.Error{
			Code:    op.ServerResponse.HTTPStatusCode,
			Message: op.Error.Message,
		}
	}
	return nil
}

func getTPUProjectURL(project string) string {
	return fmt.Sprintf("projects/%s", project)
}

func getTPUParentName(project, zone string) string {
	return fmt.Sprintf("projects/%s/locations/%s", project, zone)
}

func getTPUName(project, zone, name string) string {
	return fmt.Sprintf("projects/%s/locations/%s/nodes/%s", project, zone, name)
}
