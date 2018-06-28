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

	"github.com/golang/glog"
	"google.golang.org/api/googleapi"
	tpuapi "google.golang.org/api/tpu/v1"

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
		nodesService:      tpuapi.NewProjectsLocationsNodesService(s),
		operationsService: tpuapi.NewProjectsLocationsOperationsService(s),
	}, nil
}

// tpuService encapsulates the TPU services on nodes and the operations on the
// nodes.
type tpuService struct {
	nodesService      *tpuapi.ProjectsLocationsNodesService
	operationsService *tpuapi.ProjectsLocationsOperationsService
}

// CreateTPU creates the Cloud TPU node with the specified name in the
// specified zone.
func (gce *GCECloud) CreateTPU(ctx context.Context, name, zone string, node *tpuapi.Node) (*tpuapi.Node, error) {
	var err error
	mc := newTPUMetricContext("create", zone)
	defer mc.Observe(err)

	var op *tpuapi.Operation
	parent := getTPUParentName(gce.projectID, zone)
	op, err = gce.tpuService.nodesService.Create(parent, node).NodeId(name).Do()
	if err != nil {
		return nil, err
	}
	glog.V(2).Infof("Creating Cloud TPU %q in zone %q with operation %q", name, zone, op.Name)

	op, err = gce.waitForTPUOp(30*time.Second, 10*time.Minute, op)
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
func (gce *GCECloud) DeleteTPU(ctx context.Context, name, zone string) error {
	var err error
	mc := newTPUMetricContext("delete", zone)
	defer mc.Observe(err)

	var op *tpuapi.Operation
	name = getTPUName(gce.projectID, zone, name)
	op, err = gce.tpuService.nodesService.Delete(name).Do()
	if err != nil {
		return err
	}
	glog.V(2).Infof("Deleting Cloud TPU %q in zone %q with operation %q", name, zone, op.Name)

	op, err = gce.waitForTPUOp(30*time.Second, 10*time.Minute, op)
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
func (gce *GCECloud) GetTPU(ctx context.Context, name, zone string) (*tpuapi.Node, error) {
	mc := newTPUMetricContext("get", zone)

	name = getTPUName(gce.projectID, zone, name)
	node, err := gce.tpuService.nodesService.Get(name).Do()
	if err != nil {
		return nil, mc.Observe(err)
	}
	return node, mc.Observe(nil)
}

// ListTPUs returns Cloud TPUs in the specified zone.
func (gce *GCECloud) ListTPUs(ctx context.Context, zone string) ([]*tpuapi.Node, error) {
	mc := newTPUMetricContext("list", zone)

	parent := getTPUParentName(gce.projectID, zone)
	response, err := gce.tpuService.nodesService.List(parent).Do()
	if err != nil {
		return nil, mc.Observe(err)
	}
	return response.Nodes, mc.Observe(nil)
}

// waitForTPUOp checks whether the op is done every interval before the timeout
// occurs.
func (gce *GCECloud) waitForTPUOp(interval, timeout time.Duration, op *tpuapi.Operation) (*tpuapi.Operation, error) {
	if err := wait.PollImmediate(interval, timeout, func() (bool, error) {
		glog.V(3).Infof("Waiting for operation %q to complete...", op.Name)

		start := time.Now()
		gce.operationPollRateLimiter.Accept()
		duration := time.Now().Sub(start)
		if duration > 5*time.Second {
			glog.V(2).Infof("Getting operation %q throttled for %v", op.Name, duration)
		}

		var err error
		op, err = gce.tpuService.operationsService.Get(op.Name).Do()
		if err != nil {
			return true, err
		}
		if op.Done {
			glog.V(3).Infof("Operation %q has completed", op.Name)
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

func getTPUParentName(project, zone string) string {
	return fmt.Sprintf("projects/%s/locations/%s", project, zone)
}

func getTPUName(project, zone, name string) string {
	return fmt.Sprintf("projects/%s/locations/%s/nodes/%s", project, zone, name)
}
