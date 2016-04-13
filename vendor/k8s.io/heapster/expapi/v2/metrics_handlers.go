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

package v2

import (
	"errors"
	"fmt"
	"math"
	"net/http"
	"time"

	restful "github.com/emicklei/go-restful"
	"github.com/golang/glog"

	"k8s.io/heapster/expapi/v2/types"
	model_api "k8s.io/heapster/model"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/unversioned"
)

// errModelNotActivated is the error that is returned when manager.cluster
// has not beed initialized.
var errModelNotActivated = errors.New("the model is not activated")

// RegisterMetrics registers the Metrics API endpoints.
// All endpoints that end with a {metric-name} also receive a start time query parameter.
// The start and end times should be specified as a string, formatted according to RFC 3339.
// These apis are experimental, so they may change or disappear in the future.
func (a *Api) RegisterMetrics(container *restful.Container) {
	ws := new(restful.WebService)
	ws.
		Path("/experimental/v2").
		Doc("Root endpoint of the stats model").
		Consumes("*/*").
		Produces(restful.MIME_JSON)

	ws.Route(ws.GET("/nodeMetrics/derived/").
		To(a.derivedNodeMetricsList).
		Filter(compressionFilter).
		Doc("Get a list of all available metrics for all nodes").
		Writes([]types.DerivedNodeMetrics{}).
		Operation("derivedNodeMetricsList"))

	ws.Route(ws.GET("/nodeMetrics/derived/{node-name}").
		To(a.derivedNodeMetrics).
		Filter(compressionFilter).
		Doc("Get a list of all raw metrics for a Node entity").
		Writes(types.DerivedNodeMetrics{}).
		Operation("derivedNodeMetrics").
		Param(ws.PathParameter("node-name", "The name of the node to look up").DataType("string")))

	container.Add(ws)
}

func portStats(w *types.MetricsWindows, byName map[string]model_api.StatBundle) error {
	byDuration := make(map[time.Duration]*types.MetricsWindow)
	for _, d := range [...]time.Duration{time.Minute, time.Hour, 24 * time.Hour} {
		w.Windows = append(w.Windows, types.MetricsWindow{
			Duration: unversioned.Duration{Duration: d},
			Mean:     make(types.ResourceUsage),
			Max:      make(types.ResourceUsage),
			NinetyFifthPercentile: make(types.ResourceUsage),
		})
		byDuration[d] = &w.Windows[len(w.Windows)-1]
	}
	toQuantity := func(u uint64) (*resource.Quantity, error) {
		if u > math.MaxInt64 {
			return nil, fmt.Errorf("unexpectedly large value: %v", u)
		}
		return resource.NewQuantity(int64(u), resource.DecimalSI), nil
	}
	// Translate from the model resource names to ours
	resNames := map[string]string{
		"cpu":    "cpu-usage",
		"memory": "memory-usage",
	}
	for resName, otherName := range resNames {
		sb, e := byName[otherName]
		if !e {
			return fmt.Errorf("missing resource: %v", otherName)
		}
		for d, v := range map[time.Duration]model_api.Stats{
			time.Minute:    sb.Minute,
			time.Hour:      sb.Hour,
			24 * time.Hour: sb.Day,
		} {
			q, err := toQuantity(v.Average)
			if err != nil {
				return err
			}
			byDuration[d].Mean[resName] = *q
			q, err = toQuantity(v.Max)
			if err != nil {
				return err
			}
			byDuration[d].Max[resName] = *q
			q, err = toQuantity(v.NinetyFifth)
			if err != nil {
				return err
			}
			byDuration[d].NinetyFifthPercentile[resName] = *q
		}
	}
	return nil
}

func getNodeMetrics(model model_api.Model, name string) (*types.DerivedNodeMetrics, error) {
	res, err := model.GetNodeStats(model_api.NodeRequest{
		NodeName: name,
	})
	if err != nil {
		return nil, err
	}
	metrics := &types.DerivedNodeMetrics{
		NodeName: name,
		NodeMetrics: types.MetricsWindows{
			EndTime: unversioned.NewTime(res.Timestamp),
		},
		// TODO: fill SystemContainers
		SystemContainers: make([]types.DerivedContainerMetrics, 0),
	}
	if err := portStats(&metrics.NodeMetrics, res.ByName); err != nil {
		return nil, err
	}
	return metrics, nil
}

func (a *Api) derivedNodeMetrics(request *restful.Request, response *restful.Response) {
	model := a.manager.GetModel()
	if model == nil {
		response.WriteError(400, errModelNotActivated)
	}
	metrics, err := getNodeMetrics(model, request.PathParameter("node-name"))
	if err != nil {
		response.WriteError(400, err)
	}
	response.WriteEntity(metrics)
}

func (a *Api) derivedNodeMetricsList(request *restful.Request, response *restful.Response) {
	model := a.manager.GetModel()
	if model == nil {
		response.WriteError(400, errModelNotActivated)
	}
	nodes := model.GetNodes()
	list := make([]types.DerivedNodeMetrics, 0, len(nodes))
	for _, node := range nodes {
		metrics, err := getNodeMetrics(model, node.Name)
		if err != nil {
			response.WriteError(400, err)
		}
		list = append(list, *metrics)
	}
	response.WriteEntity(list)
}

// parseRequestParam parses a time.Time from a named QueryParam.
// parseRequestParam receives a request and a response as inputs, and returns the parsed time.
func parseRequestParam(param string, request *restful.Request, response *restful.Response) time.Time {
	var err error
	query_param := request.QueryParameter(param)
	req_stamp := time.Time{}
	if query_param != "" {
		req_stamp, err = time.Parse(time.RFC3339, query_param)
		if err != nil {
			// Timestamp parameter cannot be parsed
			response.WriteError(http.StatusInternalServerError, err)
			glog.Errorf("timestamp argument cannot be parsed: %s", err)
			return time.Time{}
		}
	}
	return req_stamp
}
