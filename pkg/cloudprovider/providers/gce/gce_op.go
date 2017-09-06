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
	"encoding/json"
	"fmt"
	"time"

	"k8s.io/apimachinery/pkg/util/wait"

	"github.com/golang/glog"
	computealpha "google.golang.org/api/compute/v0.alpha"
	computebeta "google.golang.org/api/compute/v0.beta"
	computev1 "google.golang.org/api/compute/v1"
	"google.golang.org/api/googleapi"
)

func (gce *GCECloud) waitForOp(op *computev1.Operation, getOperation func(operationName string) (*computev1.Operation, error), mc *metricContext) error {
	if op == nil {
		return mc.Observe(fmt.Errorf("operation must not be nil"))
	}

	if opIsDone(op) {
		return getErrorFromOp(op)
	}

	opStart := time.Now()
	opName := op.Name

	return wait.Poll(operationPollInterval, operationPollTimeoutDuration, func() (bool, error) {
		start := time.Now()
		gce.operationPollRateLimiter.Accept()
		duration := time.Now().Sub(start)
		if duration > 5*time.Second {
			glog.V(2).Infof("pollOperation: throttled %v for %v", duration, opName)
		}
		pollOp, err := getOperation(opName)
		if err != nil {
			glog.Warningf("GCE poll operation %s failed: pollOp: [%v] err: [%v] getErrorFromOp: [%v]",
				opName, pollOp, err, getErrorFromOp(pollOp))
		}

		done := opIsDone(pollOp)
		if done {
			duration := time.Now().Sub(opStart)
			if duration > 1*time.Minute {
				// Log the JSON. It's cleaner than the %v structure.
				enc, err := pollOp.MarshalJSON()
				if err != nil {
					glog.Warningf("waitForOperation: long operation (%v): %v (failed to encode to JSON: %v)",
						duration, pollOp, err)
				} else {
					glog.V(2).Infof("waitForOperation: long operation (%v): %v",
						duration, string(enc))
				}
			}
		}

		return done, mc.Observe(getErrorFromOp(pollOp))
	})
}

func opIsDone(op *computev1.Operation) bool {
	return op != nil && op.Status == "DONE"
}

func getErrorFromOp(op *computev1.Operation) error {
	if op != nil && op.Error != nil && len(op.Error.Errors) > 0 {
		err := &googleapi.Error{
			Code:    int(op.HttpErrorStatusCode),
			Message: op.Error.Errors[0].Message,
		}
		glog.Errorf("GCE operation failed: %v", err)
		return err
	}

	return nil
}

func (gce *GCECloud) waitForGlobalOp(op gceObject, mc *metricContext) error {
	return gce.waitForGlobalOpInProject(op, gce.ProjectID(), mc)
}

func (gce *GCECloud) waitForRegionOp(op gceObject, region string, mc *metricContext) error {
	return gce.waitForRegionOpInProject(op, gce.ProjectID(), region, mc)
}

func (gce *GCECloud) waitForZoneOp(op gceObject, zone string, mc *metricContext) error {
	return gce.waitForZoneOpInProject(op, gce.ProjectID(), zone, mc)
}

func (gce *GCECloud) waitForGlobalOpInProject(op gceObject, projectID string, mc *metricContext) error {
	switch v := op.(type) {
	case *computealpha.Operation:
		return gce.waitForOp(convertToV1Operation(op), func(operationName string) (*computev1.Operation, error) {
			op, err := gce.serviceAlpha.GlobalOperations.Get(projectID, operationName).Do()
			return convertToV1Operation(op), err
		}, mc)
	case *computebeta.Operation:
		return gce.waitForOp(convertToV1Operation(op), func(operationName string) (*computev1.Operation, error) {
			op, err := gce.serviceBeta.GlobalOperations.Get(projectID, operationName).Do()
			return convertToV1Operation(op), err
		}, mc)
	case *computev1.Operation:
		return gce.waitForOp(op.(*computev1.Operation), func(operationName string) (*computev1.Operation, error) {
			return gce.service.GlobalOperations.Get(projectID, operationName).Do()
		}, mc)
	default:
		return fmt.Errorf("unexpected type: %T", v)
	}
}

func (gce *GCECloud) waitForRegionOpInProject(op gceObject, projectID, region string, mc *metricContext) error {
	switch v := op.(type) {
	case *computealpha.Operation:
		return gce.waitForOp(convertToV1Operation(op), func(operationName string) (*computev1.Operation, error) {
			op, err := gce.serviceAlpha.RegionOperations.Get(projectID, region, operationName).Do()
			return convertToV1Operation(op), err
		}, mc)
	case *computebeta.Operation:
		return gce.waitForOp(convertToV1Operation(op), func(operationName string) (*computev1.Operation, error) {
			op, err := gce.serviceBeta.RegionOperations.Get(projectID, region, operationName).Do()
			return convertToV1Operation(op), err
		}, mc)
	case *computev1.Operation:
		return gce.waitForOp(op.(*computev1.Operation), func(operationName string) (*computev1.Operation, error) {
			return gce.service.RegionOperations.Get(projectID, region, operationName).Do()
		}, mc)
	default:
		return fmt.Errorf("unexpected type: %T", v)
	}
}

func (gce *GCECloud) waitForZoneOpInProject(op gceObject, projectID, zone string, mc *metricContext) error {
	switch v := op.(type) {
	case *computealpha.Operation:
		return gce.waitForOp(convertToV1Operation(op), func(operationName string) (*computev1.Operation, error) {
			op, err := gce.serviceAlpha.ZoneOperations.Get(projectID, zone, operationName).Do()
			return convertToV1Operation(op), err
		}, mc)
	case *computebeta.Operation:
		return gce.waitForOp(convertToV1Operation(op), func(operationName string) (*computev1.Operation, error) {
			op, err := gce.serviceBeta.ZoneOperations.Get(projectID, zone, operationName).Do()
			return convertToV1Operation(op), err
		}, mc)
	case *computev1.Operation:
		return gce.waitForOp(op.(*computev1.Operation), func(operationName string) (*computev1.Operation, error) {
			return gce.service.ZoneOperations.Get(projectID, zone, operationName).Do()
		}, mc)
	default:
		return fmt.Errorf("unexpected type: %T", v)
	}
}

func convertToV1Operation(object gceObject) *computev1.Operation {
	enc, err := object.MarshalJSON()
	if err != nil {
		panic(fmt.Sprintf("Failed to encode to json: %v", err))
	}
	var op computev1.Operation
	if err := json.Unmarshal(enc, &op); err != nil {
		panic(fmt.Sprintf("Failed to convert GCE apiObject %v to v1 operation: %v", object, err))
	}
	return &op
}
