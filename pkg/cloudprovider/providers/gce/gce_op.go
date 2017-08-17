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
	compute_alpha "google.golang.org/api/compute/v0.alpha"
	compute_beta "google.golang.org/api/compute/v0.beta"
	compute_v1 "google.golang.org/api/compute/v1"
	"google.golang.org/api/googleapi"
)

func (gce *GCECloud) waitForOp(op *compute_v1.Operation, getOperation func(operationName string) (*compute_v1.Operation, error), mc *metricContext) error {
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

func opIsDone(op *compute_v1.Operation) bool {
	return op != nil && op.Status == "DONE"
}

func getErrorFromOp(op *compute_v1.Operation) error {
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

func (gce *GCECloud) waitForGlobalOp(op *compute_v1.Operation, mc *metricContext) error {
	return gce.waitForOp(op, func(operationName string) (*compute_v1.Operation, error) {
		return gce.service.GlobalOperations.Get(gce.projectID, operationName).Do()
	}, mc)
}

func (gce *GCECloud) waitForRegionOp(op *compute_v1.Operation, region string, mc *metricContext) error {
	return gce.waitForOp(op, func(operationName string) (*compute_v1.Operation, error) {
		return gce.service.RegionOperations.Get(gce.projectID, region, operationName).Do()
	}, mc)
}

func (gce *GCECloud) waitForZoneOp(op *compute_v1.Operation, zone string, mc *metricContext) error {
	return gce.waitForOp(op, func(operationName string) (*compute_v1.Operation, error) {
		return gce.service.ZoneOperations.Get(gce.projectID, zone, operationName).Do()
	}, mc)
}

func (gce *GCECloud) waitForGlobalAlphaOp(op *compute_alpha.Operation, mc *metricContext) error {
	return gce.waitForOp(alphaOperationToV1(op), func(operationName string) (*compute_v1.Operation, error) {
		op, err := gce.serviceAlpha.GlobalOperations.Get(gce.projectID, operationName).Do()
		return alphaOperationToV1(op), err
	}, mc)
}

func (gce *GCECloud) waitForRegionAlphaOp(op *compute_alpha.Operation, region string, mc *metricContext) error {
	return gce.waitForOp(alphaOperationToV1(op), func(operationName string) (*compute_v1.Operation, error) {
		op, err := gce.serviceAlpha.RegionOperations.Get(gce.projectID, region, operationName).Do()
		return alphaOperationToV1(op), err
	}, mc)
}

func (gce *GCECloud) waitForZoneAlphaOp(op *compute_alpha.Operation, zone string, mc *metricContext) error {
	return gce.waitForOp(alphaOperationToV1(op), func(operationName string) (*compute_v1.Operation, error) {
		op, err := gce.serviceAlpha.ZoneOperations.Get(gce.projectID, zone, operationName).Do()
		return alphaOperationToV1(op), err
	}, mc)
}

func (gce *GCECloud) waitForGlobalBetaOp(op *compute_beta.Operation, mc *metricContext) error {
	return gce.waitForOp(betaOperationToV1(op), func(operationName string) (*compute_v1.Operation, error) {
		op, err := gce.serviceBeta.GlobalOperations.Get(gce.projectID, operationName).Do()
		return betaOperationToV1(op), err
	}, mc)
}

func (gce *GCECloud) waitForRegionBetaOp(op *compute_beta.Operation, region string, mc *metricContext) error {
	return gce.waitForOp(betaOperationToV1(op), func(operationName string) (*compute_v1.Operation, error) {
		op, err := gce.serviceBeta.RegionOperations.Get(gce.projectID, region, operationName).Do()
		return betaOperationToV1(op), err
	}, mc)
}

func (gce *GCECloud) waitForZoneBetaOp(op *compute_beta.Operation, zone string, mc *metricContext) error {
	return gce.waitForOp(betaOperationToV1(op), func(operationName string) (*compute_v1.Operation, error) {
		op, err := gce.serviceBeta.ZoneOperations.Get(gce.projectID, zone, operationName).Do()
		return betaOperationToV1(op), err
	}, mc)
}

func betaOperationToV1(op *compute_beta.Operation) *compute_v1.Operation {
	enc, err := op.MarshalJSON()
	if err != nil {
		panic(fmt.Sprintf("Failed to encode to json: %v", err))
	}
	return convertToV1Operation(enc)
}

func alphaOperationToV1(op *compute_alpha.Operation) *compute_v1.Operation {
	enc, err := op.MarshalJSON()
	if err != nil {
		panic(fmt.Sprintf("Failed to encode to json: %v", err))
	}
	return convertToV1Operation(enc)
}

func convertToV1Operation(enc []byte) *compute_v1.Operation {
	var op compute_v1.Operation
	if err := json.Unmarshal(enc, &op); err != nil {
		panic(fmt.Sprintf("Failed to convert alpha operation to v1: %v", err))
	}
	return &op
}
