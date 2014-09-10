/*
Copyright 2014 Google Inc. All rights reserved.

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

package gce_cloud

import (
	"errors"
	"strings"
	"time"

	compute "code.google.com/p/google-api-go-client/compute/v1"
	"github.com/golang/glog"
)

type opScope int

const (
	REGION opScope = iota
	ZONE
	GLOBAL
)

// GCEOp is an abstraction of GCE's compute.Operation, providing only necessary generic information.
type GCEOp struct {
	op    *compute.Operation
	scope opScope
}

// Status provides the status of the operation, and will be either "PENDING", "RUNNING", or "DONE".
func (op *GCEOp) Status() string {
	return op.op.Status
}

// Errors provides any errors that the operation encountered.
func (op *GCEOp) Errors() []string {
	var errors []string
	if op.op.Error != nil {
		for _, err := range op.op.Error.Errors {
			errors = append(errors, err.Message)
		}
	}
	return errors
}

// OperationType provides the type of operation represented by op.
func (op *GCEOp) OperationType() string {
	return op.op.OperationType
}

// TargetName provides the name of the affected resource.
func (op *GCEOp) TargetName() string {
	target, _ := targetInfo(op.op.TargetLink)
	return target
}

// Resource provides the type of the affected resource.
func (op *GCEOp) Resource() string {
	_, resource := targetInfo(op.op.TargetLink)
	return resource
}

func targetInfo(targetLink string) (target, resourceType string) {
	i := strings.LastIndex(targetLink, "/")
	target = targetLink[i+1:]
	j := strings.LastIndex(targetLink[:i], "/")
	resourceType = targetLink[j+1 : i-1]
	return target, resourceType
}

func (gce *GCECloud) pollOp(op *GCEOp) (*GCEOp, error) {
	var err error
	switch op.scope {
	case REGION:
		region := op.op.Region[strings.LastIndex(op.op.Region, "/")+1:]
		op.op, err = gce.service.RegionOperations.Get(gce.projectID, region, op.op.Name).Do()
	case ZONE:
		zone := op.op.Zone[strings.LastIndex(op.op.Zone, "/")+1:]
		op.op, err = gce.service.ZoneOperations.Get(gce.projectID, zone, op.op.Name).Do()
	case GLOBAL:
		op.op, err = gce.service.GlobalOperations.Get(gce.projectID, op.op.Name).Do()
	default:
		err = errors.New("unknown operation scope")
	}
	return op, err
}

// waitForOps polls the status of the specified operations until they are all "DONE"
func (gce *GCECloud) waitForOps(ops []*GCEOp) error {
	// Wait for all operations to complete
	for _, op := range ops {
		op, err := gce.pollOp(op)
		if err != nil {
			return err
		}
		for op.Status() != "DONE" {
			glog.Infof("Waiting 2s for %v of %v %v\n", op.OperationType(), op.Resource(), op.TargetName())
			time.Sleep(2 * time.Second)
			op, err = gce.pollOp(op)
			if err != nil {
				return err
			}
		}
		if op.Errors() != nil {
			return errors.New("errors in operation:\n" + strings.Join(op.Errors(), "\n"))
		}
		glog.Infof("%v of %v %v has completed\n", op.OperationType(), op.Resource(), op.TargetName())
	}
	return nil
}
