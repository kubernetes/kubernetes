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

package cloud

import (
	"context"
	"fmt"

	"github.com/golang/glog"

	alpha "google.golang.org/api/compute/v0.alpha"
	beta "google.golang.org/api/compute/v0.beta"
	ga "google.golang.org/api/compute/v1"
	"google.golang.org/api/googleapi"

	"k8s.io/kubernetes/pkg/cloudprovider/providers/gce/cloud/meta"
)

const (
	operationStatusDone = "DONE"
)

// operation is a GCE operation that can be watied on.
type operation interface {
	// isDone queries GCE for the done status. This call can block.
	isDone(ctx context.Context) (bool, error)
	// error returns the resulting error of the operation. This may be nil if the operations
	// was successful.
	error() error
	// rateLimitKey returns the rate limit key to use for the given operation.
	// This rate limit will govern how fast the server will be polled for
	// operation completion status.
	rateLimitKey() *RateLimitKey
}

type gaOperation struct {
	s         *Service
	projectID string
	key       *meta.Key
	err       error
}

func (o *gaOperation) String() string {
	return fmt.Sprintf("gaOperation{%q, %v}", o.projectID, o.key)
}

func (o *gaOperation) isDone(ctx context.Context) (bool, error) {
	var (
		op  *ga.Operation
		err error
	)

	switch o.key.Type() {
	case meta.Regional:
		op, err = o.s.GA.RegionOperations.Get(o.projectID, o.key.Region, o.key.Name).Context(ctx).Do()
		glog.V(5).Infof("GA.RegionOperations.Get(%v, %v, %v) = %+v, %v; ctx = %v", o.projectID, o.key.Region, o.key.Name, op, err, ctx)
	case meta.Zonal:
		op, err = o.s.GA.ZoneOperations.Get(o.projectID, o.key.Zone, o.key.Name).Context(ctx).Do()
		glog.V(5).Infof("GA.ZoneOperations.Get(%v, %v, %v) = %+v, %v; ctx = %v", o.projectID, o.key.Zone, o.key.Name, op, err, ctx)
	case meta.Global:
		op, err = o.s.GA.GlobalOperations.Get(o.projectID, o.key.Name).Context(ctx).Do()
		glog.V(5).Infof("GA.GlobalOperations.Get(%v, %v) = %+v, %v; ctx = %v", o.projectID, o.key.Name, op, err, ctx)
	default:
		return false, fmt.Errorf("invalid key type: %#v", o.key)
	}
	if err != nil {
		return false, err
	}
	if op == nil || op.Status != operationStatusDone {
		return false, nil
	}

	if op.Error != nil && len(op.Error.Errors) > 0 && op.Error.Errors[0] != nil {
		e := op.Error.Errors[0]
		o.err = &googleapi.Error{Code: int(op.HttpErrorStatusCode), Message: fmt.Sprintf("%v - %v", e.Code, e.Message)}
	}
	return true, nil
}

func (o *gaOperation) rateLimitKey() *RateLimitKey {
	return &RateLimitKey{
		ProjectID: o.projectID,
		Operation: "Get",
		Service:   "Operations",
		Version:   meta.VersionGA,
	}
}

func (o *gaOperation) error() error {
	return o.err
}

type alphaOperation struct {
	s         *Service
	projectID string
	key       *meta.Key
	err       error
}

func (o *alphaOperation) String() string {
	return fmt.Sprintf("alphaOperation{%q, %v}", o.projectID, o.key)
}

func (o *alphaOperation) isDone(ctx context.Context) (bool, error) {
	var (
		op  *alpha.Operation
		err error
	)

	switch o.key.Type() {
	case meta.Regional:
		op, err = o.s.Alpha.RegionOperations.Get(o.projectID, o.key.Region, o.key.Name).Context(ctx).Do()
		glog.V(5).Infof("Alpha.RegionOperations.Get(%v, %v, %v) = %+v, %v; ctx = %v", o.projectID, o.key.Region, o.key.Name, op, err, ctx)
	case meta.Zonal:
		op, err = o.s.Alpha.ZoneOperations.Get(o.projectID, o.key.Zone, o.key.Name).Context(ctx).Do()
		glog.V(5).Infof("Alpha.ZoneOperations.Get(%v, %v, %v) = %+v, %v; ctx = %v", o.projectID, o.key.Zone, o.key.Name, op, err, ctx)
	case meta.Global:
		op, err = o.s.Alpha.GlobalOperations.Get(o.projectID, o.key.Name).Context(ctx).Do()
		glog.V(5).Infof("Alpha.GlobalOperations.Get(%v, %v) = %+v, %v; ctx = %v", o.projectID, o.key.Name, op, err, ctx)
	default:
		return false, fmt.Errorf("invalid key type: %#v", o.key)
	}
	if err != nil {
		return false, err
	}
	if op == nil || op.Status != operationStatusDone {
		return false, nil
	}

	if op.Error != nil && len(op.Error.Errors) > 0 && op.Error.Errors[0] != nil {
		e := op.Error.Errors[0]
		o.err = &googleapi.Error{Code: int(op.HttpErrorStatusCode), Message: fmt.Sprintf("%v - %v", e.Code, e.Message)}
	}
	return true, nil
}

func (o *alphaOperation) rateLimitKey() *RateLimitKey {
	return &RateLimitKey{
		ProjectID: o.projectID,
		Operation: "Get",
		Service:   "Operations",
		Version:   meta.VersionAlpha,
	}
}

func (o *alphaOperation) error() error {
	return o.err
}

type betaOperation struct {
	s         *Service
	projectID string
	key       *meta.Key
	err       error
}

func (o *betaOperation) String() string {
	return fmt.Sprintf("betaOperation{%q, %v}", o.projectID, o.key)
}

func (o *betaOperation) isDone(ctx context.Context) (bool, error) {
	var (
		op  *beta.Operation
		err error
	)

	switch o.key.Type() {
	case meta.Regional:
		op, err = o.s.Beta.RegionOperations.Get(o.projectID, o.key.Region, o.key.Name).Context(ctx).Do()
		glog.V(5).Infof("Beta.RegionOperations.Get(%v, %v, %v) = %+v, %v; ctx = %v", o.projectID, o.key.Region, o.key.Name, op, err, ctx)
	case meta.Zonal:
		op, err = o.s.Beta.ZoneOperations.Get(o.projectID, o.key.Zone, o.key.Name).Context(ctx).Do()
		glog.V(5).Infof("Beta.ZoneOperations.Get(%v, %v, %v) = %+v, %v; ctx = %v", o.projectID, o.key.Zone, o.key.Name, op, err, ctx)
	case meta.Global:
		op, err = o.s.Beta.GlobalOperations.Get(o.projectID, o.key.Name).Context(ctx).Do()
		glog.V(5).Infof("Beta.GlobalOperations.Get(%v, %v) = %+v, %v; ctx = %v", o.projectID, o.key.Name, op, err, ctx)
	default:
		return false, fmt.Errorf("invalid key type: %#v", o.key)
	}
	if err != nil {
		return false, err
	}
	if op == nil || op.Status != operationStatusDone {
		return false, nil
	}

	if op.Error != nil && len(op.Error.Errors) > 0 && op.Error.Errors[0] != nil {
		e := op.Error.Errors[0]
		o.err = &googleapi.Error{Code: int(op.HttpErrorStatusCode), Message: fmt.Sprintf("%v - %v", e.Code, e.Message)}
	}
	return true, nil
}

func (o *betaOperation) rateLimitKey() *RateLimitKey {
	return &RateLimitKey{
		ProjectID: o.projectID,
		Operation: "Get",
		Service:   "Operations",
		Version:   meta.VersionBeta,
	}
}

func (o *betaOperation) error() error {
	return o.err
}
