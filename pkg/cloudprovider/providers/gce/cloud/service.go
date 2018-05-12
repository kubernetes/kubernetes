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
)

// Service is the top-level adapter for all of the different compute API
// versions.
type Service struct {
	GA            *ga.Service
	Alpha         *alpha.Service
	Beta          *beta.Service
	ProjectRouter ProjectRouter
	RateLimiter   RateLimiter
}

// wrapOperation wraps a GCE anyOP in a version generic operation type.
func (s *Service) wrapOperation(anyOp interface{}) (operation, error) {
	switch o := anyOp.(type) {
	case *ga.Operation:
		r, err := ParseResourceURL(o.SelfLink)
		if err != nil {
			return nil, err
		}
		return &gaOperation{s, r.ProjectID, r.Key}, nil
	case *alpha.Operation:
		r, err := ParseResourceURL(o.SelfLink)
		if err != nil {
			return nil, err
		}
		return &alphaOperation{s, r.ProjectID, r.Key}, nil
	case *beta.Operation:
		r, err := ParseResourceURL(o.SelfLink)
		if err != nil {
			return nil, err
		}
		return &betaOperation{s, r.ProjectID, r.Key}, nil
	default:
		return nil, fmt.Errorf("invalid type %T", anyOp)
	}
}

// WaitForCompletion of a long running operation. This will poll the state of
// GCE for the completion status of the given operation. genericOp can be one
// of alpha, beta, ga Operation types.
func (s *Service) WaitForCompletion(ctx context.Context, genericOp interface{}) error {
	op, err := s.wrapOperation(genericOp)
	if err != nil {
		glog.Errorf("wrapOperation(%+v) error: %v", genericOp, err)
		return err
	}
	for done, err := op.isDone(ctx); !done; done, err = op.isDone(ctx) {
		if err != nil {
			glog.V(4).Infof("op.isDone(%v) error; op = %v, err = %v", ctx, op, err)
			return err
		}
		glog.V(5).Infof("op.isDone(%v) waiting; op = %v", ctx, op)
		s.RateLimiter.Accept(ctx, op.rateLimitKey())
	}
	glog.V(5).Infof("op.isDone(%v) complete; op = %v", ctx, op)
	return nil
}
