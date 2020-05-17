/*
Copyright 2018 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

https://www.apache.org/licenses/LICENSE-2.0

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

	"k8s.io/klog/v2"

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
		return &gaOperation{s: s, projectID: r.ProjectID, key: r.Key}, nil
	case *alpha.Operation:
		r, err := ParseResourceURL(o.SelfLink)
		if err != nil {
			return nil, err
		}
		return &alphaOperation{s: s, projectID: r.ProjectID, key: r.Key}, nil
	case *beta.Operation:
		r, err := ParseResourceURL(o.SelfLink)
		if err != nil {
			return nil, err
		}
		return &betaOperation{s: s, projectID: r.ProjectID, key: r.Key}, nil
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
		klog.Errorf("wrapOperation(%+v) error: %v", genericOp, err)
		return err
	}

	return s.pollOperation(ctx, op)
}

// pollOperation calls operations.isDone until the function comes back true or context is Done.
// If an error occurs retrieving the operation, the loop will continue until the context is done.
// This is to prevent a transient error from bubbling up to controller-level logic.
func (s *Service) pollOperation(ctx context.Context, op operation) error {
	var pollCount int
	for {
		// Check if context has been cancelled. Note that ctx.Done() must be checked before
		// returning ctx.Err().
		select {
		case <-ctx.Done():
      klog.V(5).Infof("op.pollOperation(%v, %v) not completed, poll count = %d, ctx.Err = %v", ctx, op, pollCount, ctx.Err())
			return ctx.Err()
		default:
			// ctx is not canceled, continue immediately
		}

		pollCount++
		klog.V(5).Infof("op.isDone(%v) waiting; op = %v, poll count = %d", ctx, op, pollCount)
		s.RateLimiter.Accept(ctx, op.rateLimitKey())
		switch done, err := op.isDone(ctx); {
		case err != nil:
			klog.V(5).Infof("op.isDone(%v) error; op = %v, poll count = %d, err = %v, retrying", ctx, op, pollCount, err)
			return err
		case done:
                        klog.V(5).Infof("op.isDone(%v) complete; op = %v, poll count = %d, op.err = %v", ctx, op, pollCount, op.error())
			return op.error()
		}
	}
}
