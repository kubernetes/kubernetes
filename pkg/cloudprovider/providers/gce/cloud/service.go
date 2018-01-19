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
func (g *Service) wrapOperation(anyOp interface{}) (operation, error) {
	switch o := anyOp.(type) {
	case *ga.Operation:
		r, err := ParseResourceURL(o.SelfLink)
		if err != nil {
			return nil, err
		}
		return &gaOperation{g, o, r.ProjectID}, nil
	case *alpha.Operation:
		r, err := ParseResourceURL(o.SelfLink)
		if err != nil {
			return nil, err
		}
		return &alphaOperation{g, o, r.ProjectID}, nil
	case *beta.Operation:
		r, err := ParseResourceURL(o.SelfLink)
		if err != nil {
			return nil, err
		}
		return &betaOperation{g, o, r.ProjectID}, nil
	default:
		return nil, fmt.Errorf("invalid type %T", anyOp)
	}
}

// WaitForCompletion of a long running operation. This will poll the state of
// GCE for the completion status of the given operation. genericOp can be one
// of alpha, beta, ga Operation types.
func (g *Service) WaitForCompletion(ctx context.Context, genericOp interface{}) error {
	op, err := g.wrapOperation(genericOp)
	if err != nil {
		return err
	}
	for done, err := op.isDone(ctx); !done; done, err = op.isDone(ctx) {
		if err != nil {
			return err
		}
		g.RateLimiter.Accept(ctx, op.rateLimitKey())
	}
	return nil
}
