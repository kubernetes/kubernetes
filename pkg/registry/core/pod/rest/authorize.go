/*
Copyright 2025 The Kubernetes Authors.

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

package rest

import (
	"context"
	"errors"
	"fmt"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/apiserver/pkg/endpoints/filters"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
)

// Pod subresources differs on the REST verbs depending on the protocol used
// SPDY uses POST that at the authz layer is translated to "create".
// Websockets uses GET that is translated to "get".
// Since the defaulting to websocket for kubectl in KEP-4006 this caused an
// unexpected side effect and in order to keep existing policies backwards
// compatible we always check that the "create" verb is allowed.
// Ref: https://issues.k8s.io/133515
func ensureAuthorizedForVerb(ctx context.Context, a authorizer.Authorizer, verb string) error {
	requestInfo, ok := genericapirequest.RequestInfoFrom(ctx)
	if !ok {
		return apierrors.NewInternalError(errors.New("no request info in context"))
	}
	if requestInfo.Verb == verb {
		// already authorized
		return nil
	}

	if a == nil {
		return apierrors.NewInternalError(errors.New("no authorizer available"))
	}
	originalAttrs, err := filters.GetAuthorizerAttributes(ctx)
	if err != nil {
		return apierrors.NewInternalError(fmt.Errorf("error building authorizer attributes: %w", err))
	}
	authorized, reason, err := a.Authorize(ctx, &overrideVerb{Attributes: originalAttrs, verb: verb})
	if err != nil {
		return err
	}
	if authorized != authorizer.DecisionAllow {
		return apierrors.NewForbidden(schema.GroupResource{Group: requestInfo.APIGroup, Resource: requestInfo.Resource}, requestInfo.Name, errors.New(reason))
	}
	return nil
}

type overrideVerb struct {
	authorizer.Attributes
	verb string
}

func (o *overrideVerb) GetVerb() string {
	return o.verb
}
