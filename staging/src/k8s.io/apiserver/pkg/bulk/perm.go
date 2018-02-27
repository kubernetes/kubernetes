/*
Copyright 2014 The Kubernetes Authors.

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

package bulk

import (
	"errors"
	"time"

	"github.com/golang/glog"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime/schema"
	bulkapi "k8s.io/apiserver/pkg/apis/bulk"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/apiserver/pkg/endpoints/request"
)

// Checks Authorization / Admission
type permCheckFunc func() (err error)

type authorizationCheckerFactory struct {
	GroupInfo *LocalAPIGroupInfo
	Context   request.Context
	Resource  bulkapi.GroupVersionResource
	Name      string
	Verb      string
}

func (a authorizationCheckerFactory) newForbiddenError(reason string) error {
	return apierrors.NewForbidden(
		schema.GroupResource{Group: a.Resource.Group, Resource: a.Resource.Resource}, a.Name, errors.New(reason))
}

func (a authorizationCheckerFactory) checkAuthorization() error {
	return a.makeAuthorizationChecker()()
}

func (a authorizationCheckerFactory) makeAuthorizationChecker() permCheckFunc {
	glog.V(8).Infof("make permission checker for %v", a)
	auth := a.GroupInfo.Authorizer
	permRecheck := a.GroupInfo.AuthroizationCachingPeriod
	if auth == nil {
		// Authorization is disabled.
		return func() error { return nil }
	}

	var attribs authorizer.AttributesRecord
	if user, ok := request.UserFrom(a.Context); ok {
		attribs.User = user
	}
	attribs.APIGroup = a.Resource.Group
	attribs.APIVersion = a.Resource.Version
	attribs.Namespace = a.Resource.Namespace
	attribs.Resource = a.Resource.Resource
	attribs.Name = a.Name
	attribs.ResourceRequest = true
	attribs.Subresource = ""
	attribs.Verb = a.Verb
	// attribs.Path = generateSelfLink(s) // FIXME

	var lastCheckAt *time.Time
	var lastResult error

	return func() error {
		now := time.Now()
		if lastCheckAt == nil || lastCheckAt.Add(permRecheck).Before(now) {
			lastCheckAt = &now
			decision, reason, err := auth.Authorize(attribs)
			if err != nil {
				lastResult = err
			} else if decision == authorizer.DecisionAllow {
				lastResult = nil
			} else {
				lastResult = a.newForbiddenError(reason)
			}
		}
		return lastResult
	}
}
