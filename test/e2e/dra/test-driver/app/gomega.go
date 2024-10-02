/*
Copyright 2023 The Kubernetes Authors.

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

package app

import (
	"strings"

	"github.com/onsi/gomega/gcustom"
)

// BeRegistered checks that plugin registration has completed.
var BeRegistered = gcustom.MakeMatcher(func(actualCalls []GRPCCall) (bool, error) {
	for _, call := range actualCalls {
		if call.FullMethod == "/pluginregistration.Registration/NotifyRegistrationStatus" &&
			call.Err == nil {
			return true, nil
		}
	}
	return false, nil
}).WithMessage("contain successful NotifyRegistrationStatus call")

// NodePrepareResoucesSucceeded checks that NodePrepareResources API has been called and succeeded
var NodePrepareResourcesSucceeded = gcustom.MakeMatcher(func(actualCalls []GRPCCall) (bool, error) {
	for _, call := range actualCalls {
		if strings.HasSuffix(call.FullMethod, "/NodePrepareResources") && call.Response != nil && call.Err == nil {
			return true, nil
		}
	}
	return false, nil
}).WithMessage("contain successful NodePrepareResources call")

// NodePrepareResoucesFailed checks that NodePrepareResources API has been called and returned an error
var NodePrepareResourcesFailed = gcustom.MakeMatcher(func(actualCalls []GRPCCall) (bool, error) {
	for _, call := range actualCalls {
		if strings.HasSuffix(call.FullMethod, "/NodePrepareResources") && call.Err != nil {
			return true, nil
		}
	}
	return false, nil
}).WithMessage("contain unsuccessful NodePrepareResources call")

// NodeUnprepareResoucesSucceeded checks that NodeUnprepareResources API has been called and succeeded
var NodeUnprepareResourcesSucceeded = gcustom.MakeMatcher(func(actualCalls []GRPCCall) (bool, error) {
	for _, call := range actualCalls {
		if strings.HasSuffix(call.FullMethod, "/NodeUnprepareResources") && call.Response != nil && call.Err == nil {
			return true, nil
		}
	}
	return false, nil
}).WithMessage("contain successful NodeUnprepareResources call")

// NodeUnprepareResoucesFailed checks that NodeUnprepareResources API has been called and returned an error
var NodeUnprepareResourcesFailed = gcustom.MakeMatcher(func(actualCalls []GRPCCall) (bool, error) {
	for _, call := range actualCalls {
		if strings.HasSuffix(call.FullMethod, "/NodeUnprepareResources") && call.Err != nil {
			return true, nil
		}
	}
	return false, nil
}).WithMessage("contain unsuccessful NodeUnprepareResources call")
