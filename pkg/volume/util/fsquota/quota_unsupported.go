//go:build !linux

/*
Copyright 2018 The Kubernetes Authors.

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

package fsquota

import (
	"errors"

	"k8s.io/kubernetes/pkg/volume/util/fsquota/common"
	"k8s.io/mount-utils"

	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/types"
)

// Dummy quota implementation for systems that do not implement support
// for volume quotas

var errNotImplemented = errors.New("not implemented")

func GetQuotaOnDir(_ mount.Interface, _ string) (common.QuotaID, error) {
	return common.BadQuotaID, errNotImplemented
}

// SupportsQuotas -- dummy implementation
func SupportsQuotas(_ mount.Interface, _ string, _ bool) (bool, error) {
	return false, errNotImplemented
}

// AssignQuota -- dummy implementation
func AssignQuota(_ mount.Interface, _ string, _ types.UID, _ *resource.Quantity, _ bool) error {
	return errNotImplemented
}

// GetConsumption -- dummy implementation
func GetConsumption(_ string) (*resource.Quantity, error) {
	return nil, errNotImplemented
}

// GetInodes -- dummy implementation
func GetInodes(_ string) (*resource.Quantity, error) {
	return nil, errNotImplemented
}

// ClearQuota -- dummy implementation
func ClearQuota(_ mount.Interface, _ string, _ bool) error {
	return errNotImplemented
}
