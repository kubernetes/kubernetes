//go:build linux

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

package common

import (
	"regexp"
)

// QuotaType -- type of quota to be applied
type QuotaType int

const (
	// FSQuotaAccounting for quotas for accounting only
	FSQuotaAccounting QuotaType = 1 << iota
	// FSQuotaEnforcing for quotas for enforcement
	FSQuotaEnforcing QuotaType = 1 << iota
)

// FirstQuota is the quota ID we start with.
// XXXXXXX Need a better way of doing this...
var FirstQuota QuotaID = 1048577

// MountsFile is the location of the system mount data
var MountsFile = "/proc/self/mounts"

// MountParseRegexp parses out /proc/sys/self/mounts
var MountParseRegexp = regexp.MustCompilePOSIX("^([^ ]*)[ \t]*([^ ]*)[ \t]*([^ ]*)") // Ignore options etc.

// LinuxVolumeQuotaProvider returns an appropriate quota applier
// object if we can support quotas on this device
type LinuxVolumeQuotaProvider interface {
	// GetQuotaApplier retrieves an object that can apply
	// quotas (or nil if this provider cannot support quotas
	// on the device)
	GetQuotaApplier(mountpoint string, backingDev string) LinuxVolumeQuotaApplier
}

// LinuxVolumeQuotaApplier is a generic interface to any quota
// mechanism supported by Linux
type LinuxVolumeQuotaApplier interface {
	// GetQuotaOnDir gets the quota ID (if any) that applies to
	// this directory
	GetQuotaOnDir(path string) (QuotaID, error)

	// SetQuotaOnDir applies the specified quota ID to a directory.
	// Negative value for bytes means that a non-enforcing quota
	// should be applied (perhaps by setting a quota too large to
	// be hit)
	SetQuotaOnDir(path string, id QuotaID, bytes int64) error

	// QuotaIDIsInUse determines whether the quota ID is in use.
	// Implementations should not check /etc/project or /etc/projid,
	// only whether their underlying mechanism already has the ID in
	// use.
	// Return value of false with no error means that the ID is not
	// in use; true means that it is already in use.  An error
	// return means that any quota ID will fail.
	QuotaIDIsInUse(id QuotaID) (bool, error)

	// GetConsumption returns the consumption (in bytes) of the
	// directory, determined by the implementation's quota-based
	// mechanism.  If it is unable to do so using that mechanism,
	// it should return an error and allow higher layers to
	// enumerate the directory.
	GetConsumption(path string, id QuotaID) (int64, error)

	// GetInodes returns the number of inodes used by the
	// directory, determined by the implementation's quota-based
	// mechanism.  If it is unable to do so using that mechanism,
	// it should return an error and allow higher layers to
	// enumerate the directory.
	GetInodes(path string, id QuotaID) (int64, error)
}
