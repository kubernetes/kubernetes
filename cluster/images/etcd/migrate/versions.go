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

package main

import (
	"fmt"
	"strings"

	"github.com/blang/semver"
)

// EtcdVersion specifies an etcd server binaries SemVer.
type EtcdVersion struct {
	semver.Version
}

// ParseEtcdVersion parses a SemVer string to an EtcdVersion.
func ParseEtcdVersion(s string) (*EtcdVersion, error) {
	v, err := semver.Make(s)
	if err != nil {
		return nil, err
	}
	return &EtcdVersion{v}, nil
}

// MustParseEtcdVersion parses a SemVer string to an EtcdVersion and panics if the parse fails.
func MustParseEtcdVersion(s string) *EtcdVersion {
	return &EtcdVersion{semver.MustParse(s)}
}

// String returns the version in SemVer string format.
func (v *EtcdVersion) String() string {
	return v.Version.String()
}

// Equals returns true if the versions are exactly equal.
func (v *EtcdVersion) Equals(o *EtcdVersion) bool {
	return v.Version.Equals(o.Version)
}

// MajorMinorEquals returns true if the major and minor parts of the versions are equal;
// if only patch versions differ, this returns true.
func (v *EtcdVersion) MajorMinorEquals(o *EtcdVersion) bool {
	return v.Major == o.Major && v.Minor == o.Minor
}

// EtcdStorageVersion identifies the storage version of an etcd data directory.
type EtcdStorageVersion int

const (
	storageUnknown EtcdStorageVersion = iota
	storageEtcd2
	storageEtcd3
)

// ParseEtcdStorageVersion parses an etcd storage version string to an EtcdStorageVersion.
func ParseEtcdStorageVersion(s string) (EtcdStorageVersion, error) {
	switch s {
	case "etcd2":
		return storageEtcd2, nil
	case "etcd3":
		return storageEtcd3, nil
	default:
		return storageUnknown, fmt.Errorf("unrecognized storage version: %s", s)
	}
}

// MustParseEtcdStorageVersion parses an etcd storage version string to an EtcdStorageVersion and
// panics if the parse fails.
func MustParseEtcdStorageVersion(s string) EtcdStorageVersion {
	version, err := ParseEtcdStorageVersion(s)
	if err != nil {
		panic(err)
	}
	return version
}

// String returns the text representation of the EtcdStorageVersion, 'etcd2' or 'etcd3'.
func (v EtcdStorageVersion) String() string {
	switch v {
	case storageEtcd2:
		return "etcd2"
	case storageEtcd3:
		return "etcd3"
	default:
		panic(fmt.Sprintf("enum value %d missing from EtcdStorageVersion String() function", v))
	}
}

// EtcdVersionPair is composed of an etcd version and storage version.
type EtcdVersionPair struct {
	version        *EtcdVersion
	storageVersion EtcdStorageVersion
}

// ParseEtcdVersionPair parses a "<version>/<storage-version>" string to an EtcdVersionPair.
func ParseEtcdVersionPair(s string) (*EtcdVersionPair, error) {
	parts := strings.Split(s, "/")
	if len(parts) != 2 {
		return nil, fmt.Errorf("malformed version file, expected <major>.<minor>.<patch>/<storage> but got %s", s)
	}
	version, err := ParseEtcdVersion(parts[0])
	if err != nil {
		return nil, err
	}
	storageVersion, err := ParseEtcdStorageVersion(parts[1])
	if err != nil {
		return nil, err
	}
	return &EtcdVersionPair{version, storageVersion}, nil
}

// MustParseEtcdVersionPair parses a "<version>/<storage-version>" string to an EtcdVersionPair
// or panics if the parse fails.
func MustParseEtcdVersionPair(s string) *EtcdVersionPair {
	pair, err := ParseEtcdVersionPair(s)
	if err != nil {
		panic(err)
	}
	return pair
}

// String returns "<version>/<storage-version>" string of the EtcdVersionPair.
func (vp *EtcdVersionPair) String() string {
	return fmt.Sprintf("%s/%s", vp.version, vp.storageVersion)
}

// Equals returns true if both the versions and storage versions are exactly equal.
func (vp *EtcdVersionPair) Equals(o *EtcdVersionPair) bool {
	return vp.version.Equals(o.version) && vp.storageVersion == o.storageVersion
}

// SupportedVersions provides a list of etcd versions that are "supported" for some purpose.
// The list must be sorted from lowest semantic version to high.
type SupportedVersions []*EtcdVersion

// NextVersion returns the next supported version after the given current version, or nil if no
// next version exists.
func (sv SupportedVersions) NextVersion(current *EtcdVersion) *EtcdVersion {
	var nextVersion *EtcdVersion
	for i, supportedVersion := range sv {
		if current.MajorMinorEquals(supportedVersion) && len(sv) > i+1 {
			nextVersion = sv[i+1]
		}
	}
	return nextVersion
}

// NextVersionPair returns the next supported version after the given current version and infers
// the storage version from the major version part of the next version.
func (sv SupportedVersions) NextVersionPair(current *EtcdVersionPair) *EtcdVersionPair {
	nextVersion := sv.NextVersion(current.version)
	if nextVersion == nil {
		return nil
	}
	storageVersion := storageEtcd3
	if nextVersion.Major == 2 {
		storageVersion = storageEtcd2
	}
	return &EtcdVersionPair{version: nextVersion, storageVersion: storageVersion}
}

// ParseSupportedVersions parses a comma separated list of etcd versions.
func ParseSupportedVersions(s string) (SupportedVersions, error) {
	var err error
	list := strings.Split(s, ",")
	versions := make(SupportedVersions, len(list))
	for i, v := range list {
		versions[i], err = ParseEtcdVersion(strings.TrimSpace(v))
		if err != nil {
			return nil, err
		}
	}
	return versions, nil
}

// MustParseSupportedVersions parses a comma separated list of etcd versions or panics if the parse fails.
func MustParseSupportedVersions(s string) SupportedVersions {
	versions, err := ParseSupportedVersions(s)
	if err != nil {
		panic(err)
	}
	return versions
}
