// Copyright 2014 The rkt Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//+build linux

package common

import (
	"path/filepath"

	"github.com/appc/spec/schema/types"
	"github.com/coreos/go-systemd/unit"
	"github.com/coreos/rkt/common"
)

const (
	// UnitsDir is the default path to systemd systemd unit directory
	UnitsDir        = "/usr/lib/systemd/system"
	envDir          = "/rkt/env"
	statusDir       = "/rkt/status"
	ioMuxDir        = "/rkt/iottymux"
	defaultWantsDir = UnitsDir + "/default.target.wants"
	socketsWantsDir = UnitsDir + "/sockets.target.wants"
)

// ServiceUnitName returns a systemd service unit name for the given app name.
func ServiceUnitName(appName types.ACName) string {
	return appName.String() + ".service"
}

// ServiceUnitPath returns the path to the systemd service file for the given
// app name.
func ServiceUnitPath(root string, appName types.ACName) string {
	return filepath.Join(common.Stage1RootfsPath(root), UnitsDir, ServiceUnitName(appName))
}

// ServiceUnitPath returns the path to the systemd service file for the given
// app name.
func TargetUnitPath(root string, name string) string {
	return filepath.Join(common.Stage1RootfsPath(root), UnitsDir, name+".target")
}

// RelEnvFilePath returns the path to the environment file for the given
// app name relative to the pod's root.
func RelEnvFilePath(appName types.ACName) string {
	return filepath.Join(envDir, appName.String())
}

// EnvFilePath returns the path to the environment file for the given app name.
func EnvFilePath(root string, appName types.ACName) string {
	return filepath.Join(common.Stage1RootfsPath(root), RelEnvFilePath(appName))
}

// IOMUxFilePath returns the path to the environment file for the given app name.
func IOMuxDir(root string, appName types.ACName) string {
	return filepath.Join(common.Stage1RootfsPath(root), ioMuxDir, appName.String())
}

// ServiceWantPath returns the systemd default.target want symlink path for the
// given app name.
func ServiceWantPath(root string, appName types.ACName) string {
	return filepath.Join(common.Stage1RootfsPath(root), defaultWantsDir, ServiceUnitName(appName))
}

// InstantiatedPrepareAppUnitName returns the systemd service unit name for prepare-app
// instantiated for the given root.
func InstantiatedPrepareAppUnitName(appName types.ACName) string {
	// Naming respecting escaping rules, see systemd.unit(5) and systemd-escape(1)
	escapedRoot := unit.UnitNamePathEscape(common.RelAppRootfsPath(appName))
	return "prepare-app@-" + escapedRoot + ".service"
}

// SocketUnitName returns a systemd socket unit name for the given app name.
func SocketUnitName(appName types.ACName) string {
	return appName.String() + ".socket"
}

// SocketUnitPath returns the path to the systemd socket file for the given app name.
func SocketUnitPath(root string, appName types.ACName) string {
	return filepath.Join(common.Stage1RootfsPath(root), UnitsDir, SocketUnitName(appName))
}

// SocketWantPath returns the systemd sockets.target.wants symlink path for the
// given app name.
func SocketWantPath(root string, appName types.ACName) string {
	return filepath.Join(common.Stage1RootfsPath(root), socketsWantsDir, SocketUnitName(appName))
}

// TypedUnitPath returns the path to a custom-typed unit file
func TypedUnitPath(root string, unitName string, unitType string) string {
	return filepath.Join(common.Stage1RootfsPath(root), UnitsDir, unitName+"."+unitType)
}
