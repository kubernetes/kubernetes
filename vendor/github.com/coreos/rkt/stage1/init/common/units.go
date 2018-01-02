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
	"errors"
	"fmt"
	"io"
	"os"
	"path"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/appc/spec/schema"
	"github.com/appc/spec/schema/types"
	"github.com/coreos/rkt/common"
	"github.com/coreos/rkt/pkg/user"
	stage1commontypes "github.com/coreos/rkt/stage1/common/types"

	"github.com/coreos/go-systemd/unit"
	"github.com/hashicorp/errwrap"
)

// The maximum value for the MilliValue of an appc resource limit.
const MaxMilliValue = int64(((1 << 63) - 1) / 1000)

func MutableEnv(p *stage1commontypes.Pod) error {
	w := NewUnitWriter(p)

	w.WriteUnit(
		TargetUnitPath(p.Root, "default"),
		"failed to write default.target",
		unit.NewUnitOption("Unit", "Description", "rkt apps target"),
		unit.NewUnitOption("Unit", "DefaultDependencies", "false"),
		unit.NewUnitOption("Unit", "Requires", "systemd-journald.service"),
		unit.NewUnitOption("Unit", "After", "systemd-journald.service"),
		unit.NewUnitOption("Unit", "Wants", "supervisor-ready.service"),
		unit.NewUnitOption("Unit", "Before", "supervisor-ready.service"),
		unit.NewUnitOption("Unit", "Before", "halt.target"),
		unit.NewUnitOption("Unit", "Conflicts", "halt.target"),
	)

	w.WriteUnit(
		ServiceUnitPath(p.Root, "prepare-app@"),
		"failed to write prepare-app service template",
		unit.NewUnitOption("Unit", "Description", "Prepare minimum environment for chrooted applications"),
		unit.NewUnitOption("Unit", "DefaultDependencies", "false"),
		unit.NewUnitOption("Unit", "OnFailureJobMode", "fail"),

		// prepare-app is meant to be executed at most once.
		// We must ensure that the prepare-app service unit remains started after the prepare-app binary exits
		// such that it is not executed again during restarts of the target app.
		unit.NewUnitOption("Service", "RemainAfterExit", "yes"),

		unit.NewUnitOption("Service", "Type", "oneshot"),
		unit.NewUnitOption("Service", "Restart", "no"),
		unit.NewUnitOption("Service", "ExecStart", "/prepare-app %I"),
		unit.NewUnitOption("Service", "User", "0"),
		unit.NewUnitOption("Service", "Group", "0"),
		unit.NewUnitOption("Service", "CapabilityBoundingSet", "CAP_SYS_ADMIN CAP_DAC_OVERRIDE CAP_MKNOD"),
	)

	w.WriteUnit(
		TargetUnitPath(p.Root, "halt"),
		"failed to write halt target",
		unit.NewUnitOption("Unit", "Description", "Halt"),
		unit.NewUnitOption("Unit", "DefaultDependencies", "false"),
		unit.NewUnitOption("Unit", "AllowIsolate", "true"),
		unit.NewUnitOption("Unit", "Requires", "shutdown.service"),
		unit.NewUnitOption("Unit", "After", "shutdown.service"),
	)

	w.writeShutdownService(
		"ExecStart",
		unit.NewUnitOption("Unit", "Description", "Pod shutdown"),
		unit.NewUnitOption("Unit", "AllowIsolate", "true"),
		unit.NewUnitOption("Unit", "DefaultDependencies", "false"),
		unit.NewUnitOption("Service", "RemainAfterExit", "yes"),
	)

	return w.Error()
}

func ImmutableEnv(p *stage1commontypes.Pod) error {
	uw := NewUnitWriter(p)

	opts := []*unit.UnitOption{
		unit.NewUnitOption("Unit", "Description", "rkt apps target"),
		unit.NewUnitOption("Unit", "DefaultDependencies", "false"),
		unit.NewUnitOption("Unit", "Wants", "supervisor-ready.service"),
		unit.NewUnitOption("Unit", "Before", "supervisor-ready.service"),
	}

	for i := range p.Manifest.Apps {
		ra := &p.Manifest.Apps[i]
		serviceName := ServiceUnitName(ra.Name)
		opts = append(opts, unit.NewUnitOption("Unit", "After", serviceName))
		opts = append(opts, unit.NewUnitOption("Unit", "Wants", serviceName))
	}

	uw.WriteUnit(
		TargetUnitPath(p.Root, "default"),
		"failed to write default.target",
		opts...,
	)

	uw.WriteUnit(
		ServiceUnitPath(p.Root, "prepare-app@"),
		"failed to write prepare-app service template",
		unit.NewUnitOption("Unit", "Description", "Prepare minimum environment for chrooted applications"),
		unit.NewUnitOption("Unit", "DefaultDependencies", "false"),
		unit.NewUnitOption("Unit", "OnFailureJobMode", "fail"),
		unit.NewUnitOption("Unit", "Requires", "systemd-journald.service"),
		unit.NewUnitOption("Unit", "After", "systemd-journald.service"),
		unit.NewUnitOption("Service", "Type", "oneshot"),
		unit.NewUnitOption("Service", "Restart", "no"),
		unit.NewUnitOption("Service", "ExecStart", "/prepare-app %I"),
		unit.NewUnitOption("Service", "User", "0"),
		unit.NewUnitOption("Service", "Group", "0"),
		unit.NewUnitOption("Service", "CapabilityBoundingSet", "CAP_SYS_ADMIN CAP_DAC_OVERRIDE CAP_MKNOD"),
	)

	uw.WriteUnit(
		TargetUnitPath(p.Root, "halt"),
		"failed to write halt target",
		unit.NewUnitOption("Unit", "Description", "Halt"),
		unit.NewUnitOption("Unit", "DefaultDependencies", "false"),
		unit.NewUnitOption("Unit", "AllowIsolate", "true"),
	)

	uw.writeShutdownService(
		"ExecStop",
		unit.NewUnitOption("Unit", "Description", "Pod shutdown"),
		unit.NewUnitOption("Unit", "AllowIsolate", "true"),
		unit.NewUnitOption("Unit", "StopWhenUnneeded", "yes"),
		unit.NewUnitOption("Unit", "DefaultDependencies", "false"),
		unit.NewUnitOption("Service", "RemainAfterExit", "yes"),
	)

	if err := uw.Error(); err != nil {
		return err
	}

	for i := range p.Manifest.Apps {
		ra := &p.Manifest.Apps[i]

		if ra.App.WorkingDirectory == "" {
			ra.App.WorkingDirectory = "/"
		}

		binPath, err := FindBinPath(p, ra)
		if err != nil {
			return err
		}

		uw.AppUnit(ra, binPath,
			// When an app fails, we shut down the pod
			unit.NewUnitOption("Unit", "OnFailure", "halt.target"))

		uw.AppReaperUnit(ra.Name, binPath,
			unit.NewUnitOption("Service", "Environment", `"EXIT_POD=true"`),
			unit.NewUnitOption("Unit", "Wants", "shutdown.service"),
			unit.NewUnitOption("Unit", "After", "shutdown.service"),
		)
	}

	return uw.Error()
}

// SetupAppIO prepares all properties related to streams (stdin/stdout/stderr) and TTY
// for an application service unit.
//
// It works according to the following steps:
//  1. short-circuit interactive pods and legacy systemd, for backward compatibility
//  2. parse app-level annotations to determine stdin/stdout/stderr mode
//     2a. if an annotation is missing/invalid, it fallbacks to legacy mode (in: null, out/err: journald)
//     2b. if a valid annotation is found, it prepares:
//          - TTY and stream properties for the systemd service unit
//          - env variables for iottymux binary
//  3. if any of stdin/stdout/stderr is in TTY or streaming mode:
//     3a. the env file for iottymux is written to `/rkt/iottymux/<appname>/env` with the above content
//     3b. for TTY mode, a `TTYPath` property and an `After=ttymux@<appname>.service` dependency are added
//     3c. for streaming mode, a `Before=iomux@<appname>.service` dependency is added
//
// For complete details, see dev-docs at Documentation/devel/log-attach-design.md
func (uw *UnitWriter) SetupAppIO(p *stage1commontypes.Pod, ra *schema.RuntimeApp, binPath string, opts ...*unit.UnitOption) []*unit.UnitOption {
	if uw.err != nil {
		return opts
	}

	if p.Interactive {
		opts = append(opts, unit.NewUnitOption("Service", "StandardInput", "tty"))
		opts = append(opts, unit.NewUnitOption("Service", "StandardOutput", "tty"))
		opts = append(opts, unit.NewUnitOption("Service", "StandardError", "tty"))
		return opts
	}

	flavor, systemdVersion, err := GetFlavor(uw.p)
	if err != nil {
		uw.err = err
		return opts
	}

	stdin, _ := ra.Annotations.Get(stage1commontypes.AppStdinMode)
	stdout, _ := ra.Annotations.Get(stage1commontypes.AppStdoutMode)
	stderr, _ := ra.Annotations.Get(stage1commontypes.AppStderrMode)

	// Attach needs https://github.com/systemd/systemd/pull/4179, ie. systemd v232 or a backport
	if ((flavor == "src" || flavor == "host") && systemdVersion < 232) ||
		((flavor == "coreos" || flavor == "kvm") && systemdVersion < 231) {
		// Explicit error if systemd is too old for attaching
		if stdin != "" || stdout != "" || stderr != "" {
			uw.err = fmt.Errorf("stage1 systemd %q does not support attachable I/O", systemdVersion)
			return opts
		}
		opts = append(opts, unit.NewUnitOption("Service", "StandardInput", "null"))
		opts = append(opts, unit.NewUnitOption("Service", "StandardOutput", "journal+console"))
		opts = append(opts, unit.NewUnitOption("Service", "StandardError", "journal+console"))
		return opts
	}

	var iottymuxEnvFlags []string
	needsIOMux := false
	needsTTYMux := false

	switch stdin {
	case "stream":
		needsIOMux = true
		uw.AppSocketUnit(ra.Name, binPath, "stdin")
		iottymuxEnvFlags = append(iottymuxEnvFlags, "STAGE2_STDIN=true")
		opts = append(opts, unit.NewUnitOption("Service", "StandardInput", "fd"))
		opts = append(opts, unit.NewUnitOption("Service", "Sockets", fmt.Sprintf("%s-%s.socket", ra.Name, "stdin")))
	case "tty":
		needsTTYMux = true
		iottymuxEnvFlags = append(iottymuxEnvFlags, "STAGE2_STDIN=true")
		opts = append(opts, unit.NewUnitOption("Service", "StandardInput", "tty-force"))
	case "interactive":
		opts = append(opts, unit.NewUnitOption("Service", "StandardInput", "tty"))
	default:
		// null mode
		opts = append(opts, unit.NewUnitOption("Service", "StandardInput", "null"))
	}

	switch stdout {
	case "stream":
		needsIOMux = true
		uw.AppSocketUnit(ra.Name, binPath, "stdout")
		iottymuxEnvFlags = append(iottymuxEnvFlags, "STAGE2_STDOUT=true")
		opts = append(opts, unit.NewUnitOption("Service", "StandardOutput", "fd"))
		opts = append(opts, unit.NewUnitOption("Service", "Sockets", fmt.Sprintf("%s-%s.socket", ra.Name, "stdout")))
	case "tty":
		needsTTYMux = true
		iottymuxEnvFlags = append(iottymuxEnvFlags, "STAGE2_STDOUT=true")
		opts = append(opts, unit.NewUnitOption("Service", "StandardOutput", "tty"))
	case "interactive":
		opts = append(opts, unit.NewUnitOption("Service", "StandardOutput", "tty"))
	case "null":
		opts = append(opts, unit.NewUnitOption("Service", "StandardOutput", "null"))
	default:
		// log mode
		opts = append(opts, unit.NewUnitOption("Service", "StandardOutput", "journal+console"))
	}

	switch stderr {
	case "stream":
		needsIOMux = true
		uw.AppSocketUnit(ra.Name, binPath, "stderr")
		iottymuxEnvFlags = append(iottymuxEnvFlags, "STAGE2_STDERR=true")
		opts = append(opts, unit.NewUnitOption("Service", "StandardError", "fd"))
		opts = append(opts, unit.NewUnitOption("Service", "Sockets", fmt.Sprintf("%s-%s.socket", ra.Name, "stderr")))
	case "tty":
		needsTTYMux = true
		iottymuxEnvFlags = append(iottymuxEnvFlags, "STAGE2_STDERR=true")
		opts = append(opts, unit.NewUnitOption("Service", "StandardError", "tty"))
	case "interactive":
		opts = append(opts, unit.NewUnitOption("Service", "StandardError", "tty"))
	case "null":
		opts = append(opts, unit.NewUnitOption("Service", "StandardError", "null"))
	default:
		// log mode
		opts = append(opts, unit.NewUnitOption("Service", "StandardError", "journal+console"))
	}

	// if at least one stream requires I/O muxing, an appropriate iottymux dependency needs to be setup
	if needsIOMux || needsTTYMux {
		// an env file is written here for iottymux, containing service configuration.
		appIODir := IOMuxDir(p.Root, ra.Name)
		os.MkdirAll(appIODir, 0644)
		file, err := os.OpenFile(filepath.Join(appIODir, "env"), os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0644)
		if err != nil {
			uw.err = err
			return nil
		}
		defer file.Close()

		// env file specifies: debug verbosity, which streams to mux and whether a dedicated TTY is needed.
		file.WriteString(fmt.Sprintf("STAGE2_TTY=%t\n", needsTTYMux))
		file.WriteString(fmt.Sprintf("STAGE1_DEBUG=%t\n", p.Debug))
		for _, l := range iottymuxEnvFlags {
			file.WriteString(l + "\n")
		}

		if needsIOMux {
			// streaming mode brings in a `iomux@.service` before-dependency
			opts = append(opts, unit.NewUnitOption("Unit", "Requires", fmt.Sprintf("iomux@%s.service", ra.Name)))
			opts = append(opts, unit.NewUnitOption("Unit", "Before", fmt.Sprintf("iomux@%s.service", ra.Name)))
			logMode, ok := p.Manifest.Annotations.Get("coreos.com/rkt/experiment/logmode")
			if ok {
				file.WriteString(fmt.Sprintf("STAGE1_LOGMODE=%s\n", logMode))
			}
		} else if needsTTYMux {
			// tty mode brings in a `ttymux@.service` after-dependency (it needs to create the TTY first)
			opts = append(opts, unit.NewUnitOption("Service", "TTYPath", filepath.Join("/rkt/iottymux", ra.Name.String(), "stage2-pts")))
			opts = append(opts, unit.NewUnitOption("Unit", "Requires", fmt.Sprintf("ttymux@%s.service", ra.Name)))
			opts = append(opts, unit.NewUnitOption("Unit", "After", fmt.Sprintf("ttymux@%s.service", ra.Name)))
		}
	}
	return opts
}

// UnitWriter is the type that writes systemd units preserving the first previously occured error.
// Any method of this type can be invoked multiple times without error checking.
// If a previous invocation generated an error, any invoked method will be skipped.
// If an error occured during method invocations, it can be retrieved using Error().
type UnitWriter struct {
	err error
	p   *stage1commontypes.Pod
}

// NewUnitWriter returns a new UnitWriter for the given pod.
func NewUnitWriter(p *stage1commontypes.Pod) *UnitWriter {
	return &UnitWriter{p: p}
}

// WriteUnit writes a systemd unit in the given path with the given unit options
// if no previous error occured.
func (uw *UnitWriter) WriteUnit(path string, errmsg string, opts ...*unit.UnitOption) {
	if uw.err != nil {
		return
	}

	file, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0644)
	if err != nil {
		uw.err = errwrap.Wrap(errors.New(errmsg), err)
		return
	}
	defer file.Close()

	if _, err = io.Copy(file, unit.Serialize(opts)); err != nil {
		uw.err = errwrap.Wrap(errors.New(errmsg), err)
		return
	}
	if err := user.ShiftFiles([]string{path}, &uw.p.UidRange); err != nil {
		uw.err = errwrap.Wrap(errors.New(errmsg), err)
		return
	}
}

// writeShutdownService writes a shutdown.service unit with the given unit options
// if no previous error occured.
// exec specifies how systemctl should be invoked, i.e. ExecStart, or ExecStop.
func (uw *UnitWriter) writeShutdownService(exec string, opts ...*unit.UnitOption) {
	if uw.err != nil {
		return
	}

	flavor, systemdVersion, err := GetFlavor(uw.p)
	if err != nil {
		uw.err = errwrap.Wrap(errors.New("failed to create shutdown service"), err)
		return
	}

	opts = append(opts, []*unit.UnitOption{
		// The default stdout is /dev/console (the tty created by nspawn).
		// But the tty might be destroyed if rkt is executed via ssh and
		// the user terminates the ssh session. We still want
		// shutdown.service to succeed in that case, so don't use
		// /dev/console.
		unit.NewUnitOption("Service", "StandardInput", "null"),
		unit.NewUnitOption("Service", "StandardOutput", "null"),
		unit.NewUnitOption("Service", "StandardError", "null"),
	}...)

	shutdownVerb := "exit"
	// systemd <v227 doesn't allow the "exit" verb when running as PID 1, so
	// use "halt".
	// If systemdVersion is 0 it means it couldn't be guessed, assume it's new
	// enough for "systemctl exit".
	// This can happen, for example, when building rkt with:
	//
	// ./configure --with-stage1-flavors=src --with-stage1-systemd-version=master
	//
	// The patches for the "exit" verb are backported to the "coreos" flavor, so
	// don't rely on the systemd version on the "coreos" flavor.
	if flavor != "coreos" && systemdVersion != 0 && systemdVersion < 227 {
		shutdownVerb = "halt"
	}

	opts = append(
		opts,
		unit.NewUnitOption("Service", exec, fmt.Sprintf("/usr/bin/systemctl --force %s", shutdownVerb)),
	)

	uw.WriteUnit(
		ServiceUnitPath(uw.p.Root, "shutdown"),
		"failed to create shutdown service",
		opts...,
	)
}

// Activate actives the given unit in the given wantPath.
func (uw *UnitWriter) Activate(unit, wantPath string) {
	if uw.err != nil {
		return
	}

	if err := os.Symlink(path.Join("..", unit), wantPath); err != nil && !os.IsExist(err) {
		uw.err = errwrap.Wrap(errors.New("failed to link service want"), err)
	}
}

// error returns the first error that occured during write* invocations.
func (uw *UnitWriter) Error() error {
	return uw.err
}

// AppUnit sets up the main systemd service unit for the application.
func (uw *UnitWriter) AppUnit(ra *schema.RuntimeApp, binPath string, opts ...*unit.UnitOption) {
	if uw.err != nil {
		return
	}

	if len(ra.App.Exec) == 0 {
		uw.err = fmt.Errorf(`image %q has an empty "exec" (try --exec=BINARY)`,
			uw.p.AppNameToImageName(ra.Name))
		return
	}

	pa, err := prepareApp(uw.p, ra)
	if err != nil {
		uw.err = err
		return
	}

	appName := ra.Name.String()
	imgName := uw.p.AppNameToImageName(ra.Name)
	/* Write the generic unit options */
	opts = append(opts, []*unit.UnitOption{
		unit.NewUnitOption("Unit", "Description", fmt.Sprintf("Application=%v Image=%v", appName, imgName)),
		unit.NewUnitOption("Unit", "DefaultDependencies", "false"),
		unit.NewUnitOption("Unit", "Wants", fmt.Sprintf("reaper-%s.service", appName)),
		unit.NewUnitOption("Service", "Restart", "no"),

		// This helps working around a race
		// (https://github.com/systemd/systemd/issues/2913) that causes the
		// systemd unit name not getting written to the journal if the unit is
		// short-lived and runs as non-root.
		unit.NewUnitOption("Service", "SyslogIdentifier", appName),
	}...)

	// Setup I/O for iottymux (stdin/stdout/stderr)
	opts = append(opts, uw.SetupAppIO(uw.p, ra, binPath)...)

	if supportsNotify(uw.p, ra.Name.String()) {
		opts = append(opts, unit.NewUnitOption("Service", "Type", "notify"))
	}

	// Some pre-start jobs take a long time, set the timeout to 0
	opts = append(opts, unit.NewUnitOption("Service", "TimeoutStartSec", "0"))

	opts = append(opts, unit.NewUnitOption("Unit", "Requires", "sysusers.service"))
	opts = append(opts, unit.NewUnitOption("Unit", "After", "sysusers.service"))

	opts = uw.appSystemdUnit(pa, binPath, opts)

	uw.WriteUnit(ServiceUnitPath(uw.p.Root, ra.Name), "failed to create service unit file", opts...)
	uw.Activate(ServiceUnitName(ra.Name), ServiceWantPath(uw.p.Root, ra.Name))

}

// appSystemdUnit sets up an application for isolation via systemd
func (uw *UnitWriter) appSystemdUnit(pa *preparedApp, binPath string, opts []*unit.UnitOption) []*unit.UnitOption {
	if uw.err != nil {
		return nil
	}

	flavor, systemdVersion, err := GetFlavor(uw.p)
	if err != nil {
		uw.err = errwrap.Wrap(errors.New("unable to determine stage1 flavor"), err)
		return nil
	}

	ra := pa.app
	app := ra.App
	appName := ra.Name
	imgName := uw.p.AppNameToImageName(ra.Name)

	podAbsRoot, err := filepath.Abs(uw.p.Root)
	if err != nil {
		uw.err = err
		return nil
	}

	var supplementaryGroups []string
	for _, g := range app.SupplementaryGIDs {
		supplementaryGroups = append(supplementaryGroups, strconv.Itoa(g))
	}

	// Write env file
	if err := common.WriteEnvFile(pa.env, &uw.p.UidRange, EnvFilePath(uw.p.Root, pa.app.Name)); err != nil {
		uw.err = errwrap.Wrapf("unable to write environment file", err)
		return nil
	}

	execStart := append([]string{binPath}, app.Exec[1:]...)
	execStartString := quoteExec(execStart)
	opts = append(opts,
		unit.NewUnitOption("Service", "ExecStart", execStartString),
		unit.NewUnitOption("Service", "RootDirectory", common.RelAppRootfsPath(appName)),
		unit.NewUnitOption("Service", "WorkingDirectory", app.WorkingDirectory),
		unit.NewUnitOption("Service", "EnvironmentFile", RelEnvFilePath(appName)),
		unit.NewUnitOption("Service", "User", strconv.Itoa(int(pa.uid))),
		unit.NewUnitOption("Service", "Group", strconv.Itoa(int(pa.gid))),
		unit.NewUnitOption("Unit", "Requires", InstantiatedPrepareAppUnitName(ra.Name)),
		unit.NewUnitOption("Unit", "After", InstantiatedPrepareAppUnitName(ra.Name)),
	)

	if len(supplementaryGroups) > 0 {
		opts = appendOptionsList(opts, "Service", "SupplementaryGroups", "", supplementaryGroups...)
	}

	if !uw.p.InsecureOptions.DisableCapabilities {
		opts = append(opts, unit.NewUnitOption("Service", "CapabilityBoundingSet", strings.Join(pa.capabilities, " ")))
	}

	// Apply seccomp isolator, if any and not opt-ing out;
	// see https://www.freedesktop.org/software/systemd/man/systemd.exec.html#SystemCallFilter=
	if pa.seccomp != nil {
		opts, err = seccompUnitOptions(opts, pa.seccomp)
		if err != nil {
			uw.err = errwrap.Wrapf("unable to apply seccomp options", err)
			return nil
		}
	}
	opts = append(opts, unit.NewUnitOption("Service", "NoNewPrivileges", strconv.FormatBool(pa.noNewPrivileges)))

	if ra.ReadOnlyRootFS {
		for _, m := range pa.mounts {
			mntPath, err := EvaluateSymlinksInsideApp(podAbsRoot, m.Mount.Path)
			if err != nil {
				uw.err = err
				return nil
			}

			if !m.ReadOnly {
				rwDir := filepath.Join(common.RelAppRootfsPath(ra.Name), mntPath)
				opts = appendOptionsList(opts, "Service", "ReadWriteDirectories", "", rwDir)
			}
		}
		opts = appendOptionsList(opts, "Service", "ReadOnlyDirectories", "", common.RelAppRootfsPath(ra.Name))
	}

	// Unless we have --insecure-options=paths, then do some path protections:
	//
	// * prevent access to sensitive kernel tunables
	// * Run the app in a separate mount namespace
	//
	if !uw.p.InsecureOptions.DisablePaths {
		// Systemd 231+ has InaccessiblePaths
		// older versions only have InaccessibleDirectories
		// Paths prepended with "-" are ignored if they don't exist.
		if systemdVersion >= 231 {
			opts = appendOptionsList(opts, "Service", "InaccessiblePaths", "-", pa.relAppPaths(pa.hiddenPaths)...)
			opts = appendOptionsList(opts, "Service", "InaccessiblePaths", "-", pa.relAppPaths(pa.hiddenDirs)...)
			opts = appendOptionsList(opts, "Service", "ReadOnlyPaths", "-", pa.relAppPaths(pa.roPaths)...)
		} else {
			opts = appendOptionsList(opts, "Service", "InaccessibleDirectories", "-", pa.relAppPaths(pa.hiddenDirs)...)
			opts = appendOptionsList(opts, "Service", "ReadOnlyDirectories", "-", pa.relAppPaths(pa.roPaths)...)
		}

		if systemdVersion >= 233 {
			opts = append(opts, unit.NewUnitOption("Service", "ProtectKernelTunables", "true"))
		}

		// MountFlags=shared creates a new mount namespace and (as unintuitive
		// as it might seem) makes sure the mount is slave+shared.
		opts = append(opts, unit.NewUnitOption("Service", "MountFlags", "shared"))
	}

	// Generate default device policy for the app, as well as the list of allowed devices.
	// For kvm flavor, devices are VM-specific and restricting them is not strictly needed.
	if !uw.p.InsecureOptions.DisablePaths && flavor != "kvm" {
		opts = append(opts, unit.NewUnitOption("Service", "DevicePolicy", "closed"))
		deviceAllows, err := generateDeviceAllows(common.Stage1RootfsPath(podAbsRoot), appName, app.MountPoints, pa.mounts, &uw.p.UidRange)
		if err != nil {
			uw.err = err
			return nil
		}
		for _, dev := range deviceAllows {
			opts = append(opts, unit.NewUnitOption("Service", "DeviceAllow", dev))
		}
	}

	for _, eh := range app.EventHandlers {
		var typ string
		switch eh.Name {
		case "pre-start":
			typ = "ExecStartPre"
		case "post-stop":
			typ = "ExecStopPost"
		default:
			uw.err = fmt.Errorf("unrecognized eventHandler: %v", eh.Name)
			return nil
		}
		exec := quoteExec(eh.Exec)
		opts = append(opts, unit.NewUnitOption("Service", typ, exec))
	}

	// Resource isolators
	if pa.resources.MemoryLimit != nil {
		opts = append(opts, unit.NewUnitOption("Service", "MemoryLimit", strconv.FormatUint(*pa.resources.MemoryLimit, 10)))
	}
	if pa.resources.CPUQuota != nil {
		quota := strconv.FormatUint(*pa.resources.CPUQuota, 10) + "%"
		opts = append(opts, unit.NewUnitOption("Service", "CPUQuota", quota))
	}
	if pa.resources.LinuxCPUShares != nil {
		opts = append(opts, unit.NewUnitOption("Service", "CPUShares", strconv.FormatUint(*pa.resources.LinuxCPUShares, 10)))
	}
	if pa.resources.LinuxOOMScoreAdjust != nil {
		opts = append(opts, unit.NewUnitOption("Service", "OOMScoreAdjust", strconv.Itoa(*pa.resources.LinuxOOMScoreAdjust)))
	}

	var saPorts []types.Port
	for _, p := range ra.App.Ports {
		if p.SocketActivated {
			saPorts = append(saPorts, p)
		}
	}

	if len(saPorts) > 0 {
		sockopts := []*unit.UnitOption{
			unit.NewUnitOption("Unit", "Description", fmt.Sprintf("Application=%v Image=%v %s", appName, imgName, "socket-activated ports")),
			unit.NewUnitOption("Unit", "DefaultDependencies", "false"),
			unit.NewUnitOption("Socket", "BindIPv6Only", "both"),
			unit.NewUnitOption("Socket", "Service", ServiceUnitName(appName)),
		}

		for _, sap := range saPorts {
			var proto string
			switch sap.Protocol {
			case "tcp":
				proto = "ListenStream"
			case "udp":
				proto = "ListenDatagram"
			default:
				uw.err = fmt.Errorf("unrecognized protocol: %v", sap.Protocol)
				return nil
			}
			// We find the host port for the pod's port and use that in the
			// socket unit file.
			// This is so because systemd inside the pod will match based on
			// the socket port number, and since the socket was created on the
			// host, it will have the host port number.
			port := findHostPort(*uw.p.Manifest, sap.Name)
			if port == 0 {
				log.Printf("warning: no --port option for socket-activated port %q, assuming port %d as specified in the manifest", sap.Name, sap.Port)
				port = sap.Port
			}
			sockopts = append(sockopts, unit.NewUnitOption("Socket", proto, fmt.Sprintf("%v", port)))
		}

		file, err := os.OpenFile(SocketUnitPath(uw.p.Root, appName), os.O_WRONLY|os.O_CREATE, 0644)
		if err != nil {
			uw.err = errwrap.Wrap(errors.New("failed to create socket file"), err)
			return nil
		}
		defer file.Close()

		if _, err = io.Copy(file, unit.Serialize(sockopts)); err != nil {
			uw.err = errwrap.Wrap(errors.New("failed to write socket unit file"), err)
			return nil
		}

		if err = os.Symlink(path.Join("..", SocketUnitName(appName)), SocketWantPath(uw.p.Root, appName)); err != nil {
			uw.err = errwrap.Wrap(errors.New("failed to link socket want"), err)
			return nil
		}

		opts = append(opts, unit.NewUnitOption("Unit", "Requires", SocketUnitName(appName)))
	}
	return opts
}

// AppReaperUnit writes an app reaper service unit for the given app in the given path using the given unit options.
func (uw *UnitWriter) AppReaperUnit(appName types.ACName, binPath string, opts ...*unit.UnitOption) {
	if uw.err != nil {
		return
	}

	opts = append(opts, []*unit.UnitOption{
		unit.NewUnitOption("Unit", "Description", fmt.Sprintf("%s Reaper", appName)),
		unit.NewUnitOption("Unit", "DefaultDependencies", "false"),
		unit.NewUnitOption("Unit", "StopWhenUnneeded", "yes"),
		unit.NewUnitOption("Unit", "Before", "halt.target"),
		unit.NewUnitOption("Unit", "Conflicts", "exit.target"),
		unit.NewUnitOption("Unit", "Conflicts", "halt.target"),
		unit.NewUnitOption("Unit", "Conflicts", "poweroff.target"),
		unit.NewUnitOption("Service", "RemainAfterExit", "yes"),
		unit.NewUnitOption("Service", "ExecStop", fmt.Sprintf(
			"/reaper.sh \"%s\" \"%s\" \"%s\"",
			appName,
			common.RelAppRootfsPath(appName),
			binPath,
		)),
	}...)

	uw.WriteUnit(
		ServiceUnitPath(uw.p.Root, types.ACName(fmt.Sprintf("reaper-%s", appName))),
		fmt.Sprintf("failed to write app %q reaper service", appName),
		opts...,
	)
}

// AppSocketUnits writes a stream socket-unit for the given app in the given path.
func (uw *UnitWriter) AppSocketUnit(appName types.ACName, binPath string, streamName string, opts ...*unit.UnitOption) {
	opts = append(opts, []*unit.UnitOption{
		unit.NewUnitOption("Unit", "Description", fmt.Sprintf("%s socket for %s", streamName, appName)),
		unit.NewUnitOption("Unit", "DefaultDependencies", "no"),
		unit.NewUnitOption("Unit", "StopWhenUnneeded", "yes"),
		unit.NewUnitOption("Unit", "RefuseManualStart", "yes"),
		unit.NewUnitOption("Unit", "RefuseManualStop", "yes"),
		unit.NewUnitOption("Unit", "BindsTo", fmt.Sprintf("%s.service", appName)),
		unit.NewUnitOption("Socket", "RemoveOnStop", "yes"),
		unit.NewUnitOption("Socket", "Service", fmt.Sprintf("%s.service", appName)),
		unit.NewUnitOption("Socket", "FileDescriptorName", streamName),
		unit.NewUnitOption("Socket", "ListenFIFO", filepath.Join("/rkt/iottymux", appName.String(), "stage2-"+streamName)),
	}...)

	uw.WriteUnit(
		TypedUnitPath(uw.p.Root, appName.String()+"-"+streamName, "socket"),
		fmt.Sprintf("failed to write %s socket for %q service", streamName, appName),
		opts...,
	)
}

// appendOptionsList updates an existing unit options list appending
// an array of new properties, one entry at a time.
// This is the preferred method to avoid hitting line length limits
// in unit files. Target property must support multi-line entries.
func appendOptionsList(opts []*unit.UnitOption, section, property, prefix string, vals ...string) []*unit.UnitOption {
	for _, v := range vals {
		opts = append(opts, unit.NewUnitOption(section, property, fmt.Sprintf("%s%s", prefix, v)))
	}
	return opts
}
