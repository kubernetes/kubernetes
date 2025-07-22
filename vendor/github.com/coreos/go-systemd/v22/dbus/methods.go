// Copyright 2015, 2018 CoreOS, Inc.
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

package dbus

import (
	"context"
	"errors"
	"fmt"
	"path"
	"strconv"

	"github.com/godbus/dbus/v5"
)

// Who can be used to specify which process to kill in the unit via the KillUnitWithTarget API
type Who string

const (
	// All sends the signal to all processes in the unit
	All Who = "all"
	// Main sends the signal to the main process of the unit
	Main Who = "main"
	// Control sends the signal to the control process of the unit
	Control Who = "control"
)

func (c *Conn) jobComplete(signal *dbus.Signal) {
	var id uint32
	var job dbus.ObjectPath
	var unit string
	var result string
	dbus.Store(signal.Body, &id, &job, &unit, &result)
	c.jobListener.Lock()
	out, ok := c.jobListener.jobs[job]
	if ok {
		out <- result
		delete(c.jobListener.jobs, job)
	}
	c.jobListener.Unlock()
}

func (c *Conn) startJob(ctx context.Context, ch chan<- string, job string, args ...interface{}) (int, error) {
	if ch != nil {
		c.jobListener.Lock()
		defer c.jobListener.Unlock()
	}

	var p dbus.ObjectPath
	err := c.sysobj.CallWithContext(ctx, job, 0, args...).Store(&p)
	if err != nil {
		return 0, err
	}

	if ch != nil {
		c.jobListener.jobs[p] = ch
	}

	// ignore error since 0 is fine if conversion fails
	jobID, _ := strconv.Atoi(path.Base(string(p)))

	return jobID, nil
}

// Deprecated: use StartUnitContext instead.
func (c *Conn) StartUnit(name string, mode string, ch chan<- string) (int, error) {
	return c.StartUnitContext(context.Background(), name, mode, ch)
}

// StartUnitContext enqueues a start job and depending jobs, if any (unless otherwise
// specified by the mode string).
//
// Takes the unit to activate, plus a mode string. The mode needs to be one of
// replace, fail, isolate, ignore-dependencies, ignore-requirements. If
// "replace" the call will start the unit and its dependencies, possibly
// replacing already queued jobs that conflict with this. If "fail" the call
// will start the unit and its dependencies, but will fail if this would change
// an already queued job. If "isolate" the call will start the unit in question
// and terminate all units that aren't dependencies of it. If
// "ignore-dependencies" it will start a unit but ignore all its dependencies.
// If "ignore-requirements" it will start a unit but only ignore the
// requirement dependencies. It is not recommended to make use of the latter
// two options.
//
// If the provided channel is non-nil, a result string will be sent to it upon
// job completion: one of done, canceled, timeout, failed, dependency, skipped.
// done indicates successful execution of a job. canceled indicates that a job
// has been canceled  before it finished execution. timeout indicates that the
// job timeout was reached. failed indicates that the job failed. dependency
// indicates that a job this job has been depending on failed and the job hence
// has been removed too. skipped indicates that a job was skipped because it
// didn't apply to the units current state.
//
// If no error occurs, the ID of the underlying systemd job will be returned. There
// does exist the possibility for no error to be returned, but for the returned job
// ID to be 0. In this case, the actual underlying ID is not 0 and this datapoint
// should not be considered authoritative.
//
// If an error does occur, it will be returned to the user alongside a job ID of 0.
func (c *Conn) StartUnitContext(ctx context.Context, name string, mode string, ch chan<- string) (int, error) {
	return c.startJob(ctx, ch, "org.freedesktop.systemd1.Manager.StartUnit", name, mode)
}

// Deprecated: use StopUnitContext instead.
func (c *Conn) StopUnit(name string, mode string, ch chan<- string) (int, error) {
	return c.StopUnitContext(context.Background(), name, mode, ch)
}

// StopUnitContext is similar to StartUnitContext, but stops the specified unit
// rather than starting it.
func (c *Conn) StopUnitContext(ctx context.Context, name string, mode string, ch chan<- string) (int, error) {
	return c.startJob(ctx, ch, "org.freedesktop.systemd1.Manager.StopUnit", name, mode)
}

// Deprecated: use ReloadUnitContext instead.
func (c *Conn) ReloadUnit(name string, mode string, ch chan<- string) (int, error) {
	return c.ReloadUnitContext(context.Background(), name, mode, ch)
}

// ReloadUnitContext reloads a unit. Reloading is done only if the unit
// is already running, and fails otherwise.
func (c *Conn) ReloadUnitContext(ctx context.Context, name string, mode string, ch chan<- string) (int, error) {
	return c.startJob(ctx, ch, "org.freedesktop.systemd1.Manager.ReloadUnit", name, mode)
}

// Deprecated: use RestartUnitContext instead.
func (c *Conn) RestartUnit(name string, mode string, ch chan<- string) (int, error) {
	return c.RestartUnitContext(context.Background(), name, mode, ch)
}

// RestartUnitContext restarts a service. If a service is restarted that isn't
// running it will be started.
func (c *Conn) RestartUnitContext(ctx context.Context, name string, mode string, ch chan<- string) (int, error) {
	return c.startJob(ctx, ch, "org.freedesktop.systemd1.Manager.RestartUnit", name, mode)
}

// Deprecated: use TryRestartUnitContext instead.
func (c *Conn) TryRestartUnit(name string, mode string, ch chan<- string) (int, error) {
	return c.TryRestartUnitContext(context.Background(), name, mode, ch)
}

// TryRestartUnitContext is like RestartUnitContext, except that a service that
// isn't running is not affected by the restart.
func (c *Conn) TryRestartUnitContext(ctx context.Context, name string, mode string, ch chan<- string) (int, error) {
	return c.startJob(ctx, ch, "org.freedesktop.systemd1.Manager.TryRestartUnit", name, mode)
}

// Deprecated: use ReloadOrRestartUnitContext instead.
func (c *Conn) ReloadOrRestartUnit(name string, mode string, ch chan<- string) (int, error) {
	return c.ReloadOrRestartUnitContext(context.Background(), name, mode, ch)
}

// ReloadOrRestartUnitContext attempts a reload if the unit supports it and use
// a restart otherwise.
func (c *Conn) ReloadOrRestartUnitContext(ctx context.Context, name string, mode string, ch chan<- string) (int, error) {
	return c.startJob(ctx, ch, "org.freedesktop.systemd1.Manager.ReloadOrRestartUnit", name, mode)
}

// Deprecated: use ReloadOrTryRestartUnitContext instead.
func (c *Conn) ReloadOrTryRestartUnit(name string, mode string, ch chan<- string) (int, error) {
	return c.ReloadOrTryRestartUnitContext(context.Background(), name, mode, ch)
}

// ReloadOrTryRestartUnitContext attempts a reload if the unit supports it,
// and use a "Try" flavored restart otherwise.
func (c *Conn) ReloadOrTryRestartUnitContext(ctx context.Context, name string, mode string, ch chan<- string) (int, error) {
	return c.startJob(ctx, ch, "org.freedesktop.systemd1.Manager.ReloadOrTryRestartUnit", name, mode)
}

// Deprecated: use StartTransientUnitContext instead.
func (c *Conn) StartTransientUnit(name string, mode string, properties []Property, ch chan<- string) (int, error) {
	return c.StartTransientUnitContext(context.Background(), name, mode, properties, ch)
}

// StartTransientUnitContext may be used to create and start a transient unit, which
// will be released as soon as it is not running or referenced anymore or the
// system is rebooted. name is the unit name including suffix, and must be
// unique. mode is the same as in StartUnitContext, properties contains properties
// of the unit.
func (c *Conn) StartTransientUnitContext(ctx context.Context, name string, mode string, properties []Property, ch chan<- string) (int, error) {
	return c.startJob(ctx, ch, "org.freedesktop.systemd1.Manager.StartTransientUnit", name, mode, properties, make([]PropertyCollection, 0))
}

// Deprecated: use KillUnitContext instead.
func (c *Conn) KillUnit(name string, signal int32) {
	c.KillUnitContext(context.Background(), name, signal)
}

// KillUnitContext takes the unit name and a UNIX signal number to send.
// All of the unit's processes are killed.
func (c *Conn) KillUnitContext(ctx context.Context, name string, signal int32) {
	c.KillUnitWithTarget(ctx, name, All, signal)
}

// KillUnitWithTarget is like KillUnitContext, but allows you to specify which
// process in the unit to send the signal to.
func (c *Conn) KillUnitWithTarget(ctx context.Context, name string, target Who, signal int32) error {
	return c.sysobj.CallWithContext(ctx, "org.freedesktop.systemd1.Manager.KillUnit", 0, name, string(target), signal).Store()
}

// Deprecated: use ResetFailedUnitContext instead.
func (c *Conn) ResetFailedUnit(name string) error {
	return c.ResetFailedUnitContext(context.Background(), name)
}

// ResetFailedUnitContext resets the "failed" state of a specific unit.
func (c *Conn) ResetFailedUnitContext(ctx context.Context, name string) error {
	return c.sysobj.CallWithContext(ctx, "org.freedesktop.systemd1.Manager.ResetFailedUnit", 0, name).Store()
}

// Deprecated: use SystemStateContext instead.
func (c *Conn) SystemState() (*Property, error) {
	return c.SystemStateContext(context.Background())
}

// SystemStateContext returns the systemd state. Equivalent to
// systemctl is-system-running.
func (c *Conn) SystemStateContext(ctx context.Context) (*Property, error) {
	var err error
	var prop dbus.Variant

	obj := c.sysconn.Object("org.freedesktop.systemd1", "/org/freedesktop/systemd1")
	err = obj.CallWithContext(ctx, "org.freedesktop.DBus.Properties.Get", 0, "org.freedesktop.systemd1.Manager", "SystemState").Store(&prop)
	if err != nil {
		return nil, err
	}

	return &Property{Name: "SystemState", Value: prop}, nil
}

// getProperties takes the unit path and returns all of its dbus object properties, for the given dbus interface.
func (c *Conn) getProperties(ctx context.Context, path dbus.ObjectPath, dbusInterface string) (map[string]interface{}, error) {
	var err error
	var props map[string]dbus.Variant

	if !path.IsValid() {
		return nil, fmt.Errorf("invalid unit name: %v", path)
	}

	obj := c.sysconn.Object("org.freedesktop.systemd1", path)
	err = obj.CallWithContext(ctx, "org.freedesktop.DBus.Properties.GetAll", 0, dbusInterface).Store(&props)
	if err != nil {
		return nil, err
	}

	out := make(map[string]interface{}, len(props))
	for k, v := range props {
		out[k] = v.Value()
	}

	return out, nil
}

// Deprecated: use GetUnitPropertiesContext instead.
func (c *Conn) GetUnitProperties(unit string) (map[string]interface{}, error) {
	return c.GetUnitPropertiesContext(context.Background(), unit)
}

// GetUnitPropertiesContext takes the (unescaped) unit name and returns all of
// its dbus object properties.
func (c *Conn) GetUnitPropertiesContext(ctx context.Context, unit string) (map[string]interface{}, error) {
	path := unitPath(unit)
	return c.getProperties(ctx, path, "org.freedesktop.systemd1.Unit")
}

// Deprecated: use GetUnitPathPropertiesContext instead.
func (c *Conn) GetUnitPathProperties(path dbus.ObjectPath) (map[string]interface{}, error) {
	return c.GetUnitPathPropertiesContext(context.Background(), path)
}

// GetUnitPathPropertiesContext takes the (escaped) unit path and returns all
// of its dbus object properties.
func (c *Conn) GetUnitPathPropertiesContext(ctx context.Context, path dbus.ObjectPath) (map[string]interface{}, error) {
	return c.getProperties(ctx, path, "org.freedesktop.systemd1.Unit")
}

// Deprecated: use GetAllPropertiesContext instead.
func (c *Conn) GetAllProperties(unit string) (map[string]interface{}, error) {
	return c.GetAllPropertiesContext(context.Background(), unit)
}

// GetAllPropertiesContext takes the (unescaped) unit name and returns all of
// its dbus object properties.
func (c *Conn) GetAllPropertiesContext(ctx context.Context, unit string) (map[string]interface{}, error) {
	path := unitPath(unit)
	return c.getProperties(ctx, path, "")
}

func (c *Conn) getProperty(ctx context.Context, unit string, dbusInterface string, propertyName string) (*Property, error) {
	var err error
	var prop dbus.Variant

	path := unitPath(unit)
	if !path.IsValid() {
		return nil, errors.New("invalid unit name: " + unit)
	}

	obj := c.sysconn.Object("org.freedesktop.systemd1", path)
	err = obj.CallWithContext(ctx, "org.freedesktop.DBus.Properties.Get", 0, dbusInterface, propertyName).Store(&prop)
	if err != nil {
		return nil, err
	}

	return &Property{Name: propertyName, Value: prop}, nil
}

// Deprecated: use GetUnitPropertyContext instead.
func (c *Conn) GetUnitProperty(unit string, propertyName string) (*Property, error) {
	return c.GetUnitPropertyContext(context.Background(), unit, propertyName)
}

// GetUnitPropertyContext takes an (unescaped) unit name, and a property name,
// and returns the property value.
func (c *Conn) GetUnitPropertyContext(ctx context.Context, unit string, propertyName string) (*Property, error) {
	return c.getProperty(ctx, unit, "org.freedesktop.systemd1.Unit", propertyName)
}

// Deprecated: use GetServicePropertyContext instead.
func (c *Conn) GetServiceProperty(service string, propertyName string) (*Property, error) {
	return c.GetServicePropertyContext(context.Background(), service, propertyName)
}

// GetServiceProperty returns property for given service name and property name.
func (c *Conn) GetServicePropertyContext(ctx context.Context, service string, propertyName string) (*Property, error) {
	return c.getProperty(ctx, service, "org.freedesktop.systemd1.Service", propertyName)
}

// Deprecated: use GetUnitTypePropertiesContext instead.
func (c *Conn) GetUnitTypeProperties(unit string, unitType string) (map[string]interface{}, error) {
	return c.GetUnitTypePropertiesContext(context.Background(), unit, unitType)
}

// GetUnitTypePropertiesContext returns the extra properties for a unit, specific to the unit type.
// Valid values for unitType: Service, Socket, Target, Device, Mount, Automount, Snapshot, Timer, Swap, Path, Slice, Scope.
// Returns "dbus.Error: Unknown interface" error if the unitType is not the correct type of the unit.
func (c *Conn) GetUnitTypePropertiesContext(ctx context.Context, unit string, unitType string) (map[string]interface{}, error) {
	path := unitPath(unit)
	return c.getProperties(ctx, path, "org.freedesktop.systemd1."+unitType)
}

// Deprecated: use SetUnitPropertiesContext instead.
func (c *Conn) SetUnitProperties(name string, runtime bool, properties ...Property) error {
	return c.SetUnitPropertiesContext(context.Background(), name, runtime, properties...)
}

// SetUnitPropertiesContext may be used to modify certain unit properties at runtime.
// Not all properties may be changed at runtime, but many resource management
// settings (primarily those in systemd.cgroup(5)) may. The changes are applied
// instantly, and stored on disk for future boots, unless runtime is true, in which
// case the settings only apply until the next reboot. name is the name of the unit
// to modify. properties are the settings to set, encoded as an array of property
// name and value pairs.
func (c *Conn) SetUnitPropertiesContext(ctx context.Context, name string, runtime bool, properties ...Property) error {
	return c.sysobj.CallWithContext(ctx, "org.freedesktop.systemd1.Manager.SetUnitProperties", 0, name, runtime, properties).Store()
}

// Deprecated: use GetUnitTypePropertyContext instead.
func (c *Conn) GetUnitTypeProperty(unit string, unitType string, propertyName string) (*Property, error) {
	return c.GetUnitTypePropertyContext(context.Background(), unit, unitType, propertyName)
}

// GetUnitTypePropertyContext takes a property name, a unit name, and a unit type,
// and returns a property value. For valid values of unitType, see GetUnitTypePropertiesContext.
func (c *Conn) GetUnitTypePropertyContext(ctx context.Context, unit string, unitType string, propertyName string) (*Property, error) {
	return c.getProperty(ctx, unit, "org.freedesktop.systemd1."+unitType, propertyName)
}

type UnitStatus struct {
	Name        string          // The primary unit name as string
	Description string          // The human readable description string
	LoadState   string          // The load state (i.e. whether the unit file has been loaded successfully)
	ActiveState string          // The active state (i.e. whether the unit is currently started or not)
	SubState    string          // The sub state (a more fine-grained version of the active state that is specific to the unit type, which the active state is not)
	Followed    string          // A unit that is being followed in its state by this unit, if there is any, otherwise the empty string.
	Path        dbus.ObjectPath // The unit object path
	JobId       uint32          // If there is a job queued for the job unit the numeric job id, 0 otherwise
	JobType     string          // The job type as string
	JobPath     dbus.ObjectPath // The job object path
}

type storeFunc func(retvalues ...interface{}) error

func (c *Conn) listUnitsInternal(f storeFunc) ([]UnitStatus, error) {
	result := make([][]interface{}, 0)
	err := f(&result)
	if err != nil {
		return nil, err
	}

	resultInterface := make([]interface{}, len(result))
	for i := range result {
		resultInterface[i] = result[i]
	}

	status := make([]UnitStatus, len(result))
	statusInterface := make([]interface{}, len(status))
	for i := range status {
		statusInterface[i] = &status[i]
	}

	err = dbus.Store(resultInterface, statusInterface...)
	if err != nil {
		return nil, err
	}

	return status, nil
}

// GetUnitByPID returns the unit object path of the unit a process ID
// belongs to. It takes a UNIX PID and returns the object path. The PID must
// refer to an existing system process
func (c *Conn) GetUnitByPID(ctx context.Context, pid uint32) (dbus.ObjectPath, error) {
	var result dbus.ObjectPath

	err := c.sysobj.CallWithContext(ctx, "org.freedesktop.systemd1.Manager.GetUnitByPID", 0, pid).Store(&result)

	return result, err
}

// GetUnitNameByPID returns the name of the unit a process ID belongs to. It
// takes a UNIX PID and returns the object path. The PID must refer to an
// existing system process
func (c *Conn) GetUnitNameByPID(ctx context.Context, pid uint32) (string, error) {
	path, err := c.GetUnitByPID(ctx, pid)
	if err != nil {
		return "", err
	}

	return unitName(path), nil
}

// Deprecated: use ListUnitsContext instead.
func (c *Conn) ListUnits() ([]UnitStatus, error) {
	return c.ListUnitsContext(context.Background())
}

// ListUnitsContext returns an array with all currently loaded units. Note that
// units may be known by multiple names at the same time, and hence there might
// be more unit names loaded than actual units behind them.
// Also note that a unit is only loaded if it is active and/or enabled.
// Units that are both disabled and inactive will thus not be returned.
func (c *Conn) ListUnitsContext(ctx context.Context) ([]UnitStatus, error) {
	return c.listUnitsInternal(c.sysobj.CallWithContext(ctx, "org.freedesktop.systemd1.Manager.ListUnits", 0).Store)
}

// Deprecated: use ListUnitsFilteredContext instead.
func (c *Conn) ListUnitsFiltered(states []string) ([]UnitStatus, error) {
	return c.ListUnitsFilteredContext(context.Background(), states)
}

// ListUnitsFilteredContext returns an array with units filtered by state.
// It takes a list of units' statuses to filter.
func (c *Conn) ListUnitsFilteredContext(ctx context.Context, states []string) ([]UnitStatus, error) {
	return c.listUnitsInternal(c.sysobj.CallWithContext(ctx, "org.freedesktop.systemd1.Manager.ListUnitsFiltered", 0, states).Store)
}

// Deprecated: use ListUnitsByPatternsContext instead.
func (c *Conn) ListUnitsByPatterns(states []string, patterns []string) ([]UnitStatus, error) {
	return c.ListUnitsByPatternsContext(context.Background(), states, patterns)
}

// ListUnitsByPatternsContext returns an array with units.
// It takes a list of units' statuses and names to filter.
// Note that units may be known by multiple names at the same time,
// and hence there might be more unit names loaded than actual units behind them.
func (c *Conn) ListUnitsByPatternsContext(ctx context.Context, states []string, patterns []string) ([]UnitStatus, error) {
	return c.listUnitsInternal(c.sysobj.CallWithContext(ctx, "org.freedesktop.systemd1.Manager.ListUnitsByPatterns", 0, states, patterns).Store)
}

// Deprecated: use ListUnitsByNamesContext instead.
func (c *Conn) ListUnitsByNames(units []string) ([]UnitStatus, error) {
	return c.ListUnitsByNamesContext(context.Background(), units)
}

// ListUnitsByNamesContext returns an array with units. It takes a list of units'
// names and returns an UnitStatus array. Comparing to ListUnitsByPatternsContext
// method, this method returns statuses even for inactive or non-existing
// units. Input array should contain exact unit names, but not patterns.
//
// Requires systemd v230 or higher.
func (c *Conn) ListUnitsByNamesContext(ctx context.Context, units []string) ([]UnitStatus, error) {
	return c.listUnitsInternal(c.sysobj.CallWithContext(ctx, "org.freedesktop.systemd1.Manager.ListUnitsByNames", 0, units).Store)
}

type UnitFile struct {
	Path string
	Type string
}

func (c *Conn) listUnitFilesInternal(f storeFunc) ([]UnitFile, error) {
	result := make([][]interface{}, 0)
	err := f(&result)
	if err != nil {
		return nil, err
	}

	resultInterface := make([]interface{}, len(result))
	for i := range result {
		resultInterface[i] = result[i]
	}

	files := make([]UnitFile, len(result))
	fileInterface := make([]interface{}, len(files))
	for i := range files {
		fileInterface[i] = &files[i]
	}

	err = dbus.Store(resultInterface, fileInterface...)
	if err != nil {
		return nil, err
	}

	return files, nil
}

// Deprecated: use ListUnitFilesContext instead.
func (c *Conn) ListUnitFiles() ([]UnitFile, error) {
	return c.ListUnitFilesContext(context.Background())
}

// ListUnitFiles returns an array of all available units on disk.
func (c *Conn) ListUnitFilesContext(ctx context.Context) ([]UnitFile, error) {
	return c.listUnitFilesInternal(c.sysobj.CallWithContext(ctx, "org.freedesktop.systemd1.Manager.ListUnitFiles", 0).Store)
}

// Deprecated: use ListUnitFilesByPatternsContext instead.
func (c *Conn) ListUnitFilesByPatterns(states []string, patterns []string) ([]UnitFile, error) {
	return c.ListUnitFilesByPatternsContext(context.Background(), states, patterns)
}

// ListUnitFilesByPatternsContext returns an array of all available units on disk matched the patterns.
func (c *Conn) ListUnitFilesByPatternsContext(ctx context.Context, states []string, patterns []string) ([]UnitFile, error) {
	return c.listUnitFilesInternal(c.sysobj.CallWithContext(ctx, "org.freedesktop.systemd1.Manager.ListUnitFilesByPatterns", 0, states, patterns).Store)
}

type LinkUnitFileChange EnableUnitFileChange

// Deprecated: use LinkUnitFilesContext instead.
func (c *Conn) LinkUnitFiles(files []string, runtime bool, force bool) ([]LinkUnitFileChange, error) {
	return c.LinkUnitFilesContext(context.Background(), files, runtime, force)
}

// LinkUnitFilesContext links unit files (that are located outside of the
// usual unit search paths) into the unit search path.
//
// It takes a list of absolute paths to unit files to link and two
// booleans.
//
// The first boolean controls whether the unit shall be
// enabled for runtime only (true, /run), or persistently (false,
// /etc).
//
// The second controls whether symlinks pointing to other units shall
// be replaced if necessary.
//
// This call returns a list of the changes made. The list consists of
// structures with three strings: the type of the change (one of symlink
// or unlink), the file name of the symlink and the destination of the
// symlink.
func (c *Conn) LinkUnitFilesContext(ctx context.Context, files []string, runtime bool, force bool) ([]LinkUnitFileChange, error) {
	result := make([][]interface{}, 0)
	err := c.sysobj.CallWithContext(ctx, "org.freedesktop.systemd1.Manager.LinkUnitFiles", 0, files, runtime, force).Store(&result)
	if err != nil {
		return nil, err
	}

	resultInterface := make([]interface{}, len(result))
	for i := range result {
		resultInterface[i] = result[i]
	}

	changes := make([]LinkUnitFileChange, len(result))
	changesInterface := make([]interface{}, len(changes))
	for i := range changes {
		changesInterface[i] = &changes[i]
	}

	err = dbus.Store(resultInterface, changesInterface...)
	if err != nil {
		return nil, err
	}

	return changes, nil
}

// Deprecated: use EnableUnitFilesContext instead.
func (c *Conn) EnableUnitFiles(files []string, runtime bool, force bool) (bool, []EnableUnitFileChange, error) {
	return c.EnableUnitFilesContext(context.Background(), files, runtime, force)
}

// EnableUnitFilesContext may be used to enable one or more units in the system
// (by creating symlinks to them in /etc or /run).
//
// It takes a list of unit files to enable (either just file names or full
// absolute paths if the unit files are residing outside the usual unit
// search paths), and two booleans: the first controls whether the unit shall
// be enabled for runtime only (true, /run), or persistently (false, /etc).
// The second one controls whether symlinks pointing to other units shall
// be replaced if necessary.
//
// This call returns one boolean and an array with the changes made. The
// boolean signals whether the unit files contained any enablement
// information (i.e. an [Install]) section. The changes list consists of
// structures with three strings: the type of the change (one of symlink
// or unlink), the file name of the symlink and the destination of the
// symlink.
func (c *Conn) EnableUnitFilesContext(ctx context.Context, files []string, runtime bool, force bool) (bool, []EnableUnitFileChange, error) {
	var carries_install_info bool

	result := make([][]interface{}, 0)
	err := c.sysobj.CallWithContext(ctx, "org.freedesktop.systemd1.Manager.EnableUnitFiles", 0, files, runtime, force).Store(&carries_install_info, &result)
	if err != nil {
		return false, nil, err
	}

	resultInterface := make([]interface{}, len(result))
	for i := range result {
		resultInterface[i] = result[i]
	}

	changes := make([]EnableUnitFileChange, len(result))
	changesInterface := make([]interface{}, len(changes))
	for i := range changes {
		changesInterface[i] = &changes[i]
	}

	err = dbus.Store(resultInterface, changesInterface...)
	if err != nil {
		return false, nil, err
	}

	return carries_install_info, changes, nil
}

type EnableUnitFileChange struct {
	Type        string // Type of the change (one of symlink or unlink)
	Filename    string // File name of the symlink
	Destination string // Destination of the symlink
}

// Deprecated: use DisableUnitFilesContext instead.
func (c *Conn) DisableUnitFiles(files []string, runtime bool) ([]DisableUnitFileChange, error) {
	return c.DisableUnitFilesContext(context.Background(), files, runtime)
}

// DisableUnitFilesContext may be used to disable one or more units in the
// system (by removing symlinks to them from /etc or /run).
//
// It takes a list of unit files to disable (either just file names or full
// absolute paths if the unit files are residing outside the usual unit
// search paths), and one boolean: whether the unit was enabled for runtime
// only (true, /run), or persistently (false, /etc).
//
// This call returns an array with the changes made. The changes list
// consists of structures with three strings: the type of the change (one of
// symlink or unlink), the file name of the symlink and the destination of the
// symlink.
func (c *Conn) DisableUnitFilesContext(ctx context.Context, files []string, runtime bool) ([]DisableUnitFileChange, error) {
	result := make([][]interface{}, 0)
	err := c.sysobj.CallWithContext(ctx, "org.freedesktop.systemd1.Manager.DisableUnitFiles", 0, files, runtime).Store(&result)
	if err != nil {
		return nil, err
	}

	resultInterface := make([]interface{}, len(result))
	for i := range result {
		resultInterface[i] = result[i]
	}

	changes := make([]DisableUnitFileChange, len(result))
	changesInterface := make([]interface{}, len(changes))
	for i := range changes {
		changesInterface[i] = &changes[i]
	}

	err = dbus.Store(resultInterface, changesInterface...)
	if err != nil {
		return nil, err
	}

	return changes, nil
}

type DisableUnitFileChange struct {
	Type        string // Type of the change (one of symlink or unlink)
	Filename    string // File name of the symlink
	Destination string // Destination of the symlink
}

// Deprecated: use MaskUnitFilesContext instead.
func (c *Conn) MaskUnitFiles(files []string, runtime bool, force bool) ([]MaskUnitFileChange, error) {
	return c.MaskUnitFilesContext(context.Background(), files, runtime, force)
}

// MaskUnitFilesContext masks one or more units in the system.
//
// The files argument contains a  list of units to mask (either just file names
// or full absolute paths if the unit files are residing outside the usual unit
// search paths).
//
// The runtime argument is used to specify whether the unit was enabled for
// runtime only (true, /run/systemd/..), or persistently (false,
// /etc/systemd/..).
func (c *Conn) MaskUnitFilesContext(ctx context.Context, files []string, runtime bool, force bool) ([]MaskUnitFileChange, error) {
	result := make([][]interface{}, 0)
	err := c.sysobj.CallWithContext(ctx, "org.freedesktop.systemd1.Manager.MaskUnitFiles", 0, files, runtime, force).Store(&result)
	if err != nil {
		return nil, err
	}

	resultInterface := make([]interface{}, len(result))
	for i := range result {
		resultInterface[i] = result[i]
	}

	changes := make([]MaskUnitFileChange, len(result))
	changesInterface := make([]interface{}, len(changes))
	for i := range changes {
		changesInterface[i] = &changes[i]
	}

	err = dbus.Store(resultInterface, changesInterface...)
	if err != nil {
		return nil, err
	}

	return changes, nil
}

type MaskUnitFileChange struct {
	Type        string // Type of the change (one of symlink or unlink)
	Filename    string // File name of the symlink
	Destination string // Destination of the symlink
}

// Deprecated: use UnmaskUnitFilesContext instead.
func (c *Conn) UnmaskUnitFiles(files []string, runtime bool) ([]UnmaskUnitFileChange, error) {
	return c.UnmaskUnitFilesContext(context.Background(), files, runtime)
}

// UnmaskUnitFilesContext unmasks one or more units in the system.
//
// It takes the list of unit files to mask (either just file names or full
// absolute paths if the unit files are residing outside the usual unit search
// paths), and a boolean runtime flag to specify whether the unit was enabled
// for runtime only (true, /run/systemd/..), or persistently (false,
// /etc/systemd/..).
func (c *Conn) UnmaskUnitFilesContext(ctx context.Context, files []string, runtime bool) ([]UnmaskUnitFileChange, error) {
	result := make([][]interface{}, 0)
	err := c.sysobj.CallWithContext(ctx, "org.freedesktop.systemd1.Manager.UnmaskUnitFiles", 0, files, runtime).Store(&result)
	if err != nil {
		return nil, err
	}

	resultInterface := make([]interface{}, len(result))
	for i := range result {
		resultInterface[i] = result[i]
	}

	changes := make([]UnmaskUnitFileChange, len(result))
	changesInterface := make([]interface{}, len(changes))
	for i := range changes {
		changesInterface[i] = &changes[i]
	}

	err = dbus.Store(resultInterface, changesInterface...)
	if err != nil {
		return nil, err
	}

	return changes, nil
}

type UnmaskUnitFileChange struct {
	Type        string // Type of the change (one of symlink or unlink)
	Filename    string // File name of the symlink
	Destination string // Destination of the symlink
}

// Deprecated: use ReloadContext instead.
func (c *Conn) Reload() error {
	return c.ReloadContext(context.Background())
}

// ReloadContext instructs systemd to scan for and reload unit files. This is
// an equivalent to systemctl daemon-reload.
func (c *Conn) ReloadContext(ctx context.Context) error {
	return c.sysobj.CallWithContext(ctx, "org.freedesktop.systemd1.Manager.Reload", 0).Store()
}

func unitPath(name string) dbus.ObjectPath {
	return dbus.ObjectPath("/org/freedesktop/systemd1/unit/" + PathBusEscape(name))
}

// unitName returns the unescaped base element of the supplied escaped path.
func unitName(dpath dbus.ObjectPath) string {
	return pathBusUnescape(path.Base(string(dpath)))
}

// JobStatus holds a currently queued job definition.
type JobStatus struct {
	Id       uint32          // The numeric job id
	Unit     string          // The primary unit name for this job
	JobType  string          // The job type as string
	Status   string          // The job state as string
	JobPath  dbus.ObjectPath // The job object path
	UnitPath dbus.ObjectPath // The unit object path
}

// Deprecated: use ListJobsContext instead.
func (c *Conn) ListJobs() ([]JobStatus, error) {
	return c.ListJobsContext(context.Background())
}

// ListJobsContext returns an array with all currently queued jobs.
func (c *Conn) ListJobsContext(ctx context.Context) ([]JobStatus, error) {
	return c.listJobsInternal(ctx)
}

func (c *Conn) listJobsInternal(ctx context.Context) ([]JobStatus, error) {
	result := make([][]interface{}, 0)
	if err := c.sysobj.CallWithContext(ctx, "org.freedesktop.systemd1.Manager.ListJobs", 0).Store(&result); err != nil {
		return nil, err
	}

	resultInterface := make([]interface{}, len(result))
	for i := range result {
		resultInterface[i] = result[i]
	}

	status := make([]JobStatus, len(result))
	statusInterface := make([]interface{}, len(status))
	for i := range status {
		statusInterface[i] = &status[i]
	}

	if err := dbus.Store(resultInterface, statusInterface...); err != nil {
		return nil, err
	}

	return status, nil
}

// Freeze the cgroup associated with the unit.
// Note that FreezeUnit and ThawUnit are only supported on systems running with cgroup v2.
func (c *Conn) FreezeUnit(ctx context.Context, unit string) error {
	return c.sysobj.CallWithContext(ctx, "org.freedesktop.systemd1.Manager.FreezeUnit", 0, unit).Store()
}

// Unfreeze the cgroup associated with the unit.
func (c *Conn) ThawUnit(ctx context.Context, unit string) error {
	return c.sysobj.CallWithContext(ctx, "org.freedesktop.systemd1.Manager.ThawUnit", 0, unit).Store()
}
