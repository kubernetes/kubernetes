// Copyright 2015 CoreOS, Inc.
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
	"errors"
	"path"
	"strconv"

	"github.com/godbus/dbus"
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

func (c *Conn) startJob(ch chan<- string, job string, args ...interface{}) (int, error) {
	if ch != nil {
		c.jobListener.Lock()
		defer c.jobListener.Unlock()
	}

	var p dbus.ObjectPath
	err := c.sysobj.Call(job, 0, args...).Store(&p)
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

// StartUnit enqueues a start job and depending jobs, if any (unless otherwise
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
func (c *Conn) StartUnit(name string, mode string, ch chan<- string) (int, error) {
	return c.startJob(ch, "org.freedesktop.systemd1.Manager.StartUnit", name, mode)
}

// StopUnit is similar to StartUnit but stops the specified unit rather
// than starting it.
func (c *Conn) StopUnit(name string, mode string, ch chan<- string) (int, error) {
	return c.startJob(ch, "org.freedesktop.systemd1.Manager.StopUnit", name, mode)
}

// ReloadUnit reloads a unit.  Reloading is done only if the unit is already running and fails otherwise.
func (c *Conn) ReloadUnit(name string, mode string, ch chan<- string) (int, error) {
	return c.startJob(ch, "org.freedesktop.systemd1.Manager.ReloadUnit", name, mode)
}

// RestartUnit restarts a service.  If a service is restarted that isn't
// running it will be started.
func (c *Conn) RestartUnit(name string, mode string, ch chan<- string) (int, error) {
	return c.startJob(ch, "org.freedesktop.systemd1.Manager.RestartUnit", name, mode)
}

// TryRestartUnit is like RestartUnit, except that a service that isn't running
// is not affected by the restart.
func (c *Conn) TryRestartUnit(name string, mode string, ch chan<- string) (int, error) {
	return c.startJob(ch, "org.freedesktop.systemd1.Manager.TryRestartUnit", name, mode)
}

// ReloadOrRestart attempts a reload if the unit supports it and use a restart
// otherwise.
func (c *Conn) ReloadOrRestartUnit(name string, mode string, ch chan<- string) (int, error) {
	return c.startJob(ch, "org.freedesktop.systemd1.Manager.ReloadOrRestartUnit", name, mode)
}

// ReloadOrTryRestart attempts a reload if the unit supports it and use a "Try"
// flavored restart otherwise.
func (c *Conn) ReloadOrTryRestartUnit(name string, mode string, ch chan<- string) (int, error) {
	return c.startJob(ch, "org.freedesktop.systemd1.Manager.ReloadOrTryRestartUnit", name, mode)
}

// StartTransientUnit() may be used to create and start a transient unit, which
// will be released as soon as it is not running or referenced anymore or the
// system is rebooted. name is the unit name including suffix, and must be
// unique. mode is the same as in StartUnit(), properties contains properties
// of the unit.
func (c *Conn) StartTransientUnit(name string, mode string, properties []Property, ch chan<- string) (int, error) {
	return c.startJob(ch, "org.freedesktop.systemd1.Manager.StartTransientUnit", name, mode, properties, make([]PropertyCollection, 0))
}

// KillUnit takes the unit name and a UNIX signal number to send.  All of the unit's
// processes are killed.
func (c *Conn) KillUnit(name string, signal int32) {
	c.sysobj.Call("org.freedesktop.systemd1.Manager.KillUnit", 0, name, "all", signal).Store()
}

// ResetFailedUnit resets the "failed" state of a specific unit.
func (c *Conn) ResetFailedUnit(name string) error {
	return c.sysobj.Call("org.freedesktop.systemd1.Manager.ResetFailedUnit", 0, name).Store()
}

// getProperties takes the unit name and returns all of its dbus object properties, for the given dbus interface
func (c *Conn) getProperties(unit string, dbusInterface string) (map[string]interface{}, error) {
	var err error
	var props map[string]dbus.Variant

	path := unitPath(unit)
	if !path.IsValid() {
		return nil, errors.New("invalid unit name: " + unit)
	}

	obj := c.sysconn.Object("org.freedesktop.systemd1", path)
	err = obj.Call("org.freedesktop.DBus.Properties.GetAll", 0, dbusInterface).Store(&props)
	if err != nil {
		return nil, err
	}

	out := make(map[string]interface{}, len(props))
	for k, v := range props {
		out[k] = v.Value()
	}

	return out, nil
}

// GetUnitProperties takes the unit name and returns all of its dbus object properties.
func (c *Conn) GetUnitProperties(unit string) (map[string]interface{}, error) {
	return c.getProperties(unit, "org.freedesktop.systemd1.Unit")
}

func (c *Conn) getProperty(unit string, dbusInterface string, propertyName string) (*Property, error) {
	var err error
	var prop dbus.Variant

	path := unitPath(unit)
	if !path.IsValid() {
		return nil, errors.New("invalid unit name: " + unit)
	}

	obj := c.sysconn.Object("org.freedesktop.systemd1", path)
	err = obj.Call("org.freedesktop.DBus.Properties.Get", 0, dbusInterface, propertyName).Store(&prop)
	if err != nil {
		return nil, err
	}

	return &Property{Name: propertyName, Value: prop}, nil
}

func (c *Conn) GetUnitProperty(unit string, propertyName string) (*Property, error) {
	return c.getProperty(unit, "org.freedesktop.systemd1.Unit", propertyName)
}

// GetServiceProperty returns property for given service name and property name
func (c *Conn) GetServiceProperty(service string, propertyName string) (*Property, error) {
	return c.getProperty(service, "org.freedesktop.systemd1.Service", propertyName)
}

// GetUnitTypeProperties returns the extra properties for a unit, specific to the unit type.
// Valid values for unitType: Service, Socket, Target, Device, Mount, Automount, Snapshot, Timer, Swap, Path, Slice, Scope
// return "dbus.Error: Unknown interface" if the unitType is not the correct type of the unit
func (c *Conn) GetUnitTypeProperties(unit string, unitType string) (map[string]interface{}, error) {
	return c.getProperties(unit, "org.freedesktop.systemd1."+unitType)
}

// SetUnitProperties() may be used to modify certain unit properties at runtime.
// Not all properties may be changed at runtime, but many resource management
// settings (primarily those in systemd.cgroup(5)) may. The changes are applied
// instantly, and stored on disk for future boots, unless runtime is true, in which
// case the settings only apply until the next reboot. name is the name of the unit
// to modify. properties are the settings to set, encoded as an array of property
// name and value pairs.
func (c *Conn) SetUnitProperties(name string, runtime bool, properties ...Property) error {
	return c.sysobj.Call("org.freedesktop.systemd1.Manager.SetUnitProperties", 0, name, runtime, properties).Store()
}

func (c *Conn) GetUnitTypeProperty(unit string, unitType string, propertyName string) (*Property, error) {
	return c.getProperty(unit, "org.freedesktop.systemd1."+unitType, propertyName)
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

// ListUnits returns an array with all currently loaded units. Note that
// units may be known by multiple names at the same time, and hence there might
// be more unit names loaded than actual units behind them.
func (c *Conn) ListUnits() ([]UnitStatus, error) {
	result := make([][]interface{}, 0)
	err := c.sysobj.Call("org.freedesktop.systemd1.Manager.ListUnits", 0).Store(&result)
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

type UnitFile struct {
	Path string
	Type string
}

// ListUnitFiles returns an array of all available units on disk.
func (c *Conn) ListUnitFiles() ([]UnitFile, error) {
	result := make([][]interface{}, 0)
	err := c.sysobj.Call("org.freedesktop.systemd1.Manager.ListUnitFiles", 0).Store(&result)
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

type LinkUnitFileChange EnableUnitFileChange

// LinkUnitFiles() links unit files (that are located outside of the
// usual unit search paths) into the unit search path.
//
// It takes a list of absolute paths to unit files to link and two
// booleans. The first boolean controls whether the unit shall be
// enabled for runtime only (true, /run), or persistently (false,
// /etc).
// The second controls whether symlinks pointing to other units shall
// be replaced if necessary.
//
// This call returns a list of the changes made. The list consists of
// structures with three strings: the type of the change (one of symlink
// or unlink), the file name of the symlink and the destination of the
// symlink.
func (c *Conn) LinkUnitFiles(files []string, runtime bool, force bool) ([]LinkUnitFileChange, error) {
	result := make([][]interface{}, 0)
	err := c.sysobj.Call("org.freedesktop.systemd1.Manager.LinkUnitFiles", 0, files, runtime, force).Store(&result)
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

// EnableUnitFiles() may be used to enable one or more units in the system (by
// creating symlinks to them in /etc or /run).
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
func (c *Conn) EnableUnitFiles(files []string, runtime bool, force bool) (bool, []EnableUnitFileChange, error) {
	var carries_install_info bool

	result := make([][]interface{}, 0)
	err := c.sysobj.Call("org.freedesktop.systemd1.Manager.EnableUnitFiles", 0, files, runtime, force).Store(&carries_install_info, &result)
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

// DisableUnitFiles() may be used to disable one or more units in the system (by
// removing symlinks to them from /etc or /run).
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
func (c *Conn) DisableUnitFiles(files []string, runtime bool) ([]DisableUnitFileChange, error) {
	result := make([][]interface{}, 0)
	err := c.sysobj.Call("org.freedesktop.systemd1.Manager.DisableUnitFiles", 0, files, runtime).Store(&result)
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

// Reload instructs systemd to scan for and reload unit files. This is
// equivalent to a 'systemctl daemon-reload'.
func (c *Conn) Reload() error {
	return c.sysobj.Call("org.freedesktop.systemd1.Manager.Reload", 0).Store()
}

func unitPath(name string) dbus.ObjectPath {
	return dbus.ObjectPath("/org/freedesktop/systemd1/unit/" + PathBusEscape(name))
}
