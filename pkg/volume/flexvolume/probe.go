/*
Copyright 2017 The Kubernetes Authors.

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

package flexvolume

import (
	"k8s.io/klog"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/utils/exec"

	"os"

	"fmt"
	"path/filepath"
	"sync"

	"strings"

	"github.com/fsnotify/fsnotify"
	"k8s.io/apimachinery/pkg/util/errors"
	utilfs "k8s.io/kubernetes/pkg/util/filesystem"
	utilstrings "k8s.io/kubernetes/pkg/util/strings"
)

type flexVolumeProber struct {
	mutex          sync.Mutex
	pluginDir      string         // Flexvolume driver directory
	runner         exec.Interface // Interface to use for execing flex calls
	watcher        utilfs.FSWatcher
	factory        PluginFactory
	fs             utilfs.Filesystem
	probeAllNeeded bool
	eventsMap      map[string]volume.ProbeOperation // the key is the driver directory path, the value is the coresponding operation
}

// GetDynamicPluginProber creates dynamic plugin prober
func GetDynamicPluginProber(pluginDir string, runner exec.Interface) volume.DynamicPluginProber {
	return &flexVolumeProber{
		pluginDir: pluginDir,
		watcher:   utilfs.NewFsnotifyWatcher(),
		factory:   pluginFactory{},
		runner:    runner,
		fs:        &utilfs.DefaultFs{},
	}
}

func (prober *flexVolumeProber) Init() error {
	prober.testAndSetProbeAllNeeded(true)
	prober.eventsMap = map[string]volume.ProbeOperation{}

	if err := prober.createPluginDir(); err != nil {
		return err
	}
	if err := prober.initWatcher(); err != nil {
		return err
	}

	return nil
}

// If probeAllNeeded is true, probe all pluginDir
// else probe events in eventsMap
func (prober *flexVolumeProber) Probe() (events []volume.ProbeEvent, err error) {
	if prober.probeAllNeeded {
		prober.testAndSetProbeAllNeeded(false)
		return prober.probeAll()
	}

	return prober.probeMap()
}

func (prober *flexVolumeProber) probeMap() (events []volume.ProbeEvent, err error) {
	// TODO use a concurrent map to avoid Locking the entire map
	prober.mutex.Lock()
	defer prober.mutex.Unlock()
	probeEvents := []volume.ProbeEvent{}
	allErrs := []error{}
	for driverDirPathAbs, op := range prober.eventsMap {
		driverDirName := filepath.Base(driverDirPathAbs) // e.g. driverDirName = vendor~cifs
		probeEvent, pluginErr := prober.newProbeEvent(driverDirName, op)
		if pluginErr != nil {
			allErrs = append(allErrs, pluginErr)
			continue
		}
		probeEvents = append(probeEvents, probeEvent)

		delete(prober.eventsMap, driverDirPathAbs)
	}
	return probeEvents, errors.NewAggregate(allErrs)
}

func (prober *flexVolumeProber) probeAll() (events []volume.ProbeEvent, err error) {
	probeEvents := []volume.ProbeEvent{}
	allErrs := []error{}
	files, err := prober.fs.ReadDir(prober.pluginDir)
	if err != nil {
		return nil, fmt.Errorf("Error reading the Flexvolume directory: %s", err)
	}
	for _, f := range files {
		// only directories with names that do not begin with '.' are counted as plugins
		// and pluginDir/dirname/dirname should be an executable
		// unless dirname contains '~' for escaping namespace
		// e.g. dirname = vendor~cifs
		// then, executable will be pluginDir/dirname/cifs
		if f.IsDir() && filepath.Base(f.Name())[0] != '.' {
			probeEvent, pluginErr := prober.newProbeEvent(f.Name(), volume.ProbeAddOrUpdate)
			if pluginErr != nil {
				allErrs = append(allErrs, pluginErr)
				continue
			}
			probeEvents = append(probeEvents, probeEvent)
		}
	}
	return probeEvents, errors.NewAggregate(allErrs)
}

func (prober *flexVolumeProber) newProbeEvent(driverDirName string, op volume.ProbeOperation) (volume.ProbeEvent, error) {
	probeEvent := volume.ProbeEvent{
		Op: op,
	}
	if op == volume.ProbeAddOrUpdate {
		plugin, pluginErr := prober.factory.NewFlexVolumePlugin(prober.pluginDir, driverDirName, prober.runner)
		if pluginErr != nil {
			pluginErr = fmt.Errorf(
				"Error creating Flexvolume plugin from directory %s, skipping. Error: %s",
				driverDirName, pluginErr)
			return probeEvent, pluginErr
		}
		probeEvent.Plugin = plugin
		probeEvent.PluginName = plugin.GetPluginName()
	} else if op == volume.ProbeRemove {
		driverName := utilstrings.UnescapePluginName(driverDirName)
		probeEvent.PluginName = flexVolumePluginNamePrefix + driverName

	} else {
		return probeEvent, fmt.Errorf("Unknown Operation on directory: %s. ", driverDirName)
	}
	return probeEvent, nil
}

func (prober *flexVolumeProber) handleWatchEvent(event fsnotify.Event) error {
	// event.Name is the watched path.
	if filepath.Base(event.Name)[0] == '.' {
		// Ignore files beginning with '.'
		return nil
	}

	eventPathAbs, err := filepath.Abs(event.Name)
	if err != nil {
		return err
	}
	parentPathAbs := filepath.Dir(eventPathAbs)
	pluginDirAbs, err := filepath.Abs(prober.pluginDir)
	if err != nil {
		return err
	}

	// event of pluginDirAbs
	if eventPathAbs == pluginDirAbs {
		// If the Flexvolume plugin directory is removed, need to recreate it
		// in order to keep it under watch.
		if eventOpIs(event, fsnotify.Remove) {
			if err := prober.createPluginDir(); err != nil {
				return err
			}
			if err := prober.addWatchRecursive(pluginDirAbs); err != nil {
				return err
			}
		}
		return nil
	}

	// watch newly added subdirectories inside a driver directory
	if eventOpIs(event, fsnotify.Create) {
		if err := prober.addWatchRecursive(eventPathAbs); err != nil {
			return err
		}
	}

	eventRelPathToPluginDir, err := filepath.Rel(pluginDirAbs, eventPathAbs)
	if err != nil {
		return err
	}

	// event inside specific driver dir
	if len(eventRelPathToPluginDir) > 0 {
		driverDirName := strings.Split(eventRelPathToPluginDir, string(os.PathSeparator))[0]
		driverDirAbs := filepath.Join(pluginDirAbs, driverDirName)
		// executable is removed, will trigger ProbeRemove event
		if eventOpIs(event, fsnotify.Remove) && (eventRelPathToPluginDir == getExecutablePathRel(driverDirName) || parentPathAbs == pluginDirAbs) {
			prober.updateEventsMap(driverDirAbs, volume.ProbeRemove)
		} else {
			prober.updateEventsMap(driverDirAbs, volume.ProbeAddOrUpdate)
		}
	}

	return nil
}

// getExecutableName returns the executableName of a flex plugin
func getExecutablePathRel(driverDirName string) string {
	parts := strings.Split(driverDirName, "~")
	return filepath.Join(driverDirName, parts[len(parts)-1])
}

func (prober *flexVolumeProber) updateEventsMap(eventDirAbs string, op volume.ProbeOperation) {
	prober.mutex.Lock()
	defer prober.mutex.Unlock()
	if prober.probeAllNeeded {
		return
	}
	prober.eventsMap[eventDirAbs] = op
}

// Recursively adds to watch all directories inside and including the file specified by the given filename.
// If the file is a symlink to a directory, it will watch the symlink but not any of the subdirectories.
//
// Each file or directory change triggers two events: one from the watch on itself, another from the watch
// on its parent directory.
func (prober *flexVolumeProber) addWatchRecursive(filename string) error {
	addWatch := func(path string, info os.FileInfo, err error) error {
		if err == nil && info.IsDir() {
			if err := prober.watcher.AddWatch(path); err != nil {
				klog.Errorf("Error recursively adding watch: %v", err)
			}
		}
		return nil
	}
	return prober.fs.Walk(filename, addWatch)
}

// Creates a new filesystem watcher and adds watches for the plugin directory
// and all of its subdirectories.
func (prober *flexVolumeProber) initWatcher() error {
	err := prober.watcher.Init(func(event fsnotify.Event) {
		if err := prober.handleWatchEvent(event); err != nil {
			klog.Errorf("Flexvolume prober watch: %s", err)
		}
	}, func(err error) {
		klog.Errorf("Received an error from watcher: %s", err)
	})
	if err != nil {
		return fmt.Errorf("Error initializing watcher: %s", err)
	}

	if err := prober.addWatchRecursive(prober.pluginDir); err != nil {
		return fmt.Errorf("Error adding watch on Flexvolume directory: %s", err)
	}

	prober.watcher.Run()

	return nil
}

// Creates the plugin directory, if it doesn't already exist.
func (prober *flexVolumeProber) createPluginDir() error {
	if _, err := prober.fs.Stat(prober.pluginDir); os.IsNotExist(err) {
		klog.Warningf("Flexvolume plugin directory at %s does not exist. Recreating.", prober.pluginDir)
		err := prober.fs.MkdirAll(prober.pluginDir, 0755)
		if err != nil {
			return fmt.Errorf("Error (re-)creating driver directory: %s", err)
		}
	}

	return nil
}

func (prober *flexVolumeProber) testAndSetProbeAllNeeded(newval bool) (oldval bool) {
	prober.mutex.Lock()
	defer prober.mutex.Unlock()
	oldval, prober.probeAllNeeded = prober.probeAllNeeded, newval
	return
}

func eventOpIs(event fsnotify.Event, op fsnotify.Op) bool {
	return event.Op&op == op
}
