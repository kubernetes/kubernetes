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
	"fmt"
	"path"
	"testing"

	"github.com/fsnotify/fsnotify"
	"github.com/stretchr/testify/assert"
	utilfs "k8s.io/kubernetes/pkg/util/filesystem"
	"k8s.io/kubernetes/pkg/volume"
)

const (
	pluginDir  = "/flexvolume"
	driverName = "fake-driver"
)

// Probes a driver installed before prober initialization.
func TestProberExistingDriverBeforeInit(t *testing.T) {
	// Arrange
	driverPath, _, watcher, prober := initTestEnvironment(t)

	// Act
	updated, plugins, err := prober.Probe()

	// Assert
	// Probe occurs, 1 plugin should be returned, and 2 watches (pluginDir and all its
	// current subdirectories) registered.
	assert.True(t, updated)
	assert.Equal(t, 1, len(plugins))
	assert.Equal(t, pluginDir, watcher.watches[0])
	assert.Equal(t, driverPath, watcher.watches[1])
	assert.NoError(t, err)

	// Should no longer probe.

	// Act
	updated, plugins, err = prober.Probe()
	// Assert
	assert.False(t, updated)
	assert.Equal(t, 0, len(plugins))
	assert.NoError(t, err)
}

// Probes newly added drivers after prober is running.
func TestProberAddDriver(t *testing.T) {
	// Arrange
	_, fs, watcher, prober := initTestEnvironment(t)
	prober.Probe()
	updated, _, _ := prober.Probe()
	assert.False(t, updated)

	// Call probe after a file is added. Should return true.

	// Arrange
	const driverName2 = "fake-driver2"
	driverPath := path.Join(pluginDir, driverName2)
	installDriver(driverName2, fs)
	watcher.TriggerEvent(fsnotify.Create, driverPath)
	watcher.TriggerEvent(fsnotify.Create, path.Join(driverPath, driverName2))

	// Act
	updated, plugins, err := prober.Probe()

	// Assert
	assert.True(t, updated)
	assert.Equal(t, 2, len(plugins))                                     // 1 existing, 1 newly added
	assert.Equal(t, driverPath, watcher.watches[len(watcher.watches)-1]) // Checks most recent watch
	assert.NoError(t, err)

	// Call probe again, should return false.

	// Act
	updated, _, err = prober.Probe()
	// Assert
	assert.False(t, updated)
	assert.NoError(t, err)

	// Call probe after a non-driver file is added in a subdirectory. Should return true.

	// Arrange
	fp := path.Join(driverPath, "dummyfile")
	fs.Create(fp)
	watcher.TriggerEvent(fsnotify.Create, fp)

	// Act
	updated, plugins, err = prober.Probe()

	// Assert
	assert.True(t, updated)
	assert.Equal(t, 2, len(plugins)) // Number of plugins should not change.
	assert.NoError(t, err)

	// Call probe again, should return false.
	// Act
	updated, _, err = prober.Probe()
	// Assert
	assert.False(t, updated)
	assert.NoError(t, err)
}

// Tests the behavior when no drivers exist in the plugin directory.
func TestEmptyPluginDir(t *testing.T) {
	// Arrange
	fs := utilfs.NewFakeFs()
	watcher := NewFakeWatcher()
	prober := &flexVolumeProber{
		pluginDir: pluginDir,
		watcher:   watcher,
		fs:        fs,
		factory:   fakePluginFactory{error: false},
	}
	prober.Init()

	// Act
	updated, plugins, err := prober.Probe()

	// Assert
	assert.True(t, updated)
	assert.Equal(t, 0, len(plugins))
	assert.NoError(t, err)
}

// Issue an event to remove plugindir. New directory should still be watched.
func TestRemovePluginDir(t *testing.T) {
	// Arrange
	driverPath, fs, watcher, _ := initTestEnvironment(t)
	fs.RemoveAll(pluginDir)
	watcher.TriggerEvent(fsnotify.Remove, path.Join(driverPath, driverName))
	watcher.TriggerEvent(fsnotify.Remove, driverPath)
	watcher.TriggerEvent(fsnotify.Remove, pluginDir)

	// Act: The handler triggered by the above events should have already handled the event appropriately.

	// Assert
	assert.Equal(t, 3, len(watcher.watches)) // 2 from initial setup, 1 from new watch.
	assert.Equal(t, pluginDir, watcher.watches[len(watcher.watches)-1])
}

// Issue multiple events and probe multiple times. Should give true, false, false...
func TestProberMultipleEvents(t *testing.T) {
	const iterations = 5

	// Arrange
	_, fs, watcher, prober := initTestEnvironment(t)
	for i := 0; i < iterations; i++ {
		newDriver := fmt.Sprintf("multi-event-driver%d", 1)
		installDriver(newDriver, fs)
		driverPath := path.Join(pluginDir, newDriver)
		watcher.TriggerEvent(fsnotify.Create, driverPath)
		watcher.TriggerEvent(fsnotify.Create, path.Join(driverPath, newDriver))
	}

	// Act
	updated, _, err := prober.Probe()

	// Assert
	assert.True(t, updated)
	assert.NoError(t, err)
	for i := 0; i < iterations-1; i++ {
		updated, _, err = prober.Probe()
		assert.False(t, updated)
		assert.NoError(t, err)
	}
}

// When many events are triggered quickly in succession, events should stop triggering a probe update
// after a certain limit.
func TestProberRateLimit(t *testing.T) {
	// Arrange
	driverPath, _, watcher, prober := initTestEnvironment(t)
	for i := 0; i < watchEventLimit; i++ {
		watcher.TriggerEvent(fsnotify.Write, path.Join(driverPath, driverName))
	}

	// Act
	updated, plugins, err := prober.Probe()

	// Assert
	// The probe results should not be different from what it would be if none of the events
	// are triggered.
	assert.True(t, updated)
	assert.Equal(t, 1, len(plugins))
	assert.NoError(t, err)

	// Arrange
	watcher.TriggerEvent(fsnotify.Write, path.Join(driverPath, driverName))

	// Act
	updated, _, err = prober.Probe()

	// Assert
	// The last event is outside the event limit. Should not trigger a probe.
	assert.False(t, updated)
	assert.NoError(t, err)
}

func TestProberError(t *testing.T) {
	fs := utilfs.NewFakeFs()
	watcher := NewFakeWatcher()
	prober := &flexVolumeProber{
		pluginDir: pluginDir,
		watcher:   watcher,
		fs:        fs,
		factory:   fakePluginFactory{error: true},
	}
	installDriver(driverName, fs)
	prober.Init()

	_, _, err := prober.Probe()
	assert.Error(t, err)
}

// Installs a mock driver (an empty file) in the mock fs.
func installDriver(driverName string, fs utilfs.Filesystem) {
	driverPath := path.Join(pluginDir, driverName)
	fs.MkdirAll(driverPath, 0666)
	fs.Create(path.Join(driverPath, driverName))
}

// Initializes mocks, installs a single driver in the mock fs, then initializes prober.
func initTestEnvironment(t *testing.T) (
	driverPath string,
	fs utilfs.Filesystem,
	watcher *fakeWatcher,
	prober volume.DynamicPluginProber) {
	fs = utilfs.NewFakeFs()
	watcher = NewFakeWatcher()
	prober = &flexVolumeProber{
		pluginDir: pluginDir,
		watcher:   watcher,
		fs:        fs,
		factory:   fakePluginFactory{error: false},
	}
	driverPath = path.Join(pluginDir, driverName)
	installDriver(driverName, fs)
	prober.Init()

	assert.NotNilf(t, watcher.eventHandler,
		"Expect watch event handler to be registered after prober init, but is not.")
	return
}

// Fake Flexvolume plugin
type fakePluginFactory struct {
	error bool // Indicates whether an error should be returned.
}

var _ PluginFactory = fakePluginFactory{}

func (m fakePluginFactory) NewFlexVolumePlugin(_, driverName string) (volume.VolumePlugin, error) {
	if m.error {
		return nil, fmt.Errorf("Flexvolume plugin error")
	}
	// Dummy Flexvolume plugin. Prober never interacts with the plugin.
	return &flexVolumePlugin{driverName: driverName}, nil
}
