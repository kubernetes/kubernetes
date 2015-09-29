/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package downwardapi

import (
	"io/ioutil"
	"os"
	"path"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/fieldpath"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util"
	utilErrors "k8s.io/kubernetes/pkg/util/errors"
	"k8s.io/kubernetes/pkg/volume"

	"github.com/golang/glog"
)

// ProbeVolumePlugins is the entry point for plugin detection in a package.
func ProbeVolumePlugins() []volume.VolumePlugin {
	return []volume.VolumePlugin{&downwardAPIPlugin{}}
}

const (
	downwardAPIPluginName = "kubernetes.io/downward-api"
)

// downwardAPIPlugin implements the VolumePlugin interface.
type downwardAPIPlugin struct {
	host volume.VolumeHost
}

var _ volume.VolumePlugin = &downwardAPIPlugin{}

func (plugin *downwardAPIPlugin) Init(host volume.VolumeHost) {
	plugin.host = host
}

func (plugin *downwardAPIPlugin) Name() string {
	return downwardAPIPluginName
}

func (plugin *downwardAPIPlugin) CanSupport(spec *volume.Spec) bool {
	return spec.Volume != nil && spec.Volume.DownwardAPI != nil
}

func (plugin *downwardAPIPlugin) NewBuilder(spec *volume.Spec, pod *api.Pod, opts volume.VolumeOptions) (volume.Builder, error) {
	v := &downwardAPIVolume{
		volName: spec.Name(),
		pod:     pod,
		podUID:  pod.UID,
		plugin:  plugin,
	}
	v.fieldReferenceFileNames = make(map[string]string)
	for _, fileInfo := range spec.Volume.DownwardAPI.Items {
		v.fieldReferenceFileNames[fileInfo.FieldRef.FieldPath] = path.Clean(fileInfo.Path)
	}
	return &downwardAPIVolumeBuilder{
		downwardAPIVolume: v,
		opts:              &opts}, nil
}

func (plugin *downwardAPIPlugin) NewCleaner(volName string, podUID types.UID) (volume.Cleaner, error) {
	return &downwardAPIVolumeCleaner{&downwardAPIVolume{volName: volName, podUID: podUID, plugin: plugin}}, nil
}

// downwardAPIVolume retrieves downward API data and placing them into the volume on the host.
type downwardAPIVolume struct {
	volName                 string
	fieldReferenceFileNames map[string]string
	pod                     *api.Pod
	podUID                  types.UID // TODO: remove this redundancy as soon NewCleaner func will have *api.POD and not only types.UID
	plugin                  *downwardAPIPlugin
}

// This is the spec for the volume that this plugin wraps.
var wrappedVolumeSpec = &volume.Spec{
	Volume: &api.Volume{VolumeSource: api.VolumeSource{EmptyDir: &api.EmptyDirVolumeSource{Medium: api.StorageMediumMemory}}},
}

// downwardAPIVolumeBuilder fetches info from downward API from the pod
// and dumps it in files
type downwardAPIVolumeBuilder struct {
	*downwardAPIVolume
	opts *volume.VolumeOptions
}

// downwardAPIVolumeBuilder implements volume.Builder interface
var _ volume.Builder = &downwardAPIVolumeBuilder{}

// SetUp puts in place the volume plugin.
// This function is not idempotent by design. We want the data to be refreshed periodically.
// The internal sync interval of kubelet will drive the refresh of data.
// TODO: Add volume specific ticker and refresh loop
func (b *downwardAPIVolumeBuilder) SetUp() error {
	return b.SetUpAt(b.GetPath())
}

func (b *downwardAPIVolumeBuilder) SetUpAt(dir string) error {
	glog.V(3).Infof("Setting up a downwardAPI volume %v for pod %v/%v at %v", b.volName, b.pod.Namespace, b.pod.Name, dir)
	// Wrap EmptyDir. Here we rely on the idempotency of the wrapped plugin to avoid repeatedly mounting
	wrapped, err := b.plugin.host.NewWrapperBuilder(wrappedVolumeSpec, b.pod, *b.opts)
	if err != nil {
		glog.Errorf("Couldn't setup downwardAPI volume %v for pod %v/%v: %s", b.volName, b.pod.Namespace, b.pod.Name, err.Error())
		return err
	}
	if err := wrapped.SetUpAt(dir); err != nil {
		glog.Errorf("Unable to setup downwardAPI volume %v for pod %v/%v: %s", b.volName, b.pod.Namespace, b.pod.Name, err.Error())
		return err
	}

	data, err := b.collectData()
	if err != nil {
		glog.Errorf("Error preparing data for downwardAPI volume %v for pod %v/%v: %s", b.volName, b.pod.Namespace, b.pod.Name, err.Error())
		return err
	}

	if !b.isDataChanged(data) {
		// No data changed: nothing to write
		return nil
	}

	if err := b.writeData(data); err != nil {
		glog.Errorf("Unable to dump files for downwardAPI volume %v for pod %v/%v: %s", b.volName, b.pod.Namespace, b.pod.Name, err.Error())
		return err
	}

	glog.V(3).Infof("Data dumped for downwardAPI volume %v for pod %v/%v", b.volName, b.pod.Namespace, b.pod.Name)
	return nil
}

// IsReadOnly func to fullfill volume.Builder interface
func (d *downwardAPIVolume) IsReadOnly() bool {
	return true
}

// collectData collects requested downwardAPI in data map.
// Map's key is the requested name of file to dump
// Map's value is the (sorted) content of the field to be dumped in the file.
func (d *downwardAPIVolume) collectData() (map[string]string, error) {
	errlist := []error{}
	data := make(map[string]string)
	for fieldReference, fileName := range d.fieldReferenceFileNames {
		if values, err := fieldpath.ExtractFieldPathAsString(d.pod, fieldReference); err != nil {
			glog.Errorf("Unable to extract field %s: %s", fieldReference, err.Error())
			errlist = append(errlist, err)
		} else {
			data[fileName] = sortLines(values)
		}
	}
	return data, utilErrors.NewAggregate(errlist)
}

// isDataChanged iterate over all the entries to check wether at least one
// file needs to be updated.
func (d *downwardAPIVolume) isDataChanged(data map[string]string) bool {
	for fileName, values := range data {
		if isFileToGenerate(path.Join(d.GetPath(), fileName), values) {
			return true
		}
	}
	return false
}

// isFileToGenerate compares actual file with the new values. If
// different (or the file does not exist) return true
func isFileToGenerate(fileName, values string) bool {
	if _, err := os.Lstat(fileName); os.IsNotExist(err) {
		return true
	}
	return readFile(fileName) != values
}

const (
	downwardAPIDir    = "..downwardapi"
	downwardAPITmpDir = "..downwardapi_tmp"
	// It seems reasonable to allow dot-files in the config, so we reserved double-dot-files for the implementation".
)

// writeData writes requested downwardAPI in specified files.
//
// The file visible in this volume are symlinks to files in the '..downwardapi'
// directory. Actual files are stored in an hidden timestamped directory which is
// symlinked to by '..downwardapi'. The timestamped directory and '..downwardapi' symlink
// are created in the plugin root dir.  This scheme allows the files to be
// atomically updated by changing the target of the '..downwardapi' symlink.  When new
// data is available:
//
// 1.  A new timestamped dir is created by writeDataInTimestampDir and requested data
//     is written inside new timestamped directory
// 2.  Symlinks and directory for new files are created (if needed).
//     For example for files:
//       <volume-dir>/user_space/labels
//       <volume-dir>/k8s_space/annotations
//       <volume-dir>/podName
//     This structure is created:
//       <volume-dir>/podName               -> ..downwardapi/podName
//       <volume-dir>/user_space/labels     -> ../..downwardapi/user_space/labels
//       <volume-dir>/k8s_space/annotations -> ../..downwardapi/k8s_space/annotations
//       <volume-dir>/..downwardapi         -> ..downwardapi.12345678
//     where ..downwardapi.12345678 is a randomly generated directory which contains
//     the real data. If a file has to be dumped in subdirectory (for example <volume-dir>/user_space/labels)
//     plugin builds a relative symlink (<volume-dir>/user_space/labels -> ../..downwardapi/user_space/labels)
// 3.  The previous timestamped directory is detected reading the '..downwardapi' symlink
// 4.  In case no symlink exists then it's created
// 5.  In case symlink exists a new temporary symlink is created ..downwardapi_tmp
// 6.  ..downwardapi_tmp is renamed to ..downwardapi
// 7.  The previous timestamped directory is removed

func (d *downwardAPIVolume) writeData(data map[string]string) error {
	timestampDir, err := d.writeDataInTimestampDir(data)
	if err != nil {
		glog.Errorf("Unable to write data in temporary directory: %s", err.Error())
		return err
	}
	// update symbolic links for relative paths
	if err = d.updateSymlinksToCurrentDir(); err != nil {
		os.RemoveAll(timestampDir)
		glog.Errorf("Unable to create symlinks and/or directory: %s", err.Error())
		return err
	}

	_, timestampDirBaseName := filepath.Split(timestampDir)
	var oldTimestampDirectory string
	oldTimestampDirectory, err = os.Readlink(path.Join(d.GetPath(), downwardAPIDir))

	if err = os.Symlink(timestampDirBaseName, path.Join(d.GetPath(), downwardAPITmpDir)); err != nil {
		os.RemoveAll(timestampDir)
		glog.Errorf("Unable to create symolic link: %s", err.Error())
		return err
	}

	// Rename the symbolic link downwardAPITmpDir to downwardAPIDir
	if err = os.Rename(path.Join(d.GetPath(), downwardAPITmpDir), path.Join(d.GetPath(), downwardAPIDir)); err != nil {
		// in case of error remove latest data and downwardAPITmpDir
		os.Remove(path.Join(d.GetPath(), downwardAPITmpDir))
		os.RemoveAll(timestampDir)
		glog.Errorf("Unable to rename symbolic link: %s", err.Error())
		return err
	}
	// Remove oldTimestampDirectory
	if len(oldTimestampDirectory) > 0 {
		if err := os.RemoveAll(path.Join(d.GetPath(), oldTimestampDirectory)); err != nil {
			glog.Errorf("Unable to remove directory: %s", err.Error())
			return err
		}
	}
	return nil
}

// writeDataInTimestampDir writes the latest data into a new temporary directory with a timestamp.
func (d *downwardAPIVolume) writeDataInTimestampDir(data map[string]string) (string, error) {
	errlist := []error{}
	timestampDir, err := ioutil.TempDir(d.GetPath(), ".."+time.Now().Format("2006_01_02_15_04_05"))
	for fileName, values := range data {
		fullPathFile := path.Join(timestampDir, fileName)
		dir, _ := filepath.Split(fullPathFile)
		if err = os.MkdirAll(dir, os.ModePerm); err != nil {
			glog.Errorf("Unable to create directory `%s`: %s", dir, err.Error())
			return "", err
		}
		if err := ioutil.WriteFile(fullPathFile, []byte(values), 0644); err != nil {
			glog.Errorf("Unable to write file `%s`: %s", fullPathFile, err.Error())
			errlist = append(errlist, err)
		}
	}
	return timestampDir, utilErrors.NewAggregate(errlist)
}

// updateSymlinksToCurrentDir creates the relative symlinks for all the files configured in this volume.
// If the directory in a file path does not exist, it is created.
//
// For example for files: "bar", "foo/bar", "baz/bar", "foo/baz/blah"
// the following symlinks and subdirectory are created:
// bar          -> ..downwardapi/bar
// baz/bar      -> ../..downwardapi/baz/bar
// foo/bar      -> ../..downwardapi/foo/bar
// foo/baz/blah -> ../../..downwardapi/foo/baz/blah
func (d *downwardAPIVolume) updateSymlinksToCurrentDir() error {
	for _, f := range d.fieldReferenceFileNames {
		dir, _ := filepath.Split(f)
		nbOfSubdir := 0
		if len(dir) > 0 {
			// if dir is not empty f contains at least a subdirectory (for example: f="foo/bar")
			// since filepath.Split leaves a trailing '/'  we have dir="foo/"
			// and since len(strings.Split"foo/")=2 to count the number
			// of sub directory you need to remove 1
			nbOfSubdir = len(strings.Split(dir, "/")) - 1
			if err := os.MkdirAll(path.Join(d.GetPath(), dir), os.ModePerm); err != nil {
				return err
			}
		}
		if _, err := os.Readlink(path.Join(d.GetPath(), f)); err != nil {
			// link does not exist create it
			presentedFile := path.Join(strings.Repeat("../", nbOfSubdir), downwardAPIDir, f)
			actualFile := path.Join(d.GetPath(), f)
			if err := os.Symlink(presentedFile, actualFile); err != nil {
				return err
			}
		}
	}
	return nil
}

// readFile reads the file at the given path and returns the content as a string.
func readFile(path string) string {
	if data, err := ioutil.ReadFile(path); err == nil {
		return string(data)
	}
	return ""
}

// sortLines sorts the strings generated from map based data
// (annotations and labels)
func sortLines(values string) string {
	splitted := strings.Split(values, "\n")
	sort.Strings(splitted)
	return strings.Join(splitted, "\n")
}

func (d *downwardAPIVolume) GetPath() string {
	return d.plugin.host.GetPodVolumeDir(d.podUID, util.EscapeQualifiedNameForDisk(downwardAPIPluginName), d.volName)
}

// downwardAPIVolumeCleander handles cleaning up downwardAPI volumes
type downwardAPIVolumeCleaner struct {
	*downwardAPIVolume
}

// downwardAPIVolumeCleaner implements volume.Cleaner interface
var _ volume.Cleaner = &downwardAPIVolumeCleaner{}

func (c *downwardAPIVolumeCleaner) TearDown() error {
	return c.TearDownAt(c.GetPath())
}

func (c *downwardAPIVolumeCleaner) TearDownAt(dir string) error {
	glog.V(3).Infof("Tearing down volume %v for pod %v at %v", c.volName, c.podUID, dir)

	// Wrap EmptyDir, let it do the teardown.
	wrapped, err := c.plugin.host.NewWrapperCleaner(wrappedVolumeSpec, c.podUID)
	if err != nil {
		return err
	}
	return wrapped.TearDownAt(dir)
}

func (b *downwardAPIVolumeBuilder) getMetaDir() string {
	return path.Join(b.plugin.host.GetPodPluginDir(b.podUID, util.EscapeQualifiedNameForDisk(downwardAPIPluginName)), b.volName)
}
