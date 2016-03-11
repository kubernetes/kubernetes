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
	"sort"
	"strings"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/fieldpath"
	"k8s.io/kubernetes/pkg/types"
	utilerrors "k8s.io/kubernetes/pkg/util/errors"
	utilstrings "k8s.io/kubernetes/pkg/util/strings"
	"k8s.io/kubernetes/pkg/volume"
	volumeutil "k8s.io/kubernetes/pkg/volume/util"

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

var wrappedVolumeSpec = volume.Spec{
	Volume: &api.Volume{VolumeSource: api.VolumeSource{EmptyDir: &api.EmptyDirVolumeSource{Medium: api.StorageMediumMemory}}},
}

func (plugin *downwardAPIPlugin) Init(host volume.VolumeHost) error {
	plugin.host = host
	return nil
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
		opts:              &opts,
	}, nil
}

func (plugin *downwardAPIPlugin) NewCleaner(volName string, podUID types.UID) (volume.Cleaner, error) {
	return &downwardAPIVolumeCleaner{
		&downwardAPIVolume{
			volName: volName,
			podUID:  podUID,
			plugin:  plugin,
		},
	}, nil
}

// downwardAPIVolume retrieves downward API data and placing them into the volume on the host.
type downwardAPIVolume struct {
	volName                 string
	fieldReferenceFileNames map[string]string
	pod                     *api.Pod
	podUID                  types.UID // TODO: remove this redundancy as soon NewCleaner func will have *api.POD and not only types.UID
	plugin                  *downwardAPIPlugin
	volume.MetricsNil
}

// downwardAPIVolumeBuilder fetches info from downward API from the pod
// and dumps it in files
type downwardAPIVolumeBuilder struct {
	*downwardAPIVolume
	opts *volume.VolumeOptions
}

// downwardAPIVolumeBuilder implements volume.Builder interface
var _ volume.Builder = &downwardAPIVolumeBuilder{}

// downward API volumes are always ReadOnlyManaged
func (d *downwardAPIVolume) GetAttributes() volume.Attributes {
	return volume.Attributes{
		ReadOnly:        true,
		Managed:         true,
		SupportsSELinux: true,
	}
}

// SetUp puts in place the volume plugin.
// This function is not idempotent by design. We want the data to be refreshed periodically.
// The internal sync interval of kubelet will drive the refresh of data.
// TODO: Add volume specific ticker and refresh loop
func (b *downwardAPIVolumeBuilder) SetUp(fsGroup *int64) error {
	return b.SetUpAt(b.GetPath(), fsGroup)
}

func (b *downwardAPIVolumeBuilder) SetUpAt(dir string, fsGroup *int64) error {
	glog.V(3).Infof("Setting up a downwardAPI volume %v for pod %v/%v at %v", b.volName, b.pod.Namespace, b.pod.Name, dir)
	// Wrap EmptyDir. Here we rely on the idempotency of the wrapped plugin to avoid repeatedly mounting
	wrapped, err := b.plugin.host.NewWrapperBuilder(b.volName, wrappedVolumeSpec, b.pod, *b.opts)
	if err != nil {
		glog.Errorf("Couldn't setup downwardAPI volume %v for pod %v/%v: %s", b.volName, b.pod.Namespace, b.pod.Name, err.Error())
		return err
	}
	if err := wrapped.SetUpAt(dir, fsGroup); err != nil {
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

	// Atomic write wrapper for data key/value pair writes to a volume.
	atomicWriter := &volumeutil.AtomicDataWriter{
		VolumeRoot: b.GetPath(),
		WriteFile: func(fpath string, contents string) error {
			if err := ioutil.WriteFile(fpath, []byte(contents), 0644); err != nil {
				glog.Errorf("Unable to write file `%s`: %s", fpath, err)

				return err
			}

			return nil
		},
	}

	if err := atomicWriter.Write(data); err != nil {
		glog.Errorf("Unable to dump files for downwardAPI volume %v for pod %v/%v: %s", b.volName, b.pod.Namespace, b.pod.Name, err.Error())
		return err
	}

	glog.V(3).Infof("Data dumped for downwardAPI volume %v for pod %v/%v", b.volName, b.pod.Namespace, b.pod.Name)

	volume.SetVolumeOwnership(b, fsGroup)

	return nil
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
	return data, utilerrors.NewAggregate(errlist)
}

// isDataChanged iterate over all the entries to check whether at least one
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
	return d.plugin.host.GetPodVolumeDir(d.podUID, utilstrings.EscapeQualifiedNameForDisk(downwardAPIPluginName), d.volName)
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
	wrapped, err := c.plugin.host.NewWrapperCleaner(c.volName, wrappedVolumeSpec, c.podUID)
	if err != nil {
		return err
	}
	return wrapped.TearDownAt(dir)
}

func (b *downwardAPIVolumeBuilder) getMetaDir() string {
	return path.Join(b.plugin.host.GetPodPluginDir(b.podUID, utilstrings.EscapeQualifiedNameForDisk(downwardAPIPluginName)), b.volName)
}
