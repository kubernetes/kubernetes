/*
Copyright 2015 The Kubernetes Authors.

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

package volume

import (
	"sync"
	"sync/atomic"
	"testing"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const testPluginName = "kubernetes.io/testPlugin"

func TestSpecSourceConverters(t *testing.T) {
	v := &v1.Volume{
		Name:         "foo",
		VolumeSource: v1.VolumeSource{EmptyDir: &v1.EmptyDirVolumeSource{}},
	}

	converted := NewSpecFromVolume(v)
	if converted.Volume.EmptyDir == nil {
		t.Errorf("Unexpected nil EmptyDir: %#v", converted)
	}
	if v.Name != converted.Name() {
		t.Errorf("Expected %v but got %v", converted.Name(), v.Name)
	}

	pv := &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{Name: "bar"},
		Spec: v1.PersistentVolumeSpec{
			PersistentVolumeSource: v1.PersistentVolumeSource{AWSElasticBlockStore: &v1.AWSElasticBlockStoreVolumeSource{}},
		},
	}

	converted = NewSpecFromPersistentVolume(pv, false)
	if converted.PersistentVolume.Spec.AWSElasticBlockStore == nil {
		t.Errorf("Unexpected nil AWSElasticBlockStore: %#v", converted)
	}
	if pv.Name != converted.Name() {
		t.Errorf("Expected %v but got %v", converted.Name(), pv.Name)
	}
}

type testPlugins struct {
}

func (plugin *testPlugins) Init(host VolumeHost) error {
	return nil
}

func (plugin *testPlugins) GetPluginName() string {
	return testPluginName
}

func (plugin *testPlugins) GetVolumeName(spec *Spec) (string, error) {
	return "", nil
}

func (plugin *testPlugins) CanSupport(spec *Spec) bool {
	return true
}

func (plugin *testPlugins) RequiresRemount(spec *Spec) bool {
	return false
}

func (plugin *testPlugins) SupportsMountOption() bool {
	return false
}

func (plugin *testPlugins) SupportsSELinuxContextMount(spec *Spec) (bool, error) {
	return false, nil
}

func (plugin *testPlugins) NewMounter(spec *Spec, podRef *v1.Pod) (Mounter, error) {
	return nil, nil
}

func (plugin *testPlugins) NewUnmounter(name string, podUID types.UID) (Unmounter, error) {
	return nil, nil
}

func (plugin *testPlugins) ConstructVolumeSpec(volumeName, mountPath string) (ReconstructedVolume, error) {
	return ReconstructedVolume{}, nil
}

func newTestPlugin() []VolumePlugin {
	return []VolumePlugin{&testPlugins{}}
}

func TestVolumePluginMgrFunc(t *testing.T) {
	vpm := VolumePluginMgr{}
	var prober DynamicPluginProber = nil // TODO (#51147) inject mock
	vpm.InitPlugins(newTestPlugin(), prober, nil)

	plug, err := vpm.FindPluginByName(testPluginName)
	if err != nil {
		t.Fatal("Can't find the plugin by name")
	}
	if plug.GetPluginName() != testPluginName {
		t.Errorf("Wrong name: %s", plug.GetPluginName())
	}

	_, err = vpm.FindPluginBySpec(nil)
	if err == nil {
		t.Errorf("Should return error if volume spec is nil")
	}

	volumeSpec := &Spec{}
	_, err = vpm.FindPluginBySpec(volumeSpec)
	if err != nil {
		t.Errorf("Should return test plugin if volume spec is not nil")
	}
}

func Test_ValidatePodTemplate(t *testing.T) {
	pod := &v1.Pod{
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{
				{
					Name:         "vol",
					VolumeSource: v1.VolumeSource{},
				},
			},
		},
	}
	var want error
	if got := ValidateRecyclerPodTemplate(pod); got != want {
		t.Errorf("isPodTemplateValid(%v) returned (%v), want (%v)", pod.String(), got.Error(), want)
	}

	// Check that the default recycle pod template is valid
	pod = NewPersistentVolumeRecyclerPodTemplate()
	want = nil
	if got := ValidateRecyclerPodTemplate(pod); got != want {
		t.Errorf("isPodTemplateValid(%v) returned (%v), want (%v)", pod.String(), got.Error(), want)
	}

	pod = &v1.Pod{
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name: "pv-recycler",
				},
			},
		},
	}
	// want = an error
	if got := ValidateRecyclerPodTemplate(pod); got == nil {
		t.Errorf("isPodTemplateValid(%v) returned (%v), want (%v)", pod.String(), got, "Error: pod specification does not contain any volume(s).")
	}
}

// TestVolumePluginMultiThreaded tests FindPluginByName/FindPluginBySpec in a multi-threaded environment.
// If these are called by different threads at the same time, they should still be able to reconcile the plugins
// and return the same results (no missing plugin)
func TestVolumePluginMultiThreaded(t *testing.T) {
	vpm := VolumePluginMgr{}
	var prober DynamicPluginProber = &fakeProber{events: []ProbeEvent{{PluginName: testPluginName, Op: ProbeAddOrUpdate, Plugin: &testPlugins{}}}}
	err := vpm.InitPlugins([]VolumePlugin{}, prober, nil)
	require.NoError(t, err)

	volumeSpec := &Spec{}
	totalErrors := atomic.Int32{}
	var wg sync.WaitGroup

	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, err := vpm.FindPluginByName(testPluginName)
			if err != nil {
				totalErrors.Add(1)
			}
		}()
	}
	wg.Wait()

	assert.Equal(t, int32(0), totalErrors.Load())
	totalErrors.Store(0)

	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, err := vpm.FindPluginBySpec(volumeSpec)
			if err != nil {
				totalErrors.Add(1)
			}
		}()
	}
	wg.Wait()

	assert.Equal(t, int32(0), totalErrors.Load())
}

type fakeProber struct {
	events         []ProbeEvent
	firstExecution atomic.Bool
}

func (prober *fakeProber) Init() error {
	prober.firstExecution.Store(true)
	return nil
}

func (prober *fakeProber) Probe() (events []ProbeEvent, err error) {
	if prober.firstExecution.CompareAndSwap(true, false) {
		return prober.events, nil
	}
	return []ProbeEvent{}, nil
}
