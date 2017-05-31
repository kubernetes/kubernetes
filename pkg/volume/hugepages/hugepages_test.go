// +build linux

/*
Copyright 2014 The Kubernetes Authors.

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
package hugepages

import (
	//"io/ioutil"
	//"os"
	"testing"
	//
	//"github.com/stretchr/testify/assert"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	//"k8s.io/apimachinery/pkg/types"
	//"k8s.io/kubernetes/pkg/util/mount"
	//"k8s.io/kubernetes/pkg/volume"
	//
	"fmt"
	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/volume"
	volumetest "k8s.io/kubernetes/pkg/volume/testing"
)

func setup(t *testing.T) volume.VolumePlugin {
	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), volumetest.NewFakeVolumeHost("fake", nil, nil))
	plug, err := plugMgr.FindPluginByName(hugePagesPluginName)
	assert.Nil(t, err)
	assert.EqualValues(t, plug.GetPluginName(), hugePagesPluginName)
	return plug
}

func TestHugePagesPlugin_CanSupport(t *testing.T) {
	plug := setup(t)
	testCases := []struct {
		shouldDetectHugePages bool
		volumeSpec            *v1.Volume
		expectedResult        bool
	}{
		{
			shouldDetectHugePages: false,
			volumeSpec:            &v1.Volume{"test", v1.VolumeSource{HugePages: &v1.HugePagesVolumeSource{}}},
			expectedResult:        false,
		},
		{
			shouldDetectHugePages: true,
			volumeSpec:            nil,
			expectedResult:        false,
		},
		{
			shouldDetectHugePages: true,
			volumeSpec:            &v1.Volume{"test", v1.VolumeSource{}},
			expectedResult:        false,
		},
		{
			shouldDetectHugePages: true,
			volumeSpec:            &v1.Volume{"test", v1.VolumeSource{HugePages: &v1.HugePagesVolumeSource{}}},
			expectedResult:        true,
		},
	}

	for _, testCase := range testCases {
		readFile = func(string) ([]byte, error) {
			if testCase.shouldDetectHugePages {
				return []byte("HugePages_Total: 10"), nil
			}
			return []byte("HugePages_Total: 0"), nil
		}
		canSupport := plug.CanSupport(&volume.Spec{Volume: testCase.volumeSpec})
		assert.Equal(t, testCase.expectedResult, canSupport)
	}
}

func TestDetectHugePages(t *testing.T) {
	testCases := []struct {
		expectedOutput  int
		input           string
		isInputReadable bool
	}{
		{
			expectedOutput:  -1,
			input:           "",
			isInputReadable: false,
		},
		{
			expectedOutput:  -1,
			input:           "HugePages_Total",
			isInputReadable: true,
		},
		{
			expectedOutput:  -1,
			input:           "HugePages_Total: xyz",
			isInputReadable: true,
		},
		{
			expectedOutput:  -1,
			input:           "",
			isInputReadable: true,
		},
		{
			expectedOutput:  512,
			input:           "HugePages_Total: 512",
			isInputReadable: true,
		},
	}
	for _, testCase := range testCases {
		readFile = func(string) ([]byte, error) {
			if !testCase.isInputReadable {
				return []byte(""), fmt.Errorf("error has been occurred")
			}
			return []byte(testCase.input), nil
		}
		output := detectHugepages()
		assert.Equal(t, testCase.expectedOutput, output)
	}
}

func TestHugePagesPlugin_GetVolumeName(t *testing.T) {
	plug := setup(t)
	testCases := map[string]struct {
		inputSpec *volume.Spec
		outputStr string
		throwErr  bool
	}{
		"positiveCase": {
			inputSpec: &volume.Spec{
				Volume: &v1.Volume{"test", v1.VolumeSource{HugePages: &v1.HugePagesVolumeSource{}}},
			},
			outputStr: "test",
			throwErr:  false,
		},
		"emptyVolumeSpec": {
			inputSpec: &volume.Spec{},
			outputStr: "",
			throwErr:  true,
		},
		"emptyHugePagesVolumeSources": {
			inputSpec: &volume.Spec{
				Volume: &v1.Volume{"test", v1.VolumeSource{}},
			},
			outputStr: "",
			throwErr:  true,
		},
	}

	for _, testCase := range testCases {
		name, err := plug.GetVolumeName(testCase.inputSpec)
		assert.Equal(t, testCase.outputStr, name)
		assert.Equal(t, testCase.throwErr, err != nil)
	}
}

// Purpose of this test is a detection of hugepages volume  design.
func TestStaticMethods(t *testing.T) {
	plug := setup(t)
	assert.False(t, plug.RequiresRemount())
	assert.False(t, plug.SupportsMountOption())
	assert.False(t, plug.SupportsBulkVolumeVerification())
}

func TestHugePagesPlugin_NewMounter(t *testing.T) {
	hugePagesValidator := func(t *testing.T, expected, actual *hugePages) {
		assert.Equal(t, expected.volName, actual.volName)
		assert.Equal(t, expected.pageSize, actual.pageSize)
		assert.Equal(t, expected.minSize, actual.minSize)
		assert.Equal(t, expected.size, actual.size)
	}

	plug := setup(t)

	testCases := map[string]struct {
		name                  string
		hugePagesVolumeSource *v1.HugePagesVolumeSource
		expectedOutput        *hugePages
	}{
		"realisticCase": {
			name: "test",
			hugePagesVolumeSource: &v1.HugePagesVolumeSource{
				PageSize: "2M",
				MaxSize:  "512M",
				MinSize:  "10M",
			},
			expectedOutput: &hugePages{
				volName:  "test",
				pageSize: "10G",
				size:     "400T",
				minSize:  "10k",
			},
		},
		"unrealisticCase": {
			name: "test",
			hugePagesVolumeSource: &v1.HugePagesVolumeSource{
				PageSize: "10G",
				MaxSize:  "400T",
				MinSize:  "10k",
			},
			expectedOutput: &hugePages{
				volName:  "test",
				pageSize: "10G",
				size:     "400T",
				minSize:  "10k",
			},
		},
	}

	for _, testCase := range testCases {
		spec := &volume.Spec{
			Volume: &v1.Volume{
				Name: testCase.name,
				VolumeSource: v1.VolumeSource{
					HugePages: testCase.hugePagesVolumeSource,
				},
			},
		}
		actual, _ := plug.NewMounter(spec, nil, volume.VolumeOptions{})
		hugePagesValidator(t, testCase.expectedOutput, (actual.(*hugePages)))
	}
}

func TestHugePagesPlugin_NewUnmounter(t *testing.T) {
	hugePagesValidator := func(t *testing.T, expected, actual *hugePages) {
		assert.Equal(t, expected.volName, actual.volName)
		assert.Equal(t, expected.pod.UID, expected.pod.UID)
	}

	plug := setup(t)

	testCases := map[string]struct {
		name           string
		UID            types.UID
		expectedOutput *hugePages
	}{
		"realisticCase": {
			name: "test",
			expectedOutput: &hugePages{
				volName: "test",
				pod: &v1.Pod{
					metav1.ObjectMeta{
						UID: "xxxxxx-yyyyyyyyyyy-zzzzzzzzzz",
					},
				},
			},
		},
		"unrealisticCase": {
			name: "test",
			hugePagesVolumeSource: &v1.HugePagesVolumeSource{
				PageSize: "10G",
				MaxSize:  "400T",
				MinSize:  "10k",
			},
			expectedOutput: &hugePages{
				volName:  "test",
				pageSize: "10G",
				size:     "400T",
				minSize:  "10k",
			},
		},
	}

	for _, testCase := range testCases {
		spec := &volume.Spec{
			Volume: &v1.Volume{
				Name: testCase.name,
				VolumeSource: v1.VolumeSource{
					HugePages: testCase.hugePagesVolumeSource,
				},
			},
		}
		actual, _ := plug.NewMounter(spec, nil, volume.VolumeOptions{})
		hugePagesValidator(t, testCase.expectedOutput, (actual.(*hugePages)))
	}
}

//
//func fakeVolumeHostCreator(location string) volume.VolumeHost {
//	fakeVolumeHost := kubeTesting.NewFakeVolumeHost(location, nil, nil)
//	return volume.VolumeHost(fakeVolumeHost)
//}
//
//func TestHugePages_SetUpAt(t *testing.T) {
//	testCases := []struct {
//		prepareFunc     func() (string, func() error, error)
//		isExpectedError bool
//		mockedHP        *hugePages
//		unixGroupID     types.UnixGroupID
//	}{
//		{
//			prepareFunc: func() (string, func() error, error) {
//				location, err := ioutil.TempDir("/tmp", "hugepageTest")
//				return location, func() error {
//					return os.RemoveAll(location)
//				}, err
//			},
//			isExpectedError: false,
//			mockedHP: &hugePages{
//				pod: &v1.Pod{
//					metav1.ObjectMeta{
//						UID:
//					},
//				},
//				mounter: &mount.FakeMounter{},
//				plugin:  &hugePagesPlugin{},
//			},
//			unixGroupID: 1000,
//		},
//	}
//
//	for _, testCase := range testCases {
//		location, cleanup, err := testCase.prepareFunc()
//		assert.Nil(t, err)
//
//		t.Log(location)
//		testCase.mockedHP.plugin.Init(fakeVolumeHostCreator(location))
//		testCase.mockedHP.SetUpAt(location, &testCase.unixGroupID)
//		err = cleanup()
//		assert.Nil(t, err)
//	}
//}
