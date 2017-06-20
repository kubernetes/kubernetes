// +build linux

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

package hugepages

import (
	"fmt"
	"io/ioutil"
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
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
			volumeSpec: &v1.Volume{
				Name:         "test",
				VolumeSource: v1.VolumeSource{HugePages: &v1.HugePagesVolumeSource{}},
			},
			expectedResult: false,
		},
		{
			shouldDetectHugePages: true,
			volumeSpec:            nil,
			expectedResult:        false,
		},
		{
			shouldDetectHugePages: true,
			volumeSpec: &v1.Volume{
				Name:         "test",
				VolumeSource: v1.VolumeSource{},
			},
			expectedResult: false,
		},
		{
			shouldDetectHugePages: true,
			volumeSpec: &v1.Volume{
				Name: "test",
				VolumeSource: v1.VolumeSource{
					HugePages: &v1.HugePagesVolumeSource{
						MaxSize: "10M",
					},
				},
			},
			expectedResult: true,
		},
	}

	for _, testCase := range testCases {
		readFile = func(string) ([]byte, error) {
			if testCase.shouldDetectHugePages {
				return []byte("HugePages_Total: 10 \nHugePages_Free: 10 \nHugepagesize: 2048"), nil
			}
			return []byte(fmt.Sprintf("HugePages_Total: 0 \n Hugepagesize: 2048")), nil

		}
		canSupport := plug.CanSupport(&volume.Spec{Volume: testCase.volumeSpec})
		assert.Equal(t, testCase.expectedResult, canSupport)
	}
}

func TestGetNumHugePages(t *testing.T) {
	testCases := []struct {
		expectedHugePagesTotal int64
		expectedHugePagesFree  int64
		shouldFail             bool
		input                  string
		isInputReadable        bool
	}{
		{
			shouldFail:             true,
			input:                  "",
			isInputReadable:        false,
			expectedHugePagesFree:  0,
			expectedHugePagesTotal: 0,
		},
		{
			shouldFail:             true,
			input:                  "HugePages_Total",
			isInputReadable:        true,
			expectedHugePagesFree:  0,
			expectedHugePagesTotal: 0,
		},
		{
			shouldFail:             true,
			input:                  "HugePages_Total: xyz",
			isInputReadable:        true,
			expectedHugePagesFree:  0,
			expectedHugePagesTotal: 0,
		},
		{
			shouldFail:             true,
			input:                  "",
			isInputReadable:        true,
			expectedHugePagesFree:  0,
			expectedHugePagesTotal: 0,
		},
		{
			shouldFail:             false,
			input:                  "HugePages_Total: 512 \nHugepagesize: 2048",
			isInputReadable:        true,
			expectedHugePagesFree:  0,
			expectedHugePagesTotal: 512 * 2048,
		},
		{
			shouldFail:             false,
			input:                  "HugePages_Free: 400 \nHugepagesize: 1000",
			isInputReadable:        true,
			expectedHugePagesFree:  400 * 1000,
			expectedHugePagesTotal: 0,
		},
	}
	for _, testCase := range testCases {
		readFile = func(string) ([]byte, error) {
			if !testCase.isInputReadable {
				return []byte(""), fmt.Errorf("error has occurred")
			}
			return []byte(testCase.input), nil
		}
		hugePagesTotal, hugePagesFree, err := getNumHugepages()
		if testCase.shouldFail == true {
			assert.Nil(t, err)
		} else if testCase.shouldFail == false {
			assert.NotNil(t, err)
		}
		assert.Equal(t, testCase.expectedHugePagesFree, hugePagesFree)
		assert.Equal(t, testCase.expectedHugePagesTotal, hugePagesTotal)
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
				Volume: &v1.Volume{
					Name:         "test",
					VolumeSource: v1.VolumeSource{HugePages: &v1.HugePagesVolumeSource{}},
				},
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
				Volume: &v1.Volume{
					Name:         "test",
					VolumeSource: v1.VolumeSource{},
				},
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
				pageSize: "2M",
				size:     "512M",
				minSize:  "10M",
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
		hugePagesValidator(t, testCase.expectedOutput, actual.(*hugePages))
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
			UID:  "xxxxxx-yyyyyyyyyyy-zzzzzzzzzz",
			expectedOutput: &hugePages{
				volName: "test",
				pod: &v1.Pod{
					TypeMeta:   metav1.TypeMeta{},
					ObjectMeta: metav1.ObjectMeta{UID: "xxxxxx-yyyyyyyyyyy-zzzzzzzzzz"},
					Spec:       v1.PodSpec{},
					Status:     v1.PodStatus{},
				},
			},
		},
		"emptyUIDcase": {
			name: "test2",
			UID:  "",
			expectedOutput: &hugePages{
				volName: "test2",
				pod: &v1.Pod{
					TypeMeta:   metav1.TypeMeta{},
					ObjectMeta: metav1.ObjectMeta{UID: ""},
					Spec:       v1.PodSpec{},
					Status:     v1.PodStatus{},
				},
			},
		},
		"emptyNameCase": {
			name: "",
			UID:  "xxxxxxxxxx",
			expectedOutput: &hugePages{
				volName: "",
				pod: &v1.Pod{
					TypeMeta:   metav1.TypeMeta{},
					ObjectMeta: metav1.ObjectMeta{UID: "xxxxxxxxxx"},
					Spec:       v1.PodSpec{},
					Status:     v1.PodStatus{},
				},
			},
		},
	}

	for _, testCase := range testCases {
		actual, _ := plug.NewUnmounter(testCase.name, testCase.UID)
		hugePagesValidator(t, testCase.expectedOutput, actual.(*hugePages))
	}
}

func TestHugePagesPlugin_ConstructVolumeSpec(t *testing.T) {
	specValidator := func(t *testing.T, expected, actual *volume.Spec) {
		assert.Equal(t, expected.Volume.Name, actual.Volume.Name)
	}

	plug := setup(t)
	testCases := map[string]struct {
		expectedOutput *volume.Spec
		volName        string
	}{
		"workingCase": {
			expectedOutput: &volume.Spec{
				Volume: &v1.Volume{
					Name: "test",
				},
			},
			volName: "test",
		},
		"emptyVolNameCase": {
			expectedOutput: &volume.Spec{
				Volume: &v1.Volume{
					Name: "",
				},
			},
		},
	}
	for _, testCase := range testCases {
		actual, _ := plug.ConstructVolumeSpec(testCase.volName, "")
		specValidator(t, testCase.expectedOutput, actual)
	}
}

func hugePagesSetup(t *testing.T) *hugePages {
	plugin := setup(t).(*hugePagesPlugin)
	return &hugePages{
		mounter: plugin.host.GetMounter(),
		plugin:  plugin,
	}
}

func TestHugePages_GetAttributes(t *testing.T) {
	hp := hugePagesSetup(t)
	attributes := hp.GetAttributes()
	assert.False(t, attributes.Managed)
	assert.False(t, attributes.SupportsSELinux)
	assert.False(t, attributes.ReadOnly)
}

func TestHugePages_CanMount(t *testing.T) {
	hp := hugePagesSetup(t)
	assert.Nil(t, hp.CanMount())
}

func testSetUp(t *testing.T, setUpFunction func(gId int, location string) func() error) {
	testCases := map[string]struct {
		prepareFunc   func() (string, error)
		groupId       int
		isThrowingErr bool
		postFunc      func(location string) error
	}{
		"workingCase": {
			prepareFunc: func() (string, error) {
				return ioutil.TempDir("/tmp", "hp")
			},
			groupId:       600,
			isThrowingErr: false,
			postFunc: func(location string) error {
				return os.RemoveAll(location)
			},
		},
		"notExistingLocationCase": {
			prepareFunc: func() (string, error) {
				return "/xyz/notexisting", nil
			},
			groupId:       000,
			isThrowingErr: true,
			postFunc: func(location string) error {
				return nil
			},
		},
	}

	for _, testCase := range testCases {
		location, err := testCase.prepareFunc()
		assert.Nil(t, err)
		err = setUpFunction(testCase.groupId, location)()
		assert.Equal(t, testCase.isThrowingErr, err != nil)
		err = testCase.postFunc(location)
		assert.Nil(t, err)
	}
}

func TestHugePages_SetUpAt(t *testing.T) {
	hp := hugePagesSetup(t)
	hp.pod = &v1.Pod{
		TypeMeta:   metav1.TypeMeta{},
		ObjectMeta: metav1.ObjectMeta{UID: ""},
		Spec:       v1.PodSpec{},
		Status:     v1.PodStatus{},
	}
	setUpFunction := func(gId int, location string) func() error {
		return func() error {
			gId := types.UnixGroupID(gId)
			return hp.SetUpAt(location, &gId)
		}
	}
	testSetUp(t, setUpFunction)
}
