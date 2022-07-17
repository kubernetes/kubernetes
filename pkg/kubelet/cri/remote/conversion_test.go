/*
Copyright 2021 The Kubernetes Authors.

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

package remote

import (
	"fmt"
	"reflect"
	"regexp"
	"strings"
	"testing"

	"github.com/gogo/protobuf/proto"
	"github.com/stretchr/testify/assert"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	"k8s.io/cri-api/pkg/apis/runtime/v1alpha2"
)

func TestMemoryEqual(t *testing.T) {
	testcases := []struct {
		a interface{}
		b interface{}
	}{
		{runtimeapi.VersionResponse{}, v1alpha2.VersionResponse{}},
		{runtimeapi.PodSandboxConfig{}, v1alpha2.PodSandboxConfig{}},
		{runtimeapi.PodSandboxFilter{}, v1alpha2.PodSandboxFilter{}},
		{runtimeapi.ContainerConfig{}, v1alpha2.ContainerConfig{}},
		{runtimeapi.ContainerFilter{}, v1alpha2.ContainerFilter{}},
		{runtimeapi.LinuxContainerResources{}, v1alpha2.LinuxContainerResources{}},
		{runtimeapi.ExecRequest{}, v1alpha2.ExecRequest{}},
		{runtimeapi.AttachRequest{}, v1alpha2.AttachRequest{}},
		{runtimeapi.PortForwardRequest{}, v1alpha2.PortForwardRequest{}},
		{runtimeapi.RuntimeConfig{}, v1alpha2.RuntimeConfig{}},
		{runtimeapi.ContainerStatsFilter{}, v1alpha2.ContainerStatsFilter{}},
		{runtimeapi.PodSandboxStatsFilter{}, v1alpha2.PodSandboxStatsFilter{}},
		{runtimeapi.ImageFilter{}, v1alpha2.ImageFilter{}},
		{runtimeapi.ImageSpec{}, v1alpha2.ImageSpec{}},
		{runtimeapi.AuthConfig{}, v1alpha2.AuthConfig{}},
	}

	for _, tc := range testcases {
		aType := reflect.TypeOf(tc.a)
		bType := reflect.TypeOf(tc.b)
		t.Run(aType.String(), func(t *testing.T) {
			assertEqualTypes(t, nil, aType, bType)
		})
	}
}

func assertEqualTypes(t *testing.T, path []string, a, b reflect.Type) {
	if a == b {
		return
	}

	if a.Kind() != b.Kind() {
		fatalTypeError(t, path, a, b, "mismatched Kind")
	}

	switch a.Kind() {
	case reflect.Struct:
		aFields := a.NumField()
		bFields := b.NumField()
		if aFields != bFields {
			fatalTypeError(t, path, a, b, "mismatched field count")
		}
		for i := 0; i < aFields; i++ {
			aField := a.Field(i)
			bField := b.Field(i)
			if aField.Name != bField.Name {
				fatalTypeError(t, path, a, b, fmt.Sprintf("mismatched field name %d: %s %s", i, aField.Name, bField.Name))
			}
			if aTag, bTag := stripEnum(aField.Tag), stripEnum(bField.Tag); aTag != bTag {
				fatalTypeError(t, path, a, b, fmt.Sprintf("mismatched field tag %d:\n%s\n%s\n", i, aTag, bTag))
			}
			if aField.Offset != bField.Offset {
				fatalTypeError(t, path, a, b, fmt.Sprintf("mismatched field offset %d: %v %v", i, aField.Offset, bField.Offset))
			}
			if aField.Anonymous != bField.Anonymous {
				fatalTypeError(t, path, a, b, fmt.Sprintf("mismatched field anonymous %d: %v %v", i, aField.Anonymous, bField.Anonymous))
			}
			if !reflect.DeepEqual(aField.Index, bField.Index) {
				fatalTypeError(t, path, a, b, fmt.Sprintf("mismatched field index %d: %v %v", i, aField.Index, bField.Index))
			}
			path = append(path, aField.Name)
			assertEqualTypes(t, path, aField.Type, bField.Type)
			path = path[:len(path)-1]
		}

	case reflect.Pointer, reflect.Slice:
		aElemType := a.Elem()
		bElemType := b.Elem()
		assertEqualTypes(t, path, aElemType, bElemType)

	case reflect.Int32:
		if a.Kind() != b.Kind() {
			fatalTypeError(t, path, a, b, "incompatible types")
		}

	default:
		fatalTypeError(t, path, a, b, "unhandled kind")
	}
}

// strip the enum value from the protobuf tag, since that doesn't impact the wire serialization and differs by package
func stripEnum(tagValue reflect.StructTag) reflect.StructTag {
	return reflect.StructTag(regexp.MustCompile(",enum=[^,]+").ReplaceAllString(string(tagValue), ""))
}

func fatalTypeError(t *testing.T, path []string, a, b reflect.Type, message string) {
	t.Helper()
	t.Fatalf("%s: %s: %s %s", strings.Join(path, ""), message, a, b)
}

func fillFields(s interface{}) {
	fillFieldsOffset(s, 0)
}

func fillFieldsOffset(s interface{}, offset int) {
	reflectType := reflect.TypeOf(s).Elem()
	reflectValue := reflect.ValueOf(s).Elem()

	for i := 0; i < reflectType.NumField(); i++ {
		field := reflectValue.Field(i)
		typeName := reflectType.Field(i).Name

		// Skipping protobuf internal values
		if strings.HasPrefix(typeName, "XXX_") {
			continue
		}

		fillField(field, i+offset)
	}
}

func fillField(field reflect.Value, v int) {
	switch field.Kind() {
	case reflect.Bool:
		field.SetBool(true)

	case reflect.Float32, reflect.Float64:
		field.SetFloat(float64(v))

	case reflect.String:
		field.SetString(fmt.Sprint(v))

	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		field.SetInt(int64(v))

	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		field.SetUint(uint64(v))

	case reflect.Map:
		field.Set(reflect.MakeMap(field.Type()))

	case reflect.Array, reflect.Slice:
		slice := reflect.MakeSlice(field.Type(), 1, 1)
		field.Set(slice)
		first := slice.Index(0)

		if first.Type().Kind() == reflect.Pointer {
			first.Set(reflect.New(first.Type().Elem()))
			fillFieldsOffset(first.Interface(), v)
		} else {
			fillField(first, v)
		}

	case reflect.Pointer:
		val := reflect.New(field.Type().Elem())
		field.Set(val)
		fillFieldsOffset(field.Interface(), v)

	case reflect.Struct:
		fillFieldsOffset(field.Addr().Interface(), v)
	}

}

func assertEqual(t *testing.T, a, b proto.Message) {
	aBytes, err := proto.Marshal(a)
	assert.Nil(t, err)

	bBytes, err := proto.Marshal(b)
	assert.Nil(t, err)

	assert.Equal(t, aBytes, bBytes)
}

func TestFromV1alpha2VersionResponse(t *testing.T) {
	from := &v1alpha2.VersionResponse{}
	fillFields(from)
	to := fromV1alpha2VersionResponse(from)
	assertEqual(t, from, to)
}

func TestFromV1alpha2PodSandboxStatusResponse(t *testing.T) {
	from := &v1alpha2.PodSandboxStatusResponse{}
	fillFields(from)
	to := fromV1alpha2PodSandboxStatusResponse(from)
	assertEqual(t, from, to)
}

func TestFromV1alpha2ListPodSandboxResponse(t *testing.T) {
	from := &v1alpha2.ListPodSandboxResponse{}
	fillFields(from)
	to := fromV1alpha2ListPodSandboxResponse(from)
	assertEqual(t, from, to)
}

func TestFromV1alpha2ListContainersResponse(t *testing.T) {
	from := &v1alpha2.ListContainersResponse{}
	fillFields(from)
	to := fromV1alpha2ListContainersResponse(from)
	assertEqual(t, from, to)
}

func TestFromV1alpha2ContainerStatusResponse(t *testing.T) {
	from := &v1alpha2.ContainerStatusResponse{}
	fillFields(from)
	to := fromV1alpha2ContainerStatusResponse(from)
	assertEqual(t, from, to)
}

func TestFromV1alpha2ExecResponse(t *testing.T) {
	from := &v1alpha2.ExecResponse{}
	fillFields(from)
	to := fromV1alpha2ExecResponse(from)
	assertEqual(t, from, to)
}

func TestFromV1alpha2AttachResponse(t *testing.T) {
	from := &v1alpha2.AttachResponse{}
	fillFields(from)
	to := fromV1alpha2AttachResponse(from)
	assertEqual(t, from, to)
}

func TestFromV1alpha2PortForwardResponse(t *testing.T) {
	from := &v1alpha2.PortForwardResponse{}
	fillFields(from)
	to := fromV1alpha2PortForwardResponse(from)
	assertEqual(t, from, to)
}

func TestFromV1alpha2StatusResponse(t *testing.T) {
	from := &v1alpha2.StatusResponse{}
	fillFields(from)
	to := fromV1alpha2StatusResponse(from)
	assertEqual(t, from, to)
}

func TestFromV1alpha2ContainerStats(t *testing.T) {
	from := &v1alpha2.ContainerStats{}
	fillFields(from)
	to := fromV1alpha2ContainerStats(from)
	assertEqual(t, from, to)
}

func TestFromV1alpha2ImageFsInfoResponse(t *testing.T) {
	from := &v1alpha2.ImageFsInfoResponse{}
	fillFields(from)
	to := fromV1alpha2ImageFsInfoResponse(from)
	assertEqual(t, from, to)
}

func TestFromV1alpha2ListContainerStatsResponse(t *testing.T) {
	from := &v1alpha2.ListContainerStatsResponse{}
	fillFields(from)
	to := fromV1alpha2ListContainerStatsResponse(from)
	assertEqual(t, from, to)
}

func TestFromV1alpha2PodSandboxStats(t *testing.T) {
	from := &v1alpha2.PodSandboxStats{}
	fillFields(from)
	to := fromV1alpha2PodSandboxStats(from)
	assertEqual(t, from, to)
}

func TestFromV1alpha2ListPodSandboxStatsResponse(t *testing.T) {
	from := &v1alpha2.ListPodSandboxStatsResponse{}
	fillFields(from)
	to := fromV1alpha2ListPodSandboxStatsResponse(from)
	assertEqual(t, from, to)
}

func TestFromV1alpha2ImageStatusResponse(t *testing.T) {
	from := &v1alpha2.ImageStatusResponse{}
	fillFields(from)
	to := fromV1alpha2ImageStatusResponse(from)
	assertEqual(t, from, to)
}

func TestFromV1alpha2ListImagesResponse(t *testing.T) {
	from := &v1alpha2.ListImagesResponse{}
	fillFields(from)
	to := fromV1alpha2ListImagesResponse(from)
	assertEqual(t, from, to)
}

func TestV1alpha2PodSandboxConfig(t *testing.T) {
	from := &runtimeapi.PodSandboxConfig{}
	fillFields(from)
	to := v1alpha2PodSandboxConfig(from)
	assertEqual(t, from, to)
}

func TestV1alpha2PodSandboxFilter(t *testing.T) {
	from := &runtimeapi.PodSandboxFilter{}
	fillFields(from)
	to := v1alpha2PodSandboxFilter(from)
	assertEqual(t, from, to)
}

func TestV1alpha2ContainerConfig(t *testing.T) {
	from := &runtimeapi.ContainerConfig{}
	fillFields(from)
	to := v1alpha2ContainerConfig(from)
	assertEqual(t, from, to)
}

func TestV1alpha2ContainerFilter(t *testing.T) {
	from := &runtimeapi.ContainerFilter{}
	fillFields(from)
	to := v1alpha2ContainerFilter(from)
	assertEqual(t, from, to)
}

func TestV1alpha2LinuxContainerResources(t *testing.T) {
	from := &runtimeapi.LinuxContainerResources{}
	fillFields(from)
	to := v1alpha2LinuxContainerResources(from)
	assertEqual(t, from, to)
}

func TestV1alpha2ExecRequest(t *testing.T) {
	from := &runtimeapi.ExecRequest{}
	fillFields(from)
	to := v1alpha2ExecRequest(from)
	assertEqual(t, from, to)
}

func TestV1alpha2AttachRequest(t *testing.T) {
	from := &runtimeapi.AttachRequest{}
	fillFields(from)
	to := v1alpha2AttachRequest(from)
	assertEqual(t, from, to)
}

func TestV1alpha2PortForwardRequest(t *testing.T) {
	from := &runtimeapi.PortForwardRequest{}
	fillFields(from)
	to := v1alpha2PortForwardRequest(from)
	assertEqual(t, from, to)
}

func TestV1alpha2RuntimeConfig(t *testing.T) {
	from := &runtimeapi.RuntimeConfig{}
	fillFields(from)
	to := v1alpha2RuntimeConfig(from)
	assertEqual(t, from, to)
}

func TestV1alpha2ContainerStatsFilter(t *testing.T) {
	from := &runtimeapi.ContainerStatsFilter{}
	fillFields(from)
	to := v1alpha2ContainerStatsFilter(from)
	assertEqual(t, from, to)
}

func TestV1alpha2PodSandboxStatsFilter(t *testing.T) {
	from := &runtimeapi.PodSandboxStatsFilter{}
	fillFields(from)
	to := v1alpha2PodSandboxStatsFilter(from)
	assertEqual(t, from, to)
}

func TestV1alpha2ImageFilter(t *testing.T) {
	from := &runtimeapi.ImageFilter{}
	fillFields(from)
	to := v1alpha2ImageFilter(from)
	assertEqual(t, from, to)
}

func TestV1alpha2ImageSpec(t *testing.T) {
	from := &runtimeapi.ImageSpec{}
	fillFields(from)
	to := v1alpha2ImageSpec(from)
	assertEqual(t, from, to)
}

func TestV1alpha2AuthConfig(t *testing.T) {
	from := &runtimeapi.AuthConfig{}
	fillFields(from)
	to := v1alpha2AuthConfig(from)
	assertEqual(t, from, to)
}
