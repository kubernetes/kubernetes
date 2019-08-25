/*
Copyright 2018 The Kubernetes Authors.

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

package add

import (
	"reflect"
	"testing"

	"sigs.k8s.io/kustomize/pkg/commands/kustfile"
	"sigs.k8s.io/kustomize/pkg/fs"
	"sigs.k8s.io/kustomize/pkg/types"
	"sigs.k8s.io/kustomize/pkg/validators"
)

func makeKustomization(t *testing.T) *types.Kustomization {
	fakeFS := fs.MakeFakeFS()
	fakeFS.WriteTestKustomization()
	kf, err := kustfile.NewKustomizationFile(fakeFS)
	if err != nil {
		t.Errorf("unexpected new error %v", err)
	}
	m, err := kf.Read()
	if err != nil {
		t.Errorf("unexpected read error %v", err)
	}
	return m
}

func TestRunAddAnnotation(t *testing.T) {
	var o addMetadataOptions
	o.metadata = map[string]string{"owls": "cute", "otters": "adorable"}

	m := makeKustomization(t)
	err := o.addAnnotations(m)
	if err != nil {
		t.Errorf("unexpected error: could not write to kustomization file")
	}
	// adding the same test input should not work
	err = o.addAnnotations(m)
	if err == nil {
		t.Errorf("expected already in kustomization file error")
	}
	// adding new annotations should work
	o.metadata = map[string]string{"new": "annotation"}
	err = o.addAnnotations(m)
	if err != nil {
		t.Errorf("unexpected error: could not write to kustomization file")
	}
}

func TestAddAnnotationNoArgs(t *testing.T) {
	fakeFS := fs.MakeFakeFS()
	v := validators.MakeHappyMapValidator(t)
	cmd := newCmdAddAnnotation(fakeFS, v.Validator)
	err := cmd.Execute()
	v.VerifyNoCall()
	if err == nil {
		t.Errorf("expected an error")
	}
	if err.Error() != "must specify annotation" {
		t.Errorf("incorrect error: %v", err.Error())
	}
}

func TestAddAnnotationInvalidFormat(t *testing.T) {
	fakeFS := fs.MakeFakeFS()
	v := validators.MakeSadMapValidator(t)
	cmd := newCmdAddAnnotation(fakeFS, v.Validator)
	args := []string{"whatever:whatever"}
	err := cmd.RunE(cmd, args)
	v.VerifyCall()
	if err == nil {
		t.Errorf("expected an error")
	}
	if err.Error() != validators.SAD {
		t.Errorf("incorrect error: %v", err.Error())
	}
}

func TestAddAnnotationManyArgs(t *testing.T) {
	fakeFS := fs.MakeFakeFS()
	fakeFS.WriteTestKustomization()
	v := validators.MakeHappyMapValidator(t)
	cmd := newCmdAddAnnotation(fakeFS, v.Validator)
	args := []string{"k1:v1,k2:v2,k3:v3,k4:v5"}
	err := cmd.RunE(cmd, args)
	v.VerifyCall()
	if err != nil {
		t.Errorf("unexpected error: %v", err.Error())
	}
}

func TestAddAnnotationValueQuoted(t *testing.T) {
	fakeFS := fs.MakeFakeFS()
	fakeFS.WriteTestKustomization()
	v := validators.MakeHappyMapValidator(t)
	cmd := newCmdAddAnnotation(fakeFS, v.Validator)
	args := []string{"k1:\"v1\""}
	err := cmd.RunE(cmd, args)
	v.VerifyCall()
	if err != nil {
		t.Errorf("unexpected error: %v", err.Error())
	}
}

func TestAddAnnotationValueWithColon(t *testing.T) {
	fakeFS := fs.MakeFakeFS()
	fakeFS.WriteTestKustomization()
	v := validators.MakeHappyMapValidator(t)
	cmd := newCmdAddAnnotation(fakeFS, v.Validator)
	args := []string{"k1:\"v1:v2\""}
	err := cmd.RunE(cmd, args)
	v.VerifyCall()
	if err != nil {
		t.Errorf("unexpected error: %v", err.Error())
	}
}

func TestAddAnnotationNoKey(t *testing.T) {
	fakeFS := fs.MakeFakeFS()
	v := validators.MakeHappyMapValidator(t)
	cmd := newCmdAddAnnotation(fakeFS, v.Validator)
	args := []string{":nokey"}
	err := cmd.RunE(cmd, args)
	v.VerifyNoCall()
	if err == nil {
		t.Errorf("expected an error")
	}
	if err.Error() != "invalid annotation: ':nokey' (need k:v pair where v may be quoted)" {
		t.Errorf("incorrect error: %v", err.Error())
	}
}

func TestAddAnnotationTooManyColons(t *testing.T) {
	fakeFS := fs.MakeFakeFS()
	fakeFS.WriteTestKustomization()
	v := validators.MakeHappyMapValidator(t)
	cmd := newCmdAddAnnotation(fakeFS, v.Validator)
	args := []string{"key:v1:v2"}
	err := cmd.RunE(cmd, args)
	v.VerifyCall()
	if err != nil {
		t.Errorf("unexpected error: %v", err.Error())
	}
}

func TestAddAnnotationNoValue(t *testing.T) {
	fakeFS := fs.MakeFakeFS()
	fakeFS.WriteTestKustomization()
	v := validators.MakeHappyMapValidator(t)
	cmd := newCmdAddAnnotation(fakeFS, v.Validator)
	args := []string{"no:,value"}
	err := cmd.RunE(cmd, args)
	v.VerifyCall()
	if err != nil {
		t.Errorf("unexpected error: %v", err.Error())
	}
}

func TestAddAnnotationMultipleArgs(t *testing.T) {
	fakeFS := fs.MakeFakeFS()
	fakeFS.WriteTestKustomization()
	v := validators.MakeHappyMapValidator(t)
	cmd := newCmdAddAnnotation(fakeFS, v.Validator)
	args := []string{"this:annotation", "has:spaces"}
	err := cmd.RunE(cmd, args)
	v.VerifyNoCall()
	if err == nil {
		t.Errorf("expected an error")
	}
	if err.Error() != "annotations must be comma-separated, with no spaces" {
		t.Errorf("incorrect error: %v", err.Error())
	}
}

func TestAddAnnotationForce(t *testing.T) {
	fakeFS := fs.MakeFakeFS()
	fakeFS.WriteTestKustomization()
	v := validators.MakeHappyMapValidator(t)
	cmd := newCmdAddAnnotation(fakeFS, v.Validator)
	args := []string{"key:foo"}
	err := cmd.RunE(cmd, args)
	v.VerifyCall()
	if err != nil {
		t.Errorf("unexpected error: %v", err.Error())
	}
	// trying to add the same annotation again should not work
	args = []string{"key:bar"}
	v = validators.MakeHappyMapValidator(t)
	cmd = newCmdAddAnnotation(fakeFS, v.Validator)
	err = cmd.RunE(cmd, args)
	v.VerifyCall()
	if err == nil {
		t.Errorf("expected an error")
	}
	if err.Error() != "annotation key already in kustomization file" {
		t.Errorf("expected an error")
	}
	// but trying to add it with --force should
	v = validators.MakeHappyMapValidator(t)
	cmd = newCmdAddAnnotation(fakeFS, v.Validator)
	cmd.Flag("force").Value.Set("true")
	err = cmd.RunE(cmd, args)
	v.VerifyCall()
	if err != nil {
		t.Errorf("unexpected error: %v", err.Error())
	}
}

func TestRunAddLabel(t *testing.T) {
	var o addMetadataOptions
	o.metadata = map[string]string{"owls": "cute", "otters": "adorable"}

	m := makeKustomization(t)
	err := o.addLabels(m)
	if err != nil {
		t.Errorf("unexpected error: could not write to kustomization file")
	}
	// adding the same test input should not work
	err = o.addLabels(m)
	if err == nil {
		t.Errorf("expected already in kustomization file error")
	}
	// adding new labels should work
	o.metadata = map[string]string{"new": "label"}
	err = o.addLabels(m)
	if err != nil {
		t.Errorf("unexpected error: could not write to kustomization file")
	}
}

func TestAddLabelNoArgs(t *testing.T) {
	fakeFS := fs.MakeFakeFS()
	v := validators.MakeHappyMapValidator(t)
	cmd := newCmdAddLabel(fakeFS, v.Validator)
	err := cmd.Execute()
	v.VerifyNoCall()
	if err == nil {
		t.Errorf("expected an error")
	}
	if err.Error() != "must specify label" {
		t.Errorf("incorrect error: %v", err.Error())
	}
}

func TestAddLabelInvalidFormat(t *testing.T) {
	fakeFS := fs.MakeFakeFS()
	v := validators.MakeSadMapValidator(t)
	cmd := newCmdAddLabel(fakeFS, v.Validator)
	args := []string{"exclamation!:point"}
	err := cmd.RunE(cmd, args)
	v.VerifyCall()
	if err == nil {
		t.Errorf("expected an error")
	}
	if err.Error() != validators.SAD {
		t.Errorf("incorrect error: %v", err.Error())
	}
}

func TestAddLabelNoKey(t *testing.T) {
	fakeFS := fs.MakeFakeFS()
	v := validators.MakeHappyMapValidator(t)
	cmd := newCmdAddLabel(fakeFS, v.Validator)
	args := []string{":nokey"}
	err := cmd.RunE(cmd, args)
	v.VerifyNoCall()
	if err == nil {
		t.Errorf("expected an error")
	}
	if err.Error() != "invalid label: ':nokey' (need k:v pair where v may be quoted)" {
		t.Errorf("incorrect error: %v", err.Error())
	}
}

func TestAddLabelTooManyColons(t *testing.T) {
	fakeFS := fs.MakeFakeFS()
	fakeFS.WriteTestKustomization()
	v := validators.MakeHappyMapValidator(t)
	cmd := newCmdAddLabel(fakeFS, v.Validator)
	args := []string{"key:v1:v2"}
	err := cmd.RunE(cmd, args)
	v.VerifyCall()
	if err != nil {
		t.Errorf("unexpected error: %v", err.Error())
	}
}

func TestAddLabelNoValue(t *testing.T) {
	fakeFS := fs.MakeFakeFS()
	fakeFS.WriteTestKustomization()
	v := validators.MakeHappyMapValidator(t)
	cmd := newCmdAddLabel(fakeFS, v.Validator)
	args := []string{"no,value:"}
	err := cmd.RunE(cmd, args)
	v.VerifyCall()
	if err != nil {
		t.Errorf("unexpected error: %v", err.Error())
	}
}

func TestAddLabelMultipleArgs(t *testing.T) {
	fakeFS := fs.MakeFakeFS()
	fakeFS.WriteTestKustomization()
	v := validators.MakeHappyMapValidator(t)
	cmd := newCmdAddLabel(fakeFS, v.Validator)
	args := []string{"this:input", "has:spaces"}
	err := cmd.RunE(cmd, args)
	v.VerifyNoCall()
	if err == nil {
		t.Errorf("expected an error")
	}
	if err.Error() != "labels must be comma-separated, with no spaces" {
		t.Errorf("incorrect error: %v", err.Error())
	}
}

func TestAddLabelForce(t *testing.T) {
	fakeFS := fs.MakeFakeFS()
	fakeFS.WriteTestKustomization()
	v := validators.MakeHappyMapValidator(t)
	cmd := newCmdAddLabel(fakeFS, v.Validator)
	args := []string{"key:foo"}
	err := cmd.RunE(cmd, args)
	v.VerifyCall()
	if err != nil {
		t.Errorf("unexpected error: %v", err.Error())
	}
	// trying to add the same label again should not work
	args = []string{"key:bar"}
	v = validators.MakeHappyMapValidator(t)
	cmd = newCmdAddLabel(fakeFS, v.Validator)
	err = cmd.RunE(cmd, args)
	v.VerifyCall()
	if err == nil {
		t.Errorf("expected an error")
	}
	if err.Error() != "label key already in kustomization file" {
		t.Errorf("expected an error")
	}
	// but trying to add it with --force should
	v = validators.MakeHappyMapValidator(t)
	cmd = newCmdAddLabel(fakeFS, v.Validator)
	cmd.Flag("force").Value.Set("true")
	err = cmd.RunE(cmd, args)
	v.VerifyCall()
	if err != nil {
		t.Errorf("unexpected error: %v", err.Error())
	}
}

func TestConvertToMap(t *testing.T) {
	var o addMetadataOptions
	args := "a:b,c:\"d\",e:\"f:g\",g:h:k"
	expected := make(map[string]string)
	expected["a"] = "b"
	expected["c"] = "d"
	expected["e"] = "f:g"
	expected["g"] = "h:k"

	result, err := o.convertToMap(args)
	if err != nil {
		t.Errorf("unexpected error: %v", err.Error())
	}

	eq := reflect.DeepEqual(expected, result)
	if !eq {
		t.Errorf("Converted map does not match expected, expected: %v, result: %v\n", expected, result)
	}
}

func TestConvertToMapError(t *testing.T) {
	var o addMetadataOptions
	args := "a:b,c:\"d\",:f:g"

	_, err := o.convertToMap(args)
	if err == nil {
		t.Errorf("expected an error")
	}
	if err.Error() != "invalid annotation: ':f:g' (need k:v pair where v may be quoted)" {
		t.Errorf("incorrect error: %v", err.Error())
	}
}
