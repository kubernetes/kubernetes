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

package crd_test

import (
	"path/filepath"
	"testing"

	"github.com/ghodss/yaml"
	"io/ioutil"
	apiextensionsv1beta1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1"
	apiextensionsscheme "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset/scheme"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/csi-api/pkg/crd"
	"os"
)

func TestBootstrapCRDs(t *testing.T) {
	testObjects(t, crd.CSIDriverCRD(), "csidriver.yaml")
	testObjects(t, crd.CSINodeInfoCRD(), "csinodeinfo.yaml")
}

func testObjects(t *testing.T, crd *apiextensionsv1beta1.CustomResourceDefinition, fixtureFilename string) {
	filename := filepath.Join("testdata", fixtureFilename)
	expectedYAML, err := ioutil.ReadFile(filename)
	if err != nil {
		t.Fatal(err)
	}

	jsonData, err := runtime.Encode(apiextensionsscheme.Codecs.LegacyCodec(apiextensionsv1beta1.SchemeGroupVersion), crd)
	if err != nil {
		t.Fatal(err)
	}
	yamlData, err := yaml.JSONToYAML(jsonData)
	if err != nil {
		t.Fatal(err)
	}
	if string(yamlData) != string(expectedYAML) {
		t.Errorf("Bootstrap CRD data does not match the test fixture in %s", filename)

		const updateEnvVar = "UPDATE_CSI_CRD_FIXTURE_DATA"
		if os.Getenv(updateEnvVar) == "true" {
			if err := ioutil.WriteFile(filename, []byte(yamlData), os.FileMode(0755)); err == nil {
				t.Logf("Updated data in %s", filename)
				t.Logf("Verify the diff, commit changes, and rerun the tests")
			} else {
				t.Logf("Could not update data in %s: %v", filename, err)
			}
		} else {
			t.Logf("Diff between data in code and fixture data in %s:\n-------------\n%s", filename, diff.StringDiff(string(yamlData), string(expectedYAML)))
			t.Logf("If the change is expected, re-run with %s=true to update the fixtures", updateEnvVar)
		}
	}
}
