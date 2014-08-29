/*
Copyright 2014 Google Inc. All rights reserved.

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

package examples_test

import (
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"regexp"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	_ "github.com/GoogleCloudPlatform/kubernetes/pkg/api/v1beta1"
	"github.com/golang/glog"
)

func validateObject(obj interface{}) (errors []error) {
	switch t := obj.(type) {
	case *api.ReplicationController:
		errors = api.ValidateManifest(&t.DesiredState.PodTemplate.DesiredState.Manifest)
	case *api.ReplicationControllerList:
		for i := range t.Items {
			errors = append(errors, validateObject(&t.Items[i])...)
		}
	case *api.Service:
		errors = api.ValidateService(t)
	case *api.ServiceList:
		for i := range t.Items {
			errors = append(errors, validateObject(&t.Items[i])...)
		}
	case *api.Pod:
		errors = api.ValidateManifest(&t.DesiredState.Manifest)
	case *api.PodList:
		for i := range t.Items {
			errors = append(errors, validateObject(&t.Items[i])...)
		}
	default:
		return []error{fmt.Errorf("no validation defined for %#v", obj)}
	}
	return errors
}

func walkJSONFiles(inDir string, fn func(name, path string, data []byte)) error {
	err := filepath.Walk(inDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.IsDir() && path != inDir {
			return filepath.SkipDir
		}
		name := filepath.Base(path)
		ext := filepath.Ext(name)
		if ext != "" {
			name = name[:len(name)-len(ext)]
		}
		if ext != ".json" {
			return nil
		}
		glog.Infof("Testing %s", path)
		data, err := ioutil.ReadFile(path)
		if err != nil {
			return err
		}
		fn(name, path, data)
		return nil
	})
	return err
}

func TestApiExamples(t *testing.T) {
	expected := map[string]interface{}{
		"controller":       &api.ReplicationController{},
		"controller-list":  &api.ReplicationControllerList{},
		"pod":              &api.Pod{},
		"pod-list":         &api.PodList{},
		"service":          &api.Service{},
		"external-service": &api.Service{},
		"service-list":     &api.ServiceList{},
	}

	tested := 0
	err := walkJSONFiles("../api/examples", func(name, path string, data []byte) {
		expectedType, found := expected[name]
		if !found {
			t.Errorf("%s does not have a test case defined", path)
			return
		}
		tested += 1
		if err := api.DecodeInto(data, expectedType); err != nil {
			t.Errorf("%s did not decode correctly: %v\n%s", path, err, string(data))
			return
		}
		if errors := validateObject(expectedType); len(errors) > 0 {
			t.Errorf("%s did not validate correctly: %v", path, errors)
		}
	})
	if err != nil {
		t.Errorf("Expected no error, Got %v", err)
	}
	if tested != len(expected) {
		t.Errorf("Expected %d examples, Got %d", len(expected), tested)
	}
}

func TestExamples(t *testing.T) {
	expected := map[string]interface{}{
		"frontend-controller":    &api.ReplicationController{},
		"redis-slave-controller": &api.ReplicationController{},
		"redis-master":           &api.Pod{},
		"frontend-service":       &api.Service{},
		"redis-master-service":   &api.Service{},
		"redis-slave-service":    &api.Service{},
	}

	tested := 0
	err := walkJSONFiles("../examples/guestbook", func(name, path string, data []byte) {
		expectedType, found := expected[name]
		if !found {
			t.Errorf("%s does not have a test case defined", path)
			return
		}
		tested += 1
		if err := api.DecodeInto(data, expectedType); err != nil {
			t.Errorf("%s did not decode correctly: %v\n%s", path, err, string(data))
			return
		}
		if errors := validateObject(expectedType); len(errors) > 0 {
			t.Errorf("%s did not validate correctly: %v", path, errors)
		}
	})
	if err != nil {
		t.Errorf("Expected no error, Got %v", err)
	}
	if tested != len(expected) {
		t.Errorf("Expected %d examples, Got %d", len(expected), tested)
	}
}

var jsonRegexp = regexp.MustCompile("(?ms)^```\\w*\\n(\\{.+?\\})\\w*\\n^```")

func TestReadme(t *testing.T) {
	path := "../README.md"
	data, err := ioutil.ReadFile(path)
	if err != nil {
		t.Fatalf("Unable to read file: %v", err)
	}

	match := jsonRegexp.FindStringSubmatch(string(data))
	if match == nil {
		return
	}
	for _, json := range match[1:] {
		expectedType := &api.Pod{}
		if err := api.DecodeInto([]byte(json), expectedType); err != nil {
			t.Errorf("%s did not decode correctly: %v\n%s", path, err, string(data))
			return
		}
		if errors := validateObject(expectedType); len(errors) > 0 {
			t.Errorf("%s did not validate correctly: %v", path, errors)
		}
		encoded, err := api.Encode(expectedType)
		if err != nil {
			t.Errorf("Could not encode object: %v", err)
			continue
		}
		t.Logf("Found pod %s\n%s", json, encoded)
	}
}
