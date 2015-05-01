/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"strings"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/latest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/validation"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/yaml"
	"github.com/golang/glog"
)

func validateObject(obj runtime.Object) (errors []error) {
	switch t := obj.(type) {
	case *api.ReplicationController:
		if t.Namespace == "" {
			t.Namespace = api.NamespaceDefault
		}
		errors = validation.ValidateReplicationController(t)
	case *api.ReplicationControllerList:
		for i := range t.Items {
			errors = append(errors, validateObject(&t.Items[i])...)
		}
	case *api.Service:
		if t.Namespace == "" {
			t.Namespace = api.NamespaceDefault
		}
		errors = validation.ValidateService(t)
	case *api.ServiceList:
		for i := range t.Items {
			errors = append(errors, validateObject(&t.Items[i])...)
		}
	case *api.Pod:
		if t.Namespace == "" {
			t.Namespace = api.NamespaceDefault
		}
		errors = validation.ValidatePod(t)
	case *api.PodList:
		for i := range t.Items {
			errors = append(errors, validateObject(&t.Items[i])...)
		}
	case *api.PersistentVolume:
		errors = validation.ValidatePersistentVolume(t)
	case *api.PersistentVolumeClaim:
		if t.Namespace == "" {
			t.Namespace = api.NamespaceDefault
		}
		errors = validation.ValidatePersistentVolumeClaim(t)
	case *api.PodTemplate:
		if t.Namespace == "" {
			t.Namespace = api.NamespaceDefault
		}
		errors = validation.ValidatePodTemplate(t)
	default:
		return []error{fmt.Errorf("no validation defined for %#v", obj)}
	}
	return errors
}

func walkJSONFiles(inDir string, fn func(name, path string, data []byte)) error {
	return filepath.Walk(inDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		if info.IsDir() && path != inDir {
			return filepath.SkipDir
		}

		file := filepath.Base(path)
		if ext := filepath.Ext(file); ext == ".json" || ext == ".yaml" {
			glog.Infof("Testing %s", path)
			data, err := ioutil.ReadFile(path)
			if err != nil {
				return err
			}
			name := strings.TrimSuffix(file, ext)

			if ext == ".yaml" {
				out, err := yaml.ToJSON(data)
				if err != nil {
					return err
				}
				data = out
			}

			fn(name, path, data)
		}
		return nil
	})
}

func TestExampleObjectSchemas(t *testing.T) {
	cases := map[string]map[string]runtime.Object{
		"../docs/getting-started-guides": {
			"pod": &api.Pod{},
		},
		"../cmd/integration": {
			"v1beta1-controller": &api.ReplicationController{},
			"v1beta3-controller": &api.ReplicationController{},
		},
		"../examples/guestbook": {
			"frontend-controller":     &api.ReplicationController{},
			"redis-slave-controller":  &api.ReplicationController{},
			"redis-master-controller": &api.ReplicationController{},
			"frontend-service":        &api.Service{},
			"redis-master-service":    &api.Service{},
			"redis-slave-service":     &api.Service{},
		},
		"../examples/guestbook-go": {
			"guestbook-controller":    &api.ReplicationController{},
			"redis-slave-controller":  &api.ReplicationController{},
			"redis-master-controller": &api.ReplicationController{},
			"guestbook-service":       &api.Service{},
			"redis-master-service":    &api.Service{},
			"redis-slave-service":     &api.Service{},
		},
		"../examples/guestbook-go/v1beta3": {
			"guestbook-controller":    &api.ReplicationController{},
			"redis-slave-controller":  &api.ReplicationController{},
			"redis-master-controller": &api.ReplicationController{},
			"guestbook-service":       &api.Service{},
			"redis-master-service":    &api.Service{},
			"redis-slave-service":     &api.Service{},
		},
		"../examples/walkthrough": {
			"pod1": &api.Pod{},
			"pod2": &api.Pod{},
			"pod-with-http-healthcheck": &api.Pod{},
			"service":                   &api.Service{},
			"replication-controller":    &api.ReplicationController{},
			"podtemplate":               &api.PodTemplate{},
		},
		"../examples/update-demo": {
			"kitten-rc":   &api.ReplicationController{},
			"nautilus-rc": &api.ReplicationController{},
		},
		"../examples/persistent-volumes/volumes": {
			"local-01": &api.PersistentVolume{},
			"local-02": &api.PersistentVolume{},
			"gce":      &api.PersistentVolume{},
		},
		"../examples/persistent-volumes/claims": {
			"claim-01": &api.PersistentVolumeClaim{},
			"claim-02": &api.PersistentVolumeClaim{},
			"claim-03": &api.PersistentVolumeClaim{},
		},
		"../examples/iscsi/v1beta1": {
			"iscsi": &api.Pod{},
		},
		"../examples/iscsi/v1beta3": {
			"iscsi": &api.Pod{},
		},
		"../examples/glusterfs/v1beta3": {
			"glusterfs": &api.Pod{},
		},
	}

	for path, expected := range cases {
		tested := 0
		err := walkJSONFiles(path, func(name, path string, data []byte) {
			expectedType, found := expected[name]
			if !found {
				t.Errorf("%s: %s does not have a test case defined", path, name)
				return
			}
			tested += 1
			if err := latest.Codec.DecodeInto(data, expectedType); err != nil {
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
}

// This regex is tricky, but it works.  For future me, here is the decode:
//
// Flags: (?ms) = multiline match, allow . to match \n
// 1) Look for a line that starts with ``` (a markdown code block)
// 2) (?: ... ) = non-capturing group
// 3) (P<name>) = capture group as "name"
// 4) Look for #1 followed by either:
// 4a)    "yaml" followed by any word-characters followed by a newline (e.g. ```yamlfoo\n)
// 4b)    "any word-characters followed by a newline (e.g. ```json\n)
// 5) Look for either:
// 5a)    #4a followed by one or more characters (non-greedy)
// 5b)    #4b followed by { followed by one or more characters (non-greedy) followed by }
// 6) Look for #5 followed by a newline followed by ``` (end of the code block)
//
// This could probably be simplified, but is already too delicate.  Before any
// real changes, we should have a testscase that just tests this regex.
var sampleRegexp = regexp.MustCompile("(?ms)^```(?:(?P<type>yaml)\\w*\\n(?P<content>.+?)|\\w*\\n(?P<content>\\{.+?\\}))\\n^```")
var subsetRegexp = regexp.MustCompile("(?ms)\\.{3}")

func TestReadme(t *testing.T) {
	paths := []string{
		"../README.md",
		"../examples/walkthrough/README.md",
		"../examples/iscsi/README.md",
	}

	for _, path := range paths {
		data, err := ioutil.ReadFile(path)
		if err != nil {
			t.Errorf("Unable to read file %s: %v", path, err)
			continue
		}

		matches := sampleRegexp.FindAllStringSubmatch(string(data), -1)
		if matches == nil {
			continue
		}
		for _, match := range matches {
			var content, subtype string
			for i, name := range sampleRegexp.SubexpNames() {
				if name == "type" {
					subtype = match[i]
				}
				if name == "content" && match[i] != "" {
					content = match[i]
				}
			}
			if subtype == "yaml" && subsetRegexp.FindString(content) != "" {
				t.Logf("skipping (%s): \n%s", subtype, content)
				continue
			}

			//t.Logf("testing (%s): \n%s", subtype, content)
			expectedType := &api.Pod{}
			json, err := yaml.ToJSON([]byte(content))
			if err != nil {
				t.Errorf("%s could not be converted to JSON: %v\n%s", path, err, string(content))
			}
			if err := latest.Codec.DecodeInto(json, expectedType); err != nil {
				t.Errorf("%s did not decode correctly: %v\n%s", path, err, string(content))
				continue
			}
			if errors := validateObject(expectedType); len(errors) > 0 {
				t.Errorf("%s did not validate correctly: %v", path, errors)
			}
			_, err = latest.Codec.Encode(expectedType)
			if err != nil {
				t.Errorf("Could not encode object: %v", err)
				continue
			}
		}
	}
}
