/*
Copyright 2020 The Kubernetes Authors.

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

package behaviors

import (
	"fmt"
	"io/ioutil"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"gopkg.in/yaml.v2"
)

// BehaviorFileList returns a list of eligible behavior files in or under dir
func BehaviorFileList(dir string) ([]string, error) {
	var behaviorFiles []string

	r, _ := regexp.Compile(".+.yaml$")
	err := filepath.Walk(dir,
		func(path string, info os.FileInfo, err error) error {
			if err != nil {
				return err
			}
			if r.MatchString(path) {
				behaviorFiles = append(behaviorFiles, path)
			}
			return nil
		},
	)
	return behaviorFiles, err
}

// LoadSuite loads a Behavior Suite from .yaml file at path
func LoadSuite(path string) (*Suite, error) {
	var suite Suite
	bytes, err := ioutil.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("error loading suite %s: %v", path, err)
	}
	err = yaml.UnmarshalStrict(bytes, &suite)
	if err != nil {
		return nil, fmt.Errorf("error loading suite %s: %v", path, err)
	}
	return &suite, nil
}

// ValidateSuite validates that the given suite has no duplicate behavior IDs
func ValidateSuite(suite *Suite) error {
	var errs []error
	behaviorsByID := make(map[string]bool)
	for _, b := range suite.Behaviors {
		if _, ok := behaviorsByID[b.ID]; ok {
			errs = append(errs, fmt.Errorf("Duplicate behavior ID: %s", b.ID))
		}
		if !strings.HasPrefix(b.ID, suite.Suite) {
			errs = append(errs, fmt.Errorf("Invalid behavior ID: %s, must have suite name as prefix: %s", b.ID, suite.Suite))
		}
		behaviorsByID[b.ID] = true
	}
	return utilerrors.NewAggregate(errs)
}
