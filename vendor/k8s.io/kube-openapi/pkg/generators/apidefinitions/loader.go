/*
Copyright The Kubernetes Authors.

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

package apidefinitions

import (
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"sigs.k8s.io/yaml"
)

const (
	// We define an apiVersion and kinds for defining APIs
	// in the source tree like we do for everything else.
	schemeGroupVersion = "apidefinitions.k8s.io/v1alpha1"
	kindAPIVersion     = "APIVersion"
	kindAPIGroup       = "APIGroup"

	// We have a naming convention for the files used to define
	// APIs in the source tree.
	apiVersionFile = "apiversion.yaml"
	apiGroupFile   = "apigroup.yaml"
)

// LoadAPIVersion reads an apiversion.yaml file, returning nil if absent.
func LoadAPIVersion(dir string) (*APIVersion, error) {
	data, err := readManifest(dir, apiVersionFile)
	if err != nil || data == nil {
		return nil, err
	}
	av := &APIVersion{}
	if err := yaml.Unmarshal(data, av); err != nil {
		return nil, fmt.Errorf("%s: %w", filepath.Join(dir, apiVersionFile), err)
	}
	if err := validateTypeMeta(av.APIVersion, av.Kind, kindAPIVersion); err != nil {
		return nil, fmt.Errorf("%s: %w", filepath.Join(dir, apiVersionFile), err)
	}
	if err := validateName(av); err != nil {
		return nil, fmt.Errorf("%s: %w", filepath.Join(dir, apiVersionFile), err)
	}
	return av, nil
}

// LoadAPIGroup reads an apigroup.yaml file, returning nil if absent.
func LoadAPIGroup(dir string) (*APIGroup, error) {
	data, err := readManifest(dir, apiGroupFile)
	if err != nil || data == nil {
		return nil, err
	}
	g := &APIGroup{}
	if err := yaml.Unmarshal(data, g); err != nil {
		return nil, fmt.Errorf("%s: %w", filepath.Join(dir, apiGroupFile), err)
	}
	if err := validateTypeMeta(g.APIVersion, g.Kind, kindAPIGroup); err != nil {
		return nil, fmt.Errorf("%s: %w", filepath.Join(dir, apiGroupFile), err)
	}
	return g, nil
}

func readManifest(dir, filename string) ([]byte, error) {
	data, err := os.ReadFile(filepath.Join(dir, filename))
	if errors.Is(err, fs.ErrNotExist) {
		return nil, nil
	}
	return data, err
}

func validateTypeMeta(actualAPIVersion, actualKind, expectedKind string) error {
	if actualAPIVersion != schemeGroupVersion {
		return fmt.Errorf("expected apiVersion %s but got %s", schemeGroupVersion, actualAPIVersion)
	}
	if actualKind != expectedKind {
		return fmt.Errorf("expected kind %s but got %s", expectedKind, actualKind)
	}
	return nil
}

var (
	groupRegexp = regexp.MustCompile(`^[a-z0-9\-]+(\.[a-z0-9\-]+)*$`)
)

func validateName(av *APIVersion) error {
	g, _, err := splitGroupVersion(av.Metadata.Name)
	if err != nil {
		return fmt.Errorf("metadata.name: %w", err)
	}
	if g != "" && !groupRegexp.MatchString(g) {
		return fmt.Errorf("metadata.name: group %q must be lowercase letters, optionally dot-separated", g)
	}
	return nil
}

// splitGroupVersion parses "<group>/<version>" or "<version>" (core group).
func splitGroupVersion(name string) (string, string, error) {
	parts := strings.Split(name, "/")
	switch len(parts) {
	case 1:
		if parts[0] == "" {
			return "", "", fmt.Errorf("version is required")
		}
		return "", parts[0], nil
	case 2:
		if parts[0] == "" || parts[1] == "" {
			return "", "", fmt.Errorf("group and version are both required when using <group>/<version>: %s", name)
		}
		return parts[0], parts[1], nil
	default:
		return "", "", fmt.Errorf("expected <group>/<version> or <version> but got: %s", name)
	}
}
