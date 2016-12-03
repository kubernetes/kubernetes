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

// A set of common functions needed by cmd/kubectl and pkg/kubectl packages.
package kubectl

import (
	"errors"
	"fmt"
	"path"
	"strings"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/runtime/schema"
)

const (
	kubectlAnnotationPrefix = "kubectl.kubernetes.io/"
)

type NamespaceInfo struct {
	Namespace string
}

func listOfImages(spec *api.PodSpec) []string {
	images := make([]string, 0, len(spec.Containers))
	for _, container := range spec.Containers {
		images = append(images, container.Image)
	}
	return images
}

func makeImageList(spec *api.PodSpec) string {
	return strings.Join(listOfImages(spec), ",")
}

// OutputVersionMapper is a RESTMapper that will prefer mappings that
// correspond to a preferred output version (if feasible)
type OutputVersionMapper struct {
	meta.RESTMapper

	// output versions takes a list of preferred GroupVersions. Only the first
	// hit for a given group will have effect.  This allows different output versions
	// depending upon the group of the kind being requested
	OutputVersions []schema.GroupVersion
}

// RESTMapping implements meta.RESTMapper by prepending the output version to the preferred version list.
func (m OutputVersionMapper) RESTMapping(gk schema.GroupKind, versions ...string) (*meta.RESTMapping, error) {
	for _, preferredVersion := range m.OutputVersions {
		if gk.Group == preferredVersion.Group {
			mapping, err := m.RESTMapper.RESTMapping(gk, preferredVersion.Version)
			if err == nil {
				return mapping, nil
			}

			break
		}
	}

	return m.RESTMapper.RESTMapping(gk, versions...)
}

// ShortForms is the list of short names to their expanded names
var ShortForms = map[string]string{
	// Please keep this alphabetized
	// If you add an entry here, please also take a look at pkg/kubectl/cmd/cmd.go
	// and add an entry to valid_resources when appropriate.
	"cm":     "configmaps",
	"cs":     "componentstatuses",
	"csr":    "certificatesigningrequests",
	"deploy": "deployments",
	"ds":     "daemonsets",
	"ep":     "endpoints",
	"ev":     "events",
	"hpa":    "horizontalpodautoscalers",
	"ing":    "ingresses",
	"limits": "limitranges",
	"no":     "nodes",
	"ns":     "namespaces",
	"pdb":    "poddisruptionbudgets",
	"po":     "pods",
	"psp":    "podSecurityPolicies",
	"pvc":    "persistentvolumeclaims",
	"pv":     "persistentvolumes",
	"quota":  "resourcequotas",
	"rc":     "replicationcontrollers",
	"rs":     "replicasets",
	"sa":     "serviceaccounts",
	"svc":    "services",
}

// ResourceShortFormFor looks up for a short form of resource names.
func ResourceShortFormFor(resource string) (string, bool) {
	var alias string
	exists := false
	for k, val := range ShortForms {
		if val == resource {
			alias = k
			exists = true
			break
		}
	}
	return alias, exists
}

// ResourceAliases returns the resource shortcuts and plural forms for the given resources.
func ResourceAliases(rs []string) []string {
	as := make([]string, 0, len(rs))
	plurals := make(map[string]struct{}, len(rs))
	for _, r := range rs {
		var plural string
		switch {
		case r == "endpoints":
			plural = r // exception. "endpoint" does not exist. Why?
		case strings.HasSuffix(r, "y"):
			plural = r[0:len(r)-1] + "ies"
		case strings.HasSuffix(r, "s"):
			plural = r + "es"
		default:
			plural = r + "s"
		}
		as = append(as, plural)

		plurals[plural] = struct{}{}
	}

	for sf, r := range ShortForms {
		if _, found := plurals[r]; found {
			as = append(as, sf)
		}
	}
	return as
}

// parseFileSource parses the source given. Acceptable formats include:
//
// 1.  source-path: the basename will become the key name
// 2.  source-name=source-path: the source-name will become the key name and source-path is the path to the key file
//
// Key names cannot include '='.
func parseFileSource(source string) (keyName, filePath string, err error) {
	numSeparators := strings.Count(source, "=")
	switch {
	case numSeparators == 0:
		return path.Base(source), source, nil
	case numSeparators == 1 && strings.HasPrefix(source, "="):
		return "", "", fmt.Errorf("key name for file path %v missing.", strings.TrimPrefix(source, "="))
	case numSeparators == 1 && strings.HasSuffix(source, "="):
		return "", "", fmt.Errorf("file path for key name %v missing.", strings.TrimSuffix(source, "="))
	case numSeparators > 1:
		return "", "", errors.New("Key names or file paths cannot contain '='.")
	default:
		components := strings.Split(source, "=")
		return components[0], components[1], nil
	}
}

// parseLiteralSource parses the source key=val pair
func parseLiteralSource(source string) (keyName, value string, err error) {
	// leading equal is invalid
	if strings.Index(source, "=") == 0 {
		return "", "", fmt.Errorf("invalid literal source %v, expected key=value", source)
	}
	// split after the first equal (so values can have the = character)
	items := strings.SplitN(source, "=", 2)
	if len(items) != 2 {
		return "", "", fmt.Errorf("invalid literal source %v, expected key=value", source)
	}

	return items[0], items[1], nil
}
