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

// A set of common functions needed by cmd/kubectl and pkg/kubectl packages.
package kubectl

import (
	"errors"
	"fmt"
	"path"
	"strings"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apis/extensions"
)

const kubectlAnnotationPrefix = "kubectl.kubernetes.io/"

type NamespaceInfo struct {
	Namespace string
}

func listOfImages(spec *api.PodSpec) []string {
	var images []string
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
	OutputVersions []unversioned.GroupVersion
}

// RESTMapping implements meta.RESTMapper by prepending the output version to the preferred version list.
func (m OutputVersionMapper) RESTMapping(gk unversioned.GroupKind, versions ...string) (*meta.RESTMapping, error) {
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

// ShortcutExpander is a RESTMapper that can be used for Kubernetes
// resources.
type ShortcutExpander struct {
	meta.RESTMapper
}

var _ meta.RESTMapper = &ShortcutExpander{}

// KindFor implements meta.RESTMapper. It expands the resource first, then invokes the wrapped
// mapper.
func (e ShortcutExpander) KindFor(resource unversioned.GroupVersionResource) (unversioned.GroupVersionKind, error) {
	resource = expandResourceShortcut(resource)
	return e.RESTMapper.KindFor(resource)
}

// ResourceSingularizer expands the named resource and then singularizes it.
func (e ShortcutExpander) ResourceSingularizer(resource string) (string, error) {
	return e.RESTMapper.ResourceSingularizer(expandResourceShortcut(unversioned.GroupVersionResource{Resource: resource}).Resource)
}

// expandResourceShortcut will return the expanded version of resource
// (something that a pkg/api/meta.RESTMapper can understand), if it is
// indeed a shortcut. Otherwise, will return resource unmodified.
func expandResourceShortcut(resource unversioned.GroupVersionResource) unversioned.GroupVersionResource {
	shortForms := map[string]unversioned.GroupVersionResource{
		// Please keep this alphabetized
		// If you add an entry here, please also take a look at pkg/kubectl/cmd/cmd.go
		// and add an entry to valid_resources when appropriate.
		"cs":     api.SchemeGroupVersion.WithResource("componentstatuses"),
		"ds":     extensions.SchemeGroupVersion.WithResource("daemonsets"),
		"ep":     api.SchemeGroupVersion.WithResource("endpoints"),
		"ev":     api.SchemeGroupVersion.WithResource("events"),
		"hpa":    extensions.SchemeGroupVersion.WithResource("horizontalpodautoscalers"),
		"ing":    extensions.SchemeGroupVersion.WithResource("ingresses"),
		"limits": api.SchemeGroupVersion.WithResource("limitranges"),
		"no":     api.SchemeGroupVersion.WithResource("nodes"),
		"ns":     api.SchemeGroupVersion.WithResource("namespaces"),
		"po":     api.SchemeGroupVersion.WithResource("pods"),
		"psp":    api.SchemeGroupVersion.WithResource("podSecurityPolicies"),
		"pvc":    api.SchemeGroupVersion.WithResource("persistentvolumeclaims"),
		"pv":     api.SchemeGroupVersion.WithResource("persistentvolumes"),
		"quota":  api.SchemeGroupVersion.WithResource("resourcequotas"),
		"rc":     api.SchemeGroupVersion.WithResource("replicationcontrollers"),
		"rs":     extensions.SchemeGroupVersion.WithResource("replicasets"),
		"svc":    api.SchemeGroupVersion.WithResource("services"),
	}
	if expanded, ok := shortForms[resource.Resource]; ok {
		return expanded
	}
	return resource
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
	items := strings.Split(source, "=")
	if len(items) != 2 {
		return "", "", fmt.Errorf("invalid literal source %v, expected key=value", source)
	}

	return items[0], items[1], nil
}
