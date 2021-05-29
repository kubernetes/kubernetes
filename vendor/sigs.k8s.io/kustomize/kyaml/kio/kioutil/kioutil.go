// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package kioutil

import (
	"fmt"
	"path"
	"sort"
	"strconv"
	"strings"

	"sigs.k8s.io/kustomize/kyaml/errors"
	"sigs.k8s.io/kustomize/kyaml/yaml"
)

type AnnotationKey = string

const (
	// IndexAnnotation records the index of a specific resource in a file or input stream.
	IndexAnnotation AnnotationKey = "config.kubernetes.io/index"

	// PathAnnotation records the path to the file the Resource was read from
	PathAnnotation AnnotationKey = "config.kubernetes.io/path"
)

func GetFileAnnotations(rn *yaml.RNode) (string, string, error) {
	meta, err := rn.GetMeta()
	if err != nil {
		return "", "", err
	}
	path := meta.Annotations[PathAnnotation]
	index := meta.Annotations[IndexAnnotation]
	return path, index, nil
}

// ErrorIfMissingAnnotation validates the provided annotations are present on the given resources
func ErrorIfMissingAnnotation(nodes []*yaml.RNode, keys ...AnnotationKey) error {
	for _, key := range keys {
		for _, node := range nodes {
			val, err := node.Pipe(yaml.GetAnnotation(key))
			if err != nil {
				return errors.Wrap(err)
			}
			if val == nil {
				return errors.Errorf("missing annotation %s", key)
			}
		}
	}
	return nil
}

// CreatePathAnnotationValue creates a default path annotation value for a Resource.
// The path prefix will be dir.
func CreatePathAnnotationValue(dir string, m yaml.ResourceMeta) string {
	filename := fmt.Sprintf("%s_%s.yaml", strings.ToLower(m.Kind), m.Name)
	return path.Join(dir, m.Namespace, filename)
}

// DefaultPathAndIndexAnnotation sets a default path or index value on any nodes missing the
// annotation
func DefaultPathAndIndexAnnotation(dir string, nodes []*yaml.RNode) error {
	counts := map[string]int{}

	// check each node for the path annotation
	for i := range nodes {
		m, err := nodes[i].GetMeta()
		if err != nil {
			return err
		}

		// calculate the max index in each file in case we are appending
		if p, found := m.Annotations[PathAnnotation]; found {
			// record the max indexes into each file
			if i, found := m.Annotations[IndexAnnotation]; found {
				index, _ := strconv.Atoi(i)
				if index > counts[p] {
					counts[p] = index
				}
			}

			// has the path annotation already -- do nothing
			continue
		}

		// set a path annotation on the Resource
		path := CreatePathAnnotationValue(dir, m)
		if err := nodes[i].PipeE(yaml.SetAnnotation(PathAnnotation, path)); err != nil {
			return err
		}
	}

	// set the index annotations
	for i := range nodes {
		m, err := nodes[i].GetMeta()
		if err != nil {
			return err
		}

		if _, found := m.Annotations[IndexAnnotation]; found {
			continue
		}

		p := m.Annotations[PathAnnotation]

		// set an index annotation on the Resource
		c := counts[p]
		counts[p] = c + 1
		if err := nodes[i].PipeE(
			yaml.SetAnnotation(IndexAnnotation, fmt.Sprintf("%d", c))); err != nil {
			return err
		}
	}
	return nil
}

// DefaultPathAnnotation sets a default path annotation on any Reources
// missing it.
func DefaultPathAnnotation(dir string, nodes []*yaml.RNode) error {
	// check each node for the path annotation
	for i := range nodes {
		m, err := nodes[i].GetMeta()
		if err != nil {
			return err
		}

		if _, found := m.Annotations[PathAnnotation]; found {
			// has the path annotation already -- do nothing
			continue
		}

		// set a path annotation on the Resource
		path := CreatePathAnnotationValue(dir, m)
		if err := nodes[i].PipeE(yaml.SetAnnotation(PathAnnotation, path)); err != nil {
			return err
		}
	}
	return nil
}

// Map invokes fn for each element in nodes.
func Map(nodes []*yaml.RNode, fn func(*yaml.RNode) (*yaml.RNode, error)) ([]*yaml.RNode, error) {
	var returnNodes []*yaml.RNode
	for i := range nodes {
		n, err := fn(nodes[i])
		if err != nil {
			return nil, errors.Wrap(err)
		}
		if n != nil {
			returnNodes = append(returnNodes, n)
		}
	}
	return returnNodes, nil
}

func MapMeta(nodes []*yaml.RNode, fn func(*yaml.RNode, yaml.ResourceMeta) (*yaml.RNode, error)) (
	[]*yaml.RNode, error) {
	var returnNodes []*yaml.RNode
	for i := range nodes {
		meta, err := nodes[i].GetMeta()
		if err != nil {
			return nil, errors.Wrap(err)
		}
		n, err := fn(nodes[i], meta)
		if err != nil {
			return nil, errors.Wrap(err)
		}
		if n != nil {
			returnNodes = append(returnNodes, n)
		}
	}
	return returnNodes, nil
}

// SortNodes sorts nodes in place:
// - by PathAnnotation annotation
// - by IndexAnnotation annotation
func SortNodes(nodes []*yaml.RNode) error {
	var err error
	// use stable sort to keep ordering of equal elements
	sort.SliceStable(nodes, func(i, j int) bool {
		if err != nil {
			return false
		}
		var iMeta, jMeta yaml.ResourceMeta
		if iMeta, _ = nodes[i].GetMeta(); err != nil {
			return false
		}
		if jMeta, _ = nodes[j].GetMeta(); err != nil {
			return false
		}

		iValue := iMeta.Annotations[PathAnnotation]
		jValue := jMeta.Annotations[PathAnnotation]
		if iValue != jValue {
			return iValue < jValue
		}

		iValue = iMeta.Annotations[IndexAnnotation]
		jValue = jMeta.Annotations[IndexAnnotation]

		// put resource config without an index first
		if iValue == jValue {
			return false
		}
		if iValue == "" {
			return true
		}
		if jValue == "" {
			return false
		}

		// sort by index
		var iIndex, jIndex int
		iIndex, err = strconv.Atoi(iValue)
		if err != nil {
			err = fmt.Errorf("unable to parse config.kubernetes.io/index %s :%v", iValue, err)
			return false
		}
		jIndex, err = strconv.Atoi(jValue)
		if err != nil {
			err = fmt.Errorf("unable to parse config.kubernetes.io/index %s :%v", jValue, err)
			return false
		}
		if iIndex != jIndex {
			return iIndex < jIndex
		}

		// elements are equal
		return false
	})
	return errors.Wrap(err)
}
