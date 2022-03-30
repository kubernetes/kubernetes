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
	// internalPrefix is the prefix given to internal annotations that are used
	// internally by the orchestrator
	internalPrefix string = "internal.config.kubernetes.io/"

	// IndexAnnotation records the index of a specific resource in a file or input stream.
	IndexAnnotation AnnotationKey = internalPrefix + "index"

	// PathAnnotation records the path to the file the Resource was read from
	PathAnnotation AnnotationKey = internalPrefix + "path"

	// SeqIndentAnnotation records the sequence nodes indentation of the input resource
	SeqIndentAnnotation AnnotationKey = internalPrefix + "seqindent"

	// IdAnnotation records the id of the resource to map inputs to outputs
	IdAnnotation AnnotationKey = internalPrefix + "id"

	// Deprecated: Use IndexAnnotation instead.
	LegacyIndexAnnotation AnnotationKey = "config.kubernetes.io/index"

	// Deprecated: use PathAnnotation instead.
	LegacyPathAnnotation AnnotationKey = "config.kubernetes.io/path"

	// Deprecated: use IdAnnotation instead.
	LegacyIdAnnotation = "config.k8s.io/id"

	// InternalAnnotationsMigrationResourceIDAnnotation is used to uniquely identify
	// resources during round trip to and from a function execution. We will use it
	// to track the internal annotations and reconcile them if needed.
	InternalAnnotationsMigrationResourceIDAnnotation = internalPrefix + "annotations-migration-resource-id"
)

func GetFileAnnotations(rn *yaml.RNode) (string, string, error) {
	rm, _ := rn.GetMeta()
	annotations := rm.Annotations
	path, found := annotations[PathAnnotation]
	if !found {
		path = annotations[LegacyPathAnnotation]
	}
	index, found := annotations[IndexAnnotation]
	if !found {
		index = annotations[LegacyIndexAnnotation]
	}
	return path, index, nil
}

func GetIdAnnotation(rn *yaml.RNode) string {
	rm, _ := rn.GetMeta()
	annotations := rm.Annotations
	id, found := annotations[IdAnnotation]
	if !found {
		id = annotations[LegacyIdAnnotation]
	}
	return id
}

func CopyLegacyAnnotations(rn *yaml.RNode) error {
	meta, err := rn.GetMeta()
	if err != nil {
		if err == yaml.ErrMissingMetadata {
			// resource has no metadata, this should be a no-op
			return nil
		}
		return err
	}
	if err := copyAnnotations(meta, rn, LegacyPathAnnotation, PathAnnotation); err != nil {
		return err
	}
	if err := copyAnnotations(meta, rn, LegacyIndexAnnotation, IndexAnnotation); err != nil {
		return err
	}
	if err := copyAnnotations(meta, rn, LegacyIdAnnotation, IdAnnotation); err != nil {
		return err
	}
	return nil
}

func copyAnnotations(meta yaml.ResourceMeta, rn *yaml.RNode, legacyKey string, newKey string) error {
	newValue := meta.Annotations[newKey]
	legacyValue := meta.Annotations[legacyKey]
	if newValue != "" {
		if legacyValue == "" {
			if err := rn.PipeE(yaml.SetAnnotation(legacyKey, newValue)); err != nil {
				return err
			}
		}
	} else {
		if legacyValue != "" {
			if err := rn.PipeE(yaml.SetAnnotation(newKey, legacyValue)); err != nil {
				return err
			}
		}
	}
	return nil
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
		if err := CopyLegacyAnnotations(nodes[i]); err != nil {
			return err
		}
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
		if err := nodes[i].PipeE(yaml.SetAnnotation(LegacyPathAnnotation, path)); err != nil {
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
		if err := nodes[i].PipeE(
			yaml.SetAnnotation(LegacyIndexAnnotation, fmt.Sprintf("%d", c))); err != nil {
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
		if err := CopyLegacyAnnotations(nodes[i]); err != nil {
			return err
		}
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
		if err := nodes[i].PipeE(yaml.SetAnnotation(LegacyPathAnnotation, path)); err != nil {
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
		if err := CopyLegacyAnnotations(nodes[i]); err != nil {
			return false
		}
		if err := CopyLegacyAnnotations(nodes[j]); err != nil {
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

// CopyInternalAnnotations copies the annotations that begin with the prefix
// `internal.config.kubernetes.io` from the source RNode to the destination RNode.
// It takes a parameter exclusions, which is a list of annotation keys to ignore.
func CopyInternalAnnotations(src *yaml.RNode, dst *yaml.RNode, exclusions ...AnnotationKey) error {
	srcAnnotations := GetInternalAnnotations(src)
	for k, v := range srcAnnotations {
		if stringSliceContains(exclusions, k) {
			continue
		}
		if err := dst.PipeE(yaml.SetAnnotation(k, v)); err != nil {
			return err
		}
	}
	return nil
}

// ConfirmInternalAnnotationUnchanged compares the annotations of the RNodes that begin with the prefix
// `internal.config.kubernetes.io`, throwing an error if they differ. It takes a parameter exclusions,
// which is a list of annotation keys to ignore.
func ConfirmInternalAnnotationUnchanged(r1 *yaml.RNode, r2 *yaml.RNode, exclusions ...AnnotationKey) error {
	r1Annotations := GetInternalAnnotations(r1)
	r2Annotations := GetInternalAnnotations(r2)

	// this is a map to prevent duplicates
	diffAnnos := make(map[string]bool)

	for k, v1 := range r1Annotations {
		if stringSliceContains(exclusions, k) {
			continue
		}
		if v2, ok := r2Annotations[k]; !ok || v1 != v2 {
			diffAnnos[k] = true
		}
	}

	for k, v2 := range r2Annotations {
		if stringSliceContains(exclusions, k) {
			continue
		}
		if v1, ok := r1Annotations[k]; !ok || v2 != v1 {
			diffAnnos[k] = true
		}
	}

	if len(diffAnnos) > 0 {
		keys := make([]string, 0, len(diffAnnos))
		for k := range diffAnnos {
			keys = append(keys, k)
		}
		sort.Strings(keys)

		errorString := "internal annotations differ: "
		for _, key := range keys {
			errorString = errorString + key + ", "
		}
		return errors.Errorf(errorString[0 : len(errorString)-2])
	}

	return nil
}

// GetInternalAnnotations returns a map of all the annotations of the provided
// RNode that satisfies one of the following: 1) begin with the prefix
// `internal.config.kubernetes.io` 2) is one of `config.kubernetes.io/path`,
// `config.kubernetes.io/index` and `config.k8s.io/id`.
func GetInternalAnnotations(rn *yaml.RNode) map[string]string {
	meta, _ := rn.GetMeta()
	annotations := meta.Annotations
	result := make(map[string]string)
	for k, v := range annotations {
		if strings.HasPrefix(k, internalPrefix) || k == LegacyPathAnnotation || k == LegacyIndexAnnotation || k == LegacyIdAnnotation {
			result[k] = v
		}
	}
	return result
}

// stringSliceContains returns true if the slice has the string.
func stringSliceContains(slice []string, str string) bool {
	for _, s := range slice {
		if s == str {
			return true
		}
	}
	return false
}
