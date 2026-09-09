// Copyright 2021 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package replacement

import (
	"encoding/json"
	"fmt"
	"strings"

	"sigs.k8s.io/kustomize/api/internal/utils"
	"sigs.k8s.io/kustomize/api/resource"
	"sigs.k8s.io/kustomize/api/types"
	"sigs.k8s.io/kustomize/kyaml/errors"
	kyaml_utils "sigs.k8s.io/kustomize/kyaml/utils"
	"sigs.k8s.io/kustomize/kyaml/yaml"
)

type Filter struct {
	Replacements []types.Replacement `json:"replacements,omitempty" yaml:"replacements,omitempty"`
}

// Filter replaces values of targets with values from sources
func (f Filter) Filter(nodes []*yaml.RNode) ([]*yaml.RNode, error) {
	for i, r := range f.Replacements {
		if (r.SourceValue == nil && r.Source == nil) || r.Targets == nil {
			return nil, fmt.Errorf("replacements must specify a source and at least one target")
		}
		value, err := getReplacement(nodes, &f.Replacements[i])
		if err != nil {
			return nil, err
		}
		nodes, err = applyReplacement(nodes, value, r.Targets)
		if err != nil {
			return nil, err
		}
	}
	return nodes, nil
}

func getReplacement(nodes []*yaml.RNode, r *types.Replacement) (*yaml.RNode, error) {
	if r.SourceValue != nil && r.Source != nil {
		return nil, fmt.Errorf("value and resource selectors are mutually exclusive")
	}
	if r.SourceValue != nil {
		return yaml.NewScalarRNode(*r.SourceValue), nil
	}

	source, err := selectSourceNode(nodes, r.Source)
	if err != nil {
		return nil, err
	}

	if r.Source.FieldPath == "" {
		r.Source.FieldPath = types.DefaultReplacementFieldPath
	}
	fieldPath := kyaml_utils.SmarterPathSplitter(r.Source.FieldPath, ".")

	rn, err := source.Pipe(yaml.Lookup(fieldPath...))
	if err != nil {
		return nil, fmt.Errorf("error looking up replacement source: %w", err)
	}
	if rn.IsNilOrEmpty() {
		return nil, fmt.Errorf("fieldPath `%s` is missing for replacement source %s", r.Source.FieldPath, r.Source.ResId)
	}

	return getRefinedValue(r.Source.Options, rn)
}

// selectSourceNode finds the node that matches the selector, returning
// an error if multiple or none are found
func selectSourceNode(nodes []*yaml.RNode, selector *types.SourceSelector) (*yaml.RNode, error) {
	var matches []*yaml.RNode
	for _, n := range nodes {
		ids, err := utils.MakeResIds(n)
		if err != nil {
			return nil, fmt.Errorf("error getting node IDs: %w", err)
		}
		for _, id := range ids {
			if id.IsSelectedBy(selector.ResId) {
				if len(matches) > 0 {
					return nil, fmt.Errorf(
						"multiple matches for selector %s", selector)
				}
				matches = append(matches, n)
				break
			}
		}
	}
	if len(matches) == 0 {
		return nil, fmt.Errorf("nothing selected by %s", selector)
	}
	return matches[0], nil
}

func getRefinedValue(options *types.FieldOptions, rn *yaml.RNode) (*yaml.RNode, error) {
	if options == nil || options.Delimiter == "" {
		return rn, nil
	}
	if rn.YNode().Kind != yaml.ScalarNode {
		return nil, fmt.Errorf("delimiter option can only be used with scalar nodes")
	}
	value := strings.Split(yaml.GetValue(rn), options.Delimiter)
	if options.Index >= len(value) || options.Index < 0 {
		return nil, fmt.Errorf("options.index %d is out of bounds for value %s", options.Index, yaml.GetValue(rn))
	}
	n := rn.Copy()
	n.YNode().Value = value[options.Index]
	return n, nil
}

func applyReplacement(nodes []*yaml.RNode, value *yaml.RNode, targetSelectors []*types.TargetSelector) ([]*yaml.RNode, error) {
	for _, selector := range targetSelectors {
		if selector.Select == nil {
			return nil, errors.Errorf("target must specify resources to select")
		}
		if len(selector.FieldPaths) == 0 {
			selector.FieldPaths = []string{types.DefaultReplacementFieldPath}
		}
		tsr, err := types.NewTargetSelectorRegex(selector)
		if err != nil {
			return nil, fmt.Errorf("error creating target selector: %w", err)
		}
		for _, possibleTarget := range nodes {
			ids, err := utils.MakeResIds(possibleTarget)
			if err != nil {
				return nil, err
			}

			// filter targets by label and annotation selectors
			selectByAnnoAndLabel, err := selectByAnnoAndLabel(possibleTarget, selector)
			if err != nil {
				return nil, err
			}
			if !selectByAnnoAndLabel {
				continue
			}

			if tsr.RejectsAny(ids) {
				continue
			}

			// filter targets by matching resource IDs
			for _, id := range ids {
				if tsr.Selects(id) {
					err := copyValueToTarget(possibleTarget, value, selector)
					if err != nil {
						return nil, err
					}
					break
				}
			}
		}
	}
	return nodes, nil
}

func selectByAnnoAndLabel(n *yaml.RNode, t *types.TargetSelector) (bool, error) {
	if matchesSelect, err := matchesAnnoAndLabelSelector(n, t.Select); !matchesSelect || err != nil {
		return false, err
	}
	for _, reject := range t.Reject {
		if reject.AnnotationSelector == "" && reject.LabelSelector == "" {
			continue
		}
		if m, err := matchesAnnoAndLabelSelector(n, reject); m || err != nil {
			return false, err
		}
	}
	return true, nil
}

func matchesAnnoAndLabelSelector(n *yaml.RNode, selector *types.Selector) (bool, error) {
	r := resource.Resource{RNode: *n}
	annoMatch, err := r.MatchesAnnotationSelector(selector.AnnotationSelector)
	if err != nil {
		return false, err
	}
	labelMatch, err := r.MatchesLabelSelector(selector.LabelSelector)
	if err != nil {
		return false, err
	}
	return annoMatch && labelMatch, nil
}

func copyValueToTarget(target *yaml.RNode, value *yaml.RNode, selector *types.TargetSelector) error {
	for _, fp := range selector.FieldPaths {
		createKind := yaml.Kind(0) // do not create
		if selector.Options != nil && selector.Options.Create {
			createKind = value.YNode().Kind
		}

		// Check if this fieldPath contains structured data access
		if err := setValueInStructuredData(target, value, fp, createKind); err == nil {
			// Successfully handled as structured data
			continue
		}

		// Fall back to normal path handling
		targetFieldList, err := target.Pipe(&yaml.PathMatcher{
			Path:   kyaml_utils.SmarterPathSplitter(fp, "."),
			Create: createKind})
		if err != nil {
			return errors.WrapPrefixf(err, "%s", fieldRetrievalError(fp, createKind != 0))
		}
		targetFields, err := targetFieldList.Elements()
		if err != nil {
			return errors.WrapPrefixf(err, "%s", fieldRetrievalError(fp, createKind != 0))
		}
		if len(targetFields) == 0 {
			return errors.Errorf("%s", fieldRetrievalError(fp, createKind != 0))
		}

		for _, t := range targetFields {
			if err := setFieldValue(selector.Options, t, value); err != nil {
				return fmt.Errorf("%w", err)
			}
		}
	}
	return nil
}

func fieldRetrievalError(fieldPath string, isCreate bool) string {
	if isCreate {
		return fmt.Sprintf("unable to find or create field %q in replacement target", fieldPath)
	}
	return fmt.Sprintf("unable to find field %q in replacement target", fieldPath)
}

func setFieldValue(options *types.FieldOptions, targetField *yaml.RNode, value *yaml.RNode) error {
	value = value.Copy()
	if options != nil && options.Delimiter != "" {
		if targetField.YNode().Kind != yaml.ScalarNode {
			return fmt.Errorf("delimiter option can only be used with scalar nodes")
		}
		tv := strings.Split(targetField.YNode().Value, options.Delimiter)
		v := yaml.GetValue(value)
		// TODO: Add a way to remove an element
		switch {
		case options.Index < 0: // prefix
			tv = append([]string{v}, tv...)
		case options.Index >= len(tv): // suffix
			tv = append(tv, v)
		default: // replace an element
			tv[options.Index] = v
		}
		value.YNode().Value = strings.Join(tv, options.Delimiter)
	}

	if targetField.YNode().Kind == yaml.ScalarNode {
		// For scalar, only copy the value (leave any type intact to auto-convert int->string or string->int)
		targetField.YNode().Value = value.YNode().Value
	} else {
		targetField.SetYNode(value.YNode())
	}

	return nil
}

// setValueInStructuredData handles setting values within structured data (JSON/YAML) in scalar fields
func setValueInStructuredData(target *yaml.RNode, value *yaml.RNode, fieldPath string, createKind yaml.Kind) error {
	pathParts := kyaml_utils.SmarterPathSplitter(fieldPath, ".")
	if len(pathParts) < 2 {
		return fmt.Errorf("not a structured data path")
	}

	// Find the potential scalar field that might contain structured data
	var scalarFieldPath []string
	var structuredDataPath []string
	var foundScalar = false

	// Try to find where the scalar field ends and structured data begins
	for i := 1; i <= len(pathParts); i++ {
		potentialScalarPath := pathParts[:i]
		scalarField, err := target.Pipe(yaml.Lookup(potentialScalarPath...))
		if err != nil {
			continue
		}
		if scalarField != nil && scalarField.YNode().Kind == yaml.ScalarNode && i < len(pathParts) {
			// Try to parse the scalar value as structured data
			scalarValue := scalarField.YNode().Value
			var parsedNode yaml.Node
			if err := yaml.Unmarshal([]byte(scalarValue), &parsedNode); err == nil {
				// Successfully parsed - this is structured data
				scalarFieldPath = potentialScalarPath
				structuredDataPath = pathParts[i:]
				foundScalar = true
				break
			}
		}
	}

	if !foundScalar {
		return fmt.Errorf("no structured data found in path")
	}

	// Get the scalar field containing structured data
	scalarField, err := target.Pipe(yaml.Lookup(scalarFieldPath...))
	if err != nil {
		return fmt.Errorf("%w", err)
	}

	// Parse the structured data
	scalarValue := scalarField.YNode().Value
	var parsedNode yaml.Node
	if err := yaml.Unmarshal([]byte(scalarValue), &parsedNode); err != nil {
		return fmt.Errorf("%w", err)
	}

	structuredData := yaml.NewRNode(&parsedNode)

	// Navigate to the target location within the structured data
	targetInStructured, err := structuredData.Pipe(&yaml.PathMatcher{
		Path:   structuredDataPath,
		Create: createKind,
	})
	if err != nil {
		return fmt.Errorf("%w", err)
	}

	targetFields, err := targetInStructured.Elements()
	if err != nil {
		return fmt.Errorf("%w", err)
	}

	if len(targetFields) == 0 {
		return fmt.Errorf("unable to find field in structured data")
	}

	// Set the value in the structured data
	for _, t := range targetFields {
		if t.YNode().Kind == yaml.ScalarNode {
			t.YNode().Value = value.YNode().Value
		} else {
			t.SetYNode(value.YNode())
		}
	}

	// Serialize the modified structured data back to the scalar field
	// Try to detect if original was JSON or YAML and preserve formatting
	serializedData, err := serializeStructuredData(structuredData, scalarValue)
	if err != nil {
		return fmt.Errorf("%w", err)
	}

	// Update the original scalar field
	scalarField.YNode().Value = serializedData

	return nil
}

// serializeStructuredData handles the serialization of structured data back to string format
// preserving the original format (JSON vs YAML) and style (pretty vs compact)
func serializeStructuredData(structuredData *yaml.RNode, originalValue string) (string, error) {
	firstChar := rune(strings.TrimSpace(originalValue)[0])
	if firstChar == '{' || firstChar == '[' {
		return serializeAsJSON(structuredData, originalValue)
	}

	// Fallback to YAML format
	return serializeAsYAML(structuredData)
}

// serializeAsJSON converts structured data back to JSON format
func serializeAsJSON(structuredData *yaml.RNode, originalValue string) (string, error) {
	modifiedData, err := structuredData.String()
	if err != nil {
		return "", fmt.Errorf("failed to serialize structured data: %w", err)
	}

	// Parse the YAML output as JSON
	var jsonData interface{}
	if err := yaml.Unmarshal([]byte(modifiedData), &jsonData); err != nil {
		return "", fmt.Errorf("failed to unmarshal YAML data: %w", err)
	}

	// Check if original was pretty-printed by looking for newlines and indentation
	if strings.Contains(originalValue, "\n") && strings.Contains(originalValue, "  ") {
		// Pretty-print the JSON to match original formatting
		if prettyJSON, err := json.MarshalIndent(jsonData, "", "  "); err == nil {
			return string(prettyJSON), nil
		}
	}

	// Compact JSON
	if compactJSON, err := json.Marshal(jsonData); err == nil {
		return string(compactJSON), nil
	}

	return "", fmt.Errorf("failed to marshal JSON data")
}

// serializeAsYAML converts structured data back to YAML format
func serializeAsYAML(structuredData *yaml.RNode) (string, error) {
	modifiedData, err := structuredData.String()
	if err != nil {
		return "", fmt.Errorf("failed to serialize YAML data: %w", err)
	}

	return strings.TrimSpace(modifiedData), nil
}
