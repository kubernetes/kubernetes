// SPDX-FileCopyrightText: Copyright 2015-2025 go-swagger maintainers
// SPDX-License-Identifier: Apache-2.0

package yamlutils

import (
	json "encoding/json"
	"fmt"
	"strconv"

	"github.com/go-openapi/swag/jsonutils"
	yaml "go.yaml.in/yaml/v3"
)

// defaultMaxNestingDepth caps the recursion depth of the YAML<->JSON transforms to
// guard against stack-overflow on deeply nested (possibly adversarial) input.
//
// It matches the limit enforced by go.yaml.in/yaml/v3's own parser and by
// encoding/json's decoder.
const defaultMaxNestingDepth = 10000

// Bounds on YAML anchor/alias expansion.
//
// go.yaml.in/yaml/v3 enforces these when decoding into Go values, but that guard is
// coupled to the library's own tree walk: when we decode into a low-level [yaml.Node]
// (to preserve key order) and expand aliases ourselves in [yamlWalker.node], we bypass
// it. We therefore reproduce it here, with the same constants and ratio schedule as the
// library's decoder (see go.yaml.in/yaml/v3 decode.go, "excessive aliasing").
const (
	aliasCountThreshold  = 100
	decodeCountThreshold = 1000

	// 400,000 decode operations is ~500kb of dense object declarations, or
	// ~5kb of dense object declarations with 10000% alias expansion.
	aliasRatioRangeLow = 400000
	// 4,000,000 decode operations is ~5MB of dense object declarations.
	aliasRatioRangeHigh = 4000000
	aliasRatioRange     = float64(aliasRatioRangeHigh - aliasRatioRangeLow)

	// tolerated share of alias-driven decodes: from aliasRatioSmall (small/medium documents)
	// down to aliasRatioLarge (very large ones), interpolated with slope aliasRatioSlope.
	aliasRatioSmall = 0.99
	aliasRatioLarge = 0.10
	aliasRatioSlope = aliasRatioSmall - aliasRatioLarge
)

var (
	// errMaxNestingDepth is returned when a document nests deeper than [defaultMaxNestingDepth].
	errMaxNestingDepth = fmt.Errorf("maximum nesting depth of %d exceeded: %w", defaultMaxNestingDepth, ErrYAML)

	// errExcessiveAliasing is returned when anchor/alias expansion is disproportionate to the
	// size of the document, i.e. an "alias bomb".
	errExcessiveAliasing = fmt.Errorf("document contains excessive aliasing: %w", ErrYAML)
)

// allowedAliasRatio scales the tolerated share of alias-driven decode operations from 99%
// for small-to-medium documents down to 10% for very large ones, mirroring go.yaml.in/yaml/v3.
func allowedAliasRatio(decodeCount int) float64 {
	switch {
	case decodeCount <= aliasRatioRangeLow:
		return aliasRatioSmall
	case decodeCount >= aliasRatioRangeHigh:
		return aliasRatioLarge
	default:
		return aliasRatioSmall - aliasRatioSlope*(float64(decodeCount-aliasRatioRangeLow)/aliasRatioRange)
	}
}

// yamlWalker carries the state needed to bound a single YAML-tree traversal:
// anchor/alias expansion accounting and cycle detection.
//
// A fresh walker is created per top-level conversion; it is threaded (not copied) through
// the whole recursive walk so its counters accumulate across the entire document.
type yamlWalker struct {
	decodeCount int
	aliasCount  int
	aliasDepth  int
	aliases     map[*yaml.Node]bool // anchors currently being expanded, for cycle detection
}

func newYAMLWalker() *yamlWalker {
	return &yamlWalker{aliases: make(map[*yaml.Node]bool)}
}

// account records one processed node and fails if alias expansion has become excessive.
func (w *yamlWalker) account() error {
	w.decodeCount++
	if w.aliasDepth > 0 {
		w.aliasCount++
	}

	if w.aliasCount > aliasCountThreshold &&
		w.decodeCount > decodeCountThreshold &&
		float64(w.aliasCount)/float64(w.decodeCount) > allowedAliasRatio(w.decodeCount) {
		return errExcessiveAliasing
	}

	return nil
}

// YAMLToJSON converts a YAML document into JSON bytes.
//
// Note: a YAML document is the output from a [yaml.Marshaler], e.g a pointer to a [yaml.Node].
//
// [YAMLToJSON] is typically called after [BytesToYAMLDoc].
func YAMLToJSON(value any) (json.RawMessage, error) {
	jm, err := transformData(value, 0)
	if err != nil {
		return nil, err
	}

	b, err := jsonutils.WriteJSON(jm)

	return json.RawMessage(b), err
}

// BytesToYAMLDoc converts a byte slice into a YAML document.
//
// This function only supports root documents that are objects.
//
// A YAML document is a pointer to a [yaml.Node].
func BytesToYAMLDoc(data []byte) (any, error) {
	var document yaml.Node // preserve order that is present in the document
	if err := yaml.Unmarshal(data, &document); err != nil {
		return nil, err
	}
	if document.Kind != yaml.DocumentNode || len(document.Content) != 1 || document.Content[0].Kind != yaml.MappingNode {
		return nil, fmt.Errorf("only YAML documents that are objects are supported: %w", ErrYAML)
	}
	return &document, nil
}

func (w *yamlWalker) node(root *yaml.Node, depth int) (any, error) {
	if depth > defaultMaxNestingDepth {
		return nil, errMaxNestingDepth
	}
	if err := w.account(); err != nil {
		return nil, err
	}

	switch root.Kind {
	case yaml.DocumentNode:
		return w.document(root, depth)
	case yaml.SequenceNode:
		return w.sequence(root, depth)
	case yaml.MappingNode:
		return w.mapping(root, depth)
	case yaml.ScalarNode:
		return yamlScalar(root)
	case yaml.AliasNode:
		return w.alias(root, depth)
	default:
		return nil, fmt.Errorf("unsupported YAML node type: %v: %w", root.Kind, ErrYAML)
	}
}

// alias resolves an anchor reference, expanding the anchored subtree. It detects cycles
// (an anchor whose expansion transitively references itself) and accounts the expansion
// against the alias-bomb budget via [yamlWalker.aliasDepth].
func (w *yamlWalker) alias(node *yaml.Node, depth int) (any, error) {
	if node.Alias == nil {
		return nil, fmt.Errorf("invalid YAML alias node %q: %w", node.Value, ErrYAML)
	}
	if w.aliases[node.Alias] {
		return nil, fmt.Errorf("anchor %q contains itself: %w", node.Value, ErrYAML)
	}

	w.aliases[node.Alias] = true
	w.aliasDepth++
	out, err := w.node(node.Alias, depth+1)
	w.aliasDepth--
	delete(w.aliases, node.Alias)

	return out, err
}

func (w *yamlWalker) document(node *yaml.Node, depth int) (any, error) {
	if len(node.Content) != 1 {
		return nil, fmt.Errorf("unexpected YAML Document node content length: %d: %w", len(node.Content), ErrYAML)
	}
	return w.node(node.Content[0], depth+1)
}

func (w *yamlWalker) mapping(node *yaml.Node, depth int) (any, error) {
	const sensibleAllocDivider = 2 // nodes concatenate (key,value) sequences
	m := make(YAMLMapSlice, len(node.Content)/sensibleAllocDivider)

	if err := m.unmarshalYAML(w, node, depth); err != nil {
		return nil, err
	}

	return m, nil
}

func (w *yamlWalker) sequence(node *yaml.Node, depth int) (any, error) {
	s := make([]any, 0)

	for i := range len(node.Content) {
		v, err := w.node(node.Content[i], depth+1)
		if err != nil {
			return nil, fmt.Errorf("unable to decode YAML sequence value: %w: %w", err, ErrYAML)
		}
		s = append(s, v)
	}
	return s, nil
}

const ( // See https://yaml.org/type/
	yamlStringScalar = "tag:yaml.org,2002:str"
	yamlIntScalar    = "tag:yaml.org,2002:int"
	yamlBoolScalar   = "tag:yaml.org,2002:bool"
	yamlFloatScalar  = "tag:yaml.org,2002:float"
	yamlTimestamp    = "tag:yaml.org,2002:timestamp"
	yamlNull         = "tag:yaml.org,2002:null"
)

func yamlScalar(node *yaml.Node) (any, error) {
	switch node.LongTag() {
	case yamlStringScalar:
		return node.Value, nil
	case yamlBoolScalar:
		b, err := strconv.ParseBool(node.Value)
		if err != nil {
			return nil, fmt.Errorf("unable to process scalar node. Got %q. Expecting bool content: %w: %w", node.Value, err, ErrYAML)
		}
		return b, nil
	case yamlIntScalar:
		i, err := strconv.ParseInt(node.Value, 10, 64)
		if err != nil {
			return nil, fmt.Errorf("unable to process scalar node. Got %q. Expecting integer content: %w: %w", node.Value, err, ErrYAML)
		}
		return i, nil
	case yamlFloatScalar:
		f, err := strconv.ParseFloat(node.Value, 64)
		if err != nil {
			return nil, fmt.Errorf("unable to process scalar node. Got %q. Expecting float content: %w: %w", node.Value, err, ErrYAML)
		}
		return f, nil
	case yamlTimestamp:
		// YAML timestamp is marshaled as string, not time
		return node.Value, nil
	case yamlNull:
		return nil, nil //nolint:nilnil
	default:
		return nil, fmt.Errorf("YAML tag %q is not supported: %w", node.LongTag(), ErrYAML)
	}
}

func yamlStringScalarC(node *yaml.Node) (string, error) {
	if node.Kind != yaml.ScalarNode {
		return "", fmt.Errorf("expecting a string scalar but got %q: %w", node.Kind, ErrYAML)
	}
	switch node.LongTag() {
	case yamlStringScalar, yamlIntScalar, yamlFloatScalar:
		return node.Value, nil
	default:
		return "", fmt.Errorf("YAML tag %q is not supported as map key: %w", node.LongTag(), ErrYAML)
	}
}

func format(t any) (string, error) {
	switch k := t.(type) {
	case string:
		return k, nil
	case uint:
		return strconv.FormatUint(uint64(k), 10), nil
	case uint8:
		return strconv.FormatUint(uint64(k), 10), nil
	case uint16:
		return strconv.FormatUint(uint64(k), 10), nil
	case uint32:
		return strconv.FormatUint(uint64(k), 10), nil
	case uint64:
		return strconv.FormatUint(k, 10), nil
	case int:
		return strconv.Itoa(k), nil
	case int8:
		return strconv.FormatInt(int64(k), 10), nil
	case int16:
		return strconv.FormatInt(int64(k), 10), nil
	case int32:
		return strconv.FormatInt(int64(k), 10), nil
	case int64:
		return strconv.FormatInt(k, 10), nil
	default:
		return "", fmt.Errorf("unexpected map key type, got: %T: %w", k, ErrYAML)
	}
}

func transformData(input any, depth int) (out any, err error) {
	if depth > defaultMaxNestingDepth {
		return nil, errMaxNestingDepth
	}

	switch in := input.(type) {
	case yaml.Node:
		return newYAMLWalker().node(&in, depth)
	case *yaml.Node:
		return newYAMLWalker().node(in, depth)
	case map[any]any:
		o := make(YAMLMapSlice, 0, len(in))
		for ke, va := range in {
			var nmi YAMLMapItem
			if nmi.Key, err = format(ke); err != nil {
				return nil, err
			}

			v, ert := transformData(va, depth+1)
			if ert != nil {
				return nil, ert
			}
			nmi.Value = v
			o = append(o, nmi)
		}
		return o, nil
	case []any:
		len1 := len(in)
		o := make([]any, len1)
		for i := range len1 {
			o[i], err = transformData(in[i], depth+1)
			if err != nil {
				return nil, err
			}
		}
		return o, nil
	}
	return input, nil
}
