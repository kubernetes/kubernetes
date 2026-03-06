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

package sharding

import (
	"fmt"
	"strings"
)

// Parse parses a shard selector string into a Selector.
// The format is:
//
//	selector    = requirement ( "," requirement )*
//	requirement = "shardRange" "(" fieldPath "," hexStart "," hexEnd ")"
//	fieldPath   = "object.metadata." ( "uid" | "namespace" | "name" )
//	hexStart    = [0-9a-f]{0,16}
//	hexEnd      = [0-9a-f]{0,16}
//
// An empty string returns Everything().
func Parse(s string) (Selector, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return Everything(), nil
	}

	p := &parser{input: s}
	reqs, err := p.parseSelector()
	if err != nil {
		return nil, err
	}
	return NewSelector(reqs...), nil
}

type parser struct {
	input string
	pos   int
}

func (p *parser) parseSelector() ([]ShardRangeRequirement, error) {
	var reqs []ShardRangeRequirement

	req, err := p.parseRequirement()
	if err != nil {
		return nil, err
	}
	reqs = append(reqs, req)

	for p.pos < len(p.input) {
		if p.input[p.pos] != ',' {
			return nil, fmt.Errorf("expected ',' at position %d, got %q", p.pos, string(p.input[p.pos]))
		}
		p.pos++ // skip ','

		req, err := p.parseRequirement()
		if err != nil {
			return nil, err
		}
		reqs = append(reqs, req)
	}

	return reqs, nil
}

func (p *parser) parseRequirement() (ShardRangeRequirement, error) {
	// Expect "shardRange("
	if !p.consumePrefix("shardRange(") {
		return ShardRangeRequirement{}, fmt.Errorf("expected 'shardRange(' at position %d", p.pos)
	}

	// Parse fieldPath
	fieldPath, err := p.parseFieldPath()
	if err != nil {
		return ShardRangeRequirement{}, err
	}

	// Expect ","
	if !p.consumePrefix(",") {
		return ShardRangeRequirement{}, fmt.Errorf("expected ',' after fieldPath at position %d", p.pos)
	}

	// Parse hexStart (may be empty)
	hexStart, err := p.parseHex()
	if err != nil {
		return ShardRangeRequirement{}, fmt.Errorf("invalid hexStart: %w", err)
	}

	// Expect ","
	if !p.consumePrefix(",") {
		return ShardRangeRequirement{}, fmt.Errorf("expected ',' after hexStart at position %d", p.pos)
	}

	// Parse hexEnd (may be empty)
	hexEnd, err := p.parseHex()
	if err != nil {
		return ShardRangeRequirement{}, fmt.Errorf("invalid hexEnd: %w", err)
	}

	// Expect ")"
	if !p.consumePrefix(")") {
		return ShardRangeRequirement{}, fmt.Errorf("expected ')' at position %d", p.pos)
	}

	return ShardRangeRequirement{
		Key:   fieldPath,
		Start: hexStart,
		End:   hexEnd,
	}, nil
}

func (p *parser) parseFieldPath() (string, error) {
	// Must start with "object.metadata."
	if !p.consumePrefix("object.metadata.") {
		return "", fmt.Errorf("expected 'object.metadata.' at position %d", p.pos)
	}

	// Read the field name
	for _, name := range []string{"uid", "namespace"} {
		if p.consumePrefix(name) {
			return "object.metadata." + name, nil
		}
	}

	return "", fmt.Errorf("unsupported metadata field at position %d, expected uid or namespace", p.pos)
}

func (p *parser) parseHex() (string, error) {
	start := p.pos
	for p.pos < len(p.input) && isHexChar(p.input[p.pos]) {
		p.pos++
	}
	hex := p.input[start:p.pos]
	if len(hex) > 16 {
		return "", fmt.Errorf("hex value too long (%d chars, max 16): %q", len(hex), hex)
	}
	return hex, nil
}

func (p *parser) consumePrefix(prefix string) bool {
	if strings.HasPrefix(p.input[p.pos:], prefix) {
		p.pos += len(prefix)
		return true
	}
	return false
}

func isHexChar(c byte) bool {
	return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')
}
