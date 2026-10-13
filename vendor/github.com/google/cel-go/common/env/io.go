// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package env

import (
	"errors"
	"fmt"

	"go.yaml.in/yaml/v3"
)

type internalTypeDesc struct {
	TypeName    string      `yaml:"type_name"`
	Params      []*TypeDesc `yaml:"params,omitempty"`
	IsTypeParam bool        `yaml:"is_type_param,omitempty"`
}

// Embedding TypeDesc in variable causes issues with customizing
// unmarshalling / marshalling. Work around with a parallel type.
type internalVariable struct {
	Name        string `yaml:"name"`
	Description string `yaml:"description,omitempty"`

	// Type represents the type declaration for the variable.
	Type *TypeDesc `yaml:"type,omitempty"`

	TypeName    string      `yaml:"type_name"`
	Params      []*TypeDesc `yaml:"params,omitempty"`
	IsTypeParam bool        `yaml:"is_type_param,omitempty"`
}

// UnmarshalYAML implements yaml.Unmarshal
func (v *Variable) UnmarshalYAML(n *yaml.Node) error {
	buf := internalVariable{}
	err := n.Decode(&buf)
	if err != nil {
		return err
	}
	v.Name = buf.Name
	v.Description = buf.Description
	if buf.TypeName != "" {
		v.TypeDesc = &TypeDesc{
			TypeName:    buf.TypeName,
			Params:      buf.Params,
			IsTypeParam: buf.IsTypeParam,
		}
	} else if buf.Type != nil {
		v.TypeDesc = buf.Type
	}
	return nil
}

// MarshalYAML implements yaml.Marshaler
func (v *Variable) MarshalYAML() (any, error) {
	// The presence of an unmarshaller alters the default marshaller behavior so
	// provide a simple marshal implementation.
	buf := internalVariable{
		Name:        v.Name,
		Description: v.Description,
	}
	if t := v.GetType(); t != nil {
		buf.TypeName = t.TypeName
		buf.Params = t.Params
		buf.IsTypeParam = t.IsTypeParam
	}
	return &buf, nil
}

// UnmarshalYAML implements yaml.Unmarshaler
func (td *TypeDesc) UnmarshalYAML(n *yaml.Node) error {
	if td == nil {
		return fmt.Errorf("unexpected Unmarshal for TypeDesc at: %d", n.Line)
	}
	if n.Kind == yaml.ScalarNode {
		o, err := ParseTypeDesc(n.Value)
		if err != nil {
			return err
		}
		*td = *o
		return nil
	}

	if n.Kind != yaml.MappingNode {
		return errors.New("unsupported yaml for TypeDesc")
	}

	buf := internalTypeDesc{}
	err := n.Decode(&buf)
	if err != nil {
		return err
	}
	td.TypeName = buf.TypeName
	td.Params = buf.Params
	td.IsTypeParam = buf.IsTypeParam
	return nil
}

type typeDescParser struct {
	text   string
	pos    int
	length int
}

// ParseTypeDesc parses a TypeDesc from the type specifier format: "map<string, int>"
func ParseTypeDesc(text string) (*TypeDesc, error) {
	p := &typeDescParser{text: text, length: len(text)}
	res, err := p.parseTypeElem()
	if err != nil {
		return nil, fmt.Errorf("failed to parse type %q: %v", text, err)
	}
	p.skipWhitespace()
	if p.pos < p.length {
		return nil, fmt.Errorf("unexpected character %q at position %d in %q", p.text[p.pos], p.pos, text)
	}
	return res, nil
}

func (p *typeDescParser) parseConcreteType() (*TypeDesc, error) {
	id, err := p.parseNamespaceIdentifier()
	if err != nil {
		return nil, err
	}
	if p.pos < p.length && p.text[p.pos] == '<' {
		p.pos++ // consume '<'
		var params []*TypeDesc
		for {
			p.skipWhitespace()
			param, err := p.parseTypeElem()
			if err != nil {
				return nil, err
			}
			params = append(params, param)
			p.skipWhitespace()
			if p.pos < p.length && p.text[p.pos] == ',' {
				p.pos++ // consume ','
				continue
			}
			if p.pos < p.length && p.text[p.pos] == '>' {
				p.pos++ // consume '>'
				break
			}
			return nil, fmt.Errorf("expected ',' or '>' at position %d", p.pos)
		}
		return NewTypeDesc(id, params...), nil
	}
	return NewTypeDesc(id), nil
}

func (p *typeDescParser) parseTypeElem() (*TypeDesc, error) {
	p.skipWhitespace()
	if p.pos < p.length && p.text[p.pos] == '~' {
		p.pos++ // consume '~'
		id, err := p.parseTypeParamIdent()
		if err != nil {
			return nil, err
		}
		return NewTypeParam(id), nil
	}
	return p.parseConcreteType()
}

func (p *typeDescParser) parseNamespaceIdentifier() (string, error) {
	p.skipWhitespace()
	var id string
	for p.pos < p.length && p.text[p.pos] != '<' {
		c := p.text[p.pos]
		if c == '.' {
			id += "."
			p.pos++ // consume '.'
		}
		ident, err := p.parseIdentifier()
		if err != nil {
			return "", err
		}
		id += ident
		p.skipWhitespace()
		if p.pos < p.length && p.text[p.pos] != '.' {
			break
		}
	}
	if id == "" {
		return "", fmt.Errorf("missing identifier at position %d", p.pos)
	}
	return id, nil
}

func (p *typeDescParser) parseIdentifier() (string, error) {
	p.skipWhitespace()
	if p.pos >= p.length {
		return "", fmt.Errorf("unexpected end of input")
	}
	start := p.pos
	c := p.text[p.pos]
	if !isAlpha(c) && c != '_' {
		return "", fmt.Errorf("identifier is expected, but %q was found at position %d", c, p.pos)
	}
	p.pos++
	for p.pos < p.length {
		c := p.text[p.pos]
		if !isAlphaNumeric(c) && c != '_' {
			break
		}
		p.pos++
	}
	return p.text[start:p.pos], nil
}

func (p *typeDescParser) parseTypeParamIdent() (string, error) {
	p.skipWhitespace()
	if p.pos >= p.length {
		return "", fmt.Errorf("unexpected end of input")
	}
	c := p.text[p.pos]
	if !isAlpha(c) {
		return "", fmt.Errorf("invalid type parameter identifier %q at position %d, must be a single character from A-Z", c, p.pos)
	}
	p.pos++
	if p.pos < p.length && isAlpha(p.text[p.pos]) {
		return "", fmt.Errorf("invalid type param, must have a single alphabetic character at position %d", p.pos)
	}
	return string(c), nil
}

func (p *typeDescParser) skipWhitespace() {
	for p.pos < p.length && p.text[p.pos] == ' ' {
		p.pos++
	}
}

func isAlpha(c byte) bool {
	return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')
}

func isAlphaNumeric(c byte) bool {
	return isAlpha(c) || (c >= '0' && c <= '9')
}

// ConfigFromYAML returns a config from YAML source.
//
// Adds custom parsing logic for normalizing shorthand for specifiying some fields
// in a YAML document (mainly the type-specifier shorthand).
//
// Using yaml.Unmarshal with any implementation should be sufficient for most
// cases.
func ConfigFromYAML(data []byte) (*Config, error) {
	c := &Config{}
	e := yaml.Unmarshal(data, c)
	if e != nil {
		return nil, e
	}
	return c, nil
}

// ConfigToYAML returns the config serialized to YAML
//
// Provided as a convenience wrapper around a tested YAML Marshaler.
func ConfigToYAML(c *Config) ([]byte, error) {
	return yaml.Marshal(c)
}
