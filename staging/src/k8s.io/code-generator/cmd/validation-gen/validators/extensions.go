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

package validators

import (
	"fmt"
	"os"
	"regexp"
	"strings"

	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/gengo/v2/types"
	"sigs.k8s.io/yaml"
)

// Extensions is the set of validation definitions a project adds to the
// built-in ones, read from YAML at generation time. It is a struct rather than
// a bare list so that later sections, such as named validation rules, can be
// added without changing files people have already written.
type Extensions struct {
	// Formats are the values this project adds to the "format" tag.
	Formats []FormatExtension `json:"formats"`
}

// FormatExtension declares one project-defined value of the "format" tag.
type FormatExtension struct {
	// Name is how the format is written in the tag, as "format=<name>".
	// Lower-case alphanumerics, with dashes between words. Required.
	Name string `json:"name"`

	// Docs describes the format for "validation-gen --docs". Required.
	Docs string `json:"docs"`

	// Pattern is the regular expression a value must match. It is not
	// implicitly anchored: write ^ and $ if that is what is meant. Required.
	Pattern string `json:"pattern"`

	// Message describes what a valid value looks like. It is the whole detail
	// of the error, which does not carry the pattern. Required.
	Message string `json:"message"`
}

var formatNamePattern = regexp.MustCompile(`^[a-z0-9]+(-[a-z0-9]+)*$`)

// LoadExtensions reads, merges, and validates the extensions files at paths.
// Each file is one YAML document. Empty paths returns nil.
func LoadExtensions(paths []string) (*Extensions, error) {
	if len(paths) == 0 {
		return nil, nil
	}

	combined := &Extensions{}
	seen := map[string]string{} // format name -> the file that declared it

	for _, path := range paths {
		data, err := os.ReadFile(path)
		if err != nil {
			return nil, fmt.Errorf("reading extensions file %q: %w", path, err)
		}
		var ext Extensions
		if err := yaml.UnmarshalStrict(data, &ext); err != nil {
			return nil, fmt.Errorf("parsing extensions file %q: %w", path, err)
		}
		if err := ext.validate(path, seen); err != nil {
			return nil, fmt.Errorf("extensions file %q: %w", path, err)
		}
		combined.Formats = append(combined.Formats, ext.Formats...)
	}
	return combined, nil
}

// validate checks one file's formats. seen maps each name already declared to
// the file that declared it, so that two files cannot claim the same name.
func (e *Extensions) validate(path string, seen map[string]string) error {
	for i := range e.Formats {
		f := &e.Formats[i]
		if f.Name == "" {
			return fmt.Errorf("formats[%d]: name is required", i)
		}
		if prev, dup := seen[f.Name]; dup {
			return fmt.Errorf("formats[%d]: format %q was already defined in %q", i, f.Name, prev)
		}
		if err := f.validate(); err != nil {
			return fmt.Errorf("formats[%d] (%q): %w", i, f.Name, err)
		}
		seen[f.Name] = path
	}
	return nil
}

func (f *FormatExtension) validate() error {
	if !formatNamePattern.MatchString(f.Name) {
		return fmt.Errorf("name must be lower-case alphanumerics with dashes between words, matching %q", formatNamePattern)
	}
	// Before the "k8s-" rule, which would otherwise subsume it.
	if isBuiltInFormat(f.Name) {
		return fmt.Errorf("%q is a built-in format and cannot be redefined", f.Name)
	}
	if f.Name == "k8s" || strings.HasPrefix(f.Name, "k8s-") {
		return fmt.Errorf("name %q cannot use the reserved \"k8s-\" prefix", f.Name)
	}
	if f.Docs == "" {
		return fmt.Errorf("docs is required")
	}
	if f.Pattern == "" {
		return fmt.Errorf("pattern is required")
	}
	if _, err := regexp.Compile(f.Pattern); err != nil {
		return fmt.Errorf("pattern %q does not compile: %w", f.Pattern, err)
	}
	if f.Message == "" {
		return fmt.Errorf("message is required, so that errors explain the rule instead of printing the pattern")
	}
	return nil
}

// formats indexes the extended formats by name. Safe on a nil receiver.
func (e *Extensions) formats() map[string]FormatExtension {
	if e == nil {
		return nil
	}
	byName := make(map[string]FormatExtension, len(e.Formats))
	for _, f := range e.Formats {
		byName[f.Name] = f
	}
	return byName
}

var (
	regexpMustCompile = types.Name{Package: "regexp", Name: "MustCompile"}
	matchesValidator  = types.Name{Package: libValidationPkg, Name: "Matches"}
)

func (f FormatExtension) validations() Validations {
	var result Validations

	// Dashes become underscores rather than word breaks, so that two names
	// cannot collide onto one variable and share a pattern.
	reVar := PrivateVar{
		Name:    "FormatPattern_" + strings.ReplaceAll(f.Name, "-", "_"),
		Package: "local",
	}
	result.AddVariable(Variable(reVar, Function(formatTagName, DefaultFlags, regexpMustCompile, f.Pattern)))

	result.AddFunction(Function(formatTagName, DefaultFlags, matchesValidator,
		reVar, f.Message, f.origin()).
		WithEmits(Emission{Type: field.ErrorTypeInvalid, Origin: f.origin()}))

	return result
}

func (f *FormatExtension) origin() string {
	return formatTagName + "=" + f.Name
}
