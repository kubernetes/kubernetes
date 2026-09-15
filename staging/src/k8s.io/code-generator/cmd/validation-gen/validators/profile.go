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

// Profile is the set of validation definitions a project adds to the built-in
// ones, read from a YAML file at generation time. It is a struct rather than a
// bare list so that later sections, such as named validations, can be added
// without changing the shape of the profiles people have already written.
type Profile struct {
	// Formats are the values this project adds to the "format" tag.
	Formats []ProfileFormat `json:"formats"`
}

// ProfileFormat declares one value of the "format" tag. A regex is the only
// check today; a shared Go function or a CEL expression would be a sibling
// field, as would length bounds.
type ProfileFormat struct {
	// Name is how the format is written in the tag, as "format=<name>".
	// Lower-case alphanumerics, with dashes between words. Required.
	Name string `json:"name"`

	// Docs describes the format for "validation-gen --docs". Required.
	Docs string `json:"docs"`

	// Regex is the pattern a value must match. It is not implicitly anchored:
	// write ^ and $ if that is what is meant. Required.
	Regex string `json:"regex"`

	// Message describes what a valid value looks like. The error appends the
	// pattern, which on its own is not an explanation. Required.
	Message string `json:"message"`
}

// formatNamePattern is the naming convention the built-in formats follow. It
// also keeps formatGoIdent injective.
var formatNamePattern = regexp.MustCompile(`^[a-z0-9]+(-[a-z0-9]+)*$`)

// LoadProfile reads and validates the profile at path. Empty path returns nil.
func LoadProfile(path string) (*Profile, error) {
	if path == "" {
		return nil, nil
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("reading profile %q: %w", path, err)
	}
	profile := &Profile{}
	if err := yaml.UnmarshalStrict(data, profile); err != nil {
		return nil, fmt.Errorf("parsing profile %q: %w", path, err)
	}
	if err := profile.validate(); err != nil {
		return nil, fmt.Errorf("profile %q: %w", path, err)
	}
	return profile, nil
}

// validate checks the whole profile up front, so a mistake names the format
// rather than surfacing at the first use of the tag.
func (p *Profile) validate() error {
	seen := map[string]int{}
	for i := range p.Formats {
		f := &p.Formats[i]
		if f.Name == "" {
			return fmt.Errorf("formats[%d]: name is required", i)
		}
		if prev, dup := seen[f.Name]; dup {
			return fmt.Errorf("formats[%d]: format %q was already defined by formats[%d]", i, f.Name, prev)
		}
		seen[f.Name] = i
		if err := f.validate(); err != nil {
			return fmt.Errorf("formats[%d] (%q): %w", i, f.Name, err)
		}
	}
	return nil
}

func (f *ProfileFormat) validate() error {
	if !formatNamePattern.MatchString(f.Name) {
		return fmt.Errorf("name must be lower-case alphanumerics with dashes between words, matching %q", formatNamePattern)
	}
	// Redefining one would make the same tag mean different things per project.
	if isBuiltInFormat(f.Name) {
		return fmt.Errorf("%q is a built-in format and cannot be redefined", f.Name)
	}
	if f.Docs == "" {
		return fmt.Errorf("docs is required")
	}
	if f.Regex == "" {
		return fmt.Errorf("regex is required")
	}
	// So a bad pattern is a generation error, not a panic at package init.
	if _, err := regexp.Compile(f.Regex); err != nil {
		return fmt.Errorf("regex %q does not compile: %w", f.Regex, err)
	}
	if f.Message == "" {
		return fmt.Errorf("message is required, so that errors explain the rule instead of printing the pattern")
	}
	return nil
}

// formats indexes the profile's formats by name. Safe on a nil profile.
func (p *Profile) formats() map[string]ProfileFormat {
	if p == nil {
		return nil
	}
	byName := make(map[string]ProfileFormat, len(p.Formats))
	for _, f := range p.Formats {
		byName[f.Name] = f
	}
	return byName
}

// The symbols a profile's formats generate calls to.
var (
	regexpMustCompile = types.Name{Package: "regexp", Name: "MustCompile"}
	matchesValidator  = types.Name{Package: libValidationPkg, Name: "Matches"}
)

// validations is the code generated for one use of a profile-declared format.
func (f ProfileFormat) validations() Validations {
	var result Validations

	// Compile once at package init, not on every call; emitValidationVariables
	// collapses the repeat requests for this same variable.
	reVar := PrivateVar{Name: "FormatRegexp" + formatGoIdent(f.Name), Package: "local"}
	result.AddVariable(Variable(reVar, Function(formatTagName, DefaultFlags, regexpMustCompile, f.Regex)))

	// The generator supplies the origin, so it cannot drift from the emission
	// declared here. A regex can only produce Invalid.
	result.AddFunction(Function(formatTagName, DefaultFlags, matchesValidator,
		reVar, f.Message, f.origin()).
		WithEmits(Emission{Type: field.ErrorTypeInvalid, Origin: f.origin()}))

	return result
}

// origin is the value this format's errors carry.
func (f *ProfileFormat) origin() string {
	return formatTagName + "=" + f.Name
}

// formatGoIdent turns a format name into a Go identifier fragment. Dashes
// become underscores, not word breaks, to keep the mapping injective.
func formatGoIdent(name string) string {
	return strings.ToUpper(name[:1]) + strings.ReplaceAll(name[1:], "-", "_")
}
