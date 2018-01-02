package config

import (
	"fmt"
	"strings"
)

// Section is the representation of a section inside git configuration files.
// Each Section contains Options that are used by both the Git plumbing
// and the porcelains.
// Sections can be further divided into subsections. To begin a subsection
// put its name in double quotes, separated by space from the section name,
// in the section header, like in the example below:
//
//     [section "subsection"]
//
// All the other lines (and the remainder of the line after the section header)
// are recognized as option variables, in the form "name = value" (or just name,
// which is a short-hand to say that the variable is the boolean "true").
// The variable names are case-insensitive, allow only alphanumeric characters
// and -, and must start with an alphabetic character:
//
//     [section "subsection1"]
//         option1 = value1
//         option2
//     [section "subsection2"]
//         option3 = value2
//
type Section struct {
	Name        string
	Options     Options
	Subsections Subsections
}

type Subsection struct {
	Name    string
	Options Options
}

type Sections []*Section

func (s Sections) GoString() string {
	var strs []string
	for _, ss := range s {
		strs = append(strs, fmt.Sprintf("%#v", ss))
	}

	return strings.Join(strs, ", ")
}

type Subsections []*Subsection

func (s Subsections) GoString() string {
	var strs []string
	for _, ss := range s {
		strs = append(strs, fmt.Sprintf("%#v", ss))
	}

	return strings.Join(strs, ", ")
}

// IsName checks if the name provided is equals to the Section name, case insensitive.
func (s *Section) IsName(name string) bool {
	return strings.ToLower(s.Name) == strings.ToLower(name)
}

// Option return the value for the specified key. Empty string is returned if
// key does not exists.
func (s *Section) Option(key string) string {
	return s.Options.Get(key)
}

// AddOption adds a new Option to the Section. The updated Section is returned.
func (s *Section) AddOption(key string, value string) *Section {
	s.Options = s.Options.withAddedOption(key, value)
	return s
}

// SetOption adds a new Option to the Section. If the option already exists, is replaced.
// The updated Section is returned.
func (s *Section) SetOption(key string, value string) *Section {
	s.Options = s.Options.withSettedOption(key, value)
	return s
}

// Remove an option with the specified key. The updated Section is returned.
func (s *Section) RemoveOption(key string) *Section {
	s.Options = s.Options.withoutOption(key)
	return s
}

// Subsection returns a Subsection from the specified Section. If the
// Subsection does not exists, new one is created and added to Section.
func (s *Section) Subsection(name string) *Subsection {
	for i := len(s.Subsections) - 1; i >= 0; i-- {
		ss := s.Subsections[i]
		if ss.IsName(name) {
			return ss
		}
	}

	ss := &Subsection{Name: name}
	s.Subsections = append(s.Subsections, ss)
	return ss
}

// HasSubsection checks if the Section has a Subsection with the specified name.
func (s *Section) HasSubsection(name string) bool {
	for _, ss := range s.Subsections {
		if ss.IsName(name) {
			return true
		}
	}

	return false
}

// IsName checks if the name of the subsection is exactly the specified name.
func (s *Subsection) IsName(name string) bool {
	return s.Name == name
}

// Option returns an option with the specified key. If the option does not exists,
// empty spring will be returned.
func (s *Subsection) Option(key string) string {
	return s.Options.Get(key)
}

// AddOption adds a new Option to the Subsection. The updated Subsection is returned.
func (s *Subsection) AddOption(key string, value string) *Subsection {
	s.Options = s.Options.withAddedOption(key, value)
	return s
}

// SetOption adds a new Option to the Subsection. If the option already exists, is replaced.
// The updated Subsection is returned.
func (s *Subsection) SetOption(key string, value ...string) *Subsection {
	s.Options = s.Options.withSettedOption(key, value...)
	return s
}

// RemoveOption removes the option with the specified key. The updated Subsection is returned.
func (s *Subsection) RemoveOption(key string) *Subsection {
	s.Options = s.Options.withoutOption(key)
	return s
}
