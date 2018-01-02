package config

import (
	"bytes"
	"errors"

	format "gopkg.in/src-d/go-git.v4/plumbing/format/config"
)

var (
	ErrModuleEmptyURL  = errors.New("module config: empty URL")
	ErrModuleEmptyPath = errors.New("module config: empty path")
)

// Modules defines the submodules properties, represents a .gitmodules file
// https://www.kernel.org/pub/software/scm/git/docs/gitmodules.html
type Modules struct {
	// Submodules is a map of submodules being the key the name of the submodule.
	Submodules map[string]*Submodule

	raw *format.Config
}

// NewModules returns a new empty Modules
func NewModules() *Modules {
	return &Modules{
		Submodules: make(map[string]*Submodule, 0),
		raw:        format.New(),
	}
}

const (
	pathKey   = "path"
	branchKey = "branch"
)

// Unmarshal parses a git-config file and stores it.
func (m *Modules) Unmarshal(b []byte) error {
	r := bytes.NewBuffer(b)
	d := format.NewDecoder(r)

	m.raw = format.New()
	if err := d.Decode(m.raw); err != nil {
		return err
	}

	s := m.raw.Section(submoduleSection)
	for _, sub := range s.Subsections {
		mod := &Submodule{}
		mod.unmarshal(sub)

		m.Submodules[mod.Path] = mod
	}

	return nil
}

// Marshal returns Modules encoded as a git-config file.
func (m *Modules) Marshal() ([]byte, error) {
	s := m.raw.Section(submoduleSection)
	s.Subsections = make(format.Subsections, len(m.Submodules))

	var i int
	for _, r := range m.Submodules {
		s.Subsections[i] = r.marshal()
		i++
	}

	buf := bytes.NewBuffer(nil)
	if err := format.NewEncoder(buf).Encode(m.raw); err != nil {
		return nil, err
	}

	return buf.Bytes(), nil
}

// Submodule defines a submodule.
type Submodule struct {
	// Name module name
	Name string
	// Path defines the path, relative to the top-level directory of the Git
	// working tree.
	Path string
	// URL defines a URL from which the submodule repository can be cloned.
	URL string
	// Branch is a remote branch name for tracking updates in the upstream
	// submodule. Optional value.
	Branch string

	// raw representation of the subsection, filled by marshal or unmarshal are
	// called.
	raw *format.Subsection
}

// Validate validates the fields and sets the default values.
func (m *Submodule) Validate() error {
	if m.Path == "" {
		return ErrModuleEmptyPath
	}

	if m.URL == "" {
		return ErrModuleEmptyURL
	}

	return nil
}

func (m *Submodule) unmarshal(s *format.Subsection) {
	m.raw = s

	m.Name = m.raw.Name
	m.Path = m.raw.Option(pathKey)
	m.URL = m.raw.Option(urlKey)
	m.Branch = m.raw.Option(branchKey)
}

func (m *Submodule) marshal() *format.Subsection {
	if m.raw == nil {
		m.raw = &format.Subsection{}
	}

	m.raw.Name = m.Name
	if m.raw.Name == "" {
		m.raw.Name = m.Path
	}

	m.raw.SetOption(pathKey, m.Path)
	m.raw.SetOption(urlKey, m.URL)

	if m.Branch != "" {
		m.raw.SetOption(branchKey, m.Branch)
	}

	return m.raw
}
