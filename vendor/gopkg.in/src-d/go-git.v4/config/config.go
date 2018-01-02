// Package config contains the abstraction of multiple config files
package config

import (
	"bytes"
	"errors"
	"fmt"
	"sort"

	format "gopkg.in/src-d/go-git.v4/plumbing/format/config"
)

const (
	// DefaultFetchRefSpec is the default refspec used for fetch.
	DefaultFetchRefSpec = "+refs/heads/*:refs/remotes/%s/*"
	// DefaultPushRefSpec is the default refspec used for push.
	DefaultPushRefSpec = "refs/heads/*:refs/heads/*"
)

// ConfigStorer generic storage of Config object
type ConfigStorer interface {
	Config() (*Config, error)
	SetConfig(*Config) error
}

var (
	ErrInvalid               = errors.New("config invalid remote")
	ErrRemoteConfigNotFound  = errors.New("remote config not found")
	ErrRemoteConfigEmptyURL  = errors.New("remote config: empty URL")
	ErrRemoteConfigEmptyName = errors.New("remote config: empty name")
)

// Config contains the repository configuration
// ftp://www.kernel.org/pub/software/scm/git/docs/git-config.html#FILES
type Config struct {
	Core struct {
		// IsBare if true this repository is assumed to be bare and has no
		// working directory associated with it.
		IsBare bool
		// Worktree is the path to the root of the working tree.
		Worktree string
	}
	// Remotes list of repository remotes, the key of the map is the name
	// of the remote, should equal to RemoteConfig.Name.
	Remotes map[string]*RemoteConfig
	// Submodules list of repository submodules, the key of the map is the name
	// of the submodule, should equal to Submodule.Name.
	Submodules map[string]*Submodule

	// Raw contains the raw information of a config file. The main goal is
	// preserve the parsed information from the original format, to avoid
	// dropping unsupported fields.
	Raw *format.Config
}

// NewConfig returns a new empty Config.
func NewConfig() *Config {
	return &Config{
		Remotes:    make(map[string]*RemoteConfig, 0),
		Submodules: make(map[string]*Submodule, 0),
		Raw:        format.New(),
	}
}

// Validate validates the fields and sets the default values.
func (c *Config) Validate() error {
	for name, r := range c.Remotes {
		if r.Name != name {
			return ErrInvalid
		}

		if err := r.Validate(); err != nil {
			return err
		}
	}

	return nil
}

const (
	remoteSection    = "remote"
	submoduleSection = "submodule"
	coreSection      = "core"
	fetchKey         = "fetch"
	urlKey           = "url"
	bareKey          = "bare"
	worktreeKey      = "worktree"
)

// Unmarshal parses a git-config file and stores it.
func (c *Config) Unmarshal(b []byte) error {
	r := bytes.NewBuffer(b)
	d := format.NewDecoder(r)

	c.Raw = format.New()
	if err := d.Decode(c.Raw); err != nil {
		return err
	}

	c.unmarshalCore()
	c.unmarshalSubmodules()
	return c.unmarshalRemotes()
}

func (c *Config) unmarshalCore() {
	s := c.Raw.Section(coreSection)
	if s.Options.Get(bareKey) == "true" {
		c.Core.IsBare = true
	}

	c.Core.Worktree = s.Options.Get(worktreeKey)
}

func (c *Config) unmarshalRemotes() error {
	s := c.Raw.Section(remoteSection)
	for _, sub := range s.Subsections {
		r := &RemoteConfig{}
		if err := r.unmarshal(sub); err != nil {
			return err
		}

		c.Remotes[r.Name] = r
	}

	return nil
}

func (c *Config) unmarshalSubmodules() {
	s := c.Raw.Section(submoduleSection)
	for _, sub := range s.Subsections {
		m := &Submodule{}
		m.unmarshal(sub)

		c.Submodules[m.Name] = m
	}
}

// Marshal returns Config encoded as a git-config file.
func (c *Config) Marshal() ([]byte, error) {
	c.marshalCore()
	c.marshalRemotes()
	c.marshalSubmodules()

	buf := bytes.NewBuffer(nil)
	if err := format.NewEncoder(buf).Encode(c.Raw); err != nil {
		return nil, err
	}

	return buf.Bytes(), nil
}

func (c *Config) marshalCore() {
	s := c.Raw.Section(coreSection)
	s.SetOption(bareKey, fmt.Sprintf("%t", c.Core.IsBare))

	if c.Core.Worktree != "" {
		s.SetOption(worktreeKey, c.Core.Worktree)
	}
}

func (c *Config) marshalRemotes() {
	s := c.Raw.Section(remoteSection)
	newSubsections := make(format.Subsections, 0, len(c.Remotes))
	added := make(map[string]bool)
	for _, subsection := range s.Subsections {
		if remote, ok := c.Remotes[subsection.Name]; ok {
			newSubsections = append(newSubsections, remote.marshal())
			added[subsection.Name] = true
		}
	}

	remoteNames := make([]string, 0, len(c.Remotes))
	for name := range c.Remotes {
		remoteNames = append(remoteNames, name)
	}

	sort.Strings(remoteNames)

	for _, name := range remoteNames {
		if !added[name] {
			newSubsections = append(newSubsections, c.Remotes[name].marshal())
		}
	}

	s.Subsections = newSubsections
}

func (c *Config) marshalSubmodules() {
	s := c.Raw.Section(submoduleSection)
	s.Subsections = make(format.Subsections, len(c.Submodules))

	var i int
	for _, r := range c.Submodules {
		section := r.marshal()
		// the submodule section at config is a subset of the .gitmodule file
		// we should remove the non-valid options for the config file.
		section.RemoveOption(pathKey)
		s.Subsections[i] = section
		i++
	}
}

// RemoteConfig contains the configuration for a given remote repository.
type RemoteConfig struct {
	// Name of the remote
	Name string
	// URLs the URLs of a remote repository. It must be non-empty. Fetch will
	// always use the first URL, while push will use all of them.
	URLs []string
	// Fetch the default set of "refspec" for fetch operation
	Fetch []RefSpec

	// raw representation of the subsection, filled by marshal or unmarshal are
	// called
	raw *format.Subsection
}

// Validate validates the fields and sets the default values.
func (c *RemoteConfig) Validate() error {
	if c.Name == "" {
		return ErrRemoteConfigEmptyName
	}

	if len(c.URLs) == 0 {
		return ErrRemoteConfigEmptyURL
	}

	for _, r := range c.Fetch {
		if err := r.Validate(); err != nil {
			return err
		}
	}

	if len(c.Fetch) == 0 {
		c.Fetch = []RefSpec{RefSpec(fmt.Sprintf(DefaultFetchRefSpec, c.Name))}
	}

	return nil
}

func (c *RemoteConfig) unmarshal(s *format.Subsection) error {
	c.raw = s

	fetch := []RefSpec{}
	for _, f := range c.raw.Options.GetAll(fetchKey) {
		rs := RefSpec(f)
		if err := rs.Validate(); err != nil {
			return err
		}

		fetch = append(fetch, rs)
	}

	var urls []string
	for _, f := range c.raw.Options.GetAll(urlKey) {
		urls = append(urls, f)
	}

	c.Name = c.raw.Name
	c.URLs = urls
	c.Fetch = fetch

	return nil
}

func (c *RemoteConfig) marshal() *format.Subsection {
	if c.raw == nil {
		c.raw = &format.Subsection{}
	}

	c.raw.Name = c.Name
	if len(c.URLs) == 0 {
		c.raw.RemoveOption(urlKey)
	} else {
		c.raw.SetOption(urlKey, c.URLs...)
	}

	if len(c.Fetch) == 0 {
		c.raw.RemoveOption(fetchKey)
	} else {
		var values []string
		for _, rs := range c.Fetch {
			values = append(values, rs.String())
		}

		c.raw.SetOption(fetchKey, values...)
	}

	return c.raw
}
