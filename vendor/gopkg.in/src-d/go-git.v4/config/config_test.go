package config

import . "gopkg.in/check.v1"

type ConfigSuite struct{}

var _ = Suite(&ConfigSuite{})

func (s *ConfigSuite) TestUnmarshall(c *C) {
	input := []byte(`[core]
        bare = true
		worktree = foo
[remote "origin"]
        url = git@github.com:mcuadros/go-git.git
        fetch = +refs/heads/*:refs/remotes/origin/*
[remote "alt"]
		url = git@github.com:mcuadros/go-git.git
		url = git@github.com:src-d/go-git.git
		fetch = +refs/heads/*:refs/remotes/origin/*
		fetch = +refs/pull/*:refs/remotes/origin/pull/*
[submodule "qux"]
        path = qux
        url = https://github.com/foo/qux.git
		branch = bar
[branch "master"]
        remote = origin
        merge = refs/heads/master
`)

	cfg := NewConfig()
	err := cfg.Unmarshal(input)
	c.Assert(err, IsNil)

	c.Assert(cfg.Core.IsBare, Equals, true)
	c.Assert(cfg.Core.Worktree, Equals, "foo")
	c.Assert(cfg.Remotes, HasLen, 2)
	c.Assert(cfg.Remotes["origin"].Name, Equals, "origin")
	c.Assert(cfg.Remotes["origin"].URLs, DeepEquals, []string{"git@github.com:mcuadros/go-git.git"})
	c.Assert(cfg.Remotes["origin"].Fetch, DeepEquals, []RefSpec{"+refs/heads/*:refs/remotes/origin/*"})
	c.Assert(cfg.Remotes["alt"].Name, Equals, "alt")
	c.Assert(cfg.Remotes["alt"].URLs, DeepEquals, []string{"git@github.com:mcuadros/go-git.git", "git@github.com:src-d/go-git.git"})
	c.Assert(cfg.Remotes["alt"].Fetch, DeepEquals, []RefSpec{"+refs/heads/*:refs/remotes/origin/*", "+refs/pull/*:refs/remotes/origin/pull/*"})
	c.Assert(cfg.Submodules, HasLen, 1)
	c.Assert(cfg.Submodules["qux"].Name, Equals, "qux")
	c.Assert(cfg.Submodules["qux"].URL, Equals, "https://github.com/foo/qux.git")
	c.Assert(cfg.Submodules["qux"].Branch, Equals, "bar")

}

func (s *ConfigSuite) TestMarshall(c *C) {
	output := []byte(`[core]
	bare = true
	worktree = bar
[remote "alt"]
	url = git@github.com:mcuadros/go-git.git
	url = git@github.com:src-d/go-git.git
	fetch = +refs/heads/*:refs/remotes/origin/*
	fetch = +refs/pull/*:refs/remotes/origin/pull/*
[remote "origin"]
	url = git@github.com:mcuadros/go-git.git
[submodule "qux"]
	url = https://github.com/foo/qux.git
`)

	cfg := NewConfig()
	cfg.Core.IsBare = true
	cfg.Core.Worktree = "bar"
	cfg.Remotes["origin"] = &RemoteConfig{
		Name: "origin",
		URLs: []string{"git@github.com:mcuadros/go-git.git"},
	}

	cfg.Remotes["alt"] = &RemoteConfig{
		Name:  "alt",
		URLs:  []string{"git@github.com:mcuadros/go-git.git", "git@github.com:src-d/go-git.git"},
		Fetch: []RefSpec{"+refs/heads/*:refs/remotes/origin/*", "+refs/pull/*:refs/remotes/origin/pull/*"},
	}

	cfg.Submodules["qux"] = &Submodule{
		Name: "qux",
		URL:  "https://github.com/foo/qux.git",
	}

	b, err := cfg.Marshal()
	c.Assert(err, IsNil)

	c.Assert(string(b), Equals, string(output))
}

func (s *ConfigSuite) TestUnmarshallMarshall(c *C) {
	input := []byte(`[core]
	bare = true
	worktree = foo
	custom = ignored
[remote "origin"]
	url = git@github.com:mcuadros/go-git.git
	fetch = +refs/heads/*:refs/remotes/origin/*
	mirror = true
[branch "master"]
	remote = origin
	merge = refs/heads/master
`)

	cfg := NewConfig()
	err := cfg.Unmarshal(input)
	c.Assert(err, IsNil)

	output, err := cfg.Marshal()
	c.Assert(err, IsNil)
	c.Assert(string(output), DeepEquals, string(input))
}

func (s *ConfigSuite) TestValidateInvalidRemote(c *C) {
	config := &Config{
		Remotes: map[string]*RemoteConfig{
			"foo": {Name: "foo"},
		},
	}

	c.Assert(config.Validate(), Equals, ErrRemoteConfigEmptyURL)
}

func (s *ConfigSuite) TestValidateInvalidKey(c *C) {
	config := &Config{
		Remotes: map[string]*RemoteConfig{
			"bar": {Name: "foo"},
		},
	}

	c.Assert(config.Validate(), Equals, ErrInvalid)
}

func (s *ConfigSuite) TestRemoteConfigValidateMissingURL(c *C) {
	config := &RemoteConfig{Name: "foo"}
	c.Assert(config.Validate(), Equals, ErrRemoteConfigEmptyURL)
}

func (s *ConfigSuite) TestRemoteConfigValidateMissingName(c *C) {
	config := &RemoteConfig{}
	c.Assert(config.Validate(), Equals, ErrRemoteConfigEmptyName)
}

func (s *ConfigSuite) TestRemoteConfigValidateDefault(c *C) {
	config := &RemoteConfig{Name: "foo", URLs: []string{"http://foo/bar"}}
	c.Assert(config.Validate(), IsNil)

	fetch := config.Fetch
	c.Assert(fetch, HasLen, 1)
	c.Assert(fetch[0].String(), Equals, "+refs/heads/*:refs/remotes/foo/*")
}
