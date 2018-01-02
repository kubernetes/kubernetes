package filesystem

import (
	stdioutil "io/ioutil"
	"os"

	"gopkg.in/src-d/go-git.v4/config"
	"gopkg.in/src-d/go-git.v4/storage/filesystem/internal/dotgit"
	"gopkg.in/src-d/go-git.v4/utils/ioutil"
)

type ConfigStorage struct {
	dir *dotgit.DotGit
}

func (c *ConfigStorage) Config() (*config.Config, error) {
	cfg := config.NewConfig()

	f, err := c.dir.Config()
	if err != nil {
		if os.IsNotExist(err) {
			return cfg, nil
		}

		return nil, err
	}

	defer ioutil.CheckClose(f, &err)

	b, err := stdioutil.ReadAll(f)
	if err != nil {
		return nil, err
	}

	if err := cfg.Unmarshal(b); err != nil {
		return nil, err
	}

	return cfg, err
}

func (c *ConfigStorage) SetConfig(cfg *config.Config) error {
	if err := cfg.Validate(); err != nil {
		return err
	}

	f, err := c.dir.ConfigWriter()
	if err != nil {
		return err
	}

	defer ioutil.CheckClose(f, &err)

	b, err := cfg.Marshal()
	if err != nil {
		return err
	}

	_, err = f.Write(b)
	return err
}
