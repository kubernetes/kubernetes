package filesystem

import (
	"os"

	"github.com/go-git/go-git/v5/config"
	"github.com/go-git/go-git/v5/storage/filesystem/dotgit"
	"github.com/go-git/go-git/v5/utils/ioutil"
)

type ConfigStorage struct {
	dir *dotgit.DotGit
}

func (c *ConfigStorage) Config() (conf *config.Config, err error) {
	f, err := c.dir.Config()
	if err != nil {
		if os.IsNotExist(err) {
			return config.NewConfig(), nil
		}

		return nil, err
	}

	defer ioutil.CheckClose(f, &err)
	return config.ReadConfig(f)
}

func (c *ConfigStorage) SetConfig(cfg *config.Config) (err error) {
	if err = cfg.Validate(); err != nil {
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
