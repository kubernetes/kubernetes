package cni

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"sort"
)

func ConfFromFile(filename string) (*NetworkConfig, error) {
	bytes, err := ioutil.ReadFile(filename)
	if err != nil {
		return nil, fmt.Errorf("error reading %s: %s", filename, err)
	}
	conf := &NetworkConfig{Bytes: bytes}
	if err = json.Unmarshal(bytes, conf); err != nil {
		return nil, fmt.Errorf("error parsing %s: %s", filename, err)
	}
	return conf, nil
}

func ConfFiles(dir string) ([]string, error) {
	// In part, adapted from rkt/networking/podenv.go#listFiles
	files, err := ioutil.ReadDir(dir)
	switch {
	case err == nil: // break
	case os.IsNotExist(err):
		return nil, nil
	default:
		return nil, err
	}

	confFiles := []string{}
	for _, f := range files {
		if f.IsDir() {
			continue
		}
		if filepath.Ext(f.Name()) == ".conf" {
			confFiles = append(confFiles, filepath.Join(dir, f.Name()))
		}
	}
	return confFiles, nil
}

func LoadConf(dir, name string) (*NetworkConfig, error) {
	files, err := ConfFiles(dir)
	switch {
	case err != nil:
		return nil, err
	case files == nil || len(files) == 0:
		return nil, fmt.Errorf("No net configurations found")
	}
	sort.Strings(files)

	for _, confFile := range files {
		conf, err := ConfFromFile(confFile)
		if err != nil {
			return nil, err
		}
		if conf.Name == name {
			return conf, nil
		}
	}
	return nil, fmt.Errorf(`no net configuration with name "%s" in %s`, name, dir)
}
