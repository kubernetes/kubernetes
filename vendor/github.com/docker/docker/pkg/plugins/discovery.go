package plugins

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"net/url"
	"os"
	"path/filepath"
	"strings"
)

var (
	ErrNotFound = errors.New("Plugin not found")
	socketsPath = "/run/docker/plugins"
	specsPaths  = []string{"/etc/docker/plugins", "/usr/lib/docker/plugins"}
)

type Registry interface {
	Plugins() ([]*Plugin, error)
	Plugin(name string) (*Plugin, error)
}

type LocalRegistry struct{}

func newLocalRegistry() LocalRegistry {
	return LocalRegistry{}
}

func (l *LocalRegistry) Plugin(name string) (*Plugin, error) {
	socketpaths := pluginPaths(socketsPath, name, ".sock")

	for _, p := range socketpaths {
		if fi, err := os.Stat(p); err == nil && fi.Mode()&os.ModeSocket != 0 {
			return newLocalPlugin(name, "unix://"+p), nil
		}
	}

	var txtspecpaths []string
	for _, p := range specsPaths {
		txtspecpaths = append(txtspecpaths, pluginPaths(p, name, ".spec")...)
		txtspecpaths = append(txtspecpaths, pluginPaths(p, name, ".json")...)
	}

	for _, p := range txtspecpaths {
		if _, err := os.Stat(p); err == nil {
			if strings.HasSuffix(p, ".json") {
				return readPluginJSONInfo(name, p)
			}
			return readPluginInfo(name, p)
		}
	}
	return nil, ErrNotFound
}

func readPluginInfo(name, path string) (*Plugin, error) {
	content, err := ioutil.ReadFile(path)
	if err != nil {
		return nil, err
	}
	addr := strings.TrimSpace(string(content))

	u, err := url.Parse(addr)
	if err != nil {
		return nil, err
	}

	if len(u.Scheme) == 0 {
		return nil, fmt.Errorf("Unknown protocol")
	}

	return newLocalPlugin(name, addr), nil
}

func readPluginJSONInfo(name, path string) (*Plugin, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var p Plugin
	if err := json.NewDecoder(f).Decode(&p); err != nil {
		return nil, err
	}
	p.Name = name
	if len(p.TLSConfig.CAFile) == 0 {
		p.TLSConfig.InsecureSkipVerify = true
	}

	return &p, nil
}

func pluginPaths(base, name, ext string) []string {
	return []string{
		filepath.Join(base, name+ext),
		filepath.Join(base, name, name+ext),
	}
}
