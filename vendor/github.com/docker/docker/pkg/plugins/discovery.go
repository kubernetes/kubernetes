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
	"sync"
)

var (
	// ErrNotFound plugin not found
	ErrNotFound = errors.New("plugin not found")
	socketsPath = "/run/docker/plugins"
)

// localRegistry defines a registry that is local (using unix socket).
type localRegistry struct{}

func newLocalRegistry() localRegistry {
	return localRegistry{}
}

// Scan scans all the plugin paths and returns all the names it found
func Scan() ([]string, error) {
	var names []string
	if err := filepath.Walk(socketsPath, func(path string, fi os.FileInfo, err error) error {
		if err != nil {
			return nil
		}

		if fi.Mode()&os.ModeSocket != 0 {
			name := strings.TrimSuffix(fi.Name(), filepath.Ext(fi.Name()))
			names = append(names, name)
		}
		return nil
	}); err != nil {
		return nil, err
	}

	for _, path := range specsPaths {
		if err := filepath.Walk(path, func(p string, fi os.FileInfo, err error) error {
			if err != nil || fi.IsDir() {
				return nil
			}
			name := strings.TrimSuffix(fi.Name(), filepath.Ext(fi.Name()))
			names = append(names, name)
			return nil
		}); err != nil {
			return nil, err
		}
	}
	return names, nil
}

// Plugin returns the plugin registered with the given name (or returns an error).
func (l *localRegistry) Plugin(name string) (*Plugin, error) {
	socketpaths := pluginPaths(socketsPath, name, ".sock")

	for _, p := range socketpaths {
		if fi, err := os.Stat(p); err == nil && fi.Mode()&os.ModeSocket != 0 {
			return NewLocalPlugin(name, "unix://"+p), nil
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

	return NewLocalPlugin(name, addr), nil
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
	p.name = name
	if p.TLSConfig != nil && len(p.TLSConfig.CAFile) == 0 {
		p.TLSConfig.InsecureSkipVerify = true
	}
	p.activateWait = sync.NewCond(&sync.Mutex{})

	return &p, nil
}

func pluginPaths(base, name, ext string) []string {
	return []string{
		filepath.Join(base, name+ext),
		filepath.Join(base, name, name+ext),
	}
}
