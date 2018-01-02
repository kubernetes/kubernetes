// Copyright 2015 The rkt Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package config

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/coreos/rkt/common"
	"github.com/hashicorp/errwrap"
)

// Headerer is an interface for getting additional HTTP headers to use
// when downloading data (images, signatures).
type Headerer interface {
	GetHeader() http.Header
	SignRequest(r *http.Request) *http.Request
}

// BasicCredentials holds typical credentials used for authentication
// (user and password). Used for fetching docker images.
type BasicCredentials struct {
	User     string `json:"user"`
	Password string `json:"password"`
}

// ConfigurablePaths holds various paths defined in the configuration.
type ConfigurablePaths struct {
	DataDir         string
	Stage1ImagesDir string
}

// Stage1 holds name, version and location of a default stage1 image
// if it was specified in configuration.
type Stage1Data struct {
	Name     string
	Version  string
	Location string
}

// Config is a single place where configuration for rkt frontend needs
// resides.
type Config struct {
	AuthPerHost                  map[string]Headerer
	DockerCredentialsPerRegistry map[string]BasicCredentials
	Paths                        ConfigurablePaths
	Stage1                       Stage1Data
}

// MarshalJSON marshals the config for user output.
func (c *Config) MarshalJSON() ([]byte, error) {
	stage0 := []interface{}{}

	for host, auth := range c.AuthPerHost {
		var typ string
		var credentials interface{}

		switch h := auth.(type) {
		case *basicAuthHeaderer:
			typ = "basic"
			credentials = h.auth
		case *oAuthBearerTokenHeaderer:
			typ = "oauth"
			credentials = h.auth
		default:
			return nil, errors.New("unknown headerer type")
		}

		auth := struct {
			RktVersion  string      `json:"rktVersion"`
			RktKind     string      `json:"rktKind"`
			Domains     []string    `json:"domains"`
			Type        string      `json:"type"`
			Credentials interface{} `json:"credentials"`
		}{
			RktVersion:  "v1",
			RktKind:     "auth",
			Domains:     []string{host},
			Type:        typ,
			Credentials: credentials,
		}

		stage0 = append(stage0, auth)
	}

	for registry, credentials := range c.DockerCredentialsPerRegistry {
		dockerAuth := struct {
			RktVersion  string           `json:"rktVersion"`
			RktKind     string           `json:"rktKind"`
			Registries  []string         `json:"registries"`
			Credentials BasicCredentials `json:"credentials"`
		}{
			RktVersion:  "v1",
			RktKind:     "dockerAuth",
			Registries:  []string{registry},
			Credentials: credentials,
		}

		stage0 = append(stage0, dockerAuth)
	}

	paths := struct {
		RktVersion   string `json:"rktVersion"`
		RktKind      string `json:"rktKind"`
		Data         string `json:"data"`
		Stage1Images string `json:"stage1-images"`
	}{
		RktVersion:   "v1",
		RktKind:      "paths",
		Data:         c.Paths.DataDir,
		Stage1Images: c.Paths.Stage1ImagesDir,
	}

	stage1 := struct {
		RktVersion string `json:"rktVersion"`
		RktKind    string `json:"rktKind"`
		Name       string `json:"name"`
		Version    string `json:"version"`
		Location   string `json:"location"`
	}{
		RktVersion: "v1",
		RktKind:    "stage1",
		Name:       c.Stage1.Name,
		Version:    c.Stage1.Version,
		Location:   c.Stage1.Location,
	}

	stage0 = append(stage0, paths, stage1)

	data := map[string]interface{}{"stage0": stage0}
	return json.Marshal(data)
}

type configParser interface {
	parse(config *Config, raw []byte) error
}

var (
	// configSubDirs is a map saying what kinds of configuration
	// (values) are acceptable in a config subdirectory (key)
	configSubDirs  = make(map[string][]string)
	parsersForKind = make(map[string]map[string]configParser)
)

// ResolveAuthPerHost takes a map of strings to Headerer and resolves the
// Headerers to http.Headers
func ResolveAuthPerHost(authPerHost map[string]Headerer) map[string]http.Header {
	hostHeaders := make(map[string]http.Header, len(authPerHost))
	for k, v := range authPerHost {
		hostHeaders[k] = v.GetHeader()
	}
	return hostHeaders
}

func addParser(kind, version string, parser configParser) {
	if len(kind) == 0 {
		panic("empty kind string when registering a config parser")
	}
	if len(version) == 0 {
		panic("empty version string when registering a config parser")
	}
	if parser == nil {
		panic("trying to register a nil parser")
	}
	if _, err := getParser(kind, version); err == nil {
		panic(fmt.Sprintf("A parser for kind %q and version %q already exist", kind, version))
	}
	if _, ok := parsersForKind[kind]; !ok {
		parsersForKind[kind] = make(map[string]configParser)
	}
	parsersForKind[kind][version] = parser
}

func registerSubDir(dir string, kinds []string) {
	if len(dir) == 0 {
		panic("trying to register empty config subdirectory")
	}
	if len(kinds) == 0 {
		panic("kinds array cannot be empty when registering config subdir")
	}
	allKinds := toArray(toSet(append(configSubDirs[dir], kinds...)))
	sort.Strings(allKinds)
	configSubDirs[dir] = allKinds
}

func toSet(a []string) map[string]struct{} {
	s := make(map[string]struct{})
	for _, v := range a {
		s[v] = struct{}{}
	}
	return s
}

func toArray(s map[string]struct{}) []string {
	a := make([]string, 0, len(s))
	for k := range s {
		a = append(a, k)
	}
	return a
}

// GetConfig gets the Config instance with configuration taken from
// default system path (see common.DefaultSystemConfigDir) overridden
// with configuration from default local path (see
// common.DefaultLocalConfigDir).
func GetConfig() (*Config, error) {
	return GetConfigFrom(common.DefaultSystemConfigDir, common.DefaultLocalConfigDir)
}

// GetConfigFrom gets the Config instance with configuration taken
// from given paths. Subsequent paths override settings from the
// previous paths.
func GetConfigFrom(dirs ...string) (*Config, error) {
	cfg := newConfig()
	for _, cd := range dirs {
		subcfg, err := GetConfigFromDir(cd)
		if err != nil {
			return nil, err
		}
		mergeConfigs(cfg, subcfg)
	}
	return cfg, nil
}

// GetConfigFromDir gets the Config instance with configuration taken
// from given directory.
func GetConfigFromDir(dir string) (*Config, error) {
	subcfg := newConfig()
	if valid, err := validDir(dir); err != nil {
		return nil, err
	} else if !valid {
		return subcfg, nil
	}
	if err := readConfigDir(subcfg, dir); err != nil {
		return nil, err
	}
	return subcfg, nil
}

func newConfig() *Config {
	return &Config{
		AuthPerHost:                  make(map[string]Headerer),
		DockerCredentialsPerRegistry: make(map[string]BasicCredentials),
		Paths: ConfigurablePaths{
			DataDir: "",
		},
	}
}

func readConfigDir(config *Config, dir string) error {
	for csd, kinds := range configSubDirs {
		d := filepath.Join(dir, csd)
		if valid, err := validDir(d); err != nil {
			return err
		} else if !valid {
			continue
		}
		configWalker := getConfigWalker(config, kinds, d)
		if err := filepath.Walk(d, configWalker); err != nil {
			return err
		}
	}
	return nil
}

func validDir(path string) (bool, error) {
	fi, err := os.Stat(path)
	if err != nil {
		if os.IsNotExist(err) {
			return false, nil
		}
		return false, err
	}
	if !fi.IsDir() {
		return false, fmt.Errorf("expected %q to be a directory", path)
	}
	return true, nil
}

func getConfigWalker(config *Config, kinds []string, root string) filepath.WalkFunc {
	return func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if path == root {
			return nil
		}
		return readFile(config, info, path, kinds)
	}
}

func readFile(config *Config, info os.FileInfo, path string, kinds []string) error {
	if valid, err := validConfigFile(info); err != nil {
		return err
	} else if !valid {
		return nil
	}
	if err := parseConfigFile(config, path, kinds); err != nil {
		return err
	}
	return nil
}

func validConfigFile(info os.FileInfo) (bool, error) {
	mode := info.Mode()
	switch {
	case mode.IsDir():
		return false, filepath.SkipDir
	case mode.IsRegular():
		return filepath.Ext(info.Name()) == ".json", nil
	case mode&os.ModeSymlink == os.ModeSymlink:
		// TODO: support symlinks?
		return false, nil
	default:
		return false, nil
	}
}

type configHeader struct {
	RktVersion string `json:"rktVersion"`
	RktKind    string `json:"rktKind"`
}

func parseConfigFile(config *Config, path string, kinds []string) error {
	raw, err := ioutil.ReadFile(path)
	if err != nil {
		return err
	}
	var header configHeader
	if err := json.Unmarshal(raw, &header); err != nil {
		return err
	}
	if len(header.RktKind) == 0 {
		return fmt.Errorf("no rktKind specified in %q", path)
	}
	if len(header.RktVersion) == 0 {
		return fmt.Errorf("no rktVersion specified in %q", path)
	}
	kindOk := false
	for _, kind := range kinds {
		if header.RktKind == kind {
			kindOk = true
			break
		}
	}
	if !kindOk {
		dir := filepath.Dir(path)
		base := filepath.Base(path)
		kindsStr := strings.Join(kinds, `", "`)
		return fmt.Errorf("the configuration directory %q expects to have configuration files of kinds %q, but %q has kind of %q", dir, kindsStr, base, header.RktKind)
	}
	parser, err := getParser(header.RktKind, header.RktVersion)
	if err != nil {
		return err
	}
	if err := parser.parse(config, raw); err != nil {
		return errwrap.Wrap(fmt.Errorf("failed to parse %q", path), err)
	}
	return nil
}

func getParser(kind, version string) (configParser, error) {
	parsers, ok := parsersForKind[kind]
	if !ok {
		return nil, fmt.Errorf("no parser available for configuration of kind %q", kind)
	}
	parser, ok := parsers[version]
	if !ok {
		return nil, fmt.Errorf("no parser available for configuration of kind %q and version %q", kind, version)
	}
	return parser, nil
}

func mergeConfigs(config *Config, subconfig *Config) {
	for host, headerer := range subconfig.AuthPerHost {
		config.AuthPerHost[host] = headerer
	}
	for registry, creds := range subconfig.DockerCredentialsPerRegistry {
		config.DockerCredentialsPerRegistry[registry] = creds
	}
	if subconfig.Paths.DataDir != "" {
		config.Paths.DataDir = subconfig.Paths.DataDir
	}
	if subconfig.Paths.Stage1ImagesDir != "" {
		config.Paths.Stage1ImagesDir = subconfig.Paths.Stage1ImagesDir
	}
	if subconfig.Stage1.Name != "" {
		config.Stage1.Name = subconfig.Stage1.Name
		config.Stage1.Version = subconfig.Stage1.Version
	}
	if subconfig.Stage1.Location != "" {
		config.Stage1.Location = subconfig.Stage1.Location
	}
}
