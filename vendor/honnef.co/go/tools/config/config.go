package config

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"

	"github.com/BurntSushi/toml"
	"golang.org/x/tools/go/analysis"
)

var Analyzer = &analysis.Analyzer{
	Name: "config",
	Doc:  "loads configuration for the current package tree",
	Run: func(pass *analysis.Pass) (interface{}, error) {
		if len(pass.Files) == 0 {
			cfg := DefaultConfig
			return &cfg, nil
		}
		cache, err := os.UserCacheDir()
		if err != nil {
			cache = ""
		}
		var path string
		for _, f := range pass.Files {
			p := pass.Fset.PositionFor(f.Pos(), true).Filename
			// FIXME(dh): using strings.HasPrefix isn't technically
			// correct, but it should be good enough for now.
			if cache != "" && strings.HasPrefix(p, cache) {
				// File in the build cache of the standard Go build system
				continue
			}
			path = p
			break
		}

		if path == "" {
			// The package only consists of generated files.
			cfg := DefaultConfig
			return &cfg, nil
		}

		dir := filepath.Dir(path)
		cfg, err := Load(dir)
		if err != nil {
			return nil, fmt.Errorf("error loading staticcheck.conf: %s", err)
		}
		return &cfg, nil
	},
	RunDespiteErrors: true,
	ResultType:       reflect.TypeOf((*Config)(nil)),
}

func For(pass *analysis.Pass) *Config {
	return pass.ResultOf[Analyzer].(*Config)
}

func mergeLists(a, b []string) []string {
	out := make([]string, 0, len(a)+len(b))
	for _, el := range b {
		if el == "inherit" {
			out = append(out, a...)
		} else {
			out = append(out, el)
		}
	}

	return out
}

func normalizeList(list []string) []string {
	if len(list) > 1 {
		nlist := make([]string, 0, len(list))
		nlist = append(nlist, list[0])
		for i, el := range list[1:] {
			if el != list[i] {
				nlist = append(nlist, el)
			}
		}
		list = nlist
	}

	for _, el := range list {
		if el == "inherit" {
			// This should never happen, because the default config
			// should not use "inherit"
			panic(`unresolved "inherit"`)
		}
	}

	return list
}

func (cfg Config) Merge(ocfg Config) Config {
	if ocfg.Checks != nil {
		cfg.Checks = mergeLists(cfg.Checks, ocfg.Checks)
	}
	if ocfg.Initialisms != nil {
		cfg.Initialisms = mergeLists(cfg.Initialisms, ocfg.Initialisms)
	}
	if ocfg.DotImportWhitelist != nil {
		cfg.DotImportWhitelist = mergeLists(cfg.DotImportWhitelist, ocfg.DotImportWhitelist)
	}
	if ocfg.HTTPStatusCodeWhitelist != nil {
		cfg.HTTPStatusCodeWhitelist = mergeLists(cfg.HTTPStatusCodeWhitelist, ocfg.HTTPStatusCodeWhitelist)
	}
	return cfg
}

type Config struct {
	// TODO(dh): this implementation makes it impossible for external
	// clients to add their own checkers with configuration. At the
	// moment, we don't really care about that; we don't encourage
	// that people use this package. In the future, we may. The
	// obvious solution would be using map[string]interface{}, but
	// that's obviously subpar.

	Checks                  []string `toml:"checks"`
	Initialisms             []string `toml:"initialisms"`
	DotImportWhitelist      []string `toml:"dot_import_whitelist"`
	HTTPStatusCodeWhitelist []string `toml:"http_status_code_whitelist"`
}

func (c Config) String() string {
	buf := &bytes.Buffer{}

	fmt.Fprintf(buf, "Checks: %#v\n", c.Checks)
	fmt.Fprintf(buf, "Initialisms: %#v\n", c.Initialisms)
	fmt.Fprintf(buf, "DotImportWhitelist: %#v\n", c.DotImportWhitelist)
	fmt.Fprintf(buf, "HTTPStatusCodeWhitelist: %#v", c.HTTPStatusCodeWhitelist)

	return buf.String()
}

var DefaultConfig = Config{
	Checks: []string{"all", "-ST1000", "-ST1003", "-ST1016"},
	Initialisms: []string{
		"ACL", "API", "ASCII", "CPU", "CSS", "DNS",
		"EOF", "GUID", "HTML", "HTTP", "HTTPS", "ID",
		"IP", "JSON", "QPS", "RAM", "RPC", "SLA",
		"SMTP", "SQL", "SSH", "TCP", "TLS", "TTL",
		"UDP", "UI", "GID", "UID", "UUID", "URI",
		"URL", "UTF8", "VM", "XML", "XMPP", "XSRF",
		"XSS", "SIP", "RTP",
	},
	DotImportWhitelist:      []string{},
	HTTPStatusCodeWhitelist: []string{"200", "400", "404", "500"},
}

const configName = "staticcheck.conf"

func parseConfigs(dir string) ([]Config, error) {
	var out []Config

	// TODO(dh): consider stopping at the GOPATH/module boundary
	for dir != "" {
		f, err := os.Open(filepath.Join(dir, configName))
		if os.IsNotExist(err) {
			ndir := filepath.Dir(dir)
			if ndir == dir {
				break
			}
			dir = ndir
			continue
		}
		if err != nil {
			return nil, err
		}
		var cfg Config
		_, err = toml.DecodeReader(f, &cfg)
		f.Close()
		if err != nil {
			return nil, err
		}
		out = append(out, cfg)
		ndir := filepath.Dir(dir)
		if ndir == dir {
			break
		}
		dir = ndir
	}
	out = append(out, DefaultConfig)
	if len(out) < 2 {
		return out, nil
	}
	for i := 0; i < len(out)/2; i++ {
		out[i], out[len(out)-1-i] = out[len(out)-1-i], out[i]
	}
	return out, nil
}

func mergeConfigs(confs []Config) Config {
	if len(confs) == 0 {
		// This shouldn't happen because we always have at least a
		// default config.
		panic("trying to merge zero configs")
	}
	if len(confs) == 1 {
		return confs[0]
	}
	conf := confs[0]
	for _, oconf := range confs[1:] {
		conf = conf.Merge(oconf)
	}
	return conf
}

func Load(dir string) (Config, error) {
	confs, err := parseConfigs(dir)
	if err != nil {
		return Config{}, err
	}
	conf := mergeConfigs(confs)

	conf.Checks = normalizeList(conf.Checks)
	conf.Initialisms = normalizeList(conf.Initialisms)
	conf.DotImportWhitelist = normalizeList(conf.DotImportWhitelist)
	conf.HTTPStatusCodeWhitelist = normalizeList(conf.HTTPStatusCodeWhitelist)

	return conf, nil
}
