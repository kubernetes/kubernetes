package testjson

import (
	"bytes"
	"go/build"
	"io/ioutil"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
)

func relativePackagePath(pkgpath string) string {
	if pkgpath == pkgPathPrefix {
		return "."
	}
	return strings.TrimPrefix(pkgpath, pkgPathPrefix+"/")
}

func getPkgPathPrefix() string {
	cwd, _ := os.Getwd()
	if isGoModuleEnabled() {
		prefix := getPkgPathPrefixFromGoModule(cwd)
		if prefix != "" {
			return prefix
		}
	}
	return getPkgPathPrefixGoPath(cwd)
}

func isGoModuleEnabled() bool {
	version := runtime.Version()
	if strings.HasPrefix(version, "go1.10") {
		return false
	}
	// Go modules may not be enabled if env var is unset, or set to auto, however
	// we can always fall back to using GOPATH as the prefix if a go.mod is not
	// found.
	return os.Getenv("GO111MODULE") != "off"
}

// TODO: might not work on windows
func getPkgPathPrefixGoPath(cwd string) string {
	gopaths := strings.Split(build.Default.GOPATH, string(filepath.ListSeparator))
	for _, gopath := range gopaths {
		gosrcpath := gopath + "/src/"
		if strings.HasPrefix(cwd, gosrcpath) {
			return strings.TrimPrefix(cwd, gosrcpath)
		}
	}
	return ""
}

func getPkgPathPrefixFromGoModule(cwd string) string {
	filename := goModuleFilePath(cwd)
	if filename == "" {
		return ""
	}
	raw, err := ioutil.ReadFile(filename)
	if err != nil {
		// TODO: log.Warn
		return ""
	}
	return pkgPathFromGoModuleFile(raw)
}

var (
	slashSlash = []byte("//")
	moduleStr  = []byte("module")
)

// Copy of ModulePath from golang.org/src/cmd/go/internal/modfile/read.go
func pkgPathFromGoModuleFile(mod []byte) string {
	for len(mod) > 0 {
		line := mod
		mod = nil
		if i := bytes.IndexByte(line, '\n'); i >= 0 {
			line, mod = line[:i], line[i+1:]
		}
		if i := bytes.Index(line, slashSlash); i >= 0 {
			line = line[:i]
		}
		line = bytes.TrimSpace(line)
		if !bytes.HasPrefix(line, moduleStr) {
			continue
		}
		line = line[len(moduleStr):]
		n := len(line)
		line = bytes.TrimSpace(line)
		if len(line) == n || len(line) == 0 {
			continue
		}

		if line[0] == '"' || line[0] == '`' {
			p, err := strconv.Unquote(string(line))
			if err != nil {
				return "" // malformed quoted string or multi-line module path
			}
			return p
		}

		return string(line)
	}
	return "" // missing module path
}

// A rough re-implementation of FindModuleRoot from
// golang.org/src/cmd/go/internal/modload/init.go
func goModuleFilePath(cwd string) string {
	dir := filepath.Clean(cwd)

	for {
		path := filepath.Join(dir, "go.mod")
		if _, err := os.Stat(path); err == nil {
			return path
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return ""
		}
		dir = parent
	}
}

var pkgPathPrefix = getPkgPathPrefix()
