/* Copyright 2018 The Bazel Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package repo

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"sync"

	"github.com/bazelbuild/bazel-gazelle/label"
	"github.com/bazelbuild/bazel-gazelle/pathtools"
	"golang.org/x/tools/go/vcs"
)

// RemoteCache stores information about external repositories. The cache may
// be initialized with information about known repositories, i.e., those listed
// in the WORKSPACE file and mentioned on the command line. Other information
// is retrieved over the network.
//
// Public methods of RemoteCache may be slow in cases where a network fetch
// is needed. Public methods may be called concurrently.
//
// TODO(jayconrod): this is very Go-centric. It should be moved to language/go.
// Unfortunately, doing so would break the resolve.Resolver interface.
type RemoteCache struct {
	// RepoRootForImportPath is vcs.RepoRootForImportPath by default. It may
	// be overridden so that tests may avoid accessing the network.
	RepoRootForImportPath func(string, bool) (*vcs.RepoRoot, error)

	// HeadCmd returns the latest commit on the default branch in the given
	// repository. This is used by Head. It may be stubbed out for tests.
	HeadCmd func(remote, vcs string) (string, error)

	// ModInfo returns the module path and version that provides the package
	// with the given import path. This is used by Mod. It may be stubbed
	// out for tests.
	ModInfo func(importPath string) (modPath string, err error)

	// ModVersionInfo returns the module path, true version, and sum for
	// the module that provides the package with the given import path.
	// This is used by ModVersion. It may be stubbed out for tests.
	ModVersionInfo func(modPath, query string) (version, sum string, err error)

	root, remote, head, mod, modVersion remoteCacheMap

	tmpOnce sync.Once
	tmpDir  string
	tmpErr  error
}

// remoteCacheMap is a thread-safe, idempotent cache. It is used to store
// information which should be fetched over the network no more than once.
// This follows the Memo pattern described in The Go Programming Language,
// section 9.7.
type remoteCacheMap struct {
	mu    sync.Mutex
	cache map[string]*remoteCacheEntry
}

type remoteCacheEntry struct {
	value interface{}
	err   error

	// ready is nil for entries that were added when the cache was initialized.
	// It is non-nil for other entries. It is closed when an entry is ready,
	// i.e., the operation loading the entry completed.
	ready chan struct{}
}

type rootValue struct {
	root, name string
}

type remoteValue struct {
	remote, vcs string
}

type headValue struct {
	commit, tag string
}

type modValue struct {
	path, name string
	known      bool
}

type modVersionValue struct {
	path, name, version, sum string
}

// Repo describes details of a Go repository known in advance. It is used to
// initialize RemoteCache so that some repositories don't need to be looked up.
//
// DEPRECATED: Go-specific details should be removed from RemoteCache, and
// lookup logic should be moved to language/go. This means RemoteCache will
// need to be initialized in a different way.
type Repo struct {
	Name, GoPrefix, Remote, VCS string
}

// NewRemoteCache creates a new RemoteCache with a set of known repositories.
// The Root and Remote methods will return information about repositories listed
// here without accessing the network. However, the Head method will still
// access the network for these repositories to retrieve information about new
// versions.
//
// A cleanup function is also returned. The caller must call this when
// RemoteCache is no longer needed. RemoteCache may write files to a temporary
// directory. This will delete them.
func NewRemoteCache(knownRepos []Repo) (r *RemoteCache, cleanup func() error) {
	r = &RemoteCache{
		RepoRootForImportPath: vcs.RepoRootForImportPath,
		HeadCmd:               defaultHeadCmd,
		root:                  remoteCacheMap{cache: make(map[string]*remoteCacheEntry)},
		remote:                remoteCacheMap{cache: make(map[string]*remoteCacheEntry)},
		head:                  remoteCacheMap{cache: make(map[string]*remoteCacheEntry)},
		mod:                   remoteCacheMap{cache: make(map[string]*remoteCacheEntry)},
		modVersion:            remoteCacheMap{cache: make(map[string]*remoteCacheEntry)},
	}
	r.ModInfo = func(importPath string) (string, error) {
		return defaultModInfo(r, importPath)
	}
	r.ModVersionInfo = func(modPath, query string) (string, string, error) {
		return defaultModVersionInfo(r, modPath, query)
	}
	for _, repo := range knownRepos {
		r.root.cache[repo.GoPrefix] = &remoteCacheEntry{
			value: rootValue{
				root: repo.GoPrefix,
				name: repo.Name,
			},
		}
		if repo.Remote != "" {
			r.remote.cache[repo.GoPrefix] = &remoteCacheEntry{
				value: remoteValue{
					remote: repo.Remote,
					vcs:    repo.VCS,
				},
			}
		}
		r.mod.cache[repo.GoPrefix] = &remoteCacheEntry{
			value: modValue{
				path:  repo.GoPrefix,
				name:  repo.Name,
				known: true,
			},
		}
	}

	// Augment knownRepos with additional prefixes for
	// minimal module compatibility. For example, if repo "com_example_foo_v2"
	// has prefix "example.com/foo/v2", map "example.com/foo" to the same
	// entry.
	// TODO(jayconrod): there should probably be some control over whether
	// callers can use these mappings: packages within modules should not be
	// allowed to use them. However, we'll return the same result nearly all
	// the time, and simpler is better.
	for _, repo := range knownRepos {
		path := pathWithoutSemver(repo.GoPrefix)
		if path == "" || r.root.cache[path] != nil {
			continue
		}
		r.root.cache[path] = r.root.cache[repo.GoPrefix]
		if e := r.remote.cache[repo.GoPrefix]; e != nil {
			r.remote.cache[path] = e
		}
		r.mod.cache[path] = r.mod.cache[repo.GoPrefix]
	}

	return r, r.cleanup
}

func (r *RemoteCache) cleanup() error {
	if r.tmpDir == "" {
		return nil
	}
	return os.RemoveAll(r.tmpDir)
}

var gopkginPattern = regexp.MustCompile("^(gopkg.in/(?:[^/]+/)?[^/]+\\.v\\d+)(?:/|$)")

var knownPrefixes = []struct {
	prefix  string
	missing int
}{
	{prefix: "golang.org/x", missing: 1},
	{prefix: "google.golang.org", missing: 1},
	{prefix: "cloud.google.com", missing: 1},
	{prefix: "github.com", missing: 2},
}

// Root returns the portion of an import path that corresponds to the root
// directory of the repository containing the given import path. For example,
// given "golang.org/x/tools/go/loader", this will return "golang.org/x/tools".
// The workspace name of the repository is also returned. This may be a custom
// name set in WORKSPACE, or it may be a generated name based on the root path.
func (r *RemoteCache) Root(importPath string) (root, name string, err error) {
	// Try prefixes of the import path in the cache, but don't actually go out
	// to vcs yet. We do this before handling known special cases because
	// the cache is pre-populated with repository rules, and we want to use their
	// names if we can.
	prefix := importPath
	for {
		v, ok, err := r.root.get(prefix)
		if ok {
			if err != nil {
				return "", "", err
			}
			value := v.(rootValue)
			return value.root, value.name, nil
		}

		prefix = path.Dir(prefix)
		if prefix == "." || prefix == "/" {
			break
		}
	}

	// Try known prefixes.
	for _, p := range knownPrefixes {
		if pathtools.HasPrefix(importPath, p.prefix) {
			rest := pathtools.TrimPrefix(importPath, p.prefix)
			var components []string
			if rest != "" {
				components = strings.Split(rest, "/")
			}
			if len(components) < p.missing {
				return "", "", fmt.Errorf("import path %q is shorter than the known prefix %q", importPath, p.prefix)
			}
			root = p.prefix
			for _, c := range components[:p.missing] {
				root = path.Join(root, c)
			}
			name = label.ImportPathToBazelRepoName(root)
			return root, name, nil
		}
	}

	// gopkg.in is special, and might have either one or two levels of
	// missing paths. See http://labix.org/gopkg.in for URL patterns.
	if match := gopkginPattern.FindStringSubmatch(importPath); len(match) > 0 {
		root = match[1]
		name = label.ImportPathToBazelRepoName(root)
		return root, name, nil
	}

	// Find the prefix using vcs and cache the result.
	v, err := r.root.ensure(importPath, func() (interface{}, error) {
		res, err := r.RepoRootForImportPath(importPath, false)
		if err != nil {
			return nil, err
		}
		return rootValue{res.Root, label.ImportPathToBazelRepoName(res.Root)}, nil
	})
	if err != nil {
		return "", "", err
	}
	value := v.(rootValue)
	return value.root, value.name, nil
}

// Remote returns the VCS name and the remote URL for a repository with the
// given root import path. This is suitable for creating new repository rules.
func (r *RemoteCache) Remote(root string) (remote, vcs string, err error) {
	v, err := r.remote.ensure(root, func() (interface{}, error) {
		repo, err := r.RepoRootForImportPath(root, false)
		if err != nil {
			return nil, err
		}
		return remoteValue{remote: repo.Repo, vcs: repo.VCS.Cmd}, nil
	})
	if err != nil {
		return "", "", err
	}
	value := v.(remoteValue)
	return value.remote, value.vcs, nil
}

// Head returns the most recent commit id on the default branch and latest
// version tag for the given remote repository. The tag "" is returned if
// no latest version was found.
//
// TODO(jayconrod): support VCS other than git.
// TODO(jayconrod): support version tags. "" is always returned.
func (r *RemoteCache) Head(remote, vcs string) (commit, tag string, err error) {
	if vcs != "git" {
		return "", "", fmt.Errorf("could not locate recent commit in repo %q with unknown version control scheme %q", remote, vcs)
	}

	v, err := r.head.ensure(remote, func() (interface{}, error) {
		commit, err := r.HeadCmd(remote, vcs)
		if err != nil {
			return nil, err
		}
		return headValue{commit: commit}, nil
	})
	if err != nil {
		return "", "", err
	}
	value := v.(headValue)
	return value.commit, value.tag, nil
}

func defaultHeadCmd(remote, vcs string) (string, error) {
	switch vcs {
	case "local":
		return "", nil

	case "git":
		// Old versions of git ls-remote exit with code 129 when "--" is passed.
		// We'll try to validate the argument here instead.
		if strings.HasPrefix(remote, "-") {
			return "", fmt.Errorf("remote must not start with '-': %q", remote)
		}
		cmd := exec.Command("git", "ls-remote", remote, "HEAD")
		out, err := cmd.Output()
		if err != nil {
			var stdErr []byte
			if e, ok := err.(*exec.ExitError); ok {
				stdErr = e.Stderr
			}
			return "", fmt.Errorf("git ls-remote for %s : %v : %s", remote, err, stdErr)
		}
		ix := bytes.IndexByte(out, '\t')
		if ix < 0 {
			return "", fmt.Errorf("could not parse output for git ls-remote for %q", remote)
		}
		return string(out[:ix]), nil

	default:
		return "", fmt.Errorf("unknown version control system: %s", vcs)
	}
}

// Mod returns the module path for the module that contains the package
// named by importPath. The name of the go_repository rule for the module
// is also returned. For example, calling Mod on "github.com/foo/bar/v2/baz"
// would give the module path "github.com/foo/bar/v2" and the name
// "com_github_foo_bar_v2".
//
// If a known repository *could* provide importPath (because its "importpath"
// is a prefix of importPath), Mod will assume that it does. This may give
// inaccurate results if importPath is in an undeclared nested module. Run
// "gazelle update-repos -from_file=go.mod" first for best results.
//
// If no known repository could provide importPath, Mod will run "go list" to
// find the module. The special patterns that Root uses are ignored. Results are
// cached. Use GOPROXY for faster results.
func (r *RemoteCache) Mod(importPath string) (modPath, name string, err error) {
	// Check if any of the known repositories is a prefix.
	prefix := importPath
	for {
		v, ok, err := r.mod.get(prefix)
		if ok {
			if err != nil {
				return "", "", err
			}
			value := v.(modValue)
			if value.known {
				return value.path, value.name, nil
			} else {
				break
			}
		}

		prefix = path.Dir(prefix)
		if prefix == "." || prefix == "/" {
			break
		}
	}

	// Ask "go list".
	v, err := r.mod.ensure(importPath, func() (interface{}, error) {
		modPath, err := r.ModInfo(importPath)
		if err != nil {
			return nil, err
		}
		return modValue{
			path: modPath,
			name: label.ImportPathToBazelRepoName(modPath),
		}, nil
	})
	if err != nil {
		return "", "", err
	}
	value := v.(modValue)
	return value.path, value.name, nil
}

func defaultModInfo(rc *RemoteCache, importPath string) (modPath string, err error) {
	rc.initTmp()
	if rc.tmpErr != nil {
		return "", rc.tmpErr
	}

	goTool := findGoTool()
	cmd := exec.Command(goTool, "list", "-find", "-f", "{{.Module.Path}}", "--", importPath)
	cmd.Dir = rc.tmpDir
	cmd.Env = append(os.Environ(), "GO111MODULE=on")
	out, err := cmd.Output()
	if err != nil {
		var stdErr []byte
		if e, ok := err.(*exec.ExitError); ok {
			stdErr = e.Stderr
		}
		return "", fmt.Errorf("finding module path for import %s: %v: %s", importPath, err, stdErr)
	}
	return strings.TrimSpace(string(out)), nil
}

// ModVersion looks up information about a module at a given version.
// The path must be the module path, not a package within the module.
// The version may be a canonical semantic version, a query like "latest",
// or a branch, tag, or revision name. ModVersion returns the name of
// the repository rule providing the module (if any), the true version,
// and the sum.
func (r *RemoteCache) ModVersion(modPath, query string) (name, version, sum string, err error) {
	// Ask "go list".
	arg := modPath + "@" + query
	v, err := r.modVersion.ensure(arg, func() (interface{}, error) {
		version, sum, err := r.ModVersionInfo(modPath, query)
		if err != nil {
			return nil, err
		}
		return modVersionValue{
			path:    modPath,
			version: version,
			sum:     sum,
		}, nil
	})
	if err != nil {
		return "", "", "", err
	}
	value := v.(modVersionValue)

	// Try to find the repository name for the module, if there's already
	// a repository rule that provides it.
	v, ok, err := r.mod.get(modPath)
	if ok && err == nil {
		name = v.(modValue).name
	} else {
		name = label.ImportPathToBazelRepoName(modPath)
	}

	return name, value.version, value.sum, nil
}

func defaultModVersionInfo(rc *RemoteCache, modPath, query string) (version, sum string, err error) {
	rc.initTmp()
	if rc.tmpErr != nil {
		return "", "", rc.tmpErr
	}

	goTool := findGoTool()
	cmd := exec.Command(goTool, "mod", "download", "-json", "--", modPath+"@"+query)
	cmd.Dir = rc.tmpDir
	cmd.Env = append(os.Environ(), "GO111MODULE=on")
	out, err := cmd.Output()
	if err != nil {
		var stdErr []byte
		if e, ok := err.(*exec.ExitError); ok {
			stdErr = e.Stderr
		}
		return "", "", fmt.Errorf("finding module version and sum for %s@%s: %v: %s", modPath, query, err, stdErr)
	}

	var result struct{ Version, Sum string }
	if err := json.Unmarshal(out, &result); err != nil {
		fmt.Println(out)
		return "", "", fmt.Errorf("finding module version and sum for %s@%s: invalid output from 'go mod download': %v", modPath, query, err)
	}
	return result.Version, result.Sum, nil
}

// get retrieves a value associated with the given key from the cache. ok will
// be true if the key exists in the cache, even if it's in the process of
// being fetched.
func (m *remoteCacheMap) get(key string) (value interface{}, ok bool, err error) {
	m.mu.Lock()
	e, ok := m.cache[key]
	m.mu.Unlock()
	if !ok {
		return nil, ok, nil
	}
	if e.ready != nil {
		<-e.ready
	}
	return e.value, ok, e.err
}

// ensure retreives a value associated with the given key from the cache. If
// the key does not exist in the cache, the load function will be called,
// and its result will be associated with the key. The load function will not
// be called more than once for any key.
func (m *remoteCacheMap) ensure(key string, load func() (interface{}, error)) (interface{}, error) {
	m.mu.Lock()
	e, ok := m.cache[key]
	if !ok {
		e = &remoteCacheEntry{ready: make(chan struct{})}
		m.cache[key] = e
		m.mu.Unlock()
		e.value, e.err = load()
		close(e.ready)
	} else {
		m.mu.Unlock()
		if e.ready != nil {
			<-e.ready
		}
	}
	return e.value, e.err
}

func (rc *RemoteCache) initTmp() {
	rc.tmpOnce.Do(func() {
		rc.tmpDir, rc.tmpErr = ioutil.TempDir("", "gazelle-remotecache-")
		if rc.tmpErr != nil {
			return
		}
		rc.tmpErr = ioutil.WriteFile(filepath.Join(rc.tmpDir, "go.mod"), []byte(`module gazelle_remote_cache__\n`), 0666)
	})
}

var semverRex = regexp.MustCompile(`^.*?(/v\d+)(?:/.*)?$`)

// pathWithoutSemver removes a semantic version suffix from path.
// For example, if path is "example.com/foo/v2/bar", pathWithoutSemver
// will return "example.com/foo/bar". If there is no semantic version suffix,
// "" will be returned.
// TODO(jayconrod): copied from language/go. This whole type should be
// migrated there.
func pathWithoutSemver(path string) string {
	m := semverRex.FindStringSubmatchIndex(path)
	if m == nil {
		return ""
	}
	v := path[m[2]+2 : m[3]]
	if v == "0" || v == "1" {
		return ""
	}
	return path[:m[2]] + path[m[3]:]
}

// findGoTool attempts to locate the go executable. If GOROOT is set, we'll
// prefer the one in there; otherwise, we'll rely on PATH. If the wrapper
// script generated by the gazelle rule is invoked by Bazel, it will set
// GOROOT to the configured SDK. We don't want to rely on the host SDK in
// that situation.
//
// TODO(jayconrod): copied from language/go (though it was originally in this
// package). Go-specific details should be removed from RemoteCache, and
// this copy should be deleted.
func findGoTool() string {
	path := "go" // rely on PATH by default
	if goroot, ok := os.LookupEnv("GOROOT"); ok {
		path = filepath.Join(goroot, "bin", "go")
	}
	if runtime.GOOS == "windows" {
		path += ".exe"
	}
	return path
}
