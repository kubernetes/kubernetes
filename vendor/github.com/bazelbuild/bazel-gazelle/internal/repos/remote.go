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

package repos

import (
	"bytes"
	"fmt"
	"os/exec"
	"path"
	"regexp"
	"strings"
	"sync"

	"github.com/bazelbuild/bazel-gazelle/internal/label"
	"github.com/bazelbuild/bazel-gazelle/internal/pathtools"
	"golang.org/x/tools/go/vcs"
)

// UpdateRepo returns an object describing a repository at the most recent
// commit or version tag.
//
// This function uses RemoteCache to retrieve information about the repository.
// Depending on how the RemoteCache was initialized and used earlier, some
// information may already be locally available. Frequently though, information
// will be fetched over the network, so this function may be slow.
func UpdateRepo(rc *RemoteCache, importPath string) (Repo, error) {
	root, name, err := rc.Root(importPath)
	if err != nil {
		return Repo{}, err
	}
	remote, vcs, err := rc.Remote(root)
	if err != nil {
		return Repo{}, err
	}
	commit, tag, err := rc.Head(remote, vcs)
	if err != nil {
		return Repo{}, err
	}
	repo := Repo{
		Name:     name,
		GoPrefix: root,
		Commit:   commit,
		Tag:      tag,
		Remote:   remote,
		VCS:      vcs,
	}
	return repo, nil
}

// RemoteCache stores information about external repositories. The cache may
// be initialized with information about known repositories, i.e., those listed
// in the WORKSPACE file and mentioned on the command line. Other information
// is retrieved over the network.
//
// Public methods of RemoteCache may be slow in cases where a network fetch
// is needed. Public methods may be called concurrently.
type RemoteCache struct {
	// RepoRootForImportPath is vcs.RepoRootForImportPath by default. It may
	// be overridden so that tests may avoid accessing the network.
	RepoRootForImportPath func(string, bool) (*vcs.RepoRoot, error)

	// HeadCmd returns the latest commit on the default branch in the given
	// repository. This is used by Head. It may be stubbed out for tests.
	HeadCmd func(remote, vcs string) (string, error)

	root, remote, head remoteCacheMap
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

// NewRemoteCache creates a new RemoteCache with a set of known repositories.
// The Root and Remote methods will return information about repositories listed
// here without accessing the network. However, the Head method will still
// access the network for these repositories to retrieve information about new
// versions.
func NewRemoteCache(knownRepos []Repo) *RemoteCache {
	r := &RemoteCache{
		RepoRootForImportPath: vcs.RepoRootForImportPath,
		HeadCmd:               defaultHeadCmd,
		root:                  remoteCacheMap{cache: make(map[string]*remoteCacheEntry)},
		remote:                remoteCacheMap{cache: make(map[string]*remoteCacheEntry)},
		head:                  remoteCacheMap{cache: make(map[string]*remoteCacheEntry)},
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
	}
	return r
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
		cmd := exec.Command("git", "ls-remote", "--", remote, "HEAD")
		out, err := cmd.Output()
		if err != nil {
			return "", err
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
