// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package git

import (
	"fmt"
	"log"
	"net/url"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"

	"sigs.k8s.io/kustomize/kyaml/errors"
	"sigs.k8s.io/kustomize/kyaml/filesys"
)

// Used as a temporary non-empty occupant of the cloneDir
// field, as something distinguishable from the empty string
// in various outputs (especially tests). Not using an
// actual directory name here, as that's a temporary directory
// with a unique name that isn't created until clone time.
const notCloned = filesys.ConfirmedDir("/notCloned")

// RepoSpec specifies a git repository and a branch and path therein.
type RepoSpec struct {
	// Raw, original spec, used to look for cycles.
	// TODO(monopole): Drop raw, use processed fields instead.
	raw string

	// Host, e.g. https://github.com/
	Host string

	// RepoPath name (Path to repository),
	// e.g. kubernetes-sigs/kustomize
	RepoPath string

	// Dir is where the repository is cloned to.
	Dir filesys.ConfirmedDir

	// Relative path in the repository, and in the cloneDir,
	// to a Kustomization.
	KustRootPath string

	// Branch or tag reference.
	Ref string

	// Submodules indicates whether or not to clone git submodules.
	Submodules bool

	// Timeout is the maximum duration allowed for execing git commands.
	Timeout time.Duration
}

// CloneSpec returns a string suitable for "git clone {spec}".
func (x *RepoSpec) CloneSpec() string {
	return x.Host + x.RepoPath
}

func (x *RepoSpec) CloneDir() filesys.ConfirmedDir {
	return x.Dir
}

func (x *RepoSpec) Raw() string {
	return x.raw
}

func (x *RepoSpec) AbsPath() string {
	return x.Dir.Join(x.KustRootPath)
}

func (x *RepoSpec) Cleaner(fSys filesys.FileSystem) func() error {
	return func() error { return fSys.RemoveAll(x.Dir.String()) }
}

const (
	refQuery         = "?ref="
	gitSuffix        = ".git"
	gitRootDelimiter = "_git/"
	pathSeparator    = "/" // do not use filepath.Separator, as this is a URL
)

// NewRepoSpecFromURL parses git-like urls.
// From strings like git@github.com:someOrg/someRepo.git or
// https://github.com/someOrg/someRepo?ref=someHash, extract
// the different parts of URL, set into a RepoSpec object and return RepoSpec object.
// It MUST return an error if the input is not a git-like URL, as this is used by some code paths
// to distinguish between local and remote paths.
//
// In particular, NewRepoSpecFromURL separates the URL used to clone the repo from the
// elements Kustomize uses for other purposes (e.g. query params that turn into args, and
// the path to the kustomization root within the repo).
func NewRepoSpecFromURL(n string) (*RepoSpec, error) {
	repoSpec := &RepoSpec{raw: n, Dir: notCloned, Timeout: defaultTimeout, Submodules: defaultSubmodules}
	if filepath.IsAbs(n) {
		return nil, fmt.Errorf("uri looks like abs path: %s", n)
	}

	// Parse the query first. This is safe because according to rfc3986 "?" is only allowed in the
	// query and is not recognized %-encoded.
	// Note that parseQuery returns default values for empty parameters.
	n, query, _ := strings.Cut(n, "?")
	repoSpec.Ref, repoSpec.Timeout, repoSpec.Submodules = parseQuery(query)

	var err error

	// Parse the host (e.g. scheme, username, domain) segment.
	repoSpec.Host, n, err = extractHost(n)
	if err != nil {
		return nil, err
	}

	// In some cases, we're given a path to a git repo + a path to the kustomization root within
	// that repo. We need to split them so that we can ultimately give the repo only to the cloner.
	repoSpec.RepoPath, repoSpec.KustRootPath, err = parsePathParts(n, defaultRepoPathLength(repoSpec.Host))
	if err != nil {
		return nil, err
	}

	return repoSpec, nil
}

const allSegments = -999999
const orgRepoSegments = 2

func defaultRepoPathLength(host string) int {
	if strings.HasPrefix(host, fileScheme) {
		return allSegments
	}
	return orgRepoSegments
}

// parsePathParts splits the repo path that will ultimately be passed to git to clone the
// repo from the kustomization root path, which Kustomize will execute the build in after the repo
// is cloned.
//
// We first try to do this based on explicit markers in the URL (e.g. _git, .git or //).
// If none are present, we try to apply a historical default repo path length that is derived from
// Github URLs. If there aren't enough segments, we have historically considered the URL invalid.
func parsePathParts(n string, defaultSegmentLength int) (string, string, error) {
	repoPath, kustRootPath, success := tryExplicitMarkerSplit(n)
	if !success {
		repoPath, kustRootPath, success = tryDefaultLengthSplit(n, defaultSegmentLength)
	}

	// Validate the result
	if !success || len(repoPath) == 0 {
		return "", "", fmt.Errorf("failed to parse repo path segment")
	}
	if kustRootPathExitsRepo(kustRootPath) {
		return "", "", fmt.Errorf("url path exits repo: %s", n)
	}

	return repoPath, strings.TrimPrefix(kustRootPath, pathSeparator), nil
}

func tryExplicitMarkerSplit(n string) (string, string, bool) {
	// Look for the _git delimiter, which by convention is expected to be ONE directory above the repo root.
	// If found, split on the NEXT path element, which is the repo root.
	// Example: https://username@dev.azure.com/org/project/_git/repo/path/to/kustomization/root
	if gitRootIdx := strings.Index(n, gitRootDelimiter); gitRootIdx >= 0 {
		gitRootPath := n[:gitRootIdx+len(gitRootDelimiter)]
		subpathSegments := strings.Split(n[gitRootIdx+len(gitRootDelimiter):], pathSeparator)
		return gitRootPath + subpathSegments[0], strings.Join(subpathSegments[1:], pathSeparator), true

		// Look for a double-slash in the path, which if present separates the repo root from the kust path.
		// It is a convention, not a real path element, so do not preserve it in the returned value.
		// Example: https://github.com/org/repo//path/to/kustomozation/root
	} else if repoRootIdx := strings.Index(n, "//"); repoRootIdx >= 0 {
		return n[:repoRootIdx], n[repoRootIdx+2:], true

		// Look for .git in the path, which if present is part of the directory name of the git repo.
		// This means we want to grab everything up to and including that suffix
		// Example: https://github.com/org/repo.git/path/to/kustomozation/root
	} else if gitSuffixIdx := strings.Index(n, gitSuffix); gitSuffixIdx >= 0 {
		upToGitSuffix := n[:gitSuffixIdx+len(gitSuffix)]
		afterGitSuffix := n[gitSuffixIdx+len(gitSuffix):]
		return upToGitSuffix, afterGitSuffix, true
	}
	return "", "", false
}

func tryDefaultLengthSplit(n string, defaultSegmentLength int) (string, string, bool) {
	// If the default is to take all segments, do so.
	if defaultSegmentLength == allSegments {
		return n, "", true

		// If the default is N segments, make sure we have at least that many and take them if so.
		// If we have less than N, we have historically considered the URL invalid.
	} else if segments := strings.Split(n, pathSeparator); len(segments) >= defaultSegmentLength {
		firstNSegments := strings.Join(segments[:defaultSegmentLength], pathSeparator)
		rest := strings.Join(segments[defaultSegmentLength:], pathSeparator)
		return firstNSegments, rest, true
	}
	return "", "", false
}

func kustRootPathExitsRepo(kustRootPath string) bool {
	cleanedPath := filepath.Clean(strings.TrimPrefix(kustRootPath, string(filepath.Separator)))
	pathElements := strings.Split(cleanedPath, string(filepath.Separator))
	return len(pathElements) > 0 &&
		pathElements[0] == filesys.ParentDir
}

// Clone git submodules by default.
const defaultSubmodules = true

// Arbitrary, but non-infinite, timeout for running commands.
const defaultTimeout = 27 * time.Second

func parseQuery(query string) (string, time.Duration, bool) {
	values, err := url.ParseQuery(query)
	// in event of parse failure, return defaults
	if err != nil {
		return "", defaultTimeout, defaultSubmodules
	}

	// ref is the desired git ref to target. Can be specified by in a git URL
	// with ?ref=<string> or ?version=<string>, although ref takes precedence.
	ref := values.Get("version")
	if queryValue := values.Get("ref"); queryValue != "" {
		ref = queryValue
	}

	// depth is the desired git exec timeout. Can be specified by in a git URL
	// with ?timeout=<duration>.
	duration := defaultTimeout
	if queryValue := values.Get("timeout"); queryValue != "" {
		// Attempt to first parse as a number of integer seconds (like "61"),
		// and then attempt to parse as a suffixed duration (like "61s").
		if intValue, err := strconv.Atoi(queryValue); err == nil && intValue > 0 {
			duration = time.Duration(intValue) * time.Second
		} else if durationValue, err := time.ParseDuration(queryValue); err == nil && durationValue > 0 {
			duration = durationValue
		}
	}

	// submodules indicates if git submodule cloning is desired. Can be
	// specified by in a git URL with ?submodules=<bool>.
	submodules := defaultSubmodules
	if queryValue := values.Get("submodules"); queryValue != "" {
		if boolValue, err := strconv.ParseBool(queryValue); err == nil {
			submodules = boolValue
		}
	}

	return ref, duration, submodules
}

func extractHost(n string) (string, string, error) {
	n = ignoreForcedGitProtocol(n)
	scheme, n := extractScheme(n)
	username, n := extractUsername(n)
	stdGithub := isStandardGithubHost(n)
	acceptSCP := acceptSCPStyle(scheme, username, stdGithub)

	// Validate the username and scheme before attempting host/path parsing, because if the parsing
	// so far has not succeeded, we will not be able to extract the host and path correctly.
	if err := validateScheme(scheme, acceptSCP); err != nil {
		return "", "", err
	}

	// Now that we have extracted a valid scheme+username, we can parse host itself.

	// The file protocol specifies an absolute path to a local git repo.
	// Everything after the scheme (including any 'username' we found) is actually part of that path.
	if scheme == fileScheme {
		return scheme, username + n, nil
	}
	var host, rest = n, ""
	if sepIndex := findPathSeparator(n, acceptSCP); sepIndex >= 0 {
		host, rest = n[:sepIndex+1], n[sepIndex+1:]
	}

	// Github URLs are strictly normalized in a way that may discard scheme and username components.
	if stdGithub {
		scheme, username, host = normalizeGithubHostParts(scheme, username)
	}

	// Host is required, so do not concat the scheme and username if we didn't find one.
	if host == "" {
		return "", "", errors.Errorf("failed to parse host segment")
	}
	return scheme + username + host, rest, nil
}

// ignoreForcedGitProtocol strips the "git::" prefix from URLs.
// We used to use go-getter to handle our urls: https://github.com/hashicorp/go-getter.
// The git:: prefix signaled go-getter to use the git protocol to fetch the url's contents.
// We silently strip this prefix to allow these go-getter-style urls to continue to work,
// although the git protocol (which is insecure and unsupported on many platforms, including Github)
// will not actually be used as intended.
func ignoreForcedGitProtocol(n string) string {
	n, found := trimPrefixIgnoreCase(n, "git::")
	if found {
		log.Println("Warning: Forcing the git protocol using the 'git::' URL prefix is not supported. " +
			"Kustomize currently strips this invalid prefix, but will stop doing so in a future release. " +
			"Please remove the 'git::' prefix from your configuration.")
	}
	return n
}

// acceptSCPStyle returns true if the scheme and username indicate potential use of an SCP-style URL.
// With this style, the scheme is not explicit and the path is delimited by a colon.
// Strictly speaking the username is optional in SCP-like syntax, but Kustomize has always
// required it for non-Github URLs.
// Example: user@host.xz:path/to/repo.git/
func acceptSCPStyle(scheme, username string, isGithubURL bool) bool {
	return scheme == "" && (username != "" || isGithubURL)
}

func validateScheme(scheme string, acceptSCPStyle bool) error {
	// see https://git-scm.com/docs/git-fetch#_git_urls for info relevant to these validations
	switch scheme {
	case "":
		// Empty scheme is only ok if it's a Github URL or if it looks like SCP-style syntax
		if !acceptSCPStyle {
			return fmt.Errorf("failed to parse scheme")
		}
	case sshScheme, fileScheme, httpsScheme, httpScheme:
		// These are all supported schemes
	default:
		// At time of writing, we should never end up here because we do not parse out
		// unsupported schemes to begin with.
		return fmt.Errorf("unsupported scheme %q", scheme)
	}
	return nil
}

const fileScheme = "file://"
const httpScheme = "http://"
const httpsScheme = "https://"
const sshScheme = "ssh://"

func extractScheme(s string) (string, string) {
	for _, prefix := range []string{sshScheme, httpsScheme, httpScheme, fileScheme} {
		if rest, found := trimPrefixIgnoreCase(s, prefix); found {
			return prefix, rest
		}
	}
	return "", s
}

func extractUsername(s string) (string, string) {
	var userRegexp = regexp.MustCompile(`^([a-zA-Z][a-zA-Z0-9-]*)@`)
	if m := userRegexp.FindStringSubmatch(s); m != nil {
		username := m[1] + "@"
		return username, s[len(username):]
	}
	return "", s
}

func isStandardGithubHost(s string) bool {
	lowerCased := strings.ToLower(s)
	return strings.HasPrefix(lowerCased, "github.com/") ||
		strings.HasPrefix(lowerCased, "github.com:")
}

// trimPrefixIgnoreCase returns the rest of s and true if prefix, ignoring case, prefixes s.
// Otherwise, trimPrefixIgnoreCase returns s and false.
func trimPrefixIgnoreCase(s, prefix string) (string, bool) {
	if len(prefix) <= len(s) && strings.ToLower(s[:len(prefix)]) == prefix {
		return s[len(prefix):], true
	}
	return s, false
}

func findPathSeparator(hostPath string, acceptSCP bool) int {
	sepIndex := strings.Index(hostPath, pathSeparator)
	if acceptSCP {
		colonIndex := strings.Index(hostPath, ":")
		// The colon acts as a delimiter in scp-style ssh URLs only if not prefixed by '/'.
		if sepIndex == -1 || (colonIndex > 0 && colonIndex < sepIndex) {
			sepIndex = colonIndex
		}
	}
	return sepIndex
}

func normalizeGithubHostParts(scheme, username string) (string, string, string) {
	if strings.HasPrefix(scheme, sshScheme) || username != "" {
		return "", username, "github.com:"
	}
	return httpsScheme, "", "github.com/"
}
