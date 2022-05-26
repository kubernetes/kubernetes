package config

import (
	"errors"
	"strings"

	"github.com/go-git/go-git/v5/plumbing"
)

const (
	refSpecWildcard  = "*"
	refSpecForce     = "+"
	refSpecSeparator = ":"
)

var (
	ErrRefSpecMalformedSeparator = errors.New("malformed refspec, separators are wrong")
	ErrRefSpecMalformedWildcard  = errors.New("malformed refspec, mismatched number of wildcards")
)

// RefSpec is a mapping from local branches to remote references.
// The format of the refspec is an optional +, followed by <src>:<dst>, where
// <src> is the pattern for references on the remote side and <dst> is where
// those references will be written locally. The + tells Git to update the
// reference even if it isn’t a fast-forward.
// eg.: "+refs/heads/*:refs/remotes/origin/*"
//
// https://git-scm.com/book/en/v2/Git-Internals-The-Refspec
type RefSpec string

// Validate validates the RefSpec
func (s RefSpec) Validate() error {
	spec := string(s)
	if strings.Count(spec, refSpecSeparator) != 1 {
		return ErrRefSpecMalformedSeparator
	}

	sep := strings.Index(spec, refSpecSeparator)
	if sep == len(spec)-1 {
		return ErrRefSpecMalformedSeparator
	}

	ws := strings.Count(spec[0:sep], refSpecWildcard)
	wd := strings.Count(spec[sep+1:], refSpecWildcard)
	if ws == wd && ws < 2 && wd < 2 {
		return nil
	}

	return ErrRefSpecMalformedWildcard
}

// IsForceUpdate returns if update is allowed in non fast-forward merges.
func (s RefSpec) IsForceUpdate() bool {
	return s[0] == refSpecForce[0]
}

// IsDelete returns true if the refspec indicates a delete (empty src).
func (s RefSpec) IsDelete() bool {
	return s[0] == refSpecSeparator[0]
}

// IsExactSHA1 returns true if the source is a SHA1 hash.
func (s RefSpec) IsExactSHA1() bool {
	return plumbing.IsHash(s.Src())
}

// Src return the src side.
func (s RefSpec) Src() string {
	spec := string(s)

	var start int
	if s.IsForceUpdate() {
		start = 1
	} else {
		start = 0
	}

	end := strings.Index(spec, refSpecSeparator)
	return spec[start:end]
}

// Match match the given plumbing.ReferenceName against the source.
func (s RefSpec) Match(n plumbing.ReferenceName) bool {
	if !s.IsWildcard() {
		return s.matchExact(n)
	}

	return s.matchGlob(n)
}

// IsWildcard returns true if the RefSpec contains a wildcard.
func (s RefSpec) IsWildcard() bool {
	return strings.Contains(string(s), refSpecWildcard)
}

func (s RefSpec) matchExact(n plumbing.ReferenceName) bool {
	return s.Src() == n.String()
}

func (s RefSpec) matchGlob(n plumbing.ReferenceName) bool {
	src := s.Src()
	name := n.String()
	wildcard := strings.Index(src, refSpecWildcard)

	var prefix, suffix string
	prefix = src[0:wildcard]
	if len(src) > wildcard+1 {
		suffix = src[wildcard+1:]
	}

	return len(name) >= len(prefix)+len(suffix) &&
		strings.HasPrefix(name, prefix) &&
		strings.HasSuffix(name, suffix)
}

// Dst returns the destination for the given remote reference.
func (s RefSpec) Dst(n plumbing.ReferenceName) plumbing.ReferenceName {
	spec := string(s)
	start := strings.Index(spec, refSpecSeparator) + 1
	dst := spec[start:]
	src := s.Src()

	if !s.IsWildcard() {
		return plumbing.ReferenceName(dst)
	}

	name := n.String()
	ws := strings.Index(src, refSpecWildcard)
	wd := strings.Index(dst, refSpecWildcard)
	match := name[ws : len(name)-(len(src)-(ws+1))]

	return plumbing.ReferenceName(dst[0:wd] + match + dst[wd+1:])
}

func (s RefSpec) Reverse() RefSpec {
	spec := string(s)
	separator := strings.Index(spec, refSpecSeparator)

	return RefSpec(spec[separator+1:] + refSpecSeparator + spec[:separator])
}

func (s RefSpec) String() string {
	return string(s)
}

// MatchAny returns true if any of the RefSpec match with the given ReferenceName.
func MatchAny(l []RefSpec, n plumbing.ReferenceName) bool {
	for _, r := range l {
		if r.Match(n) {
			return true
		}
	}

	return false
}
