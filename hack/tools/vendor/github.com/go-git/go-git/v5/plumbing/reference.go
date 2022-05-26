package plumbing

import (
	"errors"
	"fmt"
	"strings"
)

const (
	refPrefix       = "refs/"
	refHeadPrefix   = refPrefix + "heads/"
	refTagPrefix    = refPrefix + "tags/"
	refRemotePrefix = refPrefix + "remotes/"
	refNotePrefix   = refPrefix + "notes/"
	symrefPrefix    = "ref: "
)

// RefRevParseRules are a set of rules to parse references into short names.
// These are the same rules as used by git in shorten_unambiguous_ref.
// See: https://github.com/git/git/blob/e0aaa1b6532cfce93d87af9bc813fb2e7a7ce9d7/refs.c#L417
var RefRevParseRules = []string{
	"refs/%s",
	"refs/tags/%s",
	"refs/heads/%s",
	"refs/remotes/%s",
	"refs/remotes/%s/HEAD",
}

var (
	ErrReferenceNotFound = errors.New("reference not found")
)

// ReferenceType reference type's
type ReferenceType int8

const (
	InvalidReference  ReferenceType = 0
	HashReference     ReferenceType = 1
	SymbolicReference ReferenceType = 2
)

func (r ReferenceType) String() string {
	switch r {
	case InvalidReference:
		return "invalid-reference"
	case HashReference:
		return "hash-reference"
	case SymbolicReference:
		return "symbolic-reference"
	}

	return ""
}

// ReferenceName reference name's
type ReferenceName string

// NewBranchReferenceName returns a reference name describing a branch based on
// his short name.
func NewBranchReferenceName(name string) ReferenceName {
	return ReferenceName(refHeadPrefix + name)
}

// NewNoteReferenceName returns a reference name describing a note based on his
// short name.
func NewNoteReferenceName(name string) ReferenceName {
	return ReferenceName(refNotePrefix + name)
}

// NewRemoteReferenceName returns a reference name describing a remote branch
// based on his short name and the remote name.
func NewRemoteReferenceName(remote, name string) ReferenceName {
	return ReferenceName(refRemotePrefix + fmt.Sprintf("%s/%s", remote, name))
}

// NewRemoteHEADReferenceName returns a reference name describing a the HEAD
// branch of a remote.
func NewRemoteHEADReferenceName(remote string) ReferenceName {
	return ReferenceName(refRemotePrefix + fmt.Sprintf("%s/%s", remote, HEAD))
}

// NewTagReferenceName returns a reference name describing a tag based on short
// his name.
func NewTagReferenceName(name string) ReferenceName {
	return ReferenceName(refTagPrefix + name)
}

// IsBranch check if a reference is a branch
func (r ReferenceName) IsBranch() bool {
	return strings.HasPrefix(string(r), refHeadPrefix)
}

// IsNote check if a reference is a note
func (r ReferenceName) IsNote() bool {
	return strings.HasPrefix(string(r), refNotePrefix)
}

// IsRemote check if a reference is a remote
func (r ReferenceName) IsRemote() bool {
	return strings.HasPrefix(string(r), refRemotePrefix)
}

// IsTag check if a reference is a tag
func (r ReferenceName) IsTag() bool {
	return strings.HasPrefix(string(r), refTagPrefix)
}

func (r ReferenceName) String() string {
	return string(r)
}

// Short returns the short name of a ReferenceName
func (r ReferenceName) Short() string {
	s := string(r)
	res := s
	for _, format := range RefRevParseRules {
		_, err := fmt.Sscanf(s, format, &res)
		if err == nil {
			continue
		}
	}

	return res
}

const (
	HEAD   ReferenceName = "HEAD"
	Master ReferenceName = "refs/heads/master"
)

// Reference is a representation of git reference
type Reference struct {
	t      ReferenceType
	n      ReferenceName
	h      Hash
	target ReferenceName
}

// NewReferenceFromStrings creates a reference from name and target as string,
// the resulting reference can be a SymbolicReference or a HashReference base
// on the target provided
func NewReferenceFromStrings(name, target string) *Reference {
	n := ReferenceName(name)

	if strings.HasPrefix(target, symrefPrefix) {
		target := ReferenceName(target[len(symrefPrefix):])
		return NewSymbolicReference(n, target)
	}

	return NewHashReference(n, NewHash(target))
}

// NewSymbolicReference creates a new SymbolicReference reference
func NewSymbolicReference(n, target ReferenceName) *Reference {
	return &Reference{
		t:      SymbolicReference,
		n:      n,
		target: target,
	}
}

// NewHashReference creates a new HashReference reference
func NewHashReference(n ReferenceName, h Hash) *Reference {
	return &Reference{
		t: HashReference,
		n: n,
		h: h,
	}
}

// Type return the type of a reference
func (r *Reference) Type() ReferenceType {
	return r.t
}

// Name return the name of a reference
func (r *Reference) Name() ReferenceName {
	return r.n
}

// Hash return the hash of a hash reference
func (r *Reference) Hash() Hash {
	return r.h
}

// Target return the target of a symbolic reference
func (r *Reference) Target() ReferenceName {
	return r.target
}

// Strings dump a reference as a [2]string
func (r *Reference) Strings() [2]string {
	var o [2]string
	o[0] = r.Name().String()

	switch r.Type() {
	case HashReference:
		o[1] = r.Hash().String()
	case SymbolicReference:
		o[1] = symrefPrefix + r.Target().String()
	}

	return o
}

func (r *Reference) String() string {
	s := r.Strings()
	return fmt.Sprintf("%s %s", s[1], s[0])
}
