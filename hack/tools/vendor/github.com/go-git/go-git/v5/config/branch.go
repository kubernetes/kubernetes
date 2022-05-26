package config

import (
	"errors"

	"github.com/go-git/go-git/v5/plumbing"
	format "github.com/go-git/go-git/v5/plumbing/format/config"
)

var (
	errBranchEmptyName     = errors.New("branch config: empty name")
	errBranchInvalidMerge  = errors.New("branch config: invalid merge")
	errBranchInvalidRebase = errors.New("branch config: rebase must be one of 'true' or 'interactive'")
)

// Branch contains information on the
// local branches and which remote to track
type Branch struct {
	// Name of branch
	Name string
	// Remote name of remote to track
	Remote string
	// Merge is the local refspec for the branch
	Merge plumbing.ReferenceName
	// Rebase instead of merge when pulling. Valid values are
	// "true" and "interactive".  "false" is undocumented and
	// typically represented by the non-existence of this field
	Rebase string

	raw *format.Subsection
}

// Validate validates fields of branch
func (b *Branch) Validate() error {
	if b.Name == "" {
		return errBranchEmptyName
	}

	if b.Merge != "" && !b.Merge.IsBranch() {
		return errBranchInvalidMerge
	}

	if b.Rebase != "" &&
		b.Rebase != "true" &&
		b.Rebase != "interactive" &&
		b.Rebase != "false" {
		return errBranchInvalidRebase
	}

	return nil
}

func (b *Branch) marshal() *format.Subsection {
	if b.raw == nil {
		b.raw = &format.Subsection{}
	}

	b.raw.Name = b.Name

	if b.Remote == "" {
		b.raw.RemoveOption(remoteSection)
	} else {
		b.raw.SetOption(remoteSection, b.Remote)
	}

	if b.Merge == "" {
		b.raw.RemoveOption(mergeKey)
	} else {
		b.raw.SetOption(mergeKey, string(b.Merge))
	}

	if b.Rebase == "" {
		b.raw.RemoveOption(rebaseKey)
	} else {
		b.raw.SetOption(rebaseKey, b.Rebase)
	}

	return b.raw
}

func (b *Branch) unmarshal(s *format.Subsection) error {
	b.raw = s

	b.Name = b.raw.Name
	b.Remote = b.raw.Options.Get(remoteSection)
	b.Merge = plumbing.ReferenceName(b.raw.Options.Get(mergeKey))
	b.Rebase = b.raw.Options.Get(rebaseKey)

	return b.Validate()
}
