package processors

import (
	"path"

	"github.com/golangci/golangci-lint/pkg/result"
)

// PathPrefixer adds a customizable prefix to every output path
type PathPrefixer struct {
	prefix string
}

var _ Processor = new(PathPrefixer)

// NewPathPrefixer returns a new path prefixer for the provided string
func NewPathPrefixer(prefix string) *PathPrefixer {
	return &PathPrefixer{prefix: prefix}
}

// Name returns the name of this processor
func (*PathPrefixer) Name() string {
	return "path_prefixer"
}

// Process adds the prefix to each path
func (p *PathPrefixer) Process(issues []result.Issue) ([]result.Issue, error) {
	if p.prefix != "" {
		for i := range issues {
			issues[i].Pos.Filename = path.Join(p.prefix, issues[i].Pos.Filename)
		}
	}
	return issues, nil
}

// Finish is implemented to satisfy the Processor interface
func (*PathPrefixer) Finish() {}
