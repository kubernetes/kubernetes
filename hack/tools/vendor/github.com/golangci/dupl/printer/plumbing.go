package printer

import (
	"sort"

	"github.com/golangci/dupl/syntax"
)

type Clone clone

func (c Clone) Filename() string {
	return c.filename
}

func (c Clone) LineStart() int {
	return c.lineStart
}

func (c Clone) LineEnd() int {
	return c.lineEnd
}

type Issue struct {
	From, To Clone
}

type Plumbing struct {
	ReadFile
}

func NewPlumbing(fread ReadFile) *Plumbing {
	return &Plumbing{fread}
}

func (p *Plumbing) MakeIssues(dups [][]*syntax.Node) ([]Issue, error) {
	clones, err := prepareClonesInfo(p.ReadFile, dups)
	if err != nil {
		return nil, err
	}
	sort.Sort(byNameAndLine(clones))
	var issues []Issue
	for i, cl := range clones {
		nextCl := clones[(i+1)%len(clones)]
		issues = append(issues, Issue{
			From: Clone(cl),
			To:   Clone(nextCl),
		})
	}
	return issues, nil
}
