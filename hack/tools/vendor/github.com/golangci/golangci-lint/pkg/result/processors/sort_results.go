package processors

import (
	"sort"
	"strings"

	"github.com/golangci/golangci-lint/pkg/config"
	"github.com/golangci/golangci-lint/pkg/result"
)

// Base propose of this functionality to sort results (issues)
// produced by various linters by analyzing code. We're achieving this
// by sorting results.Issues using processor step, and chain based
// rules that can compare different properties of the Issues struct.

var _ Processor = (*SortResults)(nil)

type SortResults struct {
	cmp comparator
	cfg *config.Config
}

func NewSortResults(cfg *config.Config) *SortResults {
	// For sorting we are comparing (in next order): file names, line numbers,
	// position, and finally - giving up.
	return &SortResults{
		cmp: ByName{
			next: ByLine{
				next: ByColumn{},
			},
		},
		cfg: cfg,
	}
}

// Process is performing sorting of the result issues.
func (sr SortResults) Process(issues []result.Issue) ([]result.Issue, error) {
	if !sr.cfg.Output.SortResults {
		return issues, nil
	}

	sort.Slice(issues, func(i, j int) bool {
		return sr.cmp.Compare(&issues[i], &issues[j]) == Less
	})

	return issues, nil
}

func (sr SortResults) Name() string { return "sort_results" }
func (sr SortResults) Finish()      {}

type compareResult int

const (
	Less compareResult = iota - 1
	Equal
	Greater
	None
)

func (c compareResult) isNeutral() bool {
	// return true if compare result is incomparable or equal.
	return c == None || c == Equal
}

func (c compareResult) String() string {
	switch c {
	case Less:
		return "Less"
	case Equal:
		return "Equal"
	case Greater:
		return "Greater"
	}

	return "None"
}

// comparator describe how to implement compare for two "issues" lexicographically
type comparator interface {
	Compare(a, b *result.Issue) compareResult
	Next() comparator
}

var (
	_ comparator = (*ByName)(nil)
	_ comparator = (*ByLine)(nil)
	_ comparator = (*ByColumn)(nil)
)

type ByName struct{ next comparator }

//nolint:golint
func (cmp ByName) Next() comparator { return cmp.next }

//nolint:golint
func (cmp ByName) Compare(a, b *result.Issue) compareResult {
	var res compareResult

	if res = compareResult(strings.Compare(a.FilePath(), b.FilePath())); !res.isNeutral() {
		return res
	}

	if next := cmp.Next(); next != nil {
		return next.Compare(a, b)
	}

	return res
}

type ByLine struct{ next comparator }

//nolint:golint
func (cmp ByLine) Next() comparator { return cmp.next }

//nolint:golint
func (cmp ByLine) Compare(a, b *result.Issue) compareResult {
	var res compareResult

	if res = numericCompare(a.Line(), b.Line()); !res.isNeutral() {
		return res
	}

	if next := cmp.Next(); next != nil {
		return next.Compare(a, b)
	}

	return res
}

type ByColumn struct{ next comparator }

//nolint:golint
func (cmp ByColumn) Next() comparator { return cmp.next }

//nolint:golint
func (cmp ByColumn) Compare(a, b *result.Issue) compareResult {
	var res compareResult

	if res = numericCompare(a.Column(), b.Column()); !res.isNeutral() {
		return res
	}

	if next := cmp.Next(); next != nil {
		return next.Compare(a, b)
	}

	return res
}

func numericCompare(a, b int) compareResult {
	var (
		isValuesInvalid  = a < 0 || b < 0
		isZeroValuesBoth = a == 0 && b == 0
		isEqual          = a == b
		isZeroValueInA   = b > 0 && a == 0
		isZeroValueInB   = a > 0 && b == 0
	)

	switch {
	case isZeroValuesBoth || isEqual:
		return Equal
	case isValuesInvalid || isZeroValueInA || isZeroValueInB:
		return None
	case a > b:
		return Greater
	case a < b:
		return Less
	}

	return Equal
}
