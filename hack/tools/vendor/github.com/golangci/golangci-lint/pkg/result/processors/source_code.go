package processors

import (
	"github.com/golangci/golangci-lint/pkg/fsutils"
	"github.com/golangci/golangci-lint/pkg/logutils"
	"github.com/golangci/golangci-lint/pkg/result"
)

type SourceCode struct {
	lineCache *fsutils.LineCache
	log       logutils.Log
}

var _ Processor = SourceCode{}

func NewSourceCode(lc *fsutils.LineCache, log logutils.Log) *SourceCode {
	return &SourceCode{
		lineCache: lc,
		log:       log,
	}
}

func (p SourceCode) Name() string {
	return "source_code"
}

func (p SourceCode) Process(issues []result.Issue) ([]result.Issue, error) {
	return transformIssues(issues, func(i *result.Issue) *result.Issue {
		newI := *i

		lineRange := i.GetLineRange()
		for lineNumber := lineRange.From; lineNumber <= lineRange.To; lineNumber++ {
			line, err := p.lineCache.GetLine(i.FilePath(), lineNumber)
			if err != nil {
				p.log.Warnf("Failed to get line %d for file %s: %s",
					lineNumber, i.FilePath(), err)
				return i
			}

			newI.SourceLines = append(newI.SourceLines, line)
		}

		return &newI
	}), nil
}

func (p SourceCode) Finish() {}
