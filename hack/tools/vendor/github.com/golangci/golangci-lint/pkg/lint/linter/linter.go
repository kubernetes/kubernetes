package linter

import (
	"context"

	"golang.org/x/tools/go/analysis"

	"github.com/golangci/golangci-lint/pkg/result"
)

type Linter interface {
	Run(ctx context.Context, lintCtx *Context) ([]result.Issue, error)
	Name() string
	Desc() string
}

type Noop struct {
	name string
	desc string
	run  func(pass *analysis.Pass) (interface{}, error)
}

func (n Noop) Run(_ context.Context, lintCtx *Context) ([]result.Issue, error) {
	lintCtx.Log.Warnf("%s is disabled because of go1.18."+
		" If you are not using go1.18, you can set `go: go1.17` in the `run` section."+
		" You can track the evolution of the go1.18 support by following the https://github.com/golangci/golangci-lint/issues/2649.", n.name)
	return nil, nil
}

func (n Noop) Name() string {
	return n.name
}

func (n Noop) Desc() string {
	return n.desc
}
