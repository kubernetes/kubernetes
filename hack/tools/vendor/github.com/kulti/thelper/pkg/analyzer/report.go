package analyzer

import (
	"go/token"

	"golang.org/x/tools/go/analysis"
)

type reports struct {
	reports  []report
	filter   map[token.Pos]struct{}
	nofilter map[token.Pos]struct{}
}

type report struct {
	pos    token.Pos
	format string
	args   []interface{}
}

func (rr *reports) Reportf(pos token.Pos, format string, args ...interface{}) {
	rr.reports = append(rr.reports, report{
		pos:    pos,
		format: format,
		args:   args,
	})
}

func (rr *reports) Filter(pos token.Pos) {
	if pos.IsValid() {
		if rr.filter == nil {
			rr.filter = make(map[token.Pos]struct{})
		}
		rr.filter[pos] = struct{}{}
	}
}

func (rr *reports) NoFilter(pos token.Pos) {
	if pos.IsValid() {
		if rr.nofilter == nil {
			rr.nofilter = make(map[token.Pos]struct{})
		}
		rr.nofilter[pos] = struct{}{}
	}
}

func (rr reports) Flush(pass *analysis.Pass) {
	for _, r := range rr.reports {
		if _, ok := rr.filter[r.pos]; ok {
			if _, ok := rr.nofilter[r.pos]; !ok {
				continue
			}
		}
		pass.Reportf(r.pos, r.format, r.args...)
	}
}
