package processors

import (
	"path/filepath"
	"strings"

	"github.com/pkg/errors"

	"github.com/golangci/golangci-lint/pkg/goutil"
	"github.com/golangci/golangci-lint/pkg/result"
)

type Cgo struct {
	goCacheDir string
}

var _ Processor = Cgo{}

func NewCgo(goenv *goutil.Env) *Cgo {
	return &Cgo{
		goCacheDir: goenv.Get(goutil.EnvGoCache),
	}
}

func (p Cgo) Name() string {
	return "cgo"
}

func (p Cgo) Process(issues []result.Issue) ([]result.Issue, error) {
	return filterIssuesErr(issues, func(i *result.Issue) (bool, error) {
		// some linters (.e.g gosec, deadcode) return incorrect filepaths for cgo issues,
		// also cgo files have strange issues looking like false positives.

		// cache dir contains all preprocessed files including cgo files

		issueFilePath := i.FilePath()
		if !filepath.IsAbs(i.FilePath()) {
			absPath, err := filepath.Abs(i.FilePath())
			if err != nil {
				return false, errors.Wrapf(err, "failed to build abs path for %q", i.FilePath())
			}
			issueFilePath = absPath
		}

		if p.goCacheDir != "" && strings.HasPrefix(issueFilePath, p.goCacheDir) {
			return false, nil
		}

		if filepath.Base(i.FilePath()) == "_cgo_gotypes.go" {
			// skip cgo warning for go1.10
			return false, nil
		}

		return true, nil
	})
}

func (Cgo) Finish() {}
