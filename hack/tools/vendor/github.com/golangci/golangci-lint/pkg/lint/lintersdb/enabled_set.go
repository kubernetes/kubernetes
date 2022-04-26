package lintersdb

import (
	"os"
	"sort"
	"strings"

	"github.com/golangci/golangci-lint/pkg/config"
	"github.com/golangci/golangci-lint/pkg/golinters/goanalysis"
	"github.com/golangci/golangci-lint/pkg/lint/linter"
	"github.com/golangci/golangci-lint/pkg/logutils"
)

type EnabledSet struct {
	m      *Manager
	v      *Validator
	log    logutils.Log
	cfg    *config.Config
	debugf logutils.DebugFunc
}

func NewEnabledSet(m *Manager, v *Validator, log logutils.Log, cfg *config.Config) *EnabledSet {
	return &EnabledSet{
		m:      m,
		v:      v,
		log:    log,
		cfg:    cfg,
		debugf: logutils.Debug("enabled_linters"),
	}
}

func (es EnabledSet) build(lcfg *config.Linters, enabledByDefaultLinters []*linter.Config) map[string]*linter.Config {
	es.debugf("Linters config: %#v", lcfg)
	resultLintersSet := map[string]*linter.Config{}
	switch {
	case len(lcfg.Presets) != 0:
		break // imply --disable-all
	case lcfg.EnableAll:
		resultLintersSet = linterConfigsToMap(es.m.GetAllSupportedLinterConfigs())
	case lcfg.DisableAll:
		break
	default:
		resultLintersSet = linterConfigsToMap(enabledByDefaultLinters)
	}

	// --presets can only add linters to default set
	for _, p := range lcfg.Presets {
		for _, lc := range es.m.GetAllLinterConfigsForPreset(p) {
			lc := lc
			resultLintersSet[lc.Name()] = lc
		}
	}

	// --fast removes slow linters from current set.
	// It should be after --presets to be able to run only fast linters in preset.
	// It should be before --enable and --disable to be able to enable or disable specific linter.
	if lcfg.Fast {
		for name, lc := range resultLintersSet {
			if lc.IsSlowLinter() {
				delete(resultLintersSet, name)
			}
		}
	}

	for _, name := range lcfg.Enable {
		for _, lc := range es.m.GetLinterConfigs(name) {
			// it's important to use lc.Name() nor name because name can be alias
			resultLintersSet[lc.Name()] = lc
		}
	}

	for _, name := range lcfg.Disable {
		for _, lc := range es.m.GetLinterConfigs(name) {
			// it's important to use lc.Name() nor name because name can be alias
			delete(resultLintersSet, lc.Name())
		}
	}

	return resultLintersSet
}

func (es EnabledSet) GetEnabledLintersMap() (map[string]*linter.Config, error) {
	if err := es.v.validateEnabledDisabledLintersConfig(&es.cfg.Linters); err != nil {
		return nil, err
	}

	enabledLinters := es.build(&es.cfg.Linters, es.m.GetAllEnabledByDefaultLinters())
	if os.Getenv("GL_TEST_RUN") == "1" {
		es.verbosePrintLintersStatus(enabledLinters)
	}
	return enabledLinters, nil
}

// GetOptimizedLinters returns enabled linters after optimization (merging) of multiple linters
// into a fewer number of linters. E.g. some go/analysis linters can be optimized into
// one metalinter for data reuse and speed up.
func (es EnabledSet) GetOptimizedLinters() ([]*linter.Config, error) {
	if err := es.v.validateEnabledDisabledLintersConfig(&es.cfg.Linters); err != nil {
		return nil, err
	}

	resultLintersSet := es.build(&es.cfg.Linters, es.m.GetAllEnabledByDefaultLinters())
	es.verbosePrintLintersStatus(resultLintersSet)
	es.combineGoAnalysisLinters(resultLintersSet)

	var resultLinters []*linter.Config
	for _, lc := range resultLintersSet {
		resultLinters = append(resultLinters, lc)
	}

	// Make order of execution of linters (go/analysis metalinter and unused) stable.
	sort.Slice(resultLinters, func(i, j int) bool {
		a, b := resultLinters[i], resultLinters[j]

		if b.Name() == linter.LastLinter {
			return true
		}

		if a.Name() == linter.LastLinter {
			return false
		}

		if a.DoesChangeTypes != b.DoesChangeTypes {
			return b.DoesChangeTypes // move type-changing linters to the end to optimize speed
		}
		return strings.Compare(a.Name(), b.Name()) < 0
	})

	return resultLinters, nil
}

func (es EnabledSet) combineGoAnalysisLinters(linters map[string]*linter.Config) {
	var goanalysisLinters []*goanalysis.Linter
	goanalysisPresets := map[string]bool{}
	for _, linter := range linters {
		lnt, ok := linter.Linter.(*goanalysis.Linter)
		if !ok {
			continue
		}
		if lnt.LoadMode() == goanalysis.LoadModeWholeProgram {
			// It's ineffective by CPU and memory to run whole-program and incremental analyzers at once.
			continue
		}
		goanalysisLinters = append(goanalysisLinters, lnt)
		for _, p := range linter.InPresets {
			goanalysisPresets[p] = true
		}
	}

	if len(goanalysisLinters) <= 1 {
		es.debugf("Didn't combine go/analysis linters: got only %d linters", len(goanalysisLinters))
		return
	}

	for _, lnt := range goanalysisLinters {
		delete(linters, lnt.Name())
	}

	// Make order of execution of go/analysis analyzers stable.
	sort.Slice(goanalysisLinters, func(i, j int) bool {
		a, b := goanalysisLinters[i], goanalysisLinters[j]

		if b.Name() == linter.LastLinter {
			return true
		}

		if a.Name() == linter.LastLinter {
			return false
		}

		return strings.Compare(a.Name(), b.Name()) <= 0
	})

	ml := goanalysis.NewMetaLinter(goanalysisLinters)

	var presets []string
	for p := range goanalysisPresets {
		presets = append(presets, p)
	}

	mlConfig := &linter.Config{
		Linter:           ml,
		EnabledByDefault: false,
		InPresets:        presets,
		AlternativeNames: nil,
		OriginalURL:      "",
	}

	mlConfig = mlConfig.WithLoadForGoAnalysis()

	linters[ml.Name()] = mlConfig
	es.debugf("Combined %d go/analysis linters into one metalinter", len(goanalysisLinters))
}

func (es EnabledSet) verbosePrintLintersStatus(lcs map[string]*linter.Config) {
	var linterNames []string
	for _, lc := range lcs {
		linterNames = append(linterNames, lc.Name())
	}
	sort.StringSlice(linterNames).Sort()
	es.log.Infof("Active %d linters: %s", len(linterNames), linterNames)

	if len(es.cfg.Linters.Presets) != 0 {
		sort.StringSlice(es.cfg.Linters.Presets).Sort()
		es.log.Infof("Active presets: %s", es.cfg.Linters.Presets)
	}
}
