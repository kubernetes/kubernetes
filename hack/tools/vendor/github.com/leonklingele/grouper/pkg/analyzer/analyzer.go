package analyzer

import (
	"fmt"
	"go/ast"

	"github.com/leonklingele/grouper/pkg/analyzer/consts"
	"github.com/leonklingele/grouper/pkg/analyzer/imports"
	"github.com/leonklingele/grouper/pkg/analyzer/types"
	"github.com/leonklingele/grouper/pkg/analyzer/vars"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
)

const (
	Name = "grouper"
	Doc  = `expression group analyzer: require 'import', 'const', 'var' and/or 'type' declaration groups`
)

func New() *analysis.Analyzer {
	return &analysis.Analyzer{ //nolint:exhaustivestruct // we do not need all fields
		Name:     Name,
		Doc:      Doc,
		Flags:    Flags(),
		Run:      run,
		Requires: []*analysis.Analyzer{inspect.Analyzer},
	}
}

func run(p *analysis.Pass) (interface{}, error) {
	flagLookupBool := func(name string) bool {
		return p.Analyzer.Flags.Lookup(name).Value.String() == "true"
	}

	c := &Config{
		ConstsConfig: &consts.Config{
			RequireSingleConst: flagLookupBool(FlagNameConstRequireSingleConst),
			RequireGrouping:    flagLookupBool(FlagNameConstRequireGrouping),
		},

		ImportsConfig: &imports.Config{
			RequireSingleImport: flagLookupBool(FlagNameImportRequireSingleImport),
			RequireGrouping:     flagLookupBool(FlagNameImportRequireGrouping),
		},

		TypesConfig: &types.Config{
			RequireSingleType: flagLookupBool(FlagNameTypeRequireSingleType),
			RequireGrouping:   flagLookupBool(FlagNameTypeRequireGrouping),
		},

		VarsConfig: &vars.Config{
			RequireSingleVar: flagLookupBool(FlagNameVarRequireSingleVar),
			RequireGrouping:  flagLookupBool(FlagNameVarRequireGrouping),
		},
	}

	return nil, pass(c, p)
}

func pass(c *Config, p *analysis.Pass) error {
	for _, f := range p.Files {
		if err := filepass(c, p, f); err != nil {
			return err
		}
	}

	return nil
}

func filepass(c *Config, p *analysis.Pass, f *ast.File) error {
	if err := consts.Filepass(c.ConstsConfig, p, f); err != nil {
		return fmt.Errorf("failed to consts.Filepass: %w", err)
	}

	if err := imports.Filepass(c.ImportsConfig, p, f); err != nil {
		return fmt.Errorf("failed to imports.Filepass: %w", err)
	}

	if err := types.Filepass(c.TypesConfig, p, f); err != nil {
		return fmt.Errorf("failed to types.Filepass: %w", err)
	}

	if err := vars.Filepass(c.VarsConfig, p, f); err != nil {
		return fmt.Errorf("failed to vars.Filepass: %w", err)
	}

	return nil
}
