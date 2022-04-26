package sections

import (
	"fmt"
	"strings"

	"github.com/daixiang0/gci/pkg/configuration"
	importPkg "github.com/daixiang0/gci/pkg/gci/imports"
	"github.com/daixiang0/gci/pkg/gci/specificity"
)

func init() {
	prefixType := &SectionType{
		generatorFun: func(parameter string, sectionPrefix, sectionSuffix Section) (Section, error) {
			return Prefix{parameter, sectionPrefix, sectionSuffix}, nil
		},
		aliases:       []string{"Prefix", "pkgPrefix"},
		parameterHelp: "gitlab.com/myorg",
		description:   "Groups all imports with the specified Prefix. Imports will be matched to the longest Prefix.",
	}
	SectionParserInst.registerSectionWithoutErr(prefixType)
}

type Prefix struct {
	ImportPrefix string
	Prefix       Section
	Suffix       Section
}

func (p Prefix) sectionPrefix() Section {
	return p.Prefix
}

func (p Prefix) sectionSuffix() Section {
	return p.Suffix
}

func (p Prefix) MatchSpecificity(spec importPkg.ImportDef) specificity.MatchSpecificity {
	if len(p.ImportPrefix) > 0 && strings.HasPrefix(spec.Path(), p.ImportPrefix) {
		return specificity.Match{len(p.ImportPrefix)}
	}
	return specificity.MisMatch{}
}

func (p Prefix) Format(imports []importPkg.ImportDef, cfg configuration.FormatterConfiguration) string {
	return inorderSectionFormat(p, imports, cfg)
}

func (p Prefix) String() string {
	return sectionStringWithPrefixSuffix(fmt.Sprintf("Prefix(%s)", p.ImportPrefix), p)
}
