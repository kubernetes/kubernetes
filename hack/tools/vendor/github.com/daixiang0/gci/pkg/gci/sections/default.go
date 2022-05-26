package sections

import (
	"github.com/daixiang0/gci/pkg/configuration"
	importPkg "github.com/daixiang0/gci/pkg/gci/imports"
	"github.com/daixiang0/gci/pkg/gci/specificity"
)

func init() {
	defaultSectionType := SectionType{
		generatorFun: func(parameter string, sectionPrefix, sectionSuffix Section) (Section, error) {
			return DefaultSection{sectionPrefix, sectionSuffix}, nil
		},
		aliases:     []string{"Def", "Default"},
		description: "Contains all imports that could not be matched to another section type",
	}.WithoutParameter()
	SectionParserInst.registerSectionWithoutErr(&defaultSectionType)
}

type DefaultSection struct {
	Prefix Section
	Suffix Section
}

func (d DefaultSection) sectionPrefix() Section {
	return d.Prefix
}

func (d DefaultSection) sectionSuffix() Section {
	return d.Suffix
}

func (d DefaultSection) MatchSpecificity(spec importPkg.ImportDef) specificity.MatchSpecificity {
	return specificity.Default{}
}

func (d DefaultSection) Format(imports []importPkg.ImportDef, cfg configuration.FormatterConfiguration) string {
	return inorderSectionFormat(d, imports, cfg)
}

func (d DefaultSection) String() string {
	return sectionStringWithPrefixSuffix("Default", d)
}
