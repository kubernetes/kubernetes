package sections

import (
	"github.com/daixiang0/gci/pkg/configuration"
	"github.com/daixiang0/gci/pkg/constants"
	importPkg "github.com/daixiang0/gci/pkg/gci/imports"
	"github.com/daixiang0/gci/pkg/gci/specificity"
)

func init() {
	newLineType := SectionType{
		generatorFun: func(parameter string, sectionPrefix, sectionSuffix Section) (Section, error) {
			return NewLine{}, nil
		},
		aliases:     []string{"NL", "NewLine"},
		description: "Prints an empty line",
	}.StandAloneSection().WithoutParameter()
	SectionParserInst.registerSectionWithoutErr(&newLineType)
}

type NewLine struct{}

func (n NewLine) sectionPrefix() Section {
	return nil
}

func (n NewLine) sectionSuffix() Section {
	return nil
}

func (n NewLine) MatchSpecificity(spec importPkg.ImportDef) specificity.MatchSpecificity {
	return specificity.MisMatch{}
}

func (n NewLine) Format(imports []importPkg.ImportDef, cfg configuration.FormatterConfiguration) string {
	return constants.Linebreak
}

func (n NewLine) String() string {
	return "NewLine"
}
