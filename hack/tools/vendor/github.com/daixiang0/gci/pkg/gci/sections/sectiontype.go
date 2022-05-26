package sections

import (
	"fmt"
)

// A SectionType is used to dynamically register Sections with the parser
type SectionType struct {
	generatorFun  func(parameter string, sectionPrefix, sectionSuffix Section) (Section, error)
	aliases       []string
	parameterHelp string
	description   string
}

func (t SectionType) WithoutParameter() SectionType {
	generatorFun := func(parameter string, sectionPrefix, sectionSuffix Section) (Section, error) {
		if parameter != "" {
			return nil, SectionTypeDoesNotAcceptParametersError
		}
		return t.generatorFun(parameter, sectionPrefix, sectionSuffix)
	}
	return SectionType{generatorFun, t.aliases, "", t.description}
}

func (t SectionType) StandAloneSection() SectionType {
	generatorFun := func(parameter string, sectionPrefix, sectionSuffix Section) (Section, error) {
		if sectionPrefix != nil {
			return nil, SectionTypeDoesNotAcceptPrefixError
		}
		if sectionSuffix != nil {
			return nil, SectionTypeDoesNotAcceptSuffixError
		}
		return t.generatorFun(parameter, sectionPrefix, sectionSuffix)
	}
	return SectionType{generatorFun, t.aliases, t.parameterHelp, t.description}
}

func (t SectionType) String() string {
	return fmt.Sprintf("Sectiontype(aliases: %v,description: %s)", t.aliases, t.description)
}
