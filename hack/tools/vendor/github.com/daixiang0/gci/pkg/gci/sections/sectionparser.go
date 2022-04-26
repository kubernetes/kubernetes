package sections

import (
	"fmt"
	"strings"

	"github.com/daixiang0/gci/pkg/constants"
)

var SectionParserInst = SectionParser{}

type SectionParser struct {
	sectionTypes []SectionType
}

func (s *SectionParser) RegisterSection(newSectionType *SectionType) error {
	for _, existingSectionType := range s.sectionTypes {
		for _, alias := range existingSectionType.aliases {
			for _, newAlias := range newSectionType.aliases {
				if alias == newAlias {
					return TypeAlreadyRegisteredError{alias, *newSectionType, existingSectionType}
				}
			}
		}
	}
	s.sectionTypes = append(s.sectionTypes, *newSectionType)
	return nil
}

func (s *SectionParser) registerSectionWithoutErr(newSectionType *SectionType) {
	err := s.RegisterSection(newSectionType)
	if err != nil {
		panic(err)
	}
}

func (s *SectionParser) ParseSectionStrings(sectionStrings []string, withSuffix, withPrefix bool) ([]Section, error) {
	var parsedSections []Section
	for _, sectionStr := range sectionStrings {
		section, err := s.parseSectionString(sectionStr, withSuffix, withPrefix)
		if err != nil {
			return nil, SectionParsingError{err}.Wrap(sectionStr)
		}
		parsedSections = append(parsedSections, section)
	}
	return parsedSections, nil
}

func (s *SectionParser) parseSectionString(sectionStr string, withSuffix, withPrefix bool) (Section, error) {
	trimmedSection := strings.TrimSpace(sectionStr)
	sectionSegments := strings.Split(trimmedSection, constants.SectionSeparator)
	switch len(sectionSegments) {
	case 1:
		// section
		return s.parseSectionStringComponents("", sectionSegments[0], "")
	case 2:
		// prefix + section
		if !withPrefix {
			return nil, PrefixNotAllowedError
		}
		return s.parseSectionStringComponents(sectionSegments[0], sectionSegments[1], "")
	case 3:
		// prefix + section + suffix
		if !withPrefix {
			return nil, PrefixNotAllowedError
		}
		if !withSuffix {
			return nil, SuffixNotAllowedError
		}
		return s.parseSectionStringComponents(sectionSegments[0], sectionSegments[1], sectionSegments[2])
	}
	return nil, SectionFormatInvalidError
}

func (s *SectionParser) parseSectionStringComponents(sectionPrefixStr string, sectionStr string, sectionSuffixStr string) (Section, error) {
	var sectionPrefix, sectionSuffix Section
	var err error
	if len(sectionPrefixStr) > 0 {
		sectionPrefix, err = s.createSectionFromString(sectionPrefixStr, nil, nil)
		if err != nil {
			return nil, fmt.Errorf("section prefix %q could not be parsed: %w", sectionPrefixStr, err)
		}
	}
	if len(sectionSuffixStr) > 0 {
		sectionSuffix, err = s.createSectionFromString(sectionSuffixStr, nil, nil)
		if err != nil {
			return nil, fmt.Errorf("section suffix %q could not be parsed: %w", sectionSuffixStr, err)
		}
	}
	section, err := s.createSectionFromString(sectionStr, sectionPrefix, sectionSuffix)
	if err != nil {
		return nil, err
	}
	return section, nil
}

func (s *SectionParser) createSectionFromString(sectionStr string, prefixSection, suffixSection Section) (Section, error) {
	// create map of all aliases
	aliasMap := map[string]SectionType{}
	for _, sectionType := range s.sectionTypes {
		for _, alias := range sectionType.aliases {
			aliasMap[strings.ToLower(alias)] = sectionType
		}
	}
	// parse everything before the parameter brackets
	sectionComponents := strings.Split(sectionStr, constants.ParameterOpeningBrackets)
	alias := sectionComponents[0]
	sectionType, exists := aliasMap[strings.ToLower(alias)]
	if !exists {
		return nil, SectionAliasNotRegisteredWithParser{alias}
	}
	switch len(sectionComponents) {
	case 1:
		return sectionType.generatorFun("", prefixSection, suffixSection)
	case 2:
		if strings.HasSuffix(sectionComponents[1], constants.ParameterClosingBrackets) {
			return sectionType.generatorFun(strings.TrimSuffix(sectionComponents[1], constants.ParameterClosingBrackets), prefixSection, suffixSection)
		} else {
			return nil, MissingParameterClosingBracketsError
		}
	}
	return nil, MoreThanOneOpeningQuotesError
}

func (s *SectionParser) SectionHelpTexts() string {
	help := ""
	for _, sectionType := range s.sectionTypes {
		var aliasesWithParameters []string
		for _, alias := range sectionType.aliases {
			parameterSuffix := ""
			if sectionType.parameterHelp != "" {
				parameterSuffix = "(" + sectionType.parameterHelp + ")"
			}
			aliasesWithParameters = append(aliasesWithParameters, alias+parameterSuffix)
		}
		help += fmt.Sprintf("%s - %s\n", strings.Join(aliasesWithParameters, " | "), sectionType.description)
	}
	return help
}
