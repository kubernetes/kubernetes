package gci

import (
	"io/ioutil"

	"github.com/daixiang0/gci/pkg/configuration"
	sectionsPkg "github.com/daixiang0/gci/pkg/gci/sections"

	"gopkg.in/yaml.v3"
)

type GciConfiguration struct {
	configuration.FormatterConfiguration
	Sections          SectionList
	SectionSeparators SectionList
}

type GciStringConfiguration struct {
	Cfg                     configuration.FormatterConfiguration `yaml:",inline"`
	SectionStrings          []string                             `yaml:"sections"`
	SectionSeparatorStrings []string                             `yaml:"sectionseparators"`
}

func (g GciStringConfiguration) Parse() (*GciConfiguration, error) {
	sections := DefaultSections()
	var err error
	if len(g.SectionStrings) > 0 {
		sections, err = sectionsPkg.SectionParserInst.ParseSectionStrings(g.SectionStrings, true, true)
		if err != nil {
			return nil, err
		}
	}
	sectionSeparators := DefaultSectionSeparators()
	if len(g.SectionSeparatorStrings) > 0 {
		sectionSeparators, err = sectionsPkg.SectionParserInst.ParseSectionStrings(g.SectionSeparatorStrings, false, false)
		if err != nil {
			return nil, err
		}
	}
	return &GciConfiguration{g.Cfg, sections, sectionSeparators}, nil
}

func initializeGciConfigFromYAML(filePath string) (*GciConfiguration, error) {
	yamlCfg := GciStringConfiguration{}
	yamlData, err := ioutil.ReadFile(filePath)
	if err != nil {
		return nil, err
	}
	err = yaml.Unmarshal(yamlData, &yamlCfg)
	if err != nil {
		return nil, err
	}
	gciCfg, err := yamlCfg.Parse()
	if err != nil {
		return nil, err
	}
	return gciCfg, nil
}
