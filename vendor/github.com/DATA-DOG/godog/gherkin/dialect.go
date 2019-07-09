package gherkin

type GherkinDialect struct {
	Language string
	Name     string
	Native   string
	Keywords map[string][]string
}

func (g *GherkinDialect) FeatureKeywords() []string {
	return g.Keywords["feature"]
}

func (g *GherkinDialect) ScenarioKeywords() []string {
	return g.Keywords["scenario"]
}

func (g *GherkinDialect) StepKeywords() []string {
	result := g.Keywords["given"]
	result = append(result, g.Keywords["when"]...)
	result = append(result, g.Keywords["then"]...)
	result = append(result, g.Keywords["and"]...)
	result = append(result, g.Keywords["but"]...)
	return result
}

func (g *GherkinDialect) BackgroundKeywords() []string {
	return g.Keywords["background"]
}

func (g *GherkinDialect) ScenarioOutlineKeywords() []string {
	return g.Keywords["scenarioOutline"]
}

func (g *GherkinDialect) ExamplesKeywords() []string {
	return g.Keywords["examples"]
}

type GherkinDialectProvider interface {
	GetDialect(language string) *GherkinDialect
}

type gherkinDialectMap map[string]*GherkinDialect

func (g gherkinDialectMap) GetDialect(language string) *GherkinDialect {
	return g[language]
}
