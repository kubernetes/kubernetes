package garif

// WithLineColumn sets a physical location with the given line and column
func (l *Location) WithLineColumn(line, column int) *Location {
	if l.PhysicalLocation == nil {
		l.PhysicalLocation = NewPhysicalLocation()
	}

	l.PhysicalLocation.Region = NewRegion()
	l.PhysicalLocation.Region.StartLine = line
	l.PhysicalLocation.Region.StartColumn = column

	return l
}

// WithURI sets a physical location with the given URI
func (l *Location) WithURI(uri string) *Location {
	if l.PhysicalLocation == nil {
		l.PhysicalLocation = NewPhysicalLocation()
	}

	l.PhysicalLocation.ArtifactLocation = NewArtifactLocation()
	l.PhysicalLocation.ArtifactLocation.Uri = uri

	return l
}

// WithKeyValue sets (overwrites) the value of the given key
func (b PropertyBag) WithKeyValue(key string, value interface{}) PropertyBag {
	b[key] = value
	return b
}

// WithHelpUri sets the help URI for this ReportingDescriptor
func (r *ReportingDescriptor) WithHelpUri(uri string) *ReportingDescriptor {
	r.HelpUri = uri
	return r
}

// WithProperties adds the key & value to the properties of this ReportingDescriptor
func (r *ReportingDescriptor) WithProperties(key string, value interface{}) *ReportingDescriptor {
	if r.Properties == nil {
		r.Properties = NewPropertyBag()
	}

	r.Properties.WithKeyValue(key, value)

	return r
}

// WithArtifactsURIs adds the given URI as artifacts of this Run
func (r *Run) WithArtifactsURIs(uris ...string) *Run {
	if r.Artifacts == nil {
		r.Artifacts = []*Artifact{}
	}

	for _, uri := range uris {
		a := NewArtifact()
		a.Location = NewArtifactLocation()
		a.Location.Uri = uri
		r.Artifacts = append(r.Artifacts, a)
	}

	return r
}

// WithResult adds a result to this Run
func (r *Run) WithResult(ruleId string, message string, uri string, line int, column int) *Run {
	if r.Results == nil {
		r.Results = []*Result{}
	}

	msg := NewMessage()
	msg.Text = message
	result := NewResult(msg)
	location := NewLocation().WithURI(uri).WithLineColumn(line, column)

	result.Locations = append(result.Locations, location)
	result.RuleId = ruleId
	r.Results = append(r.Results, result)
	return r
}

// WithInformationUri sets the information URI
func (t *ToolComponent) WithInformationUri(uri string) *ToolComponent {
	t.InformationUri = uri
	return t
}

// WithRules sets (overwrites) the rules
func (t *ToolComponent) WithRules(rules ...*ReportingDescriptor) *ToolComponent {
	t.Rules = rules
	return t
}
