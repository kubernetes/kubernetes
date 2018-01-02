package statement

// A Template contains all information to fill in templated variables in inset and query statements
type Template struct {
	Tags     []string
	Function *Function
}

// Templates are a collection of Template
type Templates []*Template

// Init makes Stringers out of the Templates for quick point creation
func (t Templates) Init(seriesCount int) Stringers {
	arr := make([]Stringer, len(t))
	for i, tmp := range t {
		if len(tmp.Tags) == 0 {
			arr[i] = tmp.Function.NewStringer(seriesCount)
			continue
		}
		arr[i] = tmp.NewTagFunc()
	}
	return arr
}

// Calculates the number of series implied by a template
func (t *Template) numSeries() int {
	// If !t.Tags then tag cardinality is t.Function.Count
	if len(t.Tags) == 0 {
		return t.Function.Count
	}
	// Else tag cardinality is len(t.Tags)
	return len(t.Tags)
}

// NewTagFunc returns a Stringer that loops through the given tags
func (t *Template) NewTagFunc() Stringer {
	if len(t.Tags) == 0 {
		return func() string { return "EMPTY TAGS" }
	}

	i := 0
	return func() string {
		s := t.Tags[i]
		i = (i + 1) % len(t.Tags)
		return s
	}
}
