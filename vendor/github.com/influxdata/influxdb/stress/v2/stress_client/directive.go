package stressClient

// Directive is a struct to enable communication between SetStatements and the stressClient backend
// Directives change state for the stress test
type Directive struct {
	Property string
	Value    string
	Tracer   *Tracer
}

// NewDirective creates a new instance of a Directive with the appropriate state variable to change
func NewDirective(property string, value string, tracer *Tracer) Directive {
	d := Directive{
		Property: property,
		Value:    value,
		Tracer:   tracer,
	}
	return d
}
