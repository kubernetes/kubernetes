package stressClient

// Package is a struct to enable communication between InsertStatements, QueryStatements and InfluxQLStatements and the stressClient backend
// Packages carry either writes or queries in the []byte that makes up the Body
type Package struct {
	T           Type
	Body        []byte
	StatementID string
	Tracer      *Tracer
}

// NewPackage creates a new package with the appropriate payload
func NewPackage(t Type, body []byte, statementID string, tracer *Tracer) Package {
	p := Package{
		T:           t,
		Body:        body,
		StatementID: statementID,
		Tracer:      tracer,
	}

	return p
}
