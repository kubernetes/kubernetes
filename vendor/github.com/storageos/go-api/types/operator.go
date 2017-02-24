package types

// Operator represents a key/field's relationship to value(s).
// See labels.Requirement and fields.Requirement for more details.
type Operator string

// Valid operators
const (
	None         Operator = ""
	DoesNotExist Operator = "!"
	Equals       Operator = "="
	DoubleEquals Operator = "=="
	In           Operator = "in"
	NotEquals    Operator = "!="
	NotIn        Operator = "notin"
	Exists       Operator = "exists"
	GreaterThan  Operator = "gt"
	LessThan     Operator = "lt"
)
