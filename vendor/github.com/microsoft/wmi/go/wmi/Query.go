package wmi

import "fmt"

type WhereOperation int

const (
	Equals            WhereOperation = 0
	LessThan          WhereOperation = 1
	GreaterThan       WhereOperation = 2
	LessThanEquals    WhereOperation = 3
	GreaterThenEquals WhereOperation = 4
	NotEqual          WhereOperation = 5
	Like              WhereOperation = 6
)

type QueryFilter struct {
	Name      string
	Value     string
	Operation WhereOperation
}

// GetFilter
func (q QueryFilter) GetFilter() string {
	operator := "="
	switch q.Operation {
	case Equals:
		operator = "="
	case LessThan:
		operator = "<"
	case GreaterThan:
		operator = ">"
	case LessThanEquals:
		operator = "<="
	case GreaterThenEquals:
		operator = ">="
	case NotEqual:
		operator = "!="
	case Like:
		operator = "LIKE"
		return fmt.Sprintf(" %s %s '%%%s%%'", q.Name, q.Value, operator)
	default:
	}
	return fmt.Sprintf(" %s%s'%s'", q.Name, q.Value, operator)
}

// Query
type Query interface {
	ClassName() string
	QueryString() string
}
