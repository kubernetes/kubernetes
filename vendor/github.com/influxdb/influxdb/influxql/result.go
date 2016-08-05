package influxql

import (
	"encoding/json"
	"errors"
	"hash/fnv"
	"sort"
)

// TagSet is a fundamental concept within the query system. It represents a composite series,
// composed of multiple individual series that share a set of tag attributes.
type TagSet struct {
	Tags       map[string]string
	Filters    []Expr
	SeriesKeys []string
	Key        []byte
}

// AddFilter adds a series-level filter to the Tagset.
func (t *TagSet) AddFilter(key string, filter Expr) {
	t.SeriesKeys = append(t.SeriesKeys, key)
	t.Filters = append(t.Filters, filter)
}

// Row represents a single row returned from the execution of a statement.
type Row struct {
	Name    string            `json:"name,omitempty"`
	Tags    map[string]string `json:"tags,omitempty"`
	Columns []string          `json:"columns,omitempty"`
	Values  [][]interface{}   `json:"values,omitempty"`
	Err     error             `json:"err,omitempty"`
}

// tagsHash returns a hash of tag key/value pairs.
func (r *Row) tagsHash() uint64 {
	h := fnv.New64a()
	keys := r.tagsKeys()
	for _, k := range keys {
		h.Write([]byte(k))
		h.Write([]byte(r.Tags[k]))
	}
	return h.Sum64()
}

// tagKeys returns a sorted list of tag keys.
func (r *Row) tagsKeys() []string {
	a := make([]string, 0, len(r.Tags))
	for k := range r.Tags {
		a = append(a, k)
	}
	sort.Strings(a)
	return a
}

// Rows represents a list of rows that can be sorted consistently by name/tag.
type Rows []*Row

func (p Rows) Len() int { return len(p) }

func (p Rows) Less(i, j int) bool {
	// Sort by name first.
	if p[i].Name != p[j].Name {
		return p[i].Name < p[j].Name
	}

	// Sort by tag set hash. Tags don't have a meaningful sort order so we
	// just compute a hash and sort by that instead. This allows the tests
	// to receive rows in a predictable order every time.
	return p[i].tagsHash() < p[j].tagsHash()
}

func (p Rows) Swap(i, j int) { p[i], p[j] = p[j], p[i] }

// Result represents a resultset returned from a single statement.
type Result struct {
	// StatementID is just the statement's position in the query. It's used
	// to combine statement results if they're being buffered in memory.
	StatementID int `json:"-"`
	Series      Rows
	Err         error
}

// MarshalJSON encodes the result into JSON.
func (r *Result) MarshalJSON() ([]byte, error) {
	// Define a struct that outputs "error" as a string.
	var o struct {
		Series []*Row `json:"series,omitempty"`
		Err    string `json:"error,omitempty"`
	}

	// Copy fields to output struct.
	o.Series = r.Series
	if r.Err != nil {
		o.Err = r.Err.Error()
	}

	return json.Marshal(&o)
}

// UnmarshalJSON decodes the data into the Result struct
func (r *Result) UnmarshalJSON(b []byte) error {
	var o struct {
		Series []*Row `json:"series,omitempty"`
		Err    string `json:"error,omitempty"`
	}

	err := json.Unmarshal(b, &o)
	if err != nil {
		return err
	}
	r.Series = o.Series
	if o.Err != "" {
		r.Err = errors.New(o.Err)
	}
	return nil
}

func GetProcessor(expr Expr, startIndex int) (Processor, int) {
	switch expr := expr.(type) {
	case *VarRef:
		return newEchoProcessor(startIndex), startIndex + 1
	case *Call:
		return newEchoProcessor(startIndex), startIndex + 1
	case *BinaryExpr:
		return getBinaryProcessor(expr, startIndex)
	case *ParenExpr:
		return GetProcessor(expr.Expr, startIndex)
	case *NumberLiteral:
		return newLiteralProcessor(expr.Val), startIndex
	case *StringLiteral:
		return newLiteralProcessor(expr.Val), startIndex
	case *BooleanLiteral:
		return newLiteralProcessor(expr.Val), startIndex
	case *TimeLiteral:
		return newLiteralProcessor(expr.Val), startIndex
	case *DurationLiteral:
		return newLiteralProcessor(expr.Val), startIndex
	}
	panic("unreachable")
}

type Processor func(values []interface{}) interface{}

func newEchoProcessor(index int) Processor {
	return func(values []interface{}) interface{} {
		return values[index]
	}
}

func newLiteralProcessor(val interface{}) Processor {
	return func(values []interface{}) interface{} {
		return val
	}
}

func getBinaryProcessor(expr *BinaryExpr, startIndex int) (Processor, int) {
	lhs, index := GetProcessor(expr.LHS, startIndex)
	rhs, index := GetProcessor(expr.RHS, index)

	return newBinaryExprEvaluator(expr.Op, lhs, rhs), index
}

func newBinaryExprEvaluator(op Token, lhs, rhs Processor) Processor {
	switch op {
	case ADD:
		return func(values []interface{}) interface{} {
			l := lhs(values)
			r := rhs(values)
			if lv, ok := l.(float64); ok {
				if rv, ok := r.(float64); ok {
					if rv != 0 {
						return lv + rv
					}
				}
			}
			return nil
		}
	case SUB:
		return func(values []interface{}) interface{} {
			l := lhs(values)
			r := rhs(values)
			if lv, ok := l.(float64); ok {
				if rv, ok := r.(float64); ok {
					if rv != 0 {
						return lv - rv
					}
				}
			}
			return nil
		}
	case MUL:
		return func(values []interface{}) interface{} {
			l := lhs(values)
			r := rhs(values)
			if lv, ok := l.(float64); ok {
				if rv, ok := r.(float64); ok {
					if rv != 0 {
						return lv * rv
					}
				}
			}
			return nil
		}
	case DIV:
		return func(values []interface{}) interface{} {
			l := lhs(values)
			r := rhs(values)
			if lv, ok := l.(float64); ok {
				if rv, ok := r.(float64); ok {
					if rv != 0 {
						return lv / rv
					}
				}
			}
			return nil
		}
	default:
		// we shouldn't get here, but give them back nils if it goes this way
		return func(values []interface{}) interface{} {
			return nil
		}
	}
}
