// Extends the Go runtime's json.Decoder enabling navigation of a stream of json tokens.
package jsonpath

import "fmt"

type jsonContext int

const (
	none jsonContext = iota
	objKey
	objValue
	arrValue
)

// AnyIndex can be used in a pattern to match any array index.
const AnyIndex = -2

// JsonPath is a slice of strings and/or integers. Each string specifies an JSON object key, and
// each integer specifies an index into a JSON array.
type JsonPath []interface{}

func (p *JsonPath) push(n interface{}) { *p = append(*p, n) }
func (p *JsonPath) pop()               { *p = (*p)[:len(*p)-1] }

// increment the index at the top of the stack (must be an array index)
func (p *JsonPath) incTop() { (*p)[len(*p)-1] = (*p)[len(*p)-1].(int) + 1 }

// name the key at the top of the stack (must be an object key)
func (p *JsonPath) nameTop(n string) { (*p)[len(*p)-1] = n }

// infer the context from the item at the top of the stack
func (p *JsonPath) inferContext() jsonContext {
	if len(*p) == 0 {
		return none
	}
	t := (*p)[len(*p)-1]
	switch t.(type) {
	case string:
		return objKey
	case int:
		return arrValue
	default:
		panic(fmt.Sprintf("Invalid stack type %T", t))
	}
}

// Equal tests for equality between two JsonPath types.
func (p *JsonPath) Equal(o JsonPath) bool {
	if len(*p) != len(o) {
		return false
	}
	for i, v := range *p {
		if v != o[i] {
			return false
		}
	}
	return true
}

func (p *JsonPath) HasPrefix(o JsonPath) bool {
	for i, v := range o {
		if v != (*p)[i] {
			return false
		}
	}
	return true
}
