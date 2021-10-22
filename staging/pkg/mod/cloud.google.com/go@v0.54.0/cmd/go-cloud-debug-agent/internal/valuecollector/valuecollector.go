// Copyright 2016 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package valuecollector is used to collect the values of variables in a program.
package valuecollector

import (
	"bytes"
	"fmt"
	"strconv"
	"strings"

	"cloud.google.com/go/cmd/go-cloud-debug-agent/internal/debug"
	cd "google.golang.org/api/clouddebugger/v2"
)

const (
	maxArrayLength = 50
	maxMapLength   = 20
)

// Collector is given references to variables from a program being debugged
// using AddVariable. Then when ReadValues is called, the Collector will fetch
// the values of those variables. Any variables referred to by those values
// will also be fetched; e.g. the targets of pointers, members of structs,
// elements of slices, etc. This continues iteratively, building a graph of
// values, until all the reachable values are fetched, or a size limit is
// reached.
//
// Variables are passed to the Collector as debug.Var, which is used by x/debug
// to represent references to variables. Values are returned as cd.Variable,
// which is used by the Debuglet Controller to represent the graph of values.
//
// For example, if the program has a struct variable:
//
//	foo := SomeStruct{a:42, b:"xyz"}
//
// and we call AddVariable with a reference to foo, we will get back a result
// like:
//
//	cd.Variable{Name:"foo", VarTableIndex:10}
//
// which denotes a variable named "foo" which will have its value stored in
// element 10 of the table that will later be returned by ReadValues. That
// element might be:
//
//	out[10] = &cd.Variable{Members:{{Name:"a", VarTableIndex:11},{Name:"b", VarTableIndex:12}}}
//
// which denotes a struct with two members a and b, whose values are in elements
// 11 and 12 of the output table:
//
//	out[11] = &cd.Variable{Value:"42"}
//	out[12] = &cd.Variable{Value:"xyz"}
type Collector struct {
	// prog is the program being debugged.
	prog debug.Program
	// limit is the maximum size of the output slice of values.
	limit int
	// index is a map from references (variables and map elements) to their
	// locations in the table.
	index map[reference]int
	// table contains the references, including those given to the
	// Collector directly and those the Collector itself found.
	// If VarTableIndex is set to 0 in a cd.Variable, it is ignored, so the first entry
	// of table can't be used. On initialization we put a dummy value there.
	table []reference
}

// reference represents a value which is in the queue to be read by the
// collector.  It is either a debug.Var, or a mapElement.
type reference interface{}

// mapElement represents an element of a map in the debugged program's memory.
type mapElement struct {
	debug.Map
	index uint64
}

// NewCollector returns a Collector for the given program and size limit.
// The limit is the maximum size of the slice of values returned by ReadValues.
func NewCollector(prog debug.Program, limit int) *Collector {
	return &Collector{
		prog:  prog,
		limit: limit,
		index: make(map[reference]int),
		table: []reference{debug.Var{}},
	}
}

// AddVariable adds another variable to be collected.
// The Collector doesn't get the value immediately; it returns a cd.Variable
// that contains an index into the table which will later be returned by
// ReadValues.
func (c *Collector) AddVariable(lv debug.LocalVar) *cd.Variable {
	ret := &cd.Variable{Name: lv.Name}
	if index, ok := c.add(lv.Var); !ok {
		// If the add call failed, it's because we reached the size limit.
		// The Debuglet Controller's convention is to pass it a "Not Captured" error
		// in this case.
		ret.Status = statusMessage(messageNotCaptured, true, refersToVariableName)
	} else {
		ret.VarTableIndex = int64(index)
	}
	return ret
}

// add adds a reference to the set of values to be read from the
// program. It returns the index in the output table that will contain the
// corresponding value. It fails if the table has reached the size limit.
// It deduplicates references, so the index may be the same as one that was
// returned from an earlier add call.
func (c *Collector) add(r reference) (outputIndex int, ok bool) {
	if i, ok := c.index[r]; ok {
		return i, true
	}
	i := len(c.table)
	if i >= c.limit {
		return 0, false
	}
	c.index[r] = i
	c.table = append(c.table, r)
	return i, true
}

func addMember(v *cd.Variable, name string) *cd.Variable {
	v2 := &cd.Variable{Name: name}
	v.Members = append(v.Members, v2)
	return v2
}

// ReadValues fetches values of the variables that were passed to the Collector
// with AddVariable. The values of any new variables found are also fetched,
// e.g. the targets of pointers or the members of structs, until we reach the
// size limit or we run out of values to fetch.
// The results are output as a []*cd.Variable, which is the type we need to send
// to the Debuglet Controller after we trigger a breakpoint.
func (c *Collector) ReadValues() (out []*cd.Variable) {
	for i := 0; i < len(c.table); i++ {
		// Create a new cd.Variable for this value, and append it to the output.
		dcv := new(cd.Variable)
		out = append(out, dcv)
		if i == 0 {
			// The first element is unused.
			continue
		}
		switch x := c.table[i].(type) {
		case mapElement:
			key, value, err := c.prog.MapElement(x.Map, x.index)
			if err != nil {
				dcv.Status = statusMessage(err.Error(), true, refersToVariableValue)
				continue
			}
			// Add a member for the key.
			member := addMember(dcv, "key")
			if index, ok := c.add(key); !ok {
				// The table is full.
				member.Status = statusMessage(messageNotCaptured, true, refersToVariableName)
				continue
			} else {
				member.VarTableIndex = int64(index)
			}
			// Add a member for the value.
			member = addMember(dcv, "value")
			if index, ok := c.add(value); !ok {
				// The table is full.
				member.Status = statusMessage(messageNotCaptured, true, refersToVariableName)
			} else {
				member.VarTableIndex = int64(index)
			}
		case debug.Var:
			if v, err := c.prog.Value(x); err != nil {
				dcv.Status = statusMessage(err.Error(), true, refersToVariableValue)
			} else {
				c.FillValue(v, dcv)
			}
		}
	}
	return out
}

// indexable is an interface for arrays, slices and channels.
type indexable interface {
	Len() uint64
	Element(uint64) debug.Var
}

// channel implements indexable.
type channel struct {
	debug.Channel
}

func (c channel) Len() uint64 {
	return c.Length
}

var (
	_ indexable = debug.Array{}
	_ indexable = debug.Slice{}
	_ indexable = channel{}
)

// FillValue copies a value into a cd.Variable.  Any variables referred to by
// that value, e.g. struct members and pointer targets, are added to the
// collector's queue, to be fetched later by ReadValues.
func (c *Collector) FillValue(v debug.Value, dcv *cd.Variable) {
	if c, ok := v.(debug.Channel); ok {
		// Convert to channel, which implements indexable.
		v = channel{c}
	}
	// Fill in dcv in a manner depending on the type of the value we got.
	switch val := v.(type) {
	case int8, int16, int32, int64, bool, uint8, uint16, uint32, uint64, float32, float64, complex64, complex128:
		// For simple types, we just print the value to dcv.Value.
		dcv.Value = fmt.Sprint(val)
	case string:
		// Put double quotes around strings.
		dcv.Value = strconv.Quote(val)
	case debug.String:
		if uint64(len(val.String)) < val.Length {
			// This string value was truncated.
			dcv.Value = strconv.Quote(val.String + "...")
		} else {
			dcv.Value = strconv.Quote(val.String)
		}
	case debug.Struct:
		// For structs, we add an entry to dcv.Members for each field in the
		// struct.
		// Each member will contain the name of the field, and the index in the
		// output table which will contain the value of that field.
		for _, f := range val.Fields {
			member := addMember(dcv, f.Name)
			if index, ok := c.add(f.Var); !ok {
				// The table is full.
				member.Status = statusMessage(messageNotCaptured, true, refersToVariableName)
			} else {
				member.VarTableIndex = int64(index)
			}
		}
	case debug.Map:
		dcv.Value = fmt.Sprintf("len = %d", val.Length)
		for i := uint64(0); i < val.Length; i++ {
			field := addMember(dcv, `⚫`)
			if i == maxMapLength {
				field.Name = "..."
				field.Status = statusMessage(messageTruncated, true, refersToVariableName)
				break
			}
			if index, ok := c.add(mapElement{val, i}); !ok {
				// The value table is full; add a member to contain the error message.
				field.Name = "..."
				field.Status = statusMessage(messageNotCaptured, true, refersToVariableName)
				break
			} else {
				field.VarTableIndex = int64(index)
			}
		}
	case debug.Pointer:
		if val.Address == 0 {
			dcv.Value = "<nil>"
		} else if val.TypeID == 0 {
			// We don't know the type of the pointer, so just output the address as
			// the value.
			dcv.Value = fmt.Sprintf("0x%X", val.Address)
			dcv.Status = statusMessage(messageUnknownPointerType, false, refersToVariableName)
		} else {
			// Adds the pointed-to variable to the table, and links this value to
			// that table entry through VarTableIndex.
			dcv.Value = fmt.Sprintf("0x%X", val.Address)
			target := addMember(dcv, "")
			if index, ok := c.add(debug.Var(val)); !ok {
				target.Status = statusMessage(messageNotCaptured, true, refersToVariableName)
			} else {
				target.VarTableIndex = int64(index)
			}
		}
	case indexable:
		// Arrays, slices and channels.
		dcv.Value = "len = " + fmt.Sprint(val.Len())
		for j := uint64(0); j < val.Len(); j++ {
			field := addMember(dcv, fmt.Sprint(`[`, j, `]`))
			if j == maxArrayLength {
				field.Name = "..."
				field.Status = statusMessage(messageTruncated, true, refersToVariableName)
				break
			}
			vr := val.Element(j)
			if index, ok := c.add(vr); !ok {
				// The value table is full; add a member to contain the error message.
				field.Name = "..."
				field.Status = statusMessage(messageNotCaptured, true, refersToVariableName)
				break
			} else {
				// Add a member with the index as the name.
				field.VarTableIndex = int64(index)
			}
		}
	default:
		dcv.Status = statusMessage(messageUnknownType, false, refersToVariableName)
	}
}

// statusMessage returns a *cd.StatusMessage with the given message, IsError
// field and refersTo field.
func statusMessage(msg string, isError bool, refersTo int) *cd.StatusMessage {
	return &cd.StatusMessage{
		Description: &cd.FormatMessage{Format: "$0", Parameters: []string{msg}},
		IsError:     isError,
		RefersTo:    refersToString[refersTo],
	}
}

// LogString produces a string for a logpoint, substituting in variable values
// using evaluatedExpressions and varTable.
func LogString(s string, evaluatedExpressions []*cd.Variable, varTable []*cd.Variable) string {
	var buf bytes.Buffer
	fmt.Fprintf(&buf, "LOGPOINT: ")
	seen := make(map[*cd.Variable]bool)
	for i := 0; i < len(s); {
		if s[i] == '$' {
			i++
			if num, n, ok := parseToken(s[i:], len(evaluatedExpressions)-1); ok {
				// This token is one of $0, $1, etc.  Write the corresponding expression.
				writeExpression(&buf, evaluatedExpressions[num], false, varTable, seen)
				i += n
			} else {
				// Something else, like $$.
				buf.WriteByte(s[i])
				i++
			}
		} else {
			buf.WriteByte(s[i])
			i++
		}
	}
	return buf.String()
}

func parseToken(s string, max int) (num int, bytesRead int, ok bool) {
	var i int
	for i < len(s) && s[i] >= '0' && s[i] <= '9' {
		i++
	}
	num, err := strconv.Atoi(s[:i])
	return num, i, err == nil && num <= max
}

// writeExpression recursively writes variables to buf, in a format suitable
// for logging.  If printName is true, writes the name of the variable.
func writeExpression(buf *bytes.Buffer, v *cd.Variable, printName bool, varTable []*cd.Variable, seen map[*cd.Variable]bool) {
	if v == nil {
		// Shouldn't happen.
		return
	}
	name, value, status, members := v.Name, v.Value, v.Status, v.Members

	// If v.VarTableIndex is not zero, it refers to an element of varTable.
	// We merge its fields with the fields we got from v.
	var other *cd.Variable
	if idx := int(v.VarTableIndex); idx > 0 && idx < len(varTable) {
		other = varTable[idx]
	}
	if other != nil {
		if name == "" {
			name = other.Name
		}
		if value == "" {
			value = other.Value
		}
		if status == nil {
			status = other.Status
		}
		if len(members) == 0 {
			members = other.Members
		}
	}
	if printName && name != "" {
		buf.WriteString(name)
		buf.WriteByte(':')
	}

	// If we have seen this value before, write "..." rather than repeating it.
	if seen[v] {
		buf.WriteString("...")
		return
	}
	seen[v] = true
	if other != nil {
		if seen[other] {
			buf.WriteString("...")
			return
		}
		seen[other] = true
	}

	if value != "" && !strings.HasPrefix(value, "len = ") {
		// A plain value.
		buf.WriteString(value)
	} else if status != nil && status.Description != nil {
		// An error.
		for _, p := range status.Description.Parameters {
			buf.WriteByte('(')
			buf.WriteString(p)
			buf.WriteByte(')')
		}
	} else if name == `⚫` {
		// A map element.
		first := true
		for _, member := range members {
			if first {
				first = false
			} else {
				buf.WriteByte(':')
			}
			writeExpression(buf, member, false, varTable, seen)
		}
	} else {
		// A map, array, slice, channel, or struct.
		isStruct := value == ""
		first := true
		buf.WriteByte('{')
		for _, member := range members {
			if first {
				first = false
			} else {
				buf.WriteString(", ")
			}
			writeExpression(buf, member, isStruct, varTable, seen)
		}
		buf.WriteByte('}')
	}
}

const (
	// Error messages for cd.StatusMessage
	messageNotCaptured        = "Not captured"
	messageTruncated          = "Truncated"
	messageUnknownPointerType = "Unknown pointer type"
	messageUnknownType        = "Unknown type"
	// RefersTo values for cd.StatusMessage.
	refersToVariableName = iota
	refersToVariableValue
)

// refersToString contains the strings for each refersTo value.
// See the definition of StatusMessage in the v2/clouddebugger package.
var refersToString = map[int]string{
	refersToVariableName:  "VARIABLE_NAME",
	refersToVariableValue: "VARIABLE_VALUE",
}
