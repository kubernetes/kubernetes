package types

import (
	"encoding/json"
)

var (
	isolatorMap map[ACName]IsolatorValueConstructor
)

func init() {
	isolatorMap = make(map[ACName]IsolatorValueConstructor)
}

type IsolatorValueConstructor func() IsolatorValue

func AddIsolatorValueConstructor(n ACName, i IsolatorValueConstructor) {
	isolatorMap[n] = i
}

type Isolators []Isolator

// GetByName returns the last isolator in the list by the given name.
func (is *Isolators) GetByName(name ACName) *Isolator {
	var i Isolator
	for j := len(*is) - 1; j >= 0; j-- {
		i = []Isolator(*is)[j]
		if i.Name == name {
			return &i
		}
	}
	return nil
}

// Unrecognized returns a set of isolators that are not recognized.
// An isolator is not recognized if it has not had an associated
// constructor registered with AddIsolatorValueConstructor.
func (is *Isolators) Unrecognized() Isolators {
	u := Isolators{}
	for _, i := range *is {
		if i.value == nil {
			u = append(u, i)
		}
	}
	return u
}

type IsolatorValue interface {
	UnmarshalJSON(b []byte) error
	AssertValid() error
}
type Isolator struct {
	Name     ACName           `json:"name"`
	ValueRaw *json.RawMessage `json:"value"`
	value    IsolatorValue
}
type isolator Isolator

func (i *Isolator) Value() IsolatorValue {
	return i.value
}

func (i *Isolator) UnmarshalJSON(b []byte) error {
	var ii isolator
	err := json.Unmarshal(b, &ii)
	if err != nil {
		return err
	}

	var dst IsolatorValue
	con, ok := isolatorMap[ii.Name]
	if ok {
		dst = con()
		err = dst.UnmarshalJSON(*ii.ValueRaw)
		if err != nil {
			return err
		}
		err = dst.AssertValid()
		if err != nil {
			return err
		}
	}

	i.value = dst
	i.ValueRaw = ii.ValueRaw
	i.Name = ii.Name

	return nil
}
