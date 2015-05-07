package types

import (
	"encoding/json"
	"fmt"
)

type Annotations []Annotation

type annotations Annotations

type Annotation struct {
	Name  ACName `json:"name"`
	Value string `json:"value"`
}

func (a Annotations) assertValid() error {
	seen := map[ACName]string{}
	for _, anno := range a {
		_, ok := seen[anno.Name]
		if ok {
			return fmt.Errorf(`duplicate annotations of name %q`, anno.Name)
		}
		seen[anno.Name] = anno.Value
	}
	if c, ok := seen["created"]; ok {
		if _, err := NewDate(c); err != nil {
			return err
		}
	}
	if h, ok := seen["homepage"]; ok {
		if _, err := NewURL(h); err != nil {
			return err
		}
	}
	if d, ok := seen["documentation"]; ok {
		if _, err := NewURL(d); err != nil {
			return err
		}
	}

	return nil
}

func (a Annotations) MarshalJSON() ([]byte, error) {
	if err := a.assertValid(); err != nil {
		return nil, err
	}
	return json.Marshal(annotations(a))
}

func (a *Annotations) UnmarshalJSON(data []byte) error {
	var ja annotations
	if err := json.Unmarshal(data, &ja); err != nil {
		return err
	}
	na := Annotations(ja)
	if err := na.assertValid(); err != nil {
		return err
	}
	*a = na
	return nil
}

// Retrieve the value of an annotation by the given name from Annotations, if
// it exists.
func (a Annotations) Get(name string) (val string, ok bool) {
	for _, anno := range a {
		if anno.Name.String() == name {
			return anno.Value, true
		}
	}
	return "", false
}

// Set sets the value of an annotation by the given name, overwriting if one already exists.
func (a *Annotations) Set(name ACName, value string) {
	for i, anno := range *a {
		if anno.Name.Equals(name) {
			(*a)[i] = Annotation{
				Name:  name,
				Value: value,
			}
			return
		}
	}
	anno := Annotation{
		Name:  name,
		Value: value,
	}
	*a = append(*a, anno)
}
