package v1

import "encoding/json"

// UnmarshalJSON implements the json.Unmarshaller interface.
// If the value is a string, it sets the Value field of the StringSource.
// Otherwise, it is unmarshaled into the StringSourceSpec struct
func (s *StringSource) UnmarshalJSON(value []byte) error {
	// If we can unmarshal to a simple string, just set the value
	var simpleValue string
	if err := json.Unmarshal(value, &simpleValue); err == nil {
		s.Value = simpleValue
		return nil
	}

	// Otherwise do the full struct unmarshal
	return json.Unmarshal(value, &s.StringSourceSpec)
}

// MarshalJSON implements the json.Marshaller interface.
// If the StringSource contains only a string Value (or is empty), it is marshaled as a JSON string.
// Otherwise, the StringSourceSpec struct is marshaled as a JSON object.
func (s *StringSource) MarshalJSON() ([]byte, error) {
	// If we have only a cleartext value set, do a simple string marshal
	if s.StringSourceSpec == (StringSourceSpec{Value: s.Value}) {
		return json.Marshal(s.Value)
	}

	// Otherwise do the full struct marshal of the externalized bits
	return json.Marshal(s.StringSourceSpec)
}
