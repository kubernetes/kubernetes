package test

type typeForTest struct {
	F *jm `json:"f,omitempty"`
}

type jm string

func (t *jm) UnmarshalJSON(b []byte) error {
	return nil
}

func (t jm) MarshalJSON() ([]byte, error) {
	return []byte(`""`), nil
}
