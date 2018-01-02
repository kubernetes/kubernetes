package test

type typeForTest struct {
	F *tm `json:"f,omitempty"`
}

type tm struct{}

func (t *tm) UnmarshalText(b []byte) error {
	return nil
}

func (t tm) MarshalText() ([]byte, error) {
	return []byte(`""`), nil
}
