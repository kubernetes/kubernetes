package test

func init() {
	testCases = append(testCases,
		(*EmbeddedFieldName)(nil),
		(*StringFieldName)(nil),
		(*StructFieldName)(nil),
		(*struct {
			F1 bool `json:"F1"`
			F2 bool `json:"F2,omitempty"`
		})(nil),
		(*EmbeddedOmitEmpty)(nil),
		(*struct {
			F1 float32 `json:"F1"`
			F2 float32 `json:"F2,omitempty"`
		})(nil),
		(*struct {
			F1 int32 `json:"F1"`
			F2 int32 `json:"F2,omitempty"`
		})(nil),
		(*struct {
			F1 map[string]string `json:"F1"`
			F2 map[string]string `json:"F2,omitempty"`
		})(nil),
		(*struct {
			F1 *bool `json:"F1"`
			F2 *bool `json:"F2,omitempty"`
		})(nil),
		(*struct {
			F1 *float32 `json:"F1"`
			F2 *float32 `json:"F2,omitempty"`
		})(nil),
		(*struct {
			F1 *int32 `json:"F1"`
			F2 *int32 `json:"F2,omitempty"`
		})(nil),
		(*struct {
			F1 *map[string]string `json:"F1"`
			F2 *map[string]string `json:"F2,omitempty"`
		})(nil),
		(*struct {
			F1 *[]string `json:"F1"`
			F2 *[]string `json:"F2,omitempty"`
		})(nil),
		(*struct {
			F1 string `json:"F1"`
			F2 string `json:"F2,omitempty"`
		})(nil),
		(*struct {
			F1 *string `json:"F1"`
			F2 *string `json:"F2,omitempty"`
		})(nil),
		(*struct {
			F *jm `json:"f,omitempty"`
		})(nil),
		(*struct {
			F *tm `json:"f,omitempty"`
		})(nil),
		(*struct {
			F *sjm `json:"f,omitempty"`
		})(nil),
		(*struct {
			F *tm `json:"f,omitempty"`
		})(nil),
		(*struct {
			F1 *uint32 `json:"F1"`
			F2 *uint32 `json:"F2,omitempty"`
		})(nil),
		(*struct {
			F1 []string `json:"F1"`
			F2 []string `json:"F2,omitempty"`
		})(nil),
		(*struct {
			F1 string `json:"F1"`
			F2 string `json:"F2,omitempty"`
		})(nil),
		(*struct {
			F jm `json:"f,omitempty"`
		})(nil),
		(*struct {
			F tm `json:"f,omitempty"`
		})(nil),
		(*struct {
			F struct{} `json:"f,omitempty"` // omitempty is meaningless here
		})(nil),
		(*struct {
			F sjm `json:"f,omitempty"`
		})(nil),
		(*struct {
			F stm `json:"f,omitempty"`
		})(nil),
		(*struct {
			F1 uint32 `json:"F1"`
			F2 uint32 `json:"F2,omitempty"`
		})(nil),
		(*struct {
			F1 bool `json:"F1"`
			F2 bool `json:"F2,string"`
		})(nil),
		(*struct {
			F1 byte `json:"F1"`
			F2 byte `json:"F2,string"`
		})(nil),
		(*struct {
			F1 float32 `json:"F1"`
			F2 float32 `json:"F2,string"`
		})(nil),
		(*struct {
			F1 float64 `json:"F1"`
			F2 float64 `json:"F2,string"`
		})(nil),
		(*struct {
			F1 int8 `json:"F1"`
			F2 int8 `json:"F2,string"`
		})(nil),
		(*struct {
			F1 int16 `json:"F1"`
			F2 int16 `json:"F2,string"`
		})(nil),
		(*struct {
			F1 int32 `json:"F1"`
			F2 int32 `json:"F2,string"`
		})(nil),
		// remove temporarily until https://github.com/golang/go/issues/38126 is fixed
		// (*struct {
		// 	F1 string `json:"F1"`
		// 	F2 string `json:"F2,string"`
		// })(nil),
		(*struct {
			F1 uint8 `json:"F1"`
			F2 uint8 `json:"F2,string"`
		})(nil),
		(*struct {
			F1 uint16 `json:"F1"`
			F2 uint16 `json:"F2,string"`
		})(nil),
		(*struct {
			F1 uint32 `json:"F1"`
			F2 uint32 `json:"F2,string"`
		})(nil),
		(*struct {
			A           string            `json:"a,omitempty"`
			B           string            `json:"b,omitempty"`
			Annotations map[string]string `json:"annotations,omitempty"`
		})(nil),
		(*struct {
			Field bool `json:",omitempty,string"`
		})(nil),
		(*struct {
			Field bool `json:"中文"`
		})(nil),
	)
}

// S1 TEST ONLY
type EmbeddedFieldNameS1 struct {
	S1F string
}

// S2 TEST ONLY
type EmbeddedFieldNameS2 struct {
	S2F string
}

// S3 TEST ONLY
type EmbeddedFieldNameS3 struct {
	S3F string
}

// S4 TEST ONLY
type EmbeddedFieldNameS4 struct {
	S4F string
}

// S5 TEST ONLY
type EmbeddedFieldNameS5 struct {
	S5F string
}

// S6 TEST ONLY
type EmbeddedFieldNameS6 struct {
	S6F string
}

type EmbeddedFieldName struct {
	EmbeddedFieldNameS1 `json:"F1"`
	EmbeddedFieldNameS2 `json:"f2"`
	EmbeddedFieldNameS3 `json:"-"`
	EmbeddedFieldNameS4 `json:"-,"`
	EmbeddedFieldNameS5 `json:","`
	EmbeddedFieldNameS6 `json:""`
}

type StringFieldNameE struct {
	E1 string
}

type StringFieldName struct {
	F1               string `json:"F1"`
	F2               string `json:"f2"`
	F3               string `json:"-"`
	F4               string `json:"-,"`
	F5               string `json:","`
	F6               string `json:""`
	StringFieldNameE `json:"e"`
}

type StructFieldNameS1 struct {
	S1F string
}

type StructFieldNameS2 struct {
	S2F string
}

type StructFieldNameS3 struct {
	S3F string
}

type StructFieldNameS4 struct {
	S4F string
}

type StructFieldNameS5 struct {
	S5F string
}

type StructFieldNameS6 struct {
	S6F string
}

type StructFieldName struct {
	F1 StructFieldNameS1 `json:"F1"`
	F2 StructFieldNameS2 `json:"f2"`
	F3 StructFieldNameS3 `json:"-"`
	F4 StructFieldNameS4 `json:"-,"`
	F5 StructFieldNameS5 `json:","`
	F6 StructFieldNameS6 `json:""`
}
type EmbeddedOmitEmptyE struct {
	F string `json:"F,omitempty"`
}

type EmbeddedOmitEmpty struct {
	EmbeddedOmitEmptyE
}

type jm string

func (t *jm) UnmarshalJSON(b []byte) error {
	return nil
}

func (t jm) MarshalJSON() ([]byte, error) {
	return []byte(`""`), nil
}

type tm string

func (t *tm) UnmarshalText(b []byte) error {
	return nil
}

func (t tm) MarshalText() ([]byte, error) {
	return []byte(`""`), nil
}

type sjm struct{}

func (t *sjm) UnmarshalJSON(b []byte) error {
	return nil
}

func (t sjm) MarshalJSON() ([]byte, error) {
	return []byte(`""`), nil
}

type stm struct{}

func (t *stm) UnmarshalText(b []byte) error {
	return nil
}

func (t stm) MarshalText() ([]byte, error) {
	return []byte(`""`), nil
}
