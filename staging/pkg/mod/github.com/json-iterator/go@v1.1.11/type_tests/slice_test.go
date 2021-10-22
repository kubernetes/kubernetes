package test

func init() {
	testCases = append(testCases,
		(*[][4]bool)(nil),
		(*[][4]byte)(nil),
		(*[][4]float64)(nil),
		(*[][4]int32)(nil),
		(*[][4]*string)(nil),
		(*[][4]string)(nil),
		(*[][4]uint8)(nil),
		(*[]bool)(nil),
		(*[]byte)(nil),
		(*[]float64)(nil),
		(*[]int32)(nil),
		(*[]int64)(nil),
		(*[]map[int32]string)(nil),
		(*[]map[string]string)(nil),
		(*[4]*[4]bool)(nil),
		(*[4]*[4]byte)(nil),
		(*[4]*[4]float64)(nil),
		(*[4]*[4]int32)(nil),
		(*[4]*[4]*string)(nil),
		(*[4]*[4]string)(nil),
		(*[4]*[4]uint8)(nil),
		(*[]*bool)(nil),
		(*[]*float64)(nil),
		(*[]*int32)(nil),
		(*[]*map[int32]string)(nil),
		(*[]*map[string]string)(nil),
		(*[]*[]bool)(nil),
		(*[]*[]byte)(nil),
		(*[]*[]float64)(nil),
		(*[]*[]int32)(nil),
		(*[]*[]*string)(nil),
		(*[]*[]string)(nil),
		(*[]*[]uint8)(nil),
		(*[]*string)(nil),
		(*[]*struct {
			String string
			Int    int32
			Float  float64
			Struct struct {
				X string
			}
			Slice []string
			Map   map[string]string
		})(nil),
		(*[]*uint8)(nil),
		(*[][]bool)(nil),
		(*[][]byte)(nil),
		(*[][]float64)(nil),
		(*[][]int32)(nil),
		(*[][]*string)(nil),
		(*[][]string)(nil),
		(*[][]uint8)(nil),
		(*[]string)(nil),
		(*[]struct{})(nil),
		(*[]structEmpty)(nil),
		(*[]struct {
			F *string
		})(nil),
		(*[]struct {
			String string
			Int    int32
			Float  float64
			Struct struct {
				X string
			}
			Slice []string
			Map   map[string]string
		})(nil),
		(*[]uint8)(nil),
		(*[]jsonMarshaler)(nil),
		(*[]jsonMarshalerMap)(nil),
		(*[]textMarshaler)(nil),
		(*[]textMarshalerMap)(nil),
	)
}

type jsonMarshaler struct {
	Id string `json:"id,omitempty" db:"id"`
}

func (p *jsonMarshaler) MarshalJSON() ([]byte, error) {
	return []byte(`{}`), nil
}

func (p *jsonMarshaler) UnmarshalJSON(input []byte) error {
	p.Id = "hello"
	return nil
}

type jsonMarshalerMap map[int]int

func (p *jsonMarshalerMap) MarshalJSON() ([]byte, error) {
	return []byte(`{}`), nil
}

func (p *jsonMarshalerMap) UnmarshalJSON(input []byte) error {
	return nil
}

type textMarshaler struct {
	Id string `json:"id,omitempty" db:"id"`
}

func (p *textMarshaler) MarshalText() ([]byte, error) {
	return []byte(`{}`), nil
}

func (p *textMarshaler) UnmarshalText(input []byte) error {
	p.Id = "hello"
	return nil
}

type textMarshalerMap map[int]int

func (p *textMarshalerMap) MarshalText() ([]byte, error) {
	return []byte(`{}`), nil
}

func (p *textMarshalerMap) UnmarshalText(input []byte) error {
	return nil
}
