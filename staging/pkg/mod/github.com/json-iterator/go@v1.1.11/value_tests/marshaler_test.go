package test

import (
	"encoding"
	"encoding/json"
)

func init() {
	jm := json.Marshaler(jmOfStruct{})
	tm1 := encoding.TextMarshaler(tmOfStruct{})
	tm2 := encoding.TextMarshaler(&tmOfStructInt{})
	marshalCases = append(marshalCases,
		jmOfStruct{},
		&jm,
		tmOfStruct{},
		&tm1,
		tmOfStructInt{},
		&tm2,
		map[tmOfStruct]int{
			{}: 100,
		},
		map[*tmOfStruct]int{
			{}: 100,
		},
		map[encoding.TextMarshaler]int{
			tm1: 100,
		},
	)
	unmarshalCases = append(unmarshalCases, unmarshalCase{
		ptr:   (*tmOfMap)(nil),
		input: `"{1:2}"`,
	}, unmarshalCase{
		ptr:   (*tmOfMapPtr)(nil),
		input: `"{1:2}"`,
	})
}

type jmOfStruct struct {
	F2 chan []byte
}

func (q jmOfStruct) MarshalJSON() ([]byte, error) {
	return []byte(`""`), nil
}

func (q *jmOfStruct) UnmarshalJSON(value []byte) error {
	return nil
}

type tmOfStruct struct {
	F2 chan []byte
}

func (q tmOfStruct) MarshalText() ([]byte, error) {
	return []byte(`""`), nil
}

func (q *tmOfStruct) UnmarshalText(value []byte) error {
	return nil
}

type tmOfStructInt struct {
	Field2 int
}

func (q *tmOfStructInt) MarshalText() ([]byte, error) {
	return []byte(`"abc"`), nil
}

func (q *tmOfStructInt) UnmarshalText(value []byte) error {
	return nil
}

type tmOfMap map[int]int

func (q tmOfMap) UnmarshalText(value []byte) error {
	return nil
}

type tmOfMapPtr map[int]int

func (q *tmOfMapPtr) UnmarshalText(value []byte) error {
	return nil
}
