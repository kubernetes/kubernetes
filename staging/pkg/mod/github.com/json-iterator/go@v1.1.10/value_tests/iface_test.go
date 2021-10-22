package test

import "io"

func init() {
	var pCloser1 = func(str string) *io.Closer {
		closer := io.Closer(strCloser1(str))
		return &closer
	}
	var pCloser2 = func(str string) *io.Closer {
		strCloser := strCloser2(str)
		closer := io.Closer(&strCloser)
		return &closer
	}
	marshalCases = append(marshalCases,
		pCloser1("hello"),
		pCloser2("hello"),
	)
	unmarshalCases = append(unmarshalCases, unmarshalCase{
		ptr:   (*[]io.Closer)(nil),
		input: "[null]",
	}, unmarshalCase{
		obj: func() interface{} {
			strCloser := strCloser2("")
			return &struct {
				Field io.Closer
			}{
				&strCloser,
			}
		},
		input: `{"Field": "hello"}`,
	})
}

type strCloser1 string

func (closer strCloser1) Close() error {
	return nil
}

type strCloser2 string

func (closer *strCloser2) Close() error {
	return nil
}
