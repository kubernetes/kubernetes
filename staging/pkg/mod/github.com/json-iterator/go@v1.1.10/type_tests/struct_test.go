package test

import "time"

func init() {
	structFields1To11()
	testCases = append(testCases,
		(*struct1Alias)(nil),
		(*struct {
			F [4]*string
		})(nil),
		(*struct {
			F [4]string
		})(nil),
		(*struct {
			F1 [4]stringAlias
			F2 arrayAlis
		})(nil),
		(*struct {
			F1 [4]string
			F2 [4]string
			F3 [4]string
		})(nil),
		(*struct {
			F [4]struct {
				F1 string
				F2 string
				F3 string
			}
		})(nil),
		(*struct{})(nil),
		(*structEmpty)(nil),
		(*struct {
			Byte1   byte
			Byte2   byte
			Bool1   bool
			Bool2   bool
			Int8    int8
			Int16   int16
			Int32   int32
			Int64   int64
			Uint8   uint8
			Uint16  uint16
			Uint32  uint32
			Uint64  uint64
			Float32 float32
			Float64 float64
			String1 string
			String2 string
		})(nil),
		(*struct {
			F float64
		})(nil),
		(*struct {
			F float64Alias
		})(nil),
		(*struct {
			F1 float64
			F2 float64
			F3 float64
		})(nil),
		(*struct {
			F1 float64Alias
			F2 float64Alias
			F3 float64Alias
		})(nil),
		(*struct {
			F int32
		})(nil),
		(*struct {
			F int32Alias
		})(nil),
		(*struct {
			F1 int32
			F2 int32
			F3 int32
		})(nil),
		(*struct {
			F1 int32Alias
			F2 int32Alias
			F3 int32Alias
		})(nil),
		(*struct {
			F int64
		})(nil),
		(*struct {
			F map[int32]*string
		})(nil),
		(*struct {
			F map[int32]string
		})(nil),
		(*struct {
			F map[int32]struct {
				F1 string
				F2 string
				F3 string
			}
		})(nil),
		(*struct {
			F map[string]*string
		})(nil),
		(*struct {
			F map[string]string
		})(nil),
		(*struct {
			F map[string]struct {
				F1 string
				F2 string
				F3 string
			}
		})(nil),
		(*struct {
			F *float64
		})(nil),
		(*struct {
			F1 *float64Alias
			F2 ptrFloat64Alias
			F3 *ptrFloat64Alias
		})(nil),
		(*struct {
			F *int32
		})(nil),
		(*struct {
			F1 *int32Alias
			F2 ptrInt32Alias
			F3 *ptrInt32Alias
		})(nil),
		(*struct {
			F **struct{}
		})(nil),
		(*struct {
			F **struct {
				F1 string
				F2 string
				F3 string
			}
		})(nil),
		(*struct {
			F *string
		})(nil),
		(*struct {
			F1 *stringAlias
			F2 ptrStringAlias
			F3 *ptrStringAlias
		})(nil),
		(*struct {
			F *struct{}
		})(nil),
		(*struct {
			F *struct {
				F1 string
				F2 string
				F3 string
			}
		})(nil),
		(*struct {
			F1 *float64
			F2 *float64
			F3 *float64
		})(nil),
		(*struct {
			F1 *int32
			F2 *int32
			F3 *int32
		})(nil),
		(*struct {
			F1 *string
			F2 *string
			F3 *string
		})(nil),
		(*struct {
			F []*string
		})(nil),
		(*struct {
			F []string
		})(nil),
		(*struct {
			F1 []stringAlias
			F2 stringAlias
		})(nil),
		(*struct {
			F1 []string
			F2 []string
			F3 []string
		})(nil),
		(*struct {
			F []struct {
				F1 string
				F2 string
				F3 string
			}
		})(nil),
		(*struct {
			F string
		})(nil),
		(*struct {
			F stringAlias
		})(nil),
		(*struct {
			F1 string
			F2 string
			F3 string
		})(nil),
		(*struct {
			F1 stringAlias
			F2 stringAlias
			F3 stringAlias
		})(nil),
		(*struct {
			F1 struct{}
			F2 struct{}
			F3 struct{}
		})(nil),
		(*struct {
			F struct{}
		})(nil),
		(*struct {
			F structEmpty
		})(nil),
		(*struct {
			F struct {
				F1 float32
				F2 float32
				F3 float32
			}
		})(nil),
		(*struct {
			F struct {
				F float32
			}
		})(nil),
		(*struct {
			F struct2
		})(nil),
		(*struct {
			F struct {
				F1 int32
				F2 int32
				F3 int32
			}
		})(nil),
		(*struct {
			F struct {
				F1 string
				F2 string
				F3 string
			}
		})(nil),
		(*struct {
			F struct3
		})(nil),
		(*struct {
			TF1 struct {
				F2 int
				F1 *withTime
			}
		})(nil),
		(*DeeplyNested)(nil),
	)
}

func structFields1To11() {
	testCases = append(testCases,
		(*struct {
			Field1 string
		})(nil),
		(*struct {
			Field1 string
			Field2 string
		})(nil),
		(*struct {
			Field1 string
			Field2 string
			Field3 string
		})(nil),
		(*struct {
			Field1 string
			Field2 string
			Field3 string
			Field4 string
		})(nil),
		(*struct {
			Field1 string
			Field2 string
			Field3 string
			Field4 string
			Field5 string
		})(nil),
		(*struct {
			Field1 string
			Field2 string
			Field3 string
			Field4 string
			Field5 string
			Field6 string
		})(nil),
		(*struct {
			Field1 string
			Field2 string
			Field3 string
			Field4 string
			Field5 string
			Field6 string
			Field7 string
		})(nil),
		(*struct {
			Field1 string
			Field2 string
			Field3 string
			Field4 string
			Field5 string
			Field6 string
			Field7 string
			Field8 string
		})(nil),
		(*struct {
			Field1 string
			Field2 string
			Field3 string
			Field4 string
			Field5 string
			Field6 string
			Field7 string
			Field8 string
			Field9 string
		})(nil),
		(*struct {
			Field1  string
			Field2  string
			Field3  string
			Field4  string
			Field5  string
			Field6  string
			Field7  string
			Field8  string
			Field9  string
			Field10 string
		})(nil),
		(*struct {
			Field1  string
			Field2  string
			Field3  string
			Field4  string
			Field5  string
			Field6  string
			Field7  string
			Field8  string
			Field9  string
			Field10 string
			Field11 string
		})(nil),
	)
}

type struct1 struct {
	Byte1   byte
	Byte2   byte
	Bool1   bool
	Bool2   bool
	Int8    int8
	Int16   int16
	Int32   int32
	Uint8   uint8
	Uint16  uint16
	Uint32  uint32
	Float32 float32
	Float64 float64
	String1 string
	String2 string
}
type struct1Alias struct1

type struct2 struct {
	F float64
}
type struct3 struct {
	F1 stringAlias
	F2 stringAlias
	F3 stringAlias
}

type withTime struct {
	time.Time
}

func (t *withTime) UnmarshalJSON(b []byte) error {
	return nil
}
func (t withTime) MarshalJSON() ([]byte, error) {
	return []byte(`"fake"`), nil
}

type YetYetAnotherObject struct {
	Field string
}
type YetAnotherObject struct {
	Field *YetYetAnotherObject
}
type AnotherObject struct {
	Field *YetAnotherObject
}
type DeeplyNested struct {
	Me *AnotherObject
}
