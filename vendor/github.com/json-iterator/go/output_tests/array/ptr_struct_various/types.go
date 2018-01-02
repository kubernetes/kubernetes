package test

type typeForTest [4]*struct {
	String string
	Int    int32
	Float  float64
	Struct struct {
		X string
	}
	Slice [4]string
	Map   map[string]string
}
