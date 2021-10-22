package test

func init() {
	testCases = append(testCases,
		(*map[int8]string)(nil),
		(*map[int16]string)(nil),
		(*map[int32]string)(nil),
		(*map[int64]string)(nil),
		(*map[string][4]string)(nil),
		(*map[string]bool)(nil),
		(*map[string]byte)(nil),
		(*map[string]float64)(nil),
		(*map[string]int32)(nil),
		(*map[string]map[string]string)(nil),
		(*map[string]*[4]string)(nil),
		(*map[string]*bool)(nil),
		(*map[string]*float64)(nil),
		(*map[string]*int32)(nil),
		(*map[string]*map[string]string)(nil),
		(*map[string]*[]string)(nil),
		(*map[string]*string)(nil),
		(*map[string]*structVarious)(nil),
		(*map[string]*uint8)(nil),
		(*map[string][]string)(nil),
		(*map[string]string)(nil),
		(*map[string]stringAlias)(nil),
		(*map[string]struct{})(nil),
		(*map[string]structEmpty)(nil),
		(*map[string]struct {
			F *string
		})(nil),
		(*map[string]struct {
			String string
			Int    int32
			Float  float64
			Struct struct {
				X string
			}
			Slice []string
			Map   map[string]string
		})(nil),
		(*map[string]uint8)(nil),
		(*map[stringAlias]string)(nil),
		(*map[stringAlias]stringAlias)(nil),
		(*map[uint8]string)(nil),
		(*map[uint16]string)(nil),
		(*map[uint32]string)(nil),
	)
}

type structVarious struct {
	String string
	Int    int32
	Float  float64
	Struct struct {
		X string
	}
	Slice []string
	Map   map[string]string
}
