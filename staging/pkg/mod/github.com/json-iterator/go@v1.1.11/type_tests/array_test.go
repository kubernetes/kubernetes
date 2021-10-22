package test

func init() {
	testCases = append(testCases,
		(*[4]bool)(nil),
		(*[4]byte)(nil),
		(*[4]float64)(nil),
		(*[4]int32)(nil),
		(*[4]map[int32]string)(nil),
		(*[4]map[string]string)(nil),
		(*[4]*bool)(nil),
		(*[4]*float64)(nil),
		(*[4]*int32)(nil),
		(*[4]*map[int32]string)(nil),
		(*[4]*map[string]string)(nil),
		(*[4]*[4]bool)(nil),
		(*[4]*[4]byte)(nil),
		(*[4]*[4]float64)(nil),
		(*[4]*[4]int32)(nil),
		(*[4]*[4]*string)(nil),
		(*[4]*[4]string)(nil),
		(*[4]*[4]uint8)(nil),
		(*[4]*string)(nil),
		(*[4]*struct {
			String string
			Int    int32
			Float  float64
			Struct struct {
				X string
			}
			Slice [4]string
			Map   map[string]string
		})(nil),
		(*[4]*uint8)(nil),
		(*[4][4]bool)(nil),
		(*[4][4]byte)(nil),
		(*[4][4]float64)(nil),
		(*[4][4]int32)(nil),
		(*[4][4]*string)(nil),
		(*[4][4]string)(nil),
		(*[4][4]uint8)(nil),
		(*[4]string)(nil),
		(*[4]struct{})(nil),
		(*[4]structEmpty)(nil),
		(*[4]struct {
			F *string
		})(nil),
		(*[4]struct {
			String string
			Int    int32
			Float  float64
			Struct struct {
				X string
			}
			Slice [4]string
			Map   map[string]string
		})(nil),
		(*[4]uint8)(nil),
	)
}

type structEmpty struct{}
type arrayAlis [4]stringAlias
