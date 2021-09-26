package test

func init() {
	testCases = append(testCases,
		(*struct {
			Upper bool `json:"M"`
			Lower bool `json:"m"`
		})(nil),
	)
	asymmetricTestCases = append(asymmetricTestCases, [][2]interface{}{
		{
			(*struct {
				Field string
			})(nil),
			(*struct {
				FIELD string
			})(nil),
		},
		{
			(*struct {
				F1 string
				F2 string
				F3 string
			})(nil),
			(*struct {
				F1 string
			})(nil),
		},
	}...)
}
