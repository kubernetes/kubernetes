package sharedcpus

import "testing"

func TestParseConfigData(t *testing.T) {
	testCases := []struct {
		data                []byte
		expectedToBeParsed  bool
		containerLimitValue int64
	}{
		{
			data: []byte(`{
					"shared_cpus": {
     					"containers_limit": 15
					}
				}`),
			expectedToBeParsed:  true,
			containerLimitValue: 15,
		},
		{
			data: []byte(`{
					"shared_cpus": {
     					"abc": "25"
  					}
				}`),
			expectedToBeParsed:  false,
			containerLimitValue: 0,
		},
	}
	for _, tc := range testCases {
		cfg, err := parseConfigData(tc.data)
		if err != nil && tc.expectedToBeParsed {
			t.Errorf("shared cpus data expected to be parsed")
		}
		if cfg.ContainersLimit != tc.containerLimitValue {
			t.Errorf("shared cpus ContainersLimit is different than expected: want: %d; got %d", tc.containerLimitValue, cfg.ContainersLimit)
		}
	}
}
