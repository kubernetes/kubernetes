package util

import "testing"

type fakeClosable struct {
	closed bool
}

func (c *fakeClosable) Close() error {
	c.closed = true
	return nil
}

func TestRevertPorts(t *testing.T) {
	testCases := []struct {
		replacementPorts []LocalPort
		existingPorts    []LocalPort
		expectToBeClose  []bool
	}{
		{
			replacementPorts: []LocalPort{
				{Port: 5001},
				{Port: 5002},
				{Port: 5003},
			},
			existingPorts:   []LocalPort{},
			expectToBeClose: []bool{true, true, true},
		},
		{
			replacementPorts: []LocalPort{},
			existingPorts: []LocalPort{
				{Port: 5001},
				{Port: 5002},
				{Port: 5003},
			},
			expectToBeClose: []bool{},
		},
		{
			replacementPorts: []LocalPort{
				{Port: 5001},
				{Port: 5002},
				{Port: 5003},
			},
			existingPorts: []LocalPort{
				{Port: 5001},
				{Port: 5002},
				{Port: 5003},
			},
			expectToBeClose: []bool{false, false, false},
		},
		{
			replacementPorts: []LocalPort{
				{Port: 5001},
				{Port: 5002},
				{Port: 5003},
			},
			existingPorts: []LocalPort{
				{Port: 5001},
				{Port: 5003},
			},
			expectToBeClose: []bool{false, true, false},
		},
		{
			replacementPorts: []LocalPort{
				{Port: 5001},
				{Port: 5002},
				{Port: 5003},
			},
			existingPorts: []LocalPort{
				{Port: 5001},
				{Port: 5002},
				{Port: 5003},
				{Port: 5004},
			},
			expectToBeClose: []bool{false, false, false},
		},
	}

	for i, tc := range testCases {
		replacementPortsMap := make(map[LocalPort]Closeable)
		for _, lp := range tc.replacementPorts {
			replacementPortsMap[lp] = &fakeClosable{}
		}
		existingPortsMap := make(map[LocalPort]Closeable)
		for _, lp := range tc.existingPorts {
			existingPortsMap[lp] = &fakeClosable{}
		}
		RevertPorts(replacementPortsMap, existingPortsMap)
		for j, expectation := range tc.expectToBeClose {
			if replacementPortsMap[tc.replacementPorts[j]].(*fakeClosable).closed != expectation {
				t.Errorf("Expect replacement localport %v to be %v in test case %v", tc.replacementPorts[j], expectation, i)
			}
		}
		for _, lp := range tc.existingPorts {
			if existingPortsMap[lp].(*fakeClosable).closed == true {
				t.Errorf("Expect existing localport %v to be false in test case %v", lp, i)
			}
		}
	}
}
