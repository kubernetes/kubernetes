package osin

import (
	"testing"
)

func TestClientIntfUserData(t *testing.T) {
	c := &DefaultClient{
		UserData: make(map[string]interface{}),
	}

	// check if the interface{} returned from the method is a reference
	c.GetUserData().(map[string]interface{})["test"] = "none"

	if _, ok := c.GetUserData().(map[string]interface{})["test"]; !ok {
		t.Error("Returned interface is not a reference")
	}
}
