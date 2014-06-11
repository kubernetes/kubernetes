package util

import (
	"encoding/json"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
)

func TestMakeJSONString(t *testing.T) {
	pod := api.Pod{
		JSONBase: api.JSONBase{ID: "foo"},
		Labels: map[string]string{
			"foo": "bar",
			"baz": "blah",
		},
	}

	body := MakeJSONString(pod)

	expectedBody, err := json.Marshal(pod)
	expectNoError(t, err)
	if string(expectedBody) != body {
		t.Errorf("JSON doesn't match.  Expected %s, saw %s", expectedBody, body)
	}
}

func TestHandleCrash(t *testing.T) {
	count := 0
	expect := 10
	for i := 0; i < expect; i = i + 1 {
		defer HandleCrash()
		if i%2 == 0 {
			panic("Test Panic")
		}
		count = count + 1
	}
	if count != expect {
		t.Errorf("Expected %d iterations, found %d", expect, count)
	}
}
