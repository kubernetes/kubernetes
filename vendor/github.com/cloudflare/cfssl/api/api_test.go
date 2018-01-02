package api

import (
	"bytes"
	"encoding/json"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"testing"
)

const (
	ty   = "Thank you!"
	deny = "That's not true!"
)

func simpleHandle(w http.ResponseWriter, r *http.Request) error {
	_, _, err := ProcessRequestOneOf(r, [][]string{
		{"compliment"},
		{"critique"},
	})
	if err != nil {
		return err
	}

	return SendResponse(w, ty)

}

func cleverHandle(w http.ResponseWriter, r *http.Request) error {
	_, matched, err := ProcessRequestFirstMatchOf(r, [][]string{
		{"compliment"},
		{"critique"},
	})
	if err != nil {
		return err
	}
	if matched[0] == "critique" {
		return SendResponse(w, deny)
	}

	return SendResponse(w, ty)
}

func post(t *testing.T, obj map[string]interface{}, ts *httptest.Server) (resp *http.Response, body []byte) {
	blob, err := json.Marshal(obj)
	if err != nil {
		t.Fatal(err)
	}

	resp, err = http.Post(ts.URL, "application/json", bytes.NewReader(blob))
	if err != nil {
		t.Fatal(err)
	}
	body, err = ioutil.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	return
}

func get(t *testing.T, ts *httptest.Server) (resp *http.Response, body []byte) {
	resp, err := http.Get(ts.URL)
	if err != nil {
		t.Fatal(err)
	}

	body, err = ioutil.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	return
}

func TestRigidHandle(t *testing.T) {
	ts := httptest.NewServer(HTTPHandler{Handler: HandlerFunc(simpleHandle), Methods: []string{"POST"}})
	defer ts.Close()

	// Response to compliment
	obj := map[string]interface{}{}
	obj["compliment"] = "it's good"
	resp, body := post(t, obj, ts)

	if resp.StatusCode != http.StatusOK {
		t.Errorf("Test expected 200, have %d", resp.StatusCode)
	}

	message := new(Response)
	err := json.Unmarshal(body, message)
	if err != nil {
		t.Errorf("failed to read response body: %v", err)
		t.Fatal("returned:", message)
	}

	if message.Result != ty {
		t.Fatal("Wrong response")
	}

	// Response to critique
	obj = map[string]interface{}{}
	obj["critique"] = "it's bad"
	resp, body = post(t, obj, ts)

	if resp.StatusCode != http.StatusOK {
		t.Errorf("Test expected 200, have %d", resp.StatusCode)
	}

	message = new(Response)
	err = json.Unmarshal(body, message)
	if err != nil {
		t.Errorf("failed to read response body: %v", err)
		t.Fatal("returned:", message)
	}

	if message.Result != ty {
		t.Fatal("Wrong response")
	}

	// reject mixed review
	obj = map[string]interface{}{}
	obj["critique"] = "it's OK"
	obj["compliment"] = "it's not bad"
	resp, body = post(t, obj, ts)

	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("Test expected 400, have %d", resp.StatusCode)
	}

	// reject empty review
	obj = map[string]interface{}{}
	resp, body = post(t, obj, ts)

	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("Test expected 400, have %d", resp.StatusCode)
	}

	// reject GET
	resp, body = get(t, ts)

	if resp.StatusCode != http.StatusMethodNotAllowed {
		t.Errorf("Test expected 405, have %d", resp.StatusCode)
	}
}

func TestCleverHandle(t *testing.T) {
	ts := httptest.NewServer(HTTPHandler{Handler: HandlerFunc(cleverHandle), Methods: []string{"POST"}})
	defer ts.Close()

	// Response ty to compliment
	obj := map[string]interface{}{}
	obj["compliment"] = "it's good"
	resp, body := post(t, obj, ts)

	if resp.StatusCode != http.StatusOK {
		t.Errorf("Test expected 200, have %d", resp.StatusCode)
	}

	message := new(Response)
	err := json.Unmarshal(body, message)
	if err != nil {
		t.Errorf("failed to read response body: %v", err)
		t.Fatal("returned:", message)
	}

	if message.Result != ty {
		t.Fatal("Wrong response")
	}

	// Response deny to critique
	obj = map[string]interface{}{}
	obj["critique"] = "it's bad"
	resp, body = post(t, obj, ts)

	if resp.StatusCode != http.StatusOK {
		t.Errorf("Test expected 200, have %d", resp.StatusCode)
	}

	message = new(Response)
	err = json.Unmarshal(body, message)
	if err != nil {
		t.Errorf("failed to read response body: %v", err)
		t.Fatal("returned:", message)
	}

	if message.Result != deny {
		t.Fatal("Wrong response")
	}

	// Be polite to mixed review
	obj = map[string]interface{}{}
	obj["critique"] = "it's OK"
	obj["compliment"] = "it's not bad"
	resp, body = post(t, obj, ts)

	message = new(Response)
	err = json.Unmarshal(body, message)
	if err != nil {
		t.Errorf("failed to read response body: %v", err)
		t.Fatal("returned:", message)
	}

	if message.Result != ty {
		t.Fatal("Wrong response")
	}

	// reject empty review
	obj = map[string]interface{}{}
	resp, body = post(t, obj, ts)

	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("Test expected 400, have %d", resp.StatusCode)
	}

	// reject GET
	resp, body = get(t, ts)

	if resp.StatusCode != http.StatusMethodNotAllowed {
		t.Errorf("Test expected 405, have %d", resp.StatusCode)
	}
}
