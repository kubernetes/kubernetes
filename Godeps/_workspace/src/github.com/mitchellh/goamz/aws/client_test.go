package aws_test

import (
	"fmt"
	"github.com/mitchellh/goamz/aws"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// Retrieve the response from handler using aws.RetryingClient
func serveAndGet(handler http.HandlerFunc) (body string, err error) {
	ts := httptest.NewServer(handler)
	defer ts.Close()
	resp, err := aws.RetryingClient.Get(ts.URL)
	if err != nil {
		return
	}
	if resp.StatusCode != 200 {
		return "", fmt.Errorf("Bad status code: %d", resp.StatusCode)
	}
	greeting, err := ioutil.ReadAll(resp.Body)
	resp.Body.Close()
	if err != nil {
		return
	}
	return strings.TrimSpace(string(greeting)), nil
}

func TestClient_expected(t *testing.T) {
	body := "foo bar"

	resp, err := serveAndGet(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintln(w, body)
	})
	if err != nil {
		t.Fatal(err)
	}
	if resp != body {
		t.Fatal("Body not as expected.")
	}
}

func TestClient_delay(t *testing.T) {
	body := "baz"
	wait := 4
	resp, err := serveAndGet(func(w http.ResponseWriter, r *http.Request) {
		if wait < 0 {
			// If we dipped to zero delay and still failed.
			t.Fatal("Never succeeded.")
		}
		wait -= 1
		time.Sleep(time.Second * time.Duration(wait))
		fmt.Fprintln(w, body)
	})
	if err != nil {
		t.Fatal(err)
	}
	if resp != body {
		t.Fatal("Body not as expected.", resp)
	}
}

func TestClient_no4xxRetry(t *testing.T) {
	tries := 0

	// Fail once before succeeding.
	_, err := serveAndGet(func(w http.ResponseWriter, r *http.Request) {
		tries += 1
		http.Error(w, "error", 404)
	})

	if err == nil {
		t.Fatal("should have error")
	}

	if tries != 1 {
		t.Fatalf("should only try once: %d", tries)
	}
}

func TestClient_retries(t *testing.T) {
	body := "biz"
	failed := false
	// Fail once before succeeding.
	resp, err := serveAndGet(func(w http.ResponseWriter, r *http.Request) {
		if !failed {
			http.Error(w, "error", 500)
			failed = true
		} else {
			fmt.Fprintln(w, body)
		}
	})
	if failed != true {
		t.Error("We didn't retry!")
	}
	if err != nil {
		t.Fatal(err)
	}
	if resp != body {
		t.Fatal("Body not as expected.")
	}
}

func TestClient_fails(t *testing.T) {
	tries := 0
	// Fail 3 times and return the last error.
	_, err := serveAndGet(func(w http.ResponseWriter, r *http.Request) {
		tries += 1
		http.Error(w, "error", 500)
	})
	if err == nil {
		t.Fatal(err)
	}
	if tries != 3 {
		t.Fatal("Didn't retry enough")
	}
}
