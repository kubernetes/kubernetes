/*
Copyright 2016 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package gcp

import (
	"fmt"
	"io/ioutil"
	"net/http"
	"os"
	"os/exec"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"golang.org/x/oauth2"
)

type fakeOutput struct {
	args   []string
	output string
}

var (
	wantCmd []string
	// Output for fakeExec, keyed by command
	execOutputs = map[string]fakeOutput{
		"/default/no/args": {
			args: []string{},
			output: `{
  "access_token": "faketoken",
  "token_expiry": "2016-10-31T22:31:09.123000000Z"
}`},
		"/default/legacy/args": {
			args: []string{"arg1", "arg2", "arg3"},
			output: `{
  "access_token": "faketoken",
  "token_expiry": "2016-10-31T22:31:09.123000000Z"
}`},
		"/space in path/customkeys": {
			args: []string{"can", "haz", "auth"},
			output: `{
  "token": "faketoken",
  "token_expiry": {
    "datetime": "2016-10-31 22:31:09.123"
  }
}`},
		"missing/tokenkey/noargs": {
			args: []string{},
			output: `{
  "broken": "faketoken",
  "token_expiry": {
    "datetime": "2016-10-31 22:31:09.123000000Z"
  }
}`},
		"missing/expirykey/legacyargs": {
			args: []string{"split", "on", "whitespace"},
			output: `{
  "access_token": "faketoken",
  "expires": "2016-10-31T22:31:09.123000000Z"
}`},
		"invalid expiry/timestamp": {
			args: []string{"foo", "--bar", "--baz=abc,def"},
			output: `{
  "access_token": "faketoken",
  "token_expiry": "sometime soon, idk"
}`},
		"badjson": {
			args: []string{},
			output: `{
  "access_token": "faketoken",
  "token_expiry": "sometime soon, idk"
  ------
`},
	}
)

func fakeExec(command string, args ...string) *exec.Cmd {
	cs := []string{"-test.run=TestHelperProcess", "--", command}
	cs = append(cs, args...)
	cmd := exec.Command(os.Args[0], cs...)
	cmd.Env = []string{"GO_WANT_HELPER_PROCESS=1"}
	return cmd
}

func TestHelperProcess(t *testing.T) {
	if os.Getenv("GO_WANT_HELPER_PROCESS") != "1" {
		return
	}
	// Strip out the leading args used to exec into this function.
	gotCmd := os.Args[3]
	gotArgs := os.Args[4:]
	output, ok := execOutputs[gotCmd]
	if !ok {
		fmt.Fprintf(os.Stdout, "unexpected call cmd=%q args=%v\n", gotCmd, gotArgs)
		os.Exit(1)
	} else if !reflect.DeepEqual(output.args, gotArgs) {
		fmt.Fprintf(os.Stdout, "call cmd=%q got args %v, want: %v\n", gotCmd, gotArgs, output.args)
		os.Exit(1)
	}
	fmt.Fprintf(os.Stdout, output.output)
	os.Exit(0)
}

func Test_isCmdTokenSource(t *testing.T) {
	c1 := map[string]string{"cmd-path": "foo"}
	if v := isCmdTokenSource(c1); !v {
		t.Fatalf("cmd-path present in config (%+v), but got %v", c1, v)
	}

	c2 := map[string]string{"cmd-args": "foo bar"}
	if v := isCmdTokenSource(c2); v {
		t.Fatalf("cmd-path not present in config (%+v), but got %v", c2, v)
	}
}

func Test_tokenSource_cmd(t *testing.T) {
	if _, err := tokenSource(true, map[string]string{}); err == nil {
		t.Fatalf("expected error, cmd-args not present in config")
	}

	c := map[string]string{
		"cmd-path": "foo",
		"cmd-args": "bar"}
	ts, err := tokenSource(true, c)
	if err != nil {
		t.Fatalf("failed to return cmd token source: %+v", err)
	}
	if ts == nil {
		t.Fatal("returned nil token source")
	}
	if _, ok := ts.(*commandTokenSource); !ok {
		t.Fatalf("returned token source type:(%T) expected:(*commandTokenSource)", ts)
	}
}

func Test_tokenSource_cmdCannotBeUsedWithScopes(t *testing.T) {
	c := map[string]string{
		"cmd-path": "foo",
		"scopes":   "A,B"}
	if _, err := tokenSource(true, c); err == nil {
		t.Fatal("expected error when scopes is used with cmd-path")
	}
}

func Test_tokenSource_applicationDefaultCredentials_fails(t *testing.T) {
	// try to use empty ADC file
	fakeTokenFile, err := ioutil.TempFile("", "adctoken")
	if err != nil {
		t.Fatalf("failed to create fake token file: +%v", err)
	}
	fakeTokenFile.Close()
	defer os.Remove(fakeTokenFile.Name())

	os.Setenv("GOOGLE_APPLICATION_CREDENTIALS", fakeTokenFile.Name())
	defer os.Unsetenv("GOOGLE_APPLICATION_CREDENTIALS")
	if _, err := tokenSource(false, map[string]string{}); err == nil {
		t.Fatalf("expected error because specified ADC token file is not a JSON")
	}
}

func Test_tokenSource_applicationDefaultCredentials(t *testing.T) {
	fakeTokenFile, err := ioutil.TempFile("", "adctoken")
	if err != nil {
		t.Fatalf("failed to create fake token file: +%v", err)
	}
	fakeTokenFile.Close()
	defer os.Remove(fakeTokenFile.Name())
	if err := ioutil.WriteFile(fakeTokenFile.Name(), []byte(`{"type":"service_account"}`), 0600); err != nil {
		t.Fatalf("failed to write to fake token file: %+v", err)
	}

	os.Setenv("GOOGLE_APPLICATION_CREDENTIALS", fakeTokenFile.Name())
	defer os.Unsetenv("GOOGLE_APPLICATION_CREDENTIALS")
	ts, err := tokenSource(false, map[string]string{})
	if err != nil {
		t.Fatalf("failed to get a token source: %+v", err)
	}
	if ts == nil {
		t.Fatal("returned nil token source")
	}
}

func Test_parseScopes(t *testing.T) {
	cases := []struct {
		in  map[string]string
		out []string
	}{
		{
			map[string]string{},
			[]string{
				"https://www.googleapis.com/auth/cloud-platform",
				"https://www.googleapis.com/auth/userinfo.email"},
		},
		{
			map[string]string{"scopes": ""},
			[]string{},
		},
		{
			map[string]string{"scopes": "A,B,C"},
			[]string{"A", "B", "C"},
		},
	}

	for _, c := range cases {
		got := parseScopes(c.in)
		if !reflect.DeepEqual(got, c.out) {
			t.Errorf("expected=%v, got=%v", c.out, got)
		}
	}
}

func errEquiv(got, want error) bool {
	if got == want {
		return true
	}
	if got != nil && want != nil {
		return strings.Contains(got.Error(), want.Error())
	}
	return false
}

func TestCmdTokenSource(t *testing.T) {
	execCommand = fakeExec
	fakeExpiry := time.Date(2016, 10, 31, 22, 31, 9, 123000000, time.UTC)
	customFmt := "2006-01-02 15:04:05.999999999"

	tests := []struct {
		name             string
		gcpConfig        map[string]string
		tok              *oauth2.Token
		newErr, tokenErr error
	}{
		{
			"default",
			map[string]string{
				"cmd-path": "/default/no/args",
			},
			&oauth2.Token{
				AccessToken: "faketoken",
				TokenType:   "Bearer",
				Expiry:      fakeExpiry,
			},
			nil,
			nil,
		},
		{
			"default legacy args",
			map[string]string{
				"cmd-path": "/default/legacy/args arg1 arg2 arg3",
			},
			&oauth2.Token{
				AccessToken: "faketoken",
				TokenType:   "Bearer",
				Expiry:      fakeExpiry,
			},
			nil,
			nil,
		},

		{
			"custom keys",
			map[string]string{
				"cmd-path":   "/space in path/customkeys",
				"cmd-args":   "can haz auth",
				"token-key":  "{.token}",
				"expiry-key": "{.token_expiry.datetime}",
				"time-fmt":   customFmt,
			},
			&oauth2.Token{
				AccessToken: "faketoken",
				TokenType:   "Bearer",
				Expiry:      fakeExpiry,
			},
			nil,
			nil,
		},
		{
			"missing cmd",
			map[string]string{
				"cmd-path": "",
			},
			nil,
			fmt.Errorf("missing access token cmd"),
			nil,
		},
		{
			"missing token-key",
			map[string]string{
				"cmd-path":  "missing/tokenkey/noargs",
				"token-key": "{.token}",
			},
			nil,
			nil,
			fmt.Errorf("error parsing token-key %q", "{.token}"),
		},

		{
			"missing expiry-key",
			map[string]string{
				"cmd-path":   "missing/expirykey/legacyargs split on whitespace",
				"expiry-key": "{.expiry}",
			},
			nil,
			nil,
			fmt.Errorf("error parsing expiry-key %q", "{.expiry}"),
		},
		{
			"invalid expiry timestamp",
			map[string]string{
				"cmd-path": "invalid expiry/timestamp",
				"cmd-args": "foo --bar --baz=abc,def",
			},
			&oauth2.Token{
				AccessToken: "faketoken",
				TokenType:   "Bearer",
				Expiry:      time.Time{},
			},
			nil,
			nil,
		},
		{
			"bad JSON",
			map[string]string{
				"cmd-path": "badjson",
			},
			nil,
			nil,
			fmt.Errorf("invalid character '-' after object key:value pair"),
		},
	}

	for _, tc := range tests {
		provider, err := newGCPAuthProvider("", tc.gcpConfig, nil /* persister */)
		if !errEquiv(err, tc.newErr) {
			t.Errorf("%q newGCPAuthProvider error: got %v, want %v", tc.name, err, tc.newErr)
			continue
		}
		if err != nil {
			continue
		}
		ts := provider.(*gcpAuthProvider).tokenSource.(*cachedTokenSource).source.(*commandTokenSource)
		wantCmd = append([]string{ts.cmd}, ts.args...)
		tok, err := ts.Token()
		if !errEquiv(err, tc.tokenErr) {
			t.Errorf("%q Token() error: got %v, want %v", tc.name, err, tc.tokenErr)
		}
		if !reflect.DeepEqual(tok, tc.tok) {
			t.Errorf("%q Token() got %v, want %v", tc.name, tok, tc.tok)
		}
	}
}

type fakePersister struct {
	lk    sync.Mutex
	cache map[string]string
}

func (f *fakePersister) Persist(cache map[string]string) error {
	f.lk.Lock()
	defer f.lk.Unlock()
	f.cache = map[string]string{}
	for k, v := range cache {
		f.cache[k] = v
	}
	return nil
}

func (f *fakePersister) read() map[string]string {
	ret := map[string]string{}
	f.lk.Lock()
	defer f.lk.Unlock()
	for k, v := range f.cache {
		ret[k] = v
	}
	return ret
}

type fakeTokenSource struct {
	token *oauth2.Token
	err   error
}

func (f *fakeTokenSource) Token() (*oauth2.Token, error) {
	return f.token, f.err
}

func TestCachedTokenSource(t *testing.T) {
	tok := &oauth2.Token{AccessToken: "fakeaccesstoken"}
	persister := &fakePersister{}
	source := &fakeTokenSource{
		token: tok,
		err:   nil,
	}
	cache := map[string]string{
		"foo": "bar",
		"baz": "bazinga",
	}
	ts, err := newCachedTokenSource("fakeaccesstoken", "", persister, source, cache)
	if err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	wg.Add(10)
	for i := 0; i < 10; i++ {
		go func() {
			_, err := ts.Token()
			if err != nil {
				t.Errorf("unexpected error: %s", err)
			}
			wg.Done()
		}()
	}
	wg.Wait()
	cache["access-token"] = "fakeaccesstoken"
	cache["expiry"] = tok.Expiry.Format(time.RFC3339Nano)
	if got := persister.read(); !reflect.DeepEqual(got, cache) {
		t.Errorf("got cache %v, want %v", got, cache)
	}
}

type MockTransport struct {
	res *http.Response
}

func (t *MockTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	return t.res, nil
}

func TestClearingCredentials(t *testing.T) {

	fakeExpiry := time.Now().Add(time.Hour)

	cache := map[string]string{
		"access-token": "fakeToken",
		"expiry":       fakeExpiry.String(),
	}

	cts := cachedTokenSource{
		source:      nil,
		accessToken: cache["access-token"],
		expiry:      fakeExpiry,
		persister:   nil,
		cache:       nil,
	}

	tests := []struct {
		name  string
		res   http.Response
		cache map[string]string
	}{
		{
			"Unauthorized",
			http.Response{StatusCode: 401},
			make(map[string]string),
		},
		{
			"Authorized",
			http.Response{StatusCode: 200},
			cache,
		},
	}

	persister := &fakePersister{}
	req := http.Request{Header: http.Header{}}

	for _, tc := range tests {
		authProvider := gcpAuthProvider{&cts, persister}

		fakeTransport := MockTransport{&tc.res}

		transport := (authProvider.WrapTransport(&fakeTransport))
		persister.Persist(cache)

		transport.RoundTrip(&req)

		if got := persister.read(); !reflect.DeepEqual(got, tc.cache) {
			t.Errorf("got cache %v, want %v", got, tc.cache)
		}
	}

}
