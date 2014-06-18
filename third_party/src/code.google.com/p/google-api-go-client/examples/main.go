// Copyright 2011 Google Inc. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"encoding/gob"
	"errors"
	"flag"
	"fmt"
	"hash/fnv"
	"io/ioutil"
	"log"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"code.google.com/p/goauth2/oauth"
)

var config = &oauth.Config{
	ClientId:     "", // Set by --clientid or --clientid_file
	ClientSecret: "", // Set by --secret or --secret_file
	Scope:        "", // filled in per-API
	AuthURL:      "https://accounts.google.com/o/oauth2/auth",
	TokenURL:     "https://accounts.google.com/o/oauth2/token",
}

// Flags
var (
	clientId     = flag.String("clientid", "", "OAuth Client ID.  If non-empty, overrides --clientid_file")
	clientIdFile = flag.String("clientid_file", "clientid.dat",
		"Name of a file containing just the project's OAuth Client ID from https://code.google.com/apis/console/")
	secret     = flag.String("secret", "", "OAuth Client Secret.  If non-empty, overrides --secret_file")
	secretFile = flag.String("secret_file", "clientsecret.dat",
		"Name of a file containing just the project's OAuth Client Secret from https://code.google.com/apis/console/")
	cacheToken = flag.Bool("cachetoken", true, "cache the OAuth token")
	debug      = flag.Bool("debug", false, "show HTTP traffic")
)

func usage() {
	fmt.Fprintf(os.Stderr, "Usage: go-api-demo <api-demo-name> [api name args]\n\nPossible APIs:\n\n")
	for n, _ := range demoFunc {
		fmt.Fprintf(os.Stderr, "  * %s\n", n)
	}
	os.Exit(2)
}

func main() {
	flag.Parse()
	if flag.NArg() == 0 {
		usage()
	}

	name := flag.Arg(0)
	demo, ok := demoFunc[name]
	if !ok {
		usage()
	}

	config.Scope = demoScope[name]
	config.ClientId = valueOrFileContents(*clientId, *clientIdFile)
	config.ClientSecret = valueOrFileContents(*secret, *secretFile)

	c := getOAuthClient(config)
	demo(c, flag.Args()[1:])
}

var (
	demoFunc  = make(map[string]func(*http.Client, []string))
	demoScope = make(map[string]string)
)

func registerDemo(name, scope string, main func(c *http.Client, argv []string)) {
	if demoFunc[name] != nil {
		panic(name + " already registered")
	}
	demoFunc[name] = main
	demoScope[name] = scope
}

func osUserCacheDir() string {
	switch runtime.GOOS {
	case "darwin":
		return filepath.Join(os.Getenv("HOME"), "Library", "Caches")
	case "linux", "freebsd":
		return filepath.Join(os.Getenv("HOME"), ".cache")
	}
	log.Printf("TODO: osUserCacheDir on GOOS %q", runtime.GOOS)
	return "."
}

func tokenCacheFile(config *oauth.Config) string {
	hash := fnv.New32a()
	hash.Write([]byte(config.ClientId))
	hash.Write([]byte(config.ClientSecret))
	hash.Write([]byte(config.Scope))
	fn := fmt.Sprintf("go-api-demo-tok%v", hash.Sum32())
	return filepath.Join(osUserCacheDir(), url.QueryEscape(fn))
}

func tokenFromFile(file string) (*oauth.Token, error) {
	if !*cacheToken {
		return nil, errors.New("--cachetoken is false")
	}
	f, err := os.Open(file)
	if err != nil {
		return nil, err
	}
	t := new(oauth.Token)
	err = gob.NewDecoder(f).Decode(t)
	return t, err
}

func saveToken(file string, token *oauth.Token) {
	f, err := os.Create(file)
	if err != nil {
		log.Printf("Warning: failed to cache oauth token: %v", err)
		return
	}
	defer f.Close()
	gob.NewEncoder(f).Encode(token)
}

func condDebugTransport(rt http.RoundTripper) http.RoundTripper {
	if *debug {
		return &logTransport{rt}
	}
	return rt
}

func getOAuthClient(config *oauth.Config) *http.Client {
	cacheFile := tokenCacheFile(config)
	token, err := tokenFromFile(cacheFile)
	if err != nil {
		token = tokenFromWeb(config)
		saveToken(cacheFile, token)
	} else {
		log.Printf("Using cached token %#v from %q", token, cacheFile)
	}

	t := &oauth.Transport{
		Token:     token,
		Config:    config,
		Transport: condDebugTransport(http.DefaultTransport),
	}
	return t.Client()
}

func tokenFromWeb(config *oauth.Config) *oauth.Token {
	ch := make(chan string)
	randState := fmt.Sprintf("st%d", time.Now().UnixNano())
	ts := httptest.NewServer(http.HandlerFunc(func(rw http.ResponseWriter, req *http.Request) {
		if req.URL.Path == "/favicon.ico" {
			http.Error(rw, "", 404)
			return
		}
		if req.FormValue("state") != randState {
			log.Printf("State doesn't match: req = %#v", req)
			http.Error(rw, "", 500)
			return
		}
		if code := req.FormValue("code"); code != "" {
			fmt.Fprintf(rw, "<h1>Success</h1>Authorized.")
			rw.(http.Flusher).Flush()
			ch <- code
			return
		}
		log.Printf("no code")
		http.Error(rw, "", 500)
	}))
	defer ts.Close()

	config.RedirectURL = ts.URL
	authUrl := config.AuthCodeURL(randState)
	go openUrl(authUrl)
	log.Printf("Authorize this app at: %s", authUrl)
	code := <-ch
	log.Printf("Got code: %s", code)

	t := &oauth.Transport{
		Config:    config,
		Transport: condDebugTransport(http.DefaultTransport),
	}
	_, err := t.Exchange(code)
	if err != nil {
		log.Fatalf("Token exchange error: %v", err)
	}
	return t.Token
}

func openUrl(url string) {
	try := []string{"xdg-open", "google-chrome", "open"}
	for _, bin := range try {
		err := exec.Command(bin, url).Run()
		if err == nil {
			return
		}
	}
	log.Printf("Error opening URL in browser.")
}

func valueOrFileContents(value string, filename string) string {
	if value != "" {
		return value
	}
	slurp, err := ioutil.ReadFile(filename)
	if err != nil {
		log.Fatalf("Error reading %q: %v", filename, err)
	}
	return strings.TrimSpace(string(slurp))
}
