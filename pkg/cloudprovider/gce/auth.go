/*
Copyright 2014 Google Inc. All rights reserved.

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

package gce_cloud

import (
	"encoding/json"
	"fmt"
	"hash/fnv"
	"io/ioutil"
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
	compute "code.google.com/p/google-api-go-client/compute/v1"
	"github.com/golang/glog"
)

var config = &oauth.Config{
	// this Id & Secret are located under the google-containers project on gce
	ClientId:     "255964991331-b0l3n9c5pqc0u0ijtniv8vls226d3d5j.apps.googleusercontent.com",
	ClientSecret: "BWm6fPAY2gS1jaRT-Xn2y-uT",
	Scope: strings.Join([]string{
		compute.DevstorageRead_onlyScope,
		compute.ComputeScope,
	}, " "),
	AuthURL:  "https://accounts.google.com/o/oauth2/auth",
	TokenURL: "https://accounts.google.com/o/oauth2/token",
}

// createOAuth2Client creates an Oauth client either from a cached token or a new token from the web.
func createOAuthClient() *http.Client {
	cacheFile := tokenCacheFile(config)
	token, err := loadToken(cacheFile)
	if err != nil {
		glog.Warningf("found a token, but couldn't open it.")
	}
	if token == nil {
		token = tokenFromWeb(config)
		saveToken(cacheFile, token)
	}
	t := &oauth.Transport{
		Token:     token,
		Config:    config,
		Transport: http.DefaultTransport,
	}
	return t.Client()
}

func saveToken(file string, token *oauth.Token) error {
	data, err := json.Marshal(token)
	if err != nil {
		return err
	}
	return ioutil.WriteFile(file, data, 0600)
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
			glog.Infof("State doesn't match: req = %#v", req)
			http.Error(rw, "", 500)
			return
		}
		if code := req.FormValue("code"); code != "" {
			fmt.Fprintf(rw, "<h1>Success</h1>Authorized.")
			rw.(http.Flusher).Flush()
			ch <- code
			return
		}
		glog.Infof("no code")
		http.Error(rw, "", 500)
	}))
	defer ts.Close()

	config.RedirectURL = ts.URL
	authUrl := config.AuthCodeURL(randState)
	go openURL(authUrl)
	glog.Infof("Authorize this app at:\n%s", authUrl)
	code := <-ch
	glog.Infof("Got code: %s", code)

	t := &oauth.Transport{
		Config:    config,
		Transport: http.DefaultTransport,
	}
	_, err := t.Exchange(code)
	if err != nil {
		glog.Fatalf("Token exchange error: %v", err)
	}
	return t.Token
}

func tokenCacheFile(config *oauth.Config) string {
	hash := fnv.New32a()
	hash.Write([]byte(config.ClientId))
	hash.Write([]byte(config.ClientSecret))
	hash.Write([]byte(config.Scope))
	fn := fmt.Sprintf("kube-tok%v", hash.Sum32())
	return filepath.Join(osUserCacheDir(), url.QueryEscape(fn))
}

func loadToken(file string) (*oauth.Token, error) {
	data, err := ioutil.ReadFile(file)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	var token oauth.Token
	err = json.Unmarshal(data, &token)
	return &token, err
}

func openURL(url string) {
	try := []string{"xdg-open", "google-chrome", "open"}
	for _, bin := range try {
		err := exec.Command(bin, url).Run()
		if err == nil {
			return
		}
	}
	glog.Fatalf("Error opening URL in browser.")
}

func osUserCacheDir() string {
	switch runtime.GOOS {
	case "darwin":
		return filepath.Join(os.Getenv("HOME"), "Library", "Caches")
	case "linux", "freebsd":
		return filepath.Join(os.Getenv("HOME"), ".cache")
	}
	return "."
}
