// Copyright 2015 go-swagger maintainers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package swag

import (
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"time"
)

// LoadHTTPTimeout the default timeout for load requests
var LoadHTTPTimeout = 30 * time.Second

// LoadHTTPBasicAuthUsername the username to use when load requests require basic auth
var LoadHTTPBasicAuthUsername = ""

// LoadHTTPBasicAuthPassword the password to use when load requests require basic auth
var LoadHTTPBasicAuthPassword = ""

// LoadHTTPCustomHeaders an optional collection of custom HTTP headers for load requests
var LoadHTTPCustomHeaders = map[string]string{}

// LoadFromFileOrHTTP loads the bytes from a file or a remote http server based on the path passed in
func LoadFromFileOrHTTP(path string) ([]byte, error) {
	return LoadStrategy(path, os.ReadFile, loadHTTPBytes(LoadHTTPTimeout))(path)
}

// LoadFromFileOrHTTPWithTimeout loads the bytes from a file or a remote http server based on the path passed in
// timeout arg allows for per request overriding of the request timeout
func LoadFromFileOrHTTPWithTimeout(path string, timeout time.Duration) ([]byte, error) {
	return LoadStrategy(path, os.ReadFile, loadHTTPBytes(timeout))(path)
}

// LoadStrategy returns a loader function for a given path or uri
func LoadStrategy(path string, local, remote func(string) ([]byte, error)) func(string) ([]byte, error) {
	if strings.HasPrefix(path, "http") {
		return remote
	}
	return func(pth string) ([]byte, error) {
		upth, err := pathUnescape(pth)
		if err != nil {
			return nil, err
		}

		if strings.HasPrefix(pth, `file://`) {
			if runtime.GOOS == "windows" {
				// support for canonical file URIs on windows.
				// Zero tolerance here for dodgy URIs.
				u, _ := url.Parse(upth)
				if u.Host != "" {
					// assume UNC name (volume share)
					// file://host/share/folder\... ==> \\host\share\path\folder
					// NOTE: UNC port not yet supported
					upth = strings.Join([]string{`\`, u.Host, u.Path}, `\`)
				} else {
					// file:///c:/folder/... ==> just remove the leading slash
					upth = strings.TrimPrefix(upth, `file:///`)
				}
			} else {
				upth = strings.TrimPrefix(upth, `file://`)
			}
		}

		return local(filepath.FromSlash(upth))
	}
}

func loadHTTPBytes(timeout time.Duration) func(path string) ([]byte, error) {
	return func(path string) ([]byte, error) {
		client := &http.Client{Timeout: timeout}
		req, err := http.NewRequest(http.MethodGet, path, nil) //nolint:noctx
		if err != nil {
			return nil, err
		}

		if LoadHTTPBasicAuthUsername != "" && LoadHTTPBasicAuthPassword != "" {
			req.SetBasicAuth(LoadHTTPBasicAuthUsername, LoadHTTPBasicAuthPassword)
		}

		for key, val := range LoadHTTPCustomHeaders {
			req.Header.Set(key, val)
		}

		resp, err := client.Do(req)
		defer func() {
			if resp != nil {
				if e := resp.Body.Close(); e != nil {
					log.Println(e)
				}
			}
		}()
		if err != nil {
			return nil, err
		}

		if resp.StatusCode != http.StatusOK {
			return nil, fmt.Errorf("could not access document at %q [%s] ", path, resp.Status)
		}

		return io.ReadAll(resp.Body)
	}
}
