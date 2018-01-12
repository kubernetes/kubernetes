/*
Copyright 2018 The Kubernetes Authors.

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

package main

import (
	"flag"
	"fmt"
	"log"
	"net/http"
	"strings"

	"github.com/euforia/go-git-server"

	"k8s.io/kubernetes/cmd/staginghub/store"
)

func init() {
	log.SetFlags(log.LstdFlags | log.Lshortfile)
}

func main() {
	port := flag.Int("p", 12345, "The TCP port to listen on.")
	addr := flag.String("l", "localhost", "The IP or hostname to listen on.")
	flag.Parse()

	repostore, err := store.NewSubdirRepoStore()
	if err != nil {
		log.Fatal(err)
	}

	objstores, err := store.NewPwdObjectStorage()
	if err != nil {
		log.Fatal(err)
	}

	gh := gitserver.NewGitHTTPService(repostore, objstores)
	rh := WithEmptyGoMetadata(fmt.Sprintf("%s:%d", *addr, *port), gitserver.NewRepoHTTPService(repostore))

	router := gitserver.NewRouter(gh, rh, nil)
	if err := router.Serve(fmt.Sprintf("%s:%d", *addr, *port)); err != nil {
		log.Println(err)
	}
}

// WithEmptyGoMetadata returns a HTTP snippet for requests with go-get=1 because
// dep expected that.
func WithEmptyGoMetadata(host string, delegate http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		if req.URL.Query().Get("go-get") != "" {
			ps := strings.Split(req.URL.Path, "/")
			repo := strings.TrimSuffix(ps[len(ps)-1], ".git")

			w.Header().Set("Content-Type", "text/html; charset=utf-8")
			w.Write([]byte(fmt.Sprintf(`<html><head>
<meta name="go-import" content="%s git http://%s">
<meta name="go-import" content="k8s.io/%s git http://%s">
</head><body></body></html>`,
				host+req.URL.Path, host+req.URL.Path,
				repo, host+req.URL.Path)))
			return
		}

		delegate.ServeHTTP(w, req)
	})
}
