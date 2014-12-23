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

// git-sync is a command that periodically sync a git repository to a local directory.

package main // import "github.com/GoogleCloudPlatform/kubernetes/git-sync"

import (
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path"
	"strings"
	"time"
)

var interval = flag.String("interval", env("GIT_SYNC_INTERVAL", "60s"), "git pull interval")
var repo = flag.String("repo", env("GIT_SYNC_REPO", ""), "git repo url")
var branch = flag.String("branch", env("GIT_SYNC_BRANCH", "master"), "git branch")
var handler = flag.String("handler", env("GIT_SYNC_HANDLER", "/"), "web hook handler")
var dest = flag.String("dest", env("GIT_SYNC_DEST", ""), "destination path")

func env(key, def string) string {
	if env := os.Getenv(key); env != "" {
		return env
	}
	return def
}

const usage = "usage: GIT_SYNC_REPO= GIT_SYNC_DEST= [GIT_SYNC_INTERVAL= GIT_SYNC_BRANCH= GIT_SYNC_HANDLER=] git-sync -repo GIT_REPO_URL -dest PATH [-interval -branch -handler]"

func main() {
	flag.Parse()
	if *repo == "" || *dest == "" {
		flag.Usage()
		log.Fatal(usage)
	}
	pullInterval, err := time.ParseDuration(*interval)
	if err != nil {
		log.Fatalf("error parsing time duration %q: %v", *interval, err)
	}
	if _, err := exec.LookPath("git"); err != nil {
		log.Fatalf("required git executable not found: %v", err)
	}
	repo, err := NewRepo()
	if err != nil {
		log.Fatalf("error creating repo: %v", err)
	}
	syncc := make(chan struct{})
	tick := time.Tick(pullInterval)
	go func() {
		for {
			repo.Sync()
			select {
			case <-tick:
			case <-syncc:
			}
		}
	}()
	http.HandleFunc(*handler, func(w http.ResponseWriter, r *http.Request) {
		syncc <- struct{}{}
	})
	log.Fatal(http.ListenAndServe(":8080", nil))
}

type Repo struct {
	basePath   string
	mirrorPath string
	lastRev    string
}

// NewRepo initalize a local bare repository mirror.
func NewRepo() (*Repo, error) {
	mirrorRepoPath := path.Join(*dest, ".git")
	_, err := os.Stat(mirrorRepoPath)
	if err == nil {
		log.Printf("found existing mirror repo %q", mirrorRepoPath)
		return &Repo{
			basePath:   *dest,
			mirrorPath: mirrorRepoPath,
		}, nil
	}
	if !os.IsNotExist(err) {
		return nil, fmt.Errorf("error checking repo %q: %v", mirrorRepoPath, err)
	}
	cmd := exec.Command("git", "clone", "--mirror", "-b", *branch, *repo, mirrorRepoPath)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return nil, fmt.Errorf("error cloning repo %q: %v:", strings.Join(cmd.Args, " "), err)
	}
	log.Printf("clone %q: %s", *repo, string(output))
	return &Repo{
		basePath:   *dest,
		mirrorPath: mirrorRepoPath,
	}, nil
}

// Sync fetch new revision from the origin remote in the bare repository
//      create a new checkout named after the revision
//	update HEAD symlink to point to the new revision
func (r *Repo) Sync() {
	cmd := exec.Command("git", "fetch", "origin", *branch)
	cmd.Dir = r.mirrorPath
	output, err := cmd.CombinedOutput()
	if err != nil {
		log.Printf("error running command %q: %v", strings.Join(cmd.Args, " "), err)
		return
	}
	log.Printf("fetch: %s", string(output))
	cmd = exec.Command("git", "rev-parse", "HEAD")
	cmd.Dir = r.mirrorPath
	output, err = cmd.CombinedOutput()
	if err != nil {
		log.Printf("error running command %q: %v", strings.Join(cmd.Args, " "), err)
		return
	}
	rev := strings.TrimSpace(string(output))
	if rev == r.lastRev {
		log.Printf("no new rev since last check %q", rev)
		return
	}
	r.lastRev = rev
	log.Printf("HEAD is: %q", rev)
	repoPath := path.Join(r.basePath, rev)
	_, err = os.Stat(repoPath)
	if err == nil {
		log.Printf("found existing repo: %q", repoPath)
		return
	}
	if !os.IsNotExist(err) {
		log.Printf("error stating repo %q: %v", repoPath, err)
		return
	}
	cmd = exec.Command("git", "clone", r.mirrorPath, repoPath)
	output, err = cmd.CombinedOutput()
	if err != nil {
		log.Printf("error running command %q : %v", strings.Join(cmd.Args, " "), err)
		return
	}
	log.Printf("clone %q: %v", repoPath, string(output))
	tempPath := path.Join(r.basePath, "HEAD."+rev)
	if err := os.Symlink(rev, tempPath); err != nil {
		log.Printf("error creating temporary symlink %q: %v", tempPath, err)
		return
	}
	linkPath := path.Join(r.basePath, "HEAD")
	if err := os.Rename(tempPath, linkPath); err != nil {
		log.Printf("error moving symlink %q: %v", linkPath, err)
		return
	}
}
