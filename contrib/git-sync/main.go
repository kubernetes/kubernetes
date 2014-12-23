package main // import "github.com/GoogleCloudPlatform/kubernetes/git-sync"

import (
	"flag"
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
	go func() {
		for _ = range time.Tick(pullInterval) {
			gitSync()
		}
	}()
	http.HandleFunc(*handler, func(w http.ResponseWriter, r *http.Request) {
		gitSync()
	})
	log.Fatal(http.ListenAndServe(":8080", nil))
}

func gitSync() {
	if _, err := os.Stat(path.Join(*dest, ".git")); os.IsNotExist(err) {
		cmd := exec.Command("git", "clone", "-b", *branch, *repo, *dest)
		output, err := cmd.CombinedOutput()
		if err != nil {
			log.Printf("command %q : %v", strings.Join(cmd.Args, " "), err)
			return
		}
		log.Println(string(output))
		return
	}
	cmd := exec.Command("git", "pull", "origin", *branch)
	cmd.Dir = *dest
	output, err := cmd.CombinedOutput()
	if err != nil {
		log.Printf("command %q : %v", strings.Join(cmd.Args, " "), err)
		return
	}
	log.Println(string(output))
}
