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

var interval = flag.String("interval", env("INTERVAL", "60s"), "git pull interval")
var repo = flag.String("repo", env("REPO", ""), "git repo url")
var branch = flag.String("branch", env("BRANCH", "master"), "git branch")
var hook = flag.String("hook", env("HOOK", "/"), "web hook path")
var dest = flag.String("dest", env("DEST", ""), "destination path")

func env(key, def string) string {
	if env := os.Getenv(key); env != "" {
		return env
	}
	return def
}

const usage = "usage: REPO= DEST= [INTERVAL= BRANCH= HOOK=] git-sync -repo GIT_REPO_URL -dest PATH [-interval -branch -hook]"

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
	http.HandleFunc(*hook, func(w http.ResponseWriter, r *http.Request) {
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
