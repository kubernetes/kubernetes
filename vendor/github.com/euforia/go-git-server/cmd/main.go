package main

import (
	"log"

	"github.com/euforia/go-git-server"
	"github.com/euforia/go-git-server/repository"
)

var (
	httpAddr = "127.0.0.1:12345"
)

func init() {
	log.SetFlags(log.LstdFlags | log.Lshortfile)
}

func main() {
	repostore := repository.NewMemRepoStore()
	objstores := gitserver.NewMemObjectStorage()

	gh := gitserver.NewGitHTTPService(repostore, objstores)
	rh := gitserver.NewRepoHTTPService(repostore)

	router := gitserver.NewRouter(gh, rh, nil)
	if err := router.Serve(httpAddr); err != nil {
		log.Println(err)
	}
}
