package main

import (
	"flag"
	"fmt"
	"log"
	"time"

	"github.com/bfirsh/funker-go"
	"github.com/docker/distribution/reference"
	"github.com/docker/docker/hack/integration-cli-on-swarm/agent/types"
)

func main() {
	if err := xmain(); err != nil {
		log.Fatalf("fatal error: %v", err)
	}
}

func validImageDigest(s string) bool {
	return reference.DigestRegexp.FindString(s) != ""
}

func xmain() error {
	workerImageDigest := flag.String("worker-image-digest", "", "Needs to be the digest of this worker image itself")
	dryRun := flag.Bool("dry-run", false, "Dry run")
	keepExecutor := flag.Bool("keep-executor", false, "Do not auto-remove executor containers, which is used for running privileged programs on Swarm")
	flag.Parse()
	if !validImageDigest(*workerImageDigest) {
		// Because of issue #29582.
		// `docker service create localregistry.example.com/blahblah:latest` pulls the image data to local, but not a tag.
		// So, `docker run localregistry.example.com/blahblah:latest` fails: `Unable to find image 'localregistry.example.com/blahblah:latest' locally`
		return fmt.Errorf("worker-image-digest must be a digest, got %q", *workerImageDigest)
	}
	executor := privilegedTestChunkExecutor(!*keepExecutor)
	if *dryRun {
		executor = dryTestChunkExecutor()
	}
	return handle(*workerImageDigest, executor)
}

func handle(workerImageDigest string, executor testChunkExecutor) error {
	log.Printf("Waiting for a funker request")
	return funker.Handle(func(args *types.Args) types.Result {
		log.Printf("Executing chunk %d, contains %d test filters",
			args.ChunkID, len(args.Tests))
		begin := time.Now()
		code, rawLog, err := executor(workerImageDigest, args.Tests)
		if err != nil {
			log.Printf("Error while executing chunk %d: %v", args.ChunkID, err)
			if code == 0 {
				// Make sure this is a failure
				code = 1
			}
			return types.Result{
				ChunkID: args.ChunkID,
				Code:    int(code),
				RawLog:  rawLog,
			}
		}
		elapsed := time.Now().Sub(begin)
		log.Printf("Finished chunk %d, code=%d, elapsed=%v", args.ChunkID, code, elapsed)
		return types.Result{
			ChunkID: args.ChunkID,
			Code:    int(code),
			RawLog:  rawLog,
		}
	})
}
