package main

import (
	"encoding/json"
	"fmt"
	"log"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/bfirsh/funker-go"
	"github.com/docker/docker/hack/integration-cli-on-swarm/agent/types"
)

const (
	// funkerRetryTimeout is for the issue https://github.com/bfirsh/funker/issues/3
	// When all the funker replicas are busy in their own job, we cannot connect to funker.
	funkerRetryTimeout  = 1 * time.Hour
	funkerRetryDuration = 1 * time.Second
)

// ticker is needed for some CI (e.g., on Travis, job is aborted when no output emitted for 10 minutes)
func ticker(d time.Duration) chan struct{} {
	t := time.NewTicker(d)
	stop := make(chan struct{})
	go func() {
		for {
			select {
			case <-t.C:
				log.Printf("tick (just for keeping CI job active) per %s", d.String())
			case <-stop:
				t.Stop()
			}
		}
	}()
	return stop
}

func executeTests(funkerName string, testChunks [][]string) error {
	tickerStopper := ticker(9*time.Minute + 55*time.Second)
	defer func() {
		close(tickerStopper)
	}()
	begin := time.Now()
	log.Printf("Executing %d chunks in parallel, against %q", len(testChunks), funkerName)
	var wg sync.WaitGroup
	var passed, failed uint32
	for chunkID, tests := range testChunks {
		log.Printf("Executing chunk %d (contains %d test filters)", chunkID, len(tests))
		wg.Add(1)
		go func(chunkID int, tests []string) {
			defer wg.Done()
			chunkBegin := time.Now()
			result, err := executeTestChunkWithRetry(funkerName, types.Args{
				ChunkID: chunkID,
				Tests:   tests,
			})
			if result.RawLog != "" {
				for _, s := range strings.Split(result.RawLog, "\n") {
					log.Printf("Log (chunk %d): %s", chunkID, s)
				}
			}
			if err != nil {
				log.Printf("Error while executing chunk %d: %v",
					chunkID, err)
				atomic.AddUint32(&failed, 1)
			} else {
				if result.Code == 0 {
					atomic.AddUint32(&passed, 1)
				} else {
					atomic.AddUint32(&failed, 1)
				}
				log.Printf("Finished chunk %d [%d/%d] with %d test filters in %s, code=%d.",
					chunkID, passed+failed, len(testChunks), len(tests),
					time.Now().Sub(chunkBegin), result.Code)
			}
		}(chunkID, tests)
	}
	wg.Wait()
	// TODO: print actual tests rather than chunks
	log.Printf("Executed %d chunks in %s. PASS: %d, FAIL: %d.",
		len(testChunks), time.Now().Sub(begin), passed, failed)
	if failed > 0 {
		return fmt.Errorf("%d chunks failed", failed)
	}
	return nil
}

func executeTestChunk(funkerName string, args types.Args) (types.Result, error) {
	ret, err := funker.Call(funkerName, args)
	if err != nil {
		return types.Result{}, err
	}
	tmp, err := json.Marshal(ret)
	if err != nil {
		return types.Result{}, err
	}
	var result types.Result
	err = json.Unmarshal(tmp, &result)
	return result, err
}

func executeTestChunkWithRetry(funkerName string, args types.Args) (types.Result, error) {
	begin := time.Now()
	for i := 0; time.Now().Sub(begin) < funkerRetryTimeout; i++ {
		result, err := executeTestChunk(funkerName, args)
		if err == nil {
			log.Printf("executeTestChunk(%q, %d) returned code %d in trial %d", funkerName, args.ChunkID, result.Code, i)
			return result, nil
		}
		if errorSeemsInteresting(err) {
			log.Printf("Error while calling executeTestChunk(%q, %d), will retry (trial %d): %v",
				funkerName, args.ChunkID, i, err)
		}
		// TODO: non-constant sleep
		time.Sleep(funkerRetryDuration)
	}
	return types.Result{}, fmt.Errorf("could not call executeTestChunk(%q, %d) in %v", funkerName, args.ChunkID, funkerRetryTimeout)
}

//  errorSeemsInteresting returns true if err does not seem about https://github.com/bfirsh/funker/issues/3
func errorSeemsInteresting(err error) bool {
	boringSubstrs := []string{"connection refused", "connection reset by peer", "no such host", "transport endpoint is not connected", "no route to host"}
	errS := err.Error()
	for _, boringS := range boringSubstrs {
		if strings.Contains(errS, boringS) {
			return false
		}
	}
	return true
}
