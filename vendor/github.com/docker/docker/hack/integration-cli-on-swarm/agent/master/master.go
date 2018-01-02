package main

import (
	"errors"
	"flag"
	"io/ioutil"
	"log"
	"strings"
)

func main() {
	if err := xmain(); err != nil {
		log.Fatalf("fatal error: %v", err)
	}
}

func xmain() error {
	workerService := flag.String("worker-service", "", "Name of worker service")
	chunks := flag.Int("chunks", 0, "Number of chunks")
	input := flag.String("input", "", "Path to input file")
	randSeed := flag.Int64("rand-seed", int64(0), "Random seed")
	shuffle := flag.Bool("shuffle", false, "Shuffle the input so as to mitigate makespan nonuniformity")
	flag.Parse()
	if *workerService == "" {
		return errors.New("worker-service unset")
	}
	if *chunks == 0 {
		return errors.New("chunks unset")
	}
	if *input == "" {
		return errors.New("input unset")
	}

	tests, err := loadTests(*input)
	if err != nil {
		return err
	}
	testChunks := chunkTests(tests, *chunks, *shuffle, *randSeed)
	log.Printf("Loaded %d tests (%d chunks)", len(tests), len(testChunks))
	return executeTests(*workerService, testChunks)
}

func chunkTests(tests []string, numChunks int, shuffle bool, randSeed int64) [][]string {
	// shuffling (experimental) mitigates makespan nonuniformity
	// Not sure this can cause some locality problem..
	if shuffle {
		shuffleStrings(tests, randSeed)
	}
	return chunkStrings(tests, numChunks)
}

func loadTests(filename string) ([]string, error) {
	b, err := ioutil.ReadFile(filename)
	if err != nil {
		return nil, err
	}
	var tests []string
	for _, line := range strings.Split(string(b), "\n") {
		s := strings.TrimSpace(line)
		if s != "" {
			tests = append(tests, s)
		}
	}
	return tests, nil
}
