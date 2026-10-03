/*
Copyright The Kubernetes Authors.

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

package resource

// testdata/quantity_decode_corpus.json records how the ParseQuantity of the
// last release, decodeCorpusRelease, decodes every input of the parse grid
// (quantity_exact_grid_test.go). An apiserver of that release persisted
// whatever spelling it was sent and decodes it the same way on every read, so
// for any input it decoded, a different decode today changes the value of an
// object that may already be stored. The file name carries no version: the
// format of Quantity must not change across releases, and a change since the
// release is tracked in parseRegressions or parseApprovedChanges, not in a
// file per release.
//
// The file is generated, never edited. It was generated against the v1.37.0
// tag. To regenerate it, copy this file and quantity_exact_grid_test.go into a
// checkout of the release tag, set decodeCorpusRelease to that tag, and run,
// from the repository root:
//
//	QUANTITY_DECODE_CORPUS_OUT=$PWD/quantity_decode_corpus.json \
//	  go test ./staging/src/k8s.io/apimachinery/pkg/api/resource/ \
//	  -run '^TestGenerateQuantityDecodeCorpus$' -count=1 -timeout 60m
//
// then move the output here. Inputs on which v1.37.0 does not return within
// decodeCorpusSilence are recorded as "hang". The generator runs the decodes in
// child processes (QUANTITY_DECODE_CORPUS_WORKERS of them, default 4) and kills
// a child that goes silent, so a hang costs one restart, not the run.
//
// The generator uses only the standard library and fields that exist in
// v1.37.0 (Quantity.i, Quantity.d, Quantity.Format), so the same file works
// unchanged in both trees.

import (
	"bufio"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"math/big"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"testing"
	"time"
)

// decodeCorpusSHA256 is the digest of decodeCorpusFile as generated.
const decodeCorpusSHA256 = "96fd69236e6a0ebc8f0767203985b9ac80987295e58cb6aecde9fb6ed738b198"

const (
	decodeCorpusOutEnv     = "QUANTITY_DECODE_CORPUS_OUT"
	decodeCorpusWorkersEnv = "QUANTITY_DECODE_CORPUS_WORKERS"
	decodeCorpusChildEnv   = "QUANTITY_DECODE_CORPUS_CHILD"
	decodeCorpusWorkerEnv  = "QUANTITY_DECODE_CORPUS_WORKER"
	decodeCorpusStartEnv   = "QUANTITY_DECODE_CORPUS_START"
	// decodeCorpusInputsEnv names a JSON file of inputs that replaces the
	// parse grid in the child processes (used by
	// TestQuantityLawParseHangPredicate).
	decodeCorpusInputsEnv = "QUANTITY_DECODE_CORPUS_INPUTS"
	decodeCorpusFile      = "testdata/quantity_decode_corpus.json"
	// decodeCorpusRelease is the release decodeCorpusFile was generated
	// against.
	decodeCorpusRelease = "v1.37.0"
	// decodeCorpusSilence is how long a child may go without reporting a
	// decode before it is killed and the pending input recorded as a hang. A
	// decode that returns does so in microseconds.
	decodeCorpusSilence = 2 * time.Second
	// decodeCorpusReady is the first line a child prints. The silence timer
	// starts at it, so a slow start cannot read as a hang; until then the
	// parent waits up to decodeCorpusStartTimeout. This file declares its own
	// constants so that it still builds on its own in a release checkout.
	decodeCorpusReady        = "QUANTITY_DECODE_CORPUS_READY"
	decodeCorpusStartTimeout = time.Minute
)

// decodeRecord is one decode: Outcome is "ok", "err" or "hang"; for "ok",
// Value is the exact decoded value as <digits>e<exponent> with no trailing
// zeros in the digits ("0" for zero), and Format is the decoded format.
type decodeRecord struct {
	Input   string
	Outcome string
	Value   string
	Format  Format
}

// decodeValue returns q's exact value in the decodeRecord.Value form.
func decodeValue(q *Quantity) string {
	var num *big.Int
	var exp int64
	if q.d.Dec != nil {
		num = new(big.Int).Set(q.d.Dec.UnscaledBig())
		exp = -int64(q.d.Dec.Scale())
	} else {
		num = big.NewInt(q.i.value)
		exp = int64(q.i.scale)
	}
	if num.Sign() == 0 {
		return "0"
	}
	ten, rem := big.NewInt(10), new(big.Int)
	for {
		quo, r := new(big.Int).QuoRem(num, ten, rem)
		if r.Sign() != 0 {
			break
		}
		num, exp = quo, exp+1
	}
	return num.String() + "e" + strconv.FormatInt(exp, 10)
}

// decodeOne decodes input with the library under test.
func decodeOne(input string) decodeRecord {
	q, err := ParseQuantity(input)
	if err != nil {
		return decodeRecord{Input: input, Outcome: "err"}
	}
	return decodeRecord{Input: input, Outcome: "ok", Value: decodeValue(&q), Format: q.Format}
}

// loadDecodeCorpus reads the recorded decodes of decodeCorpusRelease, keyed by
// input.
func loadDecodeCorpus(t *testing.T) map[string]decodeRecord {
	t.Helper()
	raw, err := os.ReadFile(decodeCorpusFile)
	if err != nil {
		t.Fatalf("read %s: %v", decodeCorpusFile, err)
	}
	// The file is generated, so any edit by hand changes this digest; a
	// regeneration updates it.
	if sum := fmt.Sprintf("%x", sha256.Sum256(raw)); sum != decodeCorpusSHA256 {
		t.Fatalf("%s has sha256 %s, recorded %s: regenerate it rather than editing it, then update decodeCorpusSHA256", decodeCorpusFile, sum, decodeCorpusSHA256)
	}
	var corpus struct {
		Release string     `json:"release"`
		Records [][]string `json:"records"`
	}
	if err := json.Unmarshal(raw, &corpus); err != nil {
		t.Fatalf("parse %s: %v", decodeCorpusFile, err)
	}
	if corpus.Release != decodeCorpusRelease {
		t.Fatalf("%s was generated against %q, the test compares with %q", decodeCorpusFile, corpus.Release, decodeCorpusRelease)
	}
	inputs := parseGridInputs()
	if len(corpus.Records) != len(inputs) {
		t.Fatalf("%s has %d records for %d grid inputs; regenerate it", decodeCorpusFile, len(corpus.Records), len(inputs))
	}
	recorded := make(map[string]decodeRecord, len(corpus.Records))
	for i, r := range corpus.Records {
		if len(r) != 4 || r[0] != inputs[i] {
			t.Fatalf("%s: record %d is %q, grid input %q; regenerate it", decodeCorpusFile, i, r, inputs[i])
		}
		recorded[r[0]] = decodeRecord{Input: r[0], Outcome: r[1], Value: r[2], Format: Format(r[3])}
	}
	return recorded
}

// TestGenerateQuantityDecodeCorpus writes the decode of every grid input by the
// library in this tree to $QUANTITY_DECODE_CORPUS_OUT. It does nothing unless
// that variable is set; see the file comment.
func TestGenerateQuantityDecodeCorpus(t *testing.T) {
	inputs := parseGridInputs()
	if path := os.Getenv(decodeCorpusInputsEnv); path != "" {
		raw, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		if err := json.Unmarshal(raw, &inputs); err != nil {
			t.Fatal(err)
		}
	}
	workers := 4
	if n, err := strconv.Atoi(os.Getenv(decodeCorpusWorkersEnv)); err == nil && n > 0 {
		workers = n
	}
	if os.Getenv(decodeCorpusChildEnv) == "1" {
		worker, _ := strconv.Atoi(os.Getenv(decodeCorpusWorkerEnv))
		start, _ := strconv.Atoi(os.Getenv(decodeCorpusStartEnv))
		out := bufio.NewWriter(os.Stdout)
		_, _ = fmt.Fprintln(out, decodeCorpusReady)
		_ = out.Flush()
		for i := start; i < len(inputs); i++ {
			if i%workers != worker {
				continue
			}
			r := decodeOne(inputs[i])
			line, _ := json.Marshal([]string{strconv.Itoa(i), r.Outcome, r.Value, string(r.Format)})
			_, _ = fmt.Fprintf(out, "%s\n", line)
			_ = out.Flush()
		}
		os.Exit(0)
	}
	path := os.Getenv(decodeCorpusOutEnv)
	if path == "" {
		t.Skipf("set %s to regenerate %s", decodeCorpusOutEnv, decodeCorpusFile)
	}

	records := make([]decodeRecord, len(inputs))
	done := make(chan error, workers)
	for w := range workers {
		go func() { done <- runDecodeWorker(inputs, workers, w, records, nil) }()
	}
	for range workers {
		if err := <-done; err != nil {
			t.Fatal(err)
		}
	}

	var b strings.Builder
	header, _ := json.Marshal(fmt.Sprintf("ParseQuantity of every parseGridInputs() input; regenerate with %s, see quantity_decode_corpus_test.go", decodeCorpusOutEnv))
	release, _ := json.Marshal(decodeCorpusRelease)
	fmt.Fprintf(&b, "{\n\"generated\": %s,\n\"release\": %s,\n\"records\": [\n", header, release)
	for i, r := range records {
		line, _ := json.Marshal([]string{r.Input, r.Outcome, r.Value, string(r.Format)})
		sep := ","
		if i == len(records)-1 {
			sep = ""
		}
		fmt.Fprintf(&b, "%s%s\n", line, sep)
	}
	b.WriteString("]\n}\n")
	if err := os.WriteFile(path, []byte(b.String()), 0o644); err != nil {
		t.Fatal(err)
	}
	hangs := 0
	for _, r := range records {
		if r.Outcome == "hang" {
			hangs++
		}
	}
	t.Logf("wrote %d records (%d hang) to %s", len(records), hangs, path)
}

// runDecodeWorker decodes the inputs with index%workers == worker in child
// processes, restarting past any input on which a child goes silent. extraEnv
// is added to the children's environment; it must make them see the same
// inputs.
func runDecodeWorker(inputs []string, workers, worker int, records []decodeRecord, extraEnv []string) error {
	next := worker
	for next < len(inputs) {
		// The child gets its own deadline, the one the regeneration command
		// gives the parent, so it stops even if the parent times out first.
		cmd := exec.Command(os.Args[0], "-test.run=^TestGenerateQuantityDecodeCorpus$", "-test.count=1",
			"-test.timeout=60m")
		cmd.Env = append(os.Environ(), decodeCorpusChildEnv+"=1",
			decodeCorpusWorkersEnv+"="+strconv.Itoa(workers),
			decodeCorpusWorkerEnv+"="+strconv.Itoa(worker),
			decodeCorpusStartEnv+"="+strconv.Itoa(next),
			"GORACE="+strings.TrimSpace(os.Getenv("GORACE")+" atexit_sleep_ms=0"))
		cmd.Env = append(cmd.Env, extraEnv...)
		stdout, err := cmd.StdoutPipe()
		if err != nil {
			return err
		}
		if err := cmd.Start(); err != nil {
			return err
		}
		lines := make(chan []string)
		go func() {
			defer close(lines)
			sc := bufio.NewScanner(stdout)
			sc.Buffer(make([]byte, 1<<20), 1<<20)
			for sc.Scan() {
				if sc.Text() == decodeCorpusReady {
					lines <- nil
					continue
				}
				var rec []string
				if json.Unmarshal(sc.Bytes(), &rec) == nil && len(rec) == 4 {
					lines <- rec
				}
			}
		}()
		silent, ready := false, false
	read:
		for {
			wait := decodeCorpusSilence
			if !ready {
				wait = decodeCorpusStartTimeout
			}
			select {
			case rec, ok := <-lines:
				if !ok {
					break read
				}
				if rec == nil {
					ready = true
					continue
				}
				i, _ := strconv.Atoi(rec[0])
				records[i] = decodeRecord{Input: inputs[i], Outcome: rec[1], Value: rec[2], Format: Format(rec[3])}
				next = i + workers
			case <-time.After(wait):
				if !ready {
					_ = cmd.Process.Kill()
					for range lines {
					}
					_ = cmd.Wait()
					return fmt.Errorf("decode child for worker %d was not ready within %s", worker, decodeCorpusStartTimeout)
				}
				silent = true
				break read
			}
		}
		_ = cmd.Process.Kill()
		for range lines {
		}
		_ = cmd.Wait()
		switch {
		case next >= len(inputs):
			// Every input of this worker has its record; a child that goes
			// quiet after its last one is not a hang.
		case silent:
			records[next] = decodeRecord{Input: inputs[next], Outcome: "hang"}
			next += workers
		default:
			return fmt.Errorf("decode child for worker %d exited before input %d", worker, next)
		}
	}
	return nil
}
