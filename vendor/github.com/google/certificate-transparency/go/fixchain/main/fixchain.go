package main

import (
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"io"
	"log"
	"net/http"
	"os"
	"sync"

	"github.com/google/certificate-transparency/go/fixchain"
	"github.com/google/certificate-transparency/go/fixchain/ratelimiter"
	"github.com/google/certificate-transparency/go/x509"
)

// Assumes chains to be stores in a file in JSON encoded with the certificates
// in DER format.
func processChains(file string, fl *fixchain.FixAndLog) {
	f, err := os.Open(file)
	if err != nil {
		log.Fatalf("Can't open %q: %s", file, err)
	}
	defer f.Close()

	type Chain struct {
		Chain [][]byte
	}

	dec := json.NewDecoder(f)
	for {
		var m Chain
		if err := dec.Decode(&m); err == io.EOF {
			break
		} else if err != nil {
			log.Fatal(err)
		}
		var chain []*x509.Certificate
		for _, derBytes := range m.Chain {
			cert, err := x509.ParseCertificate(derBytes)
			switch err.(type) {
			case nil, x509.NonFatalErrors:
				// ignore
			default:
				log.Fatalf("can't parse certificate: %s %#v",
					err, derBytes)
			}

			chain = append(chain, cert)
		}
		fl.QueueAllCertsInChain(chain)
	}
}

// A simple function to save the FixErrors that are spat out by the FixAndLog to
// a directory.  contentStore() is the function to alter to store the errors
// wherever/however they need to be stored.  Both logStringErrors() and
// logJSONErrors() use this function as a way of storing the resulting
// FixErrors.
func contentStore(baseDir string, subDir string, content []byte) {
	r := sha256.Sum256(content)
	h := base64.URLEncoding.EncodeToString(r[:])
	d := baseDir + "/" + subDir
	os.MkdirAll(d, 0777)
	fn := d + "/" + h
	f, err := os.Create(fn)
	if err != nil {
		log.Fatalf("Can't create %q: %s", fn, err)
	}
	defer f.Close()
	f.Write(content)
}

func logStringErrors(wg *sync.WaitGroup, errors chan *fixchain.FixError, baseDir string) {
	defer wg.Done()
	for err := range errors {
		contentStore(baseDir, err.TypeString(), []byte(err.String()))
	}
}

func logJSONErrors(wg *sync.WaitGroup, errors chan *fixchain.FixError, baseDir string) {
	defer wg.Done()
	for err := range errors {
		b, err2 := err.MarshalJSON()
		if err2 != nil {
			log.Fatalf("JSON encode failed: %s", err2)
		}
		contentStore(baseDir, err.TypeString(), b)
	}
}

func main() {
	logURL := os.Args[1]
	chainsFile := os.Args[2]
	errDir := os.Args[3]

	var wg sync.WaitGroup
	wg.Add(1)
	errors := make(chan *fixchain.FixError)
	// Functions to log errors as strings or as JSON are provided.
	// As-is, this will log errors as strings.
	go logStringErrors(&wg, errors, errDir)

	limiter := ratelimiter.NewLimiter(1000)
	fl := fixchain.NewFixAndLog(100, 100, errors, &http.Client{}, logURL, limiter, true)

	processChains(chainsFile, fl)

	log.Printf("Wait for fixers and loggers")
	fl.Wait()
	close(errors)
	log.Printf("Wait for errors")
	wg.Wait()
}
