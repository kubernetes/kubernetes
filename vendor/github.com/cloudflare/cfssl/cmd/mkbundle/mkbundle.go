// mkbundle is a commandline tool for building certificate pool bundles.
// All certificates in the input file paths are checked for revocation and bundled together.
//
// Usage:
//	mkbundle -f bundle_file -nw number_of_workers certificate_file_path ...
package main

import (
	"crypto/x509"
	"encoding/pem"
	"flag"
	"io/ioutil"
	"os"
	"path/filepath"
	"sync"

	"github.com/cloudflare/cfssl/log"
	"github.com/cloudflare/cfssl/revoke"
)

// worker does all the parsing and validation of the certificate(s)
// contained in a single file. It first reads all the data in the
// file, then begins parsing certificates in the file. Those
// certificates are then checked for revocation.
func worker(paths chan string, bundler chan *x509.Certificate, pool *sync.WaitGroup) {
	defer (*pool).Done()
	for {
		path, ok := <-paths
		if !ok {
			return
		}

		log.Infof("Loading %s", path)

		fileData, err := ioutil.ReadFile(path)
		if err != nil {
			log.Warningf("%v", err)
			continue
		}

		for {
			var block *pem.Block
			if len(fileData) == 0 {
				break
			}
			block, fileData = pem.Decode(fileData)
			if block == nil {
				log.Warningf("%s: no PEM data found", path)
				break
			} else if block.Type != "CERTIFICATE" {
				log.Info("Skipping non-certificate")
				continue
			}

			cert, err := x509.ParseCertificate(block.Bytes)
			if err != nil {
				log.Warningf("Invalid certificate: %v", err)
				continue
			}

			log.Infof("Validating %+v", cert.Subject)
			revoked, ok := revoke.VerifyCertificate(cert)
			if !ok {
				log.Warning("Failed to verify certificate.")
			} else if !revoked {
				bundler <- cert
			} else {
				log.Info("Skipping revoked certificate")
			}
		}
	}
}

// supervisor sets up the workers and signals the bundler that all
// certificates have been processed.
func supervisor(paths chan string, bundler chan *x509.Certificate, numWorkers int) {
	var workerPool sync.WaitGroup
	for i := 0; i < numWorkers; i++ {
		workerPool.Add(1)
		go worker(paths, bundler, &workerPool)
	}
	workerPool.Wait()
	close(bundler)
}

// makeBundle opens the file for writing, and listens for incoming
// certificates. These are PEM-encoded and written to file.
func makeBundle(filename string, bundler chan *x509.Certificate) {
	file, err := os.Create(filename)
	if err != nil {
		log.Errorf("%v", err)
		return
	}
	defer file.Close()

	var total int
	for {
		cert, ok := <-bundler
		if !ok {
			break
		}
		block := &pem.Block{
			Type:  "CERTIFICATE",
			Bytes: cert.Raw,
		}
		err = pem.Encode(file, block)
		if err != nil {
			log.Errorf("Failed to write PEM block: %v", err)
			break
		}
		total++
	}
	log.Infof("Wrote %d certificates.", total)
}

// scanFiles walks the files listed in the arguments. These files may
// be either certificate files or directories containing certificates.
func scanFiles(paths chan string) {
	walker := func(path string, info os.FileInfo, err error) error {
		log.Infof("Found %s", path)
		if err != nil {
			return err
		}

		if info.Mode().IsRegular() {
			paths <- path
		}
		return nil
	}

	for _, path := range flag.Args() {
		err := filepath.Walk(path, walker)
		if err != nil {
			log.Errorf("Walk failed: %v", err)
		}
	}
	close(paths)
}

func main() {
	bundleFile := flag.String("f", "cert-bundle.crt", "path to store certificate bundle")
	numWorkers := flag.Int("nw", 4, "number of workers")
	flag.Parse()

	paths := make(chan string)
	bundler := make(chan *x509.Certificate)

	go supervisor(paths, bundler, *numWorkers)
	go scanFiles(paths)

	makeBundle(*bundleFile, bundler)
}
