package main

import (
	"flag"
	"fmt"
	"log"
	"math/big"
	"regexp"

	"github.com/google/certificate-transparency/go"
	"github.com/google/certificate-transparency/go/client"
	"github.com/google/certificate-transparency/go/scanner"
)

const (
	// A regex which cannot match any input
	MatchesNothingRegex = "a^"
)

var logUri = flag.String("log_uri", "http://ct.googleapis.com/aviator", "CT log base URI")
var matchSubjectRegex = flag.String("match_subject_regex", ".*", "Regex to match CN/SAN")
var precertsOnly = flag.Bool("precerts_only", false, "Only match precerts")
var serialNumber = flag.String("serial_number", "", "Serial number of certificate of interest")
var batchSize = flag.Int("batch_size", 1000, "Max number of entries to request at per call to get-entries")
var numWorkers = flag.Int("num_workers", 2, "Number of concurrent matchers")
var parallelFetch = flag.Int("parallel_fetch", 2, "Number of concurrent GetEntries fetches")
var startIndex = flag.Int64("start_index", 0, "Log index to start scanning at")
var quiet = flag.Bool("quiet", false, "Don't print out extra logging messages, only matches.")

// Prints out a short bit of info about |cert|, found at |index| in the
// specified log
func logCertInfo(entry *ct.LogEntry) {
	log.Printf("Interesting cert at index %d: CN: '%s'", entry.Index, entry.X509Cert.Subject.CommonName)
}

// Prints out a short bit of info about |precert|, found at |index| in the
// specified log
func logPrecertInfo(entry *ct.LogEntry) {
	log.Printf("Interesting precert at index %d: CN: '%s' Issuer: %s", entry.Index,
		entry.Precert.TBSCertificate.Subject.CommonName, entry.Precert.TBSCertificate.Issuer.CommonName)
}

func createMatcherFromFlags() (scanner.Matcher, error) {
	if *serialNumber != "" {
		log.Printf("Using SerialNumber matcher on %s", *serialNumber)
		var sn big.Int
		_, success := sn.SetString(*serialNumber, 0)
		if !success {
			return nil, fmt.Errorf("Invalid serialNumber %s", *serialNumber)
		}
		return scanner.MatchSerialNumber{SerialNumber: sn}, nil
	} else {
		// Make a regex matcher
		var certRegex *regexp.Regexp
		precertRegex := regexp.MustCompile(*matchSubjectRegex)
		switch *precertsOnly {
		case true:
			certRegex = regexp.MustCompile(MatchesNothingRegex)
		case false:
			certRegex = precertRegex
		}
		return scanner.MatchSubjectRegex{
			CertificateSubjectRegex:    certRegex,
			PrecertificateSubjectRegex: precertRegex}, nil
	}
}

func main() {
	flag.Parse()
	logClient := client.New(*logUri)
	matcher, err := createMatcherFromFlags()
	if err != nil {
		log.Fatal(err)
	}

	opts := scanner.ScannerOptions{
		Matcher:       matcher,
		BatchSize:     *batchSize,
		NumWorkers:    *numWorkers,
		ParallelFetch: *parallelFetch,
		StartIndex:    *startIndex,
		Quiet:         *quiet,
	}
	scanner := scanner.NewScanner(logClient, opts)
	scanner.Scan(logCertInfo, logPrecertInfo)
}
