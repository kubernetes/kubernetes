package main

import (
	"compress/zlib"
	"encoding/gob"
	"flag"
	"io"
	"log"
	"os"

	"github.com/google/certificate-transparency/go/preload"
)

var sctFile = flag.String("sct_file", "", "File to load SCTs & leaf data from")

func main() {
	flag.Parse()
	var sctReader io.ReadCloser
	if *sctFile == "" {
		log.Fatal("Must specify --sct_file")
	}

	sctFileReader, err := os.Open(*sctFile)
	if err != nil {
		log.Fatal(err)
	}
	sctReader, err = zlib.NewReader(sctFileReader)
	if err != nil {
		log.Fatal(err)
	}
	defer func() {
		err := sctReader.Close()
		if err != nil && err != io.EOF {
			log.Fatalf("Error closing file: %s", err)
		}
	}()

	// TODO(alcutter) should probably store this stuff in a protobuf really.
	decoder := gob.NewDecoder(sctReader)
	var addedCert preload.AddedCert
	numAdded := 0
	numFailed := 0
	for {
		err = decoder.Decode(&addedCert)
		if err != nil {
			break
		}
		if addedCert.AddedOk {
			log.Println(addedCert.SignedCertificateTimestamp)
			numAdded++
		} else {
			log.Printf("Cert was not added: %s", addedCert.ErrorMessage)
			numFailed++
		}
	}
	log.Printf("Num certs added: %d, num failed: %d\n", numAdded, numFailed)
}
