package concurrent

import (
	"os"
	"log"
	"io/ioutil"
)

// ErrorLogger is used to print out error, can be set to writer other than stderr
var ErrorLogger = log.New(os.Stderr, "", 0)

// InfoLogger is used to print informational message, default to off
var InfoLogger = log.New(ioutil.Discard, "", 0)
