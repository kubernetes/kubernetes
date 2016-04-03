/*
Package logrus is a structured logger for Go, completely API compatible with the standard library logger.


The simplest way to use Logrus is simply the package-level exported logger:

  package main

  import (
    log "github.com/Sirupsen/logrus"
  )

  func main() {
    log.WithFields(log.Fields{
      "animal": "walrus",
      "number": 1,
      "size":   10,
    }).Info("A walrus appears")
  }

Output:
  time="2015-09-07T08:48:33Z" level=info msg="A walrus appears" animal=walrus number=1 size=10

For a full guide visit https://github.com/Sirupsen/logrus
*/
package logrus
