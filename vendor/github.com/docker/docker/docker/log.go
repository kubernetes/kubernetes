package main

import (
	"github.com/Sirupsen/logrus"
	"io"
)

func setLogLevel(lvl logrus.Level) {
	logrus.SetLevel(lvl)
}

func initLogging(stderr io.Writer) {
	logrus.SetOutput(stderr)
}
