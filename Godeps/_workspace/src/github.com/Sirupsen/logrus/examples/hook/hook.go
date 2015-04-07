package main

import (
	"github.com/Sirupsen/logrus"
	"github.com/Sirupsen/logrus/hooks/airbrake"
	"github.com/tobi/airbrake-go"
)

var log = logrus.New()

func init() {
	log.Formatter = new(logrus.TextFormatter) // default
	log.Hooks.Add(new(logrus_airbrake.AirbrakeHook))
}

func main() {
	airbrake.Endpoint = "https://exceptions.whatever.com/notifier_api/v2/notices.xml"
	airbrake.ApiKey = "whatever"
	airbrake.Environment = "production"

	log.WithFields(logrus.Fields{
		"animal": "walrus",
		"size":   10,
	}).Info("A group of walrus emerges from the ocean")

	log.WithFields(logrus.Fields{
		"omg":    true,
		"number": 122,
	}).Warn("The group's number increased tremendously!")

	log.WithFields(logrus.Fields{
		"omg":    true,
		"number": 100,
	}).Fatal("The ice breaks!")
}
