package logrus_airbrake

import (
	"github.com/Sirupsen/logrus"
	"github.com/tobi/airbrake-go"
)

// AirbrakeHook to send exceptions to an exception-tracking service compatible
// with the Airbrake API. You must set:
// * airbrake.Endpoint
// * airbrake.ApiKey
// * airbrake.Environment (only sends exceptions when set to "production")
//
// Before using this hook, to send an error. Entries that trigger an Error,
// Fatal or Panic should now include an "error" field to send to Airbrake.
type AirbrakeHook struct{}

func (hook *AirbrakeHook) Fire(entry *logrus.Entry) error {
	if entry.Data["error"] == nil {
		entry.Logger.WithFields(logrus.Fields{
			"source":   "airbrake",
			"endpoint": airbrake.Endpoint,
		}).Warn("Exceptions sent to Airbrake must have an 'error' key with the error")
		return nil
	}

	err, ok := entry.Data["error"].(error)
	if !ok {
		entry.Logger.WithFields(logrus.Fields{
			"source":   "airbrake",
			"endpoint": airbrake.Endpoint,
		}).Warn("Exceptions sent to Airbrake must have an `error` key of type `error`")
		return nil
	}

	airErr := airbrake.Notify(err)
	if airErr != nil {
		entry.Logger.WithFields(logrus.Fields{
			"source":   "airbrake",
			"endpoint": airbrake.Endpoint,
			"error":    airErr,
		}).Warn("Failed to send error to Airbrake")
	}

	return nil
}

func (hook *AirbrakeHook) Levels() []logrus.Level {
	return []logrus.Level{
		logrus.ErrorLevel,
		logrus.FatalLevel,
		logrus.PanicLevel,
	}
}
