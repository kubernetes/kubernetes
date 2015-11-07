package airbrake

import (
	"errors"
	"fmt"

	"github.com/Sirupsen/logrus"
	"github.com/tobi/airbrake-go"
)

// AirbrakeHook to send exceptions to an exception-tracking service compatible
// with the Airbrake API.
type airbrakeHook struct {
	APIKey      string
	Endpoint    string
	Environment string
}

func NewHook(endpoint, apiKey, env string) *airbrakeHook {
	return &airbrakeHook{
		APIKey:      apiKey,
		Endpoint:    endpoint,
		Environment: env,
	}
}

func (hook *airbrakeHook) Fire(entry *logrus.Entry) error {
	airbrake.ApiKey = hook.APIKey
	airbrake.Endpoint = hook.Endpoint
	airbrake.Environment = hook.Environment

	var notifyErr error
	err, ok := entry.Data["error"].(error)
	if ok {
		notifyErr = err
	} else {
		notifyErr = errors.New(entry.Message)
	}

	airErr := airbrake.Notify(notifyErr)
	if airErr != nil {
		return fmt.Errorf("Failed to send error to Airbrake: %s", airErr)
	}

	return nil
}

func (hook *airbrakeHook) Levels() []logrus.Level {
	return []logrus.Level{
		logrus.ErrorLevel,
		logrus.FatalLevel,
		logrus.PanicLevel,
	}
}
