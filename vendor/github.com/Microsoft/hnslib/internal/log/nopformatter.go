package log

import (
	"github.com/sirupsen/logrus"
)

type NopFormatter struct{}

var _ logrus.Formatter = NopFormatter{}

// Format does nothing and returns a nil slice.
func (NopFormatter) Format(*logrus.Entry) ([]byte, error) { return nil, nil }
