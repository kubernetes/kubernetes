package hcs

import "github.com/sirupsen/logrus"

func logOperationBegin(ctx logrus.Fields, msg string) {
	logrus.WithFields(ctx).Debug(msg)
}

func logOperationEnd(ctx logrus.Fields, msg string, err error) {
	if err == nil {
		logrus.WithFields(ctx).Debug(msg)
	} else {
		logrus.WithFields(ctx).WithError(err).Error(msg)
	}
}
