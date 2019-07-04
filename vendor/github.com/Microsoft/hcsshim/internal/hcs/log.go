package hcs

import "github.com/sirupsen/logrus"

func logOperationBegin(ctx logrus.Fields, msg string) {
	logrus.WithFields(ctx).Debug(msg)
}

func logOperationEnd(ctx logrus.Fields, msg string, err error) {
	// Copy the log and fields first.
	log := logrus.WithFields(ctx)
	if err == nil {
		log.Debug(msg)
	} else {
		// Edit only the copied field data to avoid race conditions on the
		// write.
		log.Data[logrus.ErrorKey] = err
		log.Error(msg)
	}
}
