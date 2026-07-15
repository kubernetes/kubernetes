package ginkgo

import (
	"github.com/go-logr/logr"
	"github.com/go-logr/logr/funcr"
	"github.com/onsi/ginkgo/v2"
)

// this is copied from ginkgo because ginkgo made it internal and then hardcoded an init block
// using these functions to wire to os.stdout and we want to wire to stderr (or a different buffer) so we can
// have json output.

func GinkgoLogrFunc(writer ginkgo.GinkgoWriterInterface) logr.Logger {
	return funcr.New(func(prefix, args string) {
		if prefix == "" {
			writer.Printf("%s\n", args)
		} else {
			writer.Printf("%s %s\n", prefix, args)
		}
	}, funcr.Options{
		// LogTimestamp adds timestamps to log lines using the format "2006-01-02 15:04:05.000000"
		// See: https://github.com/go-logr/logr/blob/bb8ea8159175ccb4eddf4ac8704f84e40ac6d9b0/funcr/funcr.go#L211
		LogTimestamp: true,
	})
}
