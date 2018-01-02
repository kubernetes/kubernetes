package log

import (
	"io/ioutil"
	"log"

	"google.golang.org/grpc/grpclog"
)

func init() {
	grpclog.SetLogger(log.New(ioutil.Discard, "", log.LstdFlags))
}
