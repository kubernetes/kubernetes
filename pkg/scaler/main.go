package main

import (
	"github.com/GoogleCloudPlatform/kubernetes/pkg/scaler/core"
	"github.com/golang/glog"
)

func main() {
	autoScaler := core.New()
	err := autoScaler.AutoScale()
	if err != nil {
		glog.Fatal(err)
	}
}
