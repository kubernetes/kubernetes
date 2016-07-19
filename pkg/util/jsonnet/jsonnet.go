// JSONNET Decoder
package jsonnet

import (
	"bytes"
	"os"
	"path"
	"strings"

	"github.com/golang/glog"
	utilexec "k8s.io/kubernetes/pkg/util/exec"
)

const cmdJsonnet string = "jsonnet"

// Run converts a single JSONNET document into a JSON document
func Run(args ...string) ([]byte, error) {
	glog.V(4).Infof("running jsonnet %v", args)
	return utilexec.New().Command(cmdJsonnet, args...).CombinedOutput()
}

// Open convert a single JSONNET file into a io.reader
func Open(args ...string) (*bytes.Reader, error) {
	output, err := Run(args...)
	return bytes.NewReader(output), err
}

// IsJsonnet reports whether the filename is a standard JSONNET file
func IsJsonnet(filename string) bool {
	if _, err := os.Stat(filename); err != nil {
		glog.V(4).Infof("file %v is not existing %v", filename, err)
		return false
	}
	if strings.ToLower(strings.TrimSpace(path.Ext(filename))) == ".jsonnet" {
		glog.V(4).Infof("file %v is jsonnet", filename)
		return true
	}
	glog.V(4).Infof("file %v is not jsonnet", filename)
	return false
}
