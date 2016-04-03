package dashfilename

import (
	"os"
	"os/exec"
	"testing"
)

//Issue 16 : https://github.com/gogo/protobuf/issues/detail?id=16
func TestDashFilename(t *testing.T) {
	name := "dash-filename"
	cmd := exec.Command("protoc", "--gogo_out=.", "-I=../../../../../:../../protobuf/:.", name+".proto")
	data, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("err = %v: %s", err, string(data))
	}
	if err := os.Remove(name + ".pb.go"); err != nil {
		panic(err)
	}
}
