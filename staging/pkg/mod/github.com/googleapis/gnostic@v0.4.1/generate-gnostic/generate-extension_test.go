package main

import (
	"io/ioutil"
	"os"
	"os/exec"
	"testing"
)

func TestErrorExtensionGeneratorUnsupportedPrimitive(t *testing.T) {
	var err error

	output, err := exec.Command(
		"generate-gnostic",
		"--extension",
		"test/x-unsupportedprimitives.json",
		"--out_dir=/tmp",
	).Output()

	outputFile := "x-unsupportedprimitives.errors"
	_ = ioutil.WriteFile(outputFile, output, 0644)
	err = exec.Command("diff", outputFile, "test/errors/x-unsupportedprimitives.errors").Run()
	if err != nil {
		t.Logf("Diff failed: %+v", err)
		t.FailNow()
	} else {
		// if the test succeeded, clean up
		os.Remove(outputFile)
	}
}

func TestErrorExtensionGeneratorNameCollision(t *testing.T) {
	var err error

	output, err := exec.Command(
		"generate-gnostic",
		"--extension",
		"test/x-extension-name-collision.json",
		"--out_dir=/tmp",
	).Output()

	outputFile := "x-extension-name-collision.errors"
	_ = ioutil.WriteFile(outputFile, output, 0644)
	err = exec.Command("diff", outputFile, "test/errors/x-extension-name-collision.errors").Run()
	if err != nil {
		t.Logf("Diff failed: %+v", err)
		t.FailNow()
	} else {
		// if the test succeeded, clean up
		os.Remove(outputFile)
	}
}
