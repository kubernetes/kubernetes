package main

import (
	"os"
	"os/exec"
	"testing"
)

func testBuilder(version string, t *testing.T) {
	var err error

	pbFile := "petstore-" + version + ".pb"
	yamlFile := "petstore.yaml"
	jsonFile := "petstore.json"
	textFile := "petstore.text"
	textReference := "../../testdata/" + version + ".0/petstore.text"

	os.Remove(pbFile)
	os.Remove(textFile)
	os.Remove(yamlFile)
	os.Remove(jsonFile)

	// Generate petstore.pb.
	command := exec.Command(
		"petstore-builder",
		"--"+version)
	_, err = command.Output()
	if err != nil {
		t.Logf("Command %v failed: %+v", command, err)
		t.FailNow()
	}

	// Convert petstore.pb to yaml and json.
	command = exec.Command(
		"gnostic",
		pbFile,
		"--json-out="+jsonFile,
		"--yaml-out="+yamlFile)
	_, err = command.Output()
	if err != nil {
		t.Logf("Command %v failed: %+v", command, err)
		t.FailNow()
	}

	// Read petstore.yaml, resolve references, and export text.
	command = exec.Command(
		"gnostic",
		yamlFile,
		"--resolve-refs",
		"--text-out="+textFile)
	_, err = command.Output()
	if err != nil {
		t.Logf("Command %v failed: %+v", command, err)
		t.FailNow()
	}

	// Verify that the generated text matches our reference.
	err = exec.Command("diff", textFile, textReference).Run()
	if err != nil {
		t.Logf("Diff %s vs %s failed: %+v", textFile, textReference, err)
		t.FailNow()
	}

	// Read petstore.json, resolve references, and export text.
	command = exec.Command(
		"gnostic",
		jsonFile,
		"--resolve-refs",
		"--text-out="+textFile)
	_, err = command.Output()
	if err != nil {
		t.Logf("Command %v failed: %+v", command, err)
		t.FailNow()
	}

	// Verify that the generated text matches our reference.
	err = exec.Command("diff", textFile, textReference).Run()
	if err != nil {
		t.Logf("Diff failed: %+v", err)
		t.FailNow()
	}

	// if the test succeeded, clean up
	os.Remove(pbFile)
	os.Remove(textFile)
	os.Remove(yamlFile)
	os.Remove(jsonFile)
}

func TestBuilderV2(t *testing.T) {
	testBuilder("v2", t)
}

func TestBuilderV3(t *testing.T) {
	testBuilder("v3", t)
}
