package main

import (
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func testCompiler(t *testing.T, inputFile string, referenceFile string, expectErrors bool) {
	textFile := strings.Replace(filepath.Base(inputFile), filepath.Ext(inputFile), ".text", 1)
	errorsFile := strings.Replace(filepath.Base(inputFile), filepath.Ext(inputFile), ".errors", 1)
	// remove any preexisting output files
	os.Remove(textFile)
	os.Remove(errorsFile)
	// run the compiler
	var err error
	var cmd = exec.Command(
		"gnostic",
		inputFile,
		"--text-out=.",
		"--errors-out=.",
		"--resolve-refs")
	//t.Log(cmd.Args)
	err = cmd.Run()
	if err != nil && !expectErrors {
		t.Logf("Compile failed: %+v", err)
		t.FailNow()
	}
	// verify the output against a reference
	var outputFile string
	if expectErrors {
		outputFile = errorsFile
	} else {
		outputFile = textFile
	}
	err = exec.Command("diff", outputFile, referenceFile).Run()
	if err != nil {
		t.Logf("Diff failed: %+v", err)
		t.FailNow()
	} else {
		// if the test succeeded, clean up
		os.Remove(textFile)
		os.Remove(errorsFile)
	}
}

func testNormal(t *testing.T, inputFile string, referenceFile string) {
	testCompiler(t, inputFile, referenceFile, false)
}

func testErrors(t *testing.T, inputFile string, referenceFile string) {
	testCompiler(t, inputFile, referenceFile, true)
}

func TestPetstoreJSON(t *testing.T) {
	testNormal(t,
		"examples/v2.0/json/petstore.json",
		"test/v2.0/petstore.text")
}

func TestPetstoreYAML(t *testing.T) {
	testNormal(t,
		"examples/v2.0/yaml/petstore.yaml",
		"test/v2.0/petstore.text")
}

func TestSeparateYAML(t *testing.T) {
	testNormal(t,
		"examples/v2.0/yaml/petstore-separate/spec/swagger.yaml",
		"test/v2.0/yaml/petstore-separate/spec/swagger.text")
}

func TestSeparateJSON(t *testing.T) {
	testNormal(t,
		"examples/v2.0/json/petstore-separate/spec/swagger.json",
		"test/v2.0/yaml/petstore-separate/spec/swagger.text") // yaml and json results should be identical
}

func TestRemotePetstoreJSON(t *testing.T) {
	testNormal(t,
		"https://raw.githubusercontent.com/googleapis/openapi-compiler/master/examples/v2.0/json/petstore.json",
		"test/v2.0/petstore.text")
}

func TestRemotePetstoreYAML(t *testing.T) {
	testNormal(t,
		"https://raw.githubusercontent.com/googleapis/openapi-compiler/master/examples/v2.0/yaml/petstore.yaml",
		"test/v2.0/petstore.text")
}

func TestRemoteSeparateYAML(t *testing.T) {
	testNormal(t,
		"https://raw.githubusercontent.com/googleapis/openapi-compiler/master/examples/v2.0/yaml/petstore-separate/spec/swagger.yaml",
		"test/v2.0/yaml/petstore-separate/spec/swagger.text")
}

func TestRemoteSeparateJSON(t *testing.T) {
	testNormal(t,
		"https://raw.githubusercontent.com/googleapis/openapi-compiler/master/examples/v2.0/json/petstore-separate/spec/swagger.json",
		"test/v2.0/yaml/petstore-separate/spec/swagger.text")
}

func TestErrorBadProperties(t *testing.T) {
	testErrors(t,
		"examples/errors/petstore-badproperties.yaml",
		"test/errors/petstore-badproperties.errors")
}

func TestErrorUnresolvedRefs(t *testing.T) {
	testErrors(t,
		"examples/errors/petstore-unresolvedrefs.yaml",
		"test/errors/petstore-unresolvedrefs.errors")
}

func TestErrorMissingVersion(t *testing.T) {
	testErrors(t,
		"examples/errors/petstore-missingversion.yaml",
		"test/errors/petstore-missingversion.errors")
}

func testPlugin(t *testing.T, plugin string, inputFile string, outputFile string, referenceFile string) {
	// remove any preexisting output files
	os.Remove(outputFile)
	// run the compiler
	var err error
	output, err := exec.Command(
		"gnostic",
		"--"+plugin+"-out=-",
		inputFile).Output()
	if err != nil {
		t.Logf("Compile failed: %+v", err)
		t.FailNow()
	}
	_ = ioutil.WriteFile(outputFile, output, 0644)
	err = exec.Command("diff", outputFile, referenceFile).Run()
	if err != nil {
		t.Logf("Diff failed: %+v", err)
		t.FailNow()
	} else {
		// if the test succeeded, clean up
		os.Remove(outputFile)
	}
}

func TestSamplePluginWithPetstore(t *testing.T) {
	testPlugin(t,
		"go-sample",
		"examples/v2.0/yaml/petstore.yaml",
		"sample-petstore.out",
		"test/v2.0/yaml/sample-petstore.out")
}

func TestErrorInvalidPluginInvocations(t *testing.T) {
	var err error
	output, err := exec.Command(
		"gnostic",
		"examples/v2.0/yaml/petstore.yaml",
		"--errors-out=-",
		"--plugin-out=foo=bar,:abc",
		"--plugin-out=,foo=bar:abc",
		"--plugin-out=foo=:abc",
		"--plugin-out==bar:abc",
		"--plugin-out=,,:abc",
		"--plugin-out=foo=bar=baz:abc",
	).Output()
	if err == nil {
		t.Logf("Invalid invocations were accepted")
		t.FailNow()
	}
	outputFile := "invalid-plugin-invocation.errors"
	_ = ioutil.WriteFile(outputFile, output, 0644)
	err = exec.Command("diff", outputFile, "test/errors/invalid-plugin-invocation.errors").Run()
	if err != nil {
		t.Logf("Diff failed: %+v", err)
		t.FailNow()
	} else {
		// if the test succeeded, clean up
		os.Remove(outputFile)
	}
}

func TestValidPluginInvocations(t *testing.T) {
	var err error
	output, err := exec.Command(
		"gnostic",
		"examples/v2.0/yaml/petstore.yaml",
		"--errors-out=-",
		// verify an invocation with no parameters
		"--go-sample-out=!", // "!" indicates that no output should be generated
		// verify single pair of parameters
		"--go-sample-out=a=b:!",
		// verify multiple parameters
		"--go-sample-out=a=b,c=123,xyz=alphabetagammadelta:!",
		// verify that special characters / . - _ can be included in parameter keys and values
		"--go-sample-out=a/b/c=x/y/z:!",
		"--go-sample-out=a.b.c=x.y.z:!",
		"--go-sample-out=a-b-c=x-y-z:!",
		"--go-sample-out=a_b_c=x_y_z:!",
	).Output()
	if len(output) != 0 {
		t.Logf("Valid invocations generated invalid errors\n%s", string(output))
		t.FailNow()
	}
	if err != nil {
		t.Logf("Valid invocations were not accepted")
		t.FailNow()
	}
}

func TestExtensionHandlerWithLibraryExample(t *testing.T) {
	outputFile := "library-example-with-ext.text.out"
	inputFile := "test/library-example-with-ext.json"
	referenceFile := "test/library-example-with-ext.text.out"

	os.Remove(outputFile)
	// run the compiler
	var err error

	command := exec.Command(
		"gnostic",
		"--x-sampleone",
		"--x-sampletwo",
		"--text-out="+outputFile,
		"--resolve-refs",
		inputFile)

	_, err = command.Output()
	if err != nil {
		t.Logf("Compile failed for command %v: %+v", command, err)
		t.FailNow()
	}
	//_ = ioutil.WriteFile(outputFile, output, 0644)
	err = exec.Command("diff", outputFile, referenceFile).Run()
	if err != nil {
		t.Logf("Diff failed: %+v", err)
		t.FailNow()
	} else {
		// if the test succeeded, clean up
		os.Remove(outputFile)
	}
}

func TestJSONOutput(t *testing.T) {
	inputFile := "test/library-example-with-ext.json"

	textFile := "sample.text"
	jsonFile := "sample.json"
	textFile2 := "sample2.text"
	jsonFile2 := "sample2.json"

	os.Remove(textFile)
	os.Remove(jsonFile)
	os.Remove(textFile2)
	os.Remove(jsonFile2)

	var err error

	// Run the compiler once.
	command := exec.Command(
		"gnostic",
		"--text-out="+textFile,
		"--json-out="+jsonFile,
		inputFile)
	_, err = command.Output()
	if err != nil {
		t.Logf("Compile failed for command %v: %+v", command, err)
		t.FailNow()
	}

	// Run the compiler again, this time on the generated output.
	command = exec.Command(
		"gnostic",
		"--text-out="+textFile2,
		"--json-out="+jsonFile2,
		jsonFile)
	_, err = command.Output()
	if err != nil {
		t.Logf("Compile failed for command %v: %+v", command, err)
		t.FailNow()
	}

	// Verify that both models have the same internal representation.
	err = exec.Command("diff", textFile, textFile2).Run()
	if err != nil {
		t.Logf("Diff failed: %+v", err)
		t.FailNow()
	} else {
		// if the test succeeded, clean up
		os.Remove(textFile)
		os.Remove(jsonFile)
		os.Remove(textFile2)
		os.Remove(jsonFile2)
	}
}

func TestYAMLOutput(t *testing.T) {
	inputFile := "test/library-example-with-ext.json"

	textFile := "sample.text"
	yamlFile := "sample.yaml"
	textFile2 := "sample2.text"
	yamlFile2 := "sample2.yaml"

	os.Remove(textFile)
	os.Remove(yamlFile)
	os.Remove(textFile2)
	os.Remove(yamlFile2)

	var err error

	// Run the compiler once.
	command := exec.Command(
		"gnostic",
		"--text-out="+textFile,
		"--yaml-out="+yamlFile,
		inputFile)
	_, err = command.Output()
	if err != nil {
		t.Logf("Compile failed for command %v: %+v", command, err)
		t.FailNow()
	}

	// Run the compiler again, this time on the generated output.
	command = exec.Command(
		"gnostic",
		"--text-out="+textFile2,
		"--yaml-out="+yamlFile2,
		yamlFile)
	_, err = command.Output()
	if err != nil {
		t.Logf("Compile failed for command %v: %+v", command, err)
		t.FailNow()
	}

	// Verify that both models have the same internal representation.
	err = exec.Command("diff", textFile, textFile2).Run()
	if err != nil {
		t.Logf("Diff failed: %+v", err)
		t.FailNow()
	} else {
		// if the test succeeded, clean up
		os.Remove(textFile)
		os.Remove(yamlFile)
		os.Remove(textFile2)
		os.Remove(yamlFile2)
	}
}

func testBuilder(version string, t *testing.T) {
	var err error

	pbFile := "petstore-" + version + ".pb"
	yamlFile := "petstore.yaml"
	jsonFile := "petstore.json"
	textFile := "petstore.text"
	textReference := "test/" + version + ".0/petstore.text"

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
		t.Logf("Diff failed: %+v", err)
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

// OpenAPI 3.0 tests

func TestPetstoreYAML_30(t *testing.T) {
	testNormal(t,
		"examples/v3.0/yaml/petstore.yaml",
		"test/v3.0/petstore.text")
}

func TestPetstoreJSON_30(t *testing.T) {
	testNormal(t,
		"examples/v3.0/json/petstore.json",
		"test/v3.0/petstore.text")
}
