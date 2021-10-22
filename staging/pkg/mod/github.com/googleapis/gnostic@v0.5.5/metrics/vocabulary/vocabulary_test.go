// Copyright 2020 Google LLC. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package vocabulary

import (
	"io/ioutil"
	"os"
	"os/exec"
	"testing"

	"github.com/golang/protobuf/jsonpb"
	discovery "github.com/googleapis/gnostic/discovery"
	metrics "github.com/googleapis/gnostic/metrics"
	openapiv2 "github.com/googleapis/gnostic/openapiv2"
	openapiv3 "github.com/googleapis/gnostic/openapiv3"
)

func fillTestProtoStructure(words []string, count []int) []*metrics.WordCount {
	counts := make([]*metrics.WordCount, 0)
	for i := 0; i < len(words); i++ {
		temp := &metrics.WordCount{
			Word:  words[i],
			Count: int32(count[i]),
		}
		counts = append(counts, temp)
	}
	return counts
}

func testVocabulary(t *testing.T, outputVocab *metrics.Vocabulary, referencePb *metrics.Vocabulary) {
	results := Difference([]*metrics.Vocabulary{outputVocab, referencePb})
	results2 := Difference([]*metrics.Vocabulary{referencePb, outputVocab})

	if !isEmpty(results) && !isEmpty(results2) {
		t.Logf("Difference failed: Output does not match")
		t.FailNow()
	} else {
		// if the test succeeded, clean up
		os.Remove("vocabulary-operation.pb")
	}
}

func testVocabularyOutput(t *testing.T, outputFile string, referenceFile string) {
	err := exec.Command("diff", outputFile, referenceFile).Run()
	if err != nil {
		t.Logf("Diff failed: %s vs %s %+v", outputFile, referenceFile, err)
		t.FailNow()
	} else {
		// if the test succeeded, clean up
		os.Remove(outputFile)
	}
}

func TestSampleVocabularyUnion(t *testing.T) {
	v1 := metrics.Vocabulary{
		Schemas:    fillTestProtoStructure([]string{"heelo", "random", "funcName", "google"}, []int{1, 2, 3, 4}),
		Properties: fillTestProtoStructure([]string{"Hello", "dog", "funcName", "cat"}, []int{4, 3, 2, 1}),
		Operations: fillTestProtoStructure([]string{"countGreetings", "print", "funcName"}, []int{12, 11, 4}),
		Parameters: fillTestProtoStructure([]string{"name", "id", "tag", "suggester"}, []int{5, 1, 1, 15}),
	}

	v2 := metrics.Vocabulary{
		Schemas:    fillTestProtoStructure([]string{"Hello", "random", "status", "google"}, []int{5, 6, 1, 4}),
		Properties: fillTestProtoStructure([]string{"cat", "dog", "thing"}, []int{4, 3, 2}),
		Operations: fillTestProtoStructure([]string{"countPrint", "print", "funcName"}, []int{17, 12, 19}),
		Parameters: fillTestProtoStructure([]string{"name", "id", "tag", "suggester"}, []int{5, 1, 1, 15}),
	}

	vocabularies := make([]*metrics.Vocabulary, 0)
	vocabularies = append(vocabularies, &v1, &v2)

	reference := metrics.Vocabulary{
		Schemas:    fillTestProtoStructure([]string{"Hello", "funcName", "google", "heelo", "random", "status"}, []int{5, 3, 8, 1, 8, 1}),
		Properties: fillTestProtoStructure([]string{"Hello", "cat", "dog", "funcName", "thing"}, []int{4, 5, 6, 2, 2}),
		Operations: fillTestProtoStructure([]string{"countGreetings", "countPrint", "funcName", "print"}, []int{12, 17, 23, 23}),
		Parameters: fillTestProtoStructure([]string{"id", "name", "suggester", "tag"}, []int{2, 10, 30, 2}),
	}

	unionResult := Union(vocabularies)

	testVocabulary(t,
		unionResult,
		&reference,
	)
}

func TestSampleVocabularyIntersection(t *testing.T) {
	v1 := metrics.Vocabulary{
		Schemas:    fillTestProtoStructure([]string{"heelo", "random", "funcName", "google"}, []int{1, 2, 3, 4}),
		Properties: fillTestProtoStructure([]string{"Hello", "dog", "funcName", "cat"}, []int{4, 3, 2, 1}),
		Operations: fillTestProtoStructure([]string{"countGreetings", "print", "funcName"}, []int{12, 11, 4}),
		Parameters: fillTestProtoStructure([]string{"name", "id", "tag", "suggester"}, []int{5, 1, 1, 15}),
	}

	v2 := metrics.Vocabulary{
		Schemas:    fillTestProtoStructure([]string{"Hello", "random", "status", "google"}, []int{5, 6, 1, 4}),
		Properties: fillTestProtoStructure([]string{"cat", "dog", "thing"}, []int{4, 3, 2}),
		Operations: fillTestProtoStructure([]string{"countPrint", "print", "funcName"}, []int{17, 12, 19}),
		Parameters: fillTestProtoStructure([]string{"name", "id", "tag", "suggester"}, []int{5, 1, 1, 15}),
	}

	vocabularies := make([]*metrics.Vocabulary, 0)
	vocabularies = append(vocabularies, &v1, &v2)

	reference := metrics.Vocabulary{
		Schemas:    fillTestProtoStructure([]string{"google", "random"}, []int{8, 8}),
		Properties: fillTestProtoStructure([]string{"cat", "dog"}, []int{5, 6}),
		Operations: fillTestProtoStructure([]string{"funcName", "print"}, []int{23, 23}),
		Parameters: fillTestProtoStructure([]string{"id", "name", "suggester", "tag"}, []int{2, 10, 30, 2}),
	}

	intersectionResult := Intersection(vocabularies)

	testVocabulary(t,
		intersectionResult,
		&reference,
	)
}
func TestSampleVocabularyDifference(t *testing.T) {
	v1 := metrics.Vocabulary{
		Schemas:    fillTestProtoStructure([]string{"heelo", "random", "funcName", "google"}, []int{1, 2, 3, 4}),
		Properties: fillTestProtoStructure([]string{"Hello", "dog", "funcName", "cat"}, []int{4, 3, 2, 1}),
		Operations: fillTestProtoStructure([]string{"countGreetings", "print", "funcName"}, []int{12, 11, 4}),
		Parameters: fillTestProtoStructure([]string{"name", "id", "tag", "suggester"}, []int{5, 1, 1, 15}),
	}

	v2 := metrics.Vocabulary{
		Schemas:    fillTestProtoStructure([]string{"Hello", "random", "status", "google"}, []int{5, 6, 1, 4}),
		Properties: fillTestProtoStructure([]string{"cat", "dog", "thing"}, []int{4, 3, 2}),
		Operations: fillTestProtoStructure([]string{"countPrint", "print", "funcName"}, []int{17, 12, 19}),
		Parameters: fillTestProtoStructure([]string{"name", "id", "tag", "suggester"}, []int{5, 1, 1, 15}),
	}

	vocabularies := make([]*metrics.Vocabulary, 0)
	vocabularies = append(vocabularies, &v1, &v2)

	reference := metrics.Vocabulary{
		Schemas:    fillTestProtoStructure([]string{"funcName", "heelo"}, []int{3, 1}),
		Properties: fillTestProtoStructure([]string{"Hello", "funcName"}, []int{4, 2}),
		Operations: fillTestProtoStructure([]string{"countGreetings"}, []int{12}),
	}

	differenceResult := Difference(vocabularies)

	testVocabulary(t,
		differenceResult,
		&reference,
	)
}

func TestSampleVocabularyFilterCommon(t *testing.T) {
	v1 := metrics.Vocabulary{
		Schemas:    fillTestProtoStructure([]string{"heelo", "random", "funcName", "google"}, []int{1, 2, 3, 4}),
		Properties: fillTestProtoStructure([]string{"Hello", "dog", "funcName", "cat"}, []int{4, 3, 2, 1}),
		Operations: fillTestProtoStructure([]string{"countGreetings", "print", "funcName"}, []int{12, 11, 4}),
		Parameters: fillTestProtoStructure([]string{"name", "id", "tag", "suggester"}, []int{5, 1, 1, 15}),
	}

	v2 := metrics.Vocabulary{
		Schemas:    fillTestProtoStructure([]string{"Hello", "random", "status", "google"}, []int{5, 6, 1, 4}),
		Properties: fillTestProtoStructure([]string{"cat", "dog", "thing"}, []int{4, 3, 2}),
		Operations: fillTestProtoStructure([]string{"countPrint", "print", "funcName"}, []int{17, 12, 19}),
		Parameters: fillTestProtoStructure([]string{"name", "id", "tag", "suggester"}, []int{5, 1, 1, 15}),
	}

	vocabularies := make([]*metrics.Vocabulary, 0)
	vocabularies = append(vocabularies, &v1, &v2)

	reference := metrics.Vocabulary{
		Schemas:    fillTestProtoStructure([]string{"funcName", "heelo"}, []int{3, 1}),
		Properties: fillTestProtoStructure([]string{"Hello", "funcName"}, []int{4, 2}),
		Operations: fillTestProtoStructure([]string{"countGreetings"}, []int{12}),
	}

	reference2 := metrics.Vocabulary{
		Schemas:    fillTestProtoStructure([]string{"Hello", "status"}, []int{5, 1}),
		Properties: fillTestProtoStructure([]string{"thing"}, []int{2}),
		Operations: fillTestProtoStructure([]string{"countPrint"}, []int{17}),
	}

	differenceResult := FilterCommon(vocabularies)

	testVocabulary(t,
		differenceResult.Vocabularies[0],
		&reference,
	)

	testVocabulary(t,
		differenceResult.Vocabularies[1],
		&reference2,
	)
}

func TestSampleVocabularyCSV(t *testing.T) {
	v1 := metrics.Vocabulary{
		Schemas:    fillTestProtoStructure([]string{"heelo", "random", "funcName", "google"}, []int{1, 2, 3, 4}),
		Properties: fillTestProtoStructure([]string{"Hello", "dog", "funcName", "cat"}, []int{4, 3, 2, 1}),
		Operations: fillTestProtoStructure([]string{"countGreetings", "print", "funcName"}, []int{12, 11, 4}),
		Parameters: fillTestProtoStructure([]string{"name", "id", "tag", "suggester"}, []int{5, 1, 1, 15}),
	}

	WriteCSV(&v1, "")

	testVocabularyOutput(t,
		"vocabulary-operation.csv",
		"../../testdata/v3.0/csv/sample-vocabulary.csv",
	)
}

func TestSampleVocabularyV2(t *testing.T) {
	inputFile := "../../examples/v2.0/json/petstore.json"
	referenceFile := "../../testdata/metrics/vocabulary/petstore-v2.json"
	data, err := ioutil.ReadFile(inputFile)
	if err != nil {
		t.Logf("ReadFile failed: %+v", err)
		t.FailNow()
	}
	document, err := openapiv2.ParseDocument(data)
	if err != nil {
		t.Logf("Parse failed: %+v", err)
		t.FailNow()
	}
	v1 := NewVocabularyFromOpenAPIv2(document)
	// uncomment the following line to write reference data
	//err = ioutil.WriteFile(referenceFile, []byte(protojson.Format(v1)), 0644)
	referenceData, err := ioutil.ReadFile(referenceFile)
	if err != nil {
		t.Logf("ReadFile failed: %+v", err)
		t.FailNow()
	}
	reference := metrics.Vocabulary{}
	jsonpb.UnmarshalString(string(referenceData), &reference)
	testVocabulary(t,
		v1,
		&reference,
	)
}

func TestSampleVocabularyV3(t *testing.T) {
	inputFile := "../../examples/v3.0/json/petstore.json"
	referenceFile := "../../testdata/metrics/vocabulary/petstore-v3.json"
	data, err := ioutil.ReadFile(inputFile)
	if err != nil {
		t.Logf("ReadFile failed: %+v", err)
		t.FailNow()
	}
	document, err := openapiv3.ParseDocument(data)
	if err != nil {
		t.Logf("Parse failed: %+v", err)
		t.FailNow()
	}
	v1 := NewVocabularyFromOpenAPIv3(document)
	// uncomment the following line to write reference data
	//err = ioutil.WriteFile(referenceFile, []byte(protojson.Format(v1)), 0644)
	referenceData, err := ioutil.ReadFile(referenceFile)
	if err != nil {
		t.Logf("ReadFile failed: %+v", err)
		t.FailNow()
	}
	reference := metrics.Vocabulary{}
	jsonpb.UnmarshalString(string(referenceData), &reference)
	testVocabulary(t,
		v1,
		&reference,
	)
}

func TestSampleVocabularyDiscovery(t *testing.T) {
	inputFile := "../../examples/discovery/discovery-v1.json"
	referenceFile := "../../testdata/metrics/vocabulary/discovery.json"
	data, err := ioutil.ReadFile(inputFile)
	if err != nil {
		t.Logf("ReadFile failed: %+v", err)
		t.FailNow()
	}
	document, err := discovery.ParseDocument(data)
	if err != nil {
		t.Logf("Parse failed: %+v", err)
		t.FailNow()
	}
	v1 := NewVocabularyFromDiscovery(document)
	// uncomment the following line to write reference data
	//err = ioutil.WriteFile(referenceFile, []byte(protojson.Format(v1)), 0644)
	referenceData, err := ioutil.ReadFile(referenceFile)
	if err != nil {
		t.Logf("ReadFile failed: %+v", err)
		t.FailNow()
	}
	reference := metrics.Vocabulary{}
	jsonpb.UnmarshalString(string(referenceData), &reference)
	testVocabulary(t,
		v1,
		&reference,
	)
}
