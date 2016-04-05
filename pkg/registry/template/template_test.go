package template

import (
	"io/ioutil"
	"testing"
	// TODO: fix this import
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/diff"
)

func TestProcessTemplateParameters(t *testing.T) {
	var template, expectedTemplate extensions.Template
	jsonData, _ := ioutil.ReadFile("./guestbook_template.json")
	if err := runtime.DecodeInto(api.Codecs.UniversalDecoder(), jsonData, &template); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	expectedData, _ := ioutil.ReadFile("./guestbook.json")
	if err := runtime.DecodeInto(api.Codecs.UniversalDecoder(), expectedData, &expectedTemplate); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	processor := Processor{}
	// Transform the template config into the result config
	errs := processor.Process(&template)
	if len(errs) > 0 {
		t.Fatalf("unexpected error: %v", errs)
	}
	result := runtime.EncodeOrDie(testapi.Extensions.Codec(), &template)
	exp := runtime.EncodeOrDie(testapi.Extensions.Codec(), &expectedTemplate)
	if string(result) != string(exp) {
		t.Errorf("unexpected output: %s", diff.StringDiff(string(exp), string(result)))
	}
}
