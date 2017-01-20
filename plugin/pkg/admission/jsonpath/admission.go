/*
Copyright 2016 The Kubernetes Authors All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package jsonpath

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"regexp"

	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"

	"k8s.io/kubernetes/pkg/admission"
	"k8s.io/kubernetes/pkg/api"
	util_jsonpath "k8s.io/kubernetes/pkg/util/jsonpath"
	"k8s.io/kubernetes/pkg/util/yaml"
)

var jsonRegexp = regexp.MustCompile("^\\{\\.?([^{}]+)\\}$|^\\.?([^{}]+)$")

// MassageJSONPath attempts to be flexible with JSONPath expressions, it accepts:
//   * metadata.namespace (no leading '.' or curly brances '{...}'
//   * {metadata.namespace} (no leading '.')
//   * .metadata.namespace (no curly braces '{...}')
//   * {.metadata.namespace} (complete expression)
// And transforms them all into a valid jsonpat expression:
//   {.metadata.namespace}
// TODO: This is copied from pkg/kubectl/custom_column_printer.go move into the pkg/util/jsonpath package.
func massageJSONPath(pathExpression string) (string, error) {
	if len(pathExpression) == 0 {
		return pathExpression, nil
	}
	submatches := jsonRegexp.FindStringSubmatch(pathExpression)
	if submatches == nil {
		return "", fmt.Errorf("unexpected path string, expected a 'name1.name2' or '.name1.name2' or '{name1.name2}' or '{.name1.name2}'")
	}
	if len(submatches) != 3 {
		return "", fmt.Errorf("unexpected submatch list: %v", submatches)
	}
	var fieldSpec string
	if len(submatches[1]) != 0 {
		fieldSpec = submatches[1]
	} else {
		fieldSpec = submatches[2]
	}
	return fmt.Sprintf("{.%s}", fieldSpec), nil
}

type JSONPathAdmissionRule struct {
	// Version indicates the version for the rule.  For now this has to be "v1"
	Version string `json:"version" yaml:"version"`

	// Name is the name of this rule
	Name string `json:"name" yaml:"name"`

	// KindRegexp holds a regular expression used to determine which Kinds of
	// objects this rule applies to.
	KindRegexp string `json:"kindRegexp" yaml:"kindRegexp"`

	// Path holds a JSONPath expression to select particular fields
	Path string `json:"path" yaml:"path"`

	// APIVersion holds the version of the object to use with the
	// JSONPath above.  (e.g. "v1" or "v1beta1")
	APIVersion string `json:"apiVersion" yaml:"apiVersion"`

	// MatchRegexp holds a matching expression, all values returned by the JSONPath expression (above)
	// must match this expression for the object to be admitted
	MatchRegexp string `json:"matchRegexp" yaml:"matchRegexp"`

	// AcceptEmptyMatch indicates if an empty match is acceptable.  Empty match means that the
	// JSONPath expression above returns no values.  Default is false, meaning that at least one
	// value that matches MatchRegexp is required for admission
	AcceptEmptyMatch bool `json:"acceptEmptyMatch" yaml:"acceptEmptyMatch"`
}

type JSONPathAdmissionConfig struct {
	Rules []JSONPathAdmissionRule `json:"rules" yaml:"rules"`
}

func init() {
	admission.RegisterPlugin("jsonpath", func(client clientset.Interface, config io.Reader) (admission.Interface, error) {
		if config == nil {
			return nil, errors.New("Config file is required for JSONPath Admission Controller")
		}
		jsonConfig := JSONPathAdmissionConfig{}
		data, err := ioutil.ReadAll(config)
		if err != nil {
			return nil, err
		}
		jsonData, err := yaml.ToJSON(data)
		if err != nil {
			return nil, err
		}
		if err := json.Unmarshal(jsonData, &jsonConfig); err != nil {
			return nil, err
		}
		return NewJSONPathAdmission(jsonConfig)
	})
}

type jsonPathRule struct {
	name            string
	kindRE          *regexp.Regexp
	path            *util_jsonpath.JSONPath
	apiVersion      string
	matchRE         *regexp.Regexp
	acceptNoMatches bool
}

func (r *jsonPathRule) String() string {
	return fmt.Sprintf("%s: %s %s %v", r.name, r.kindRE.String(), r.matchRE.String(), r.acceptNoMatches)
}

func (r *jsonPathRule) admit(a admission.Attributes) error {
	if !r.kindRE.MatchString(a.GetKind().Kind) {
		return nil
	}
	gvk := a.GetKind()
	gvk.Version = r.apiVersion
	vObj, err := api.Scheme.ConvertToVersion(a.GetObject(), gvk.GroupVersion())
	if err != nil {
		return err
	}
	vals, err := r.path.FindResults(vObj)
	if err != nil {
		return err
	}
	resultCount := 0
	for _, arr := range vals {
		resultCount += len(arr)
		for _, val := range arr {
			text, err := r.path.EvalToText(val)
			if err != nil {
				return err
			}
			if !r.matchRE.Match(text) {
				return errors.New(fmt.Sprintf("Rule %s: failed to match value (%s, %s)", r.name, r.matchRE.String(), string(text)))
			}
		}
	}
	if resultCount == 0 && !r.acceptNoMatches {
		return errors.New(fmt.Sprintf("Rule %s failed: No matches", r.name))
	}

	return nil
}

type jsonPath struct {
	*admission.Handler
	rules []jsonPathRule
}

func (j *jsonPath) Admit(a admission.Attributes) (err error) {
	for ix := range j.rules {
		if err := j.rules[ix].admit(a); err != nil {
			return err
		}
	}
	return nil
}

func makeRule(rule *JSONPathAdmissionRule) (*jsonPathRule, error) {
	kre, err := regexp.Compile(rule.KindRegexp)
	if err != nil {
		return nil, err
	}
	path := util_jsonpath.New(rule.Name)
	mPath, err := massageJSONPath(rule.Path)
	if err != nil {
		return nil, err
	}
	if err := path.Parse(mPath); err != nil {
		return nil, err
	}
	mre, err := regexp.Compile(rule.MatchRegexp)
	if err != nil {
		return nil, err
	}
	return &jsonPathRule{
		apiVersion:      rule.APIVersion,
		kindRE:          kre,
		path:            path,
		matchRE:         mre,
		name:            rule.Name,
		acceptNoMatches: rule.AcceptEmptyMatch,
	}, nil
}

// NewJSONPathAdmission creates an admission controller based on a JSONPathConfig
func NewJSONPathAdmission(config JSONPathAdmissionConfig) (admission.Interface, error) {
	result := &jsonPath{
		Handler: admission.NewHandler(admission.Create, admission.Update),
	}

	for ix := range config.Rules {
		rule, err := makeRule(&config.Rules[ix])
		if err != nil {
			return nil, err
		}
		result.rules = append(result.rules, *rule)
	}
	return result, nil
}
