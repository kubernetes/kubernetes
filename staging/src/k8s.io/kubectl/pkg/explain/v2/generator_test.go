/*
Copyright 2022 The Kubernetes Authors.

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

package v2

import (
	"bytes"
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"
)

// Shows generic throws error when attempting to `Render“ an invalid output name
// And if it is then added as a template, no error is thrown upon `Render`
func TestGeneratorMissingOutput(t *testing.T) {
	var buf bytes.Buffer
	var doc map[string]interface{}

	err := json.Unmarshal([]byte(apiDiscoveryJSON), &doc)
	require.NoError(t, err)

	gen := NewGenerator()
	badTemplateName := "bad-template"
	err = gen.Render(badTemplateName, doc, apiGroupsGVR, nil, false, &buf)

	require.ErrorContains(t, err, "unrecognized format: "+badTemplateName)
	require.Zero(t, buf.Len())

	err = gen.AddTemplate(badTemplateName, "ok")
	require.NoError(t, err)

	err = gen.Render(badTemplateName, doc, apiGroupsGVR, nil, false, &buf)
	require.NoError(t, err)
	require.Equal(t, "ok", buf.String())
}

// Shows that correct context with the passed object is passed to the template
func TestGeneratorContext(t *testing.T) {
	var buf bytes.Buffer
	var doc map[string]interface{}

	err := json.Unmarshal([]byte(apiDiscoveryJSON), &doc)
	require.NoError(t, err)

	gen := NewGenerator()
	err = gen.AddTemplate("Context", "{{ toJson . }}")
	require.NoError(t, err)

	expectedContext := TemplateContext{
		Document:  doc,
		GVR:       apiGroupsGVR,
		Recursive: false,
		FieldPath: nil,
	}

	err = gen.Render("Context",
		expectedContext.Document,
		expectedContext.GVR,
		expectedContext.FieldPath,
		expectedContext.Recursive,
		&buf)
	require.NoError(t, err)

	var actualContext TemplateContext
	err = json.Unmarshal(buf.Bytes(), &actualContext)
	require.NoError(t, err)
	require.Equal(t, expectedContext, actualContext)
}
