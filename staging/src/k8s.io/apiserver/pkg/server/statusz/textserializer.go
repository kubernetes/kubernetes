/*
Copyright 2025 The Kubernetes Authors.

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

package statusz

import (
	"fmt"
	"html"
	"io"
	"math/rand"
	"strings"
	"time"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"

	v1alpha1 "k8s.io/apiserver/pkg/server/statusz/api/v1alpha1"
)

// statuszTextSerializer implements runtime.Serializer for text/plain output.
type statuszTextSerializer struct {
	componentName string
	reg           statuszRegistry
}

// Encode writes the statusz information in plain text format to the given writer, using the provided obj.
func (s statuszTextSerializer) Encode(obj runtime.Object, w io.Writer) error {
	if _, err := fmt.Fprintf(w, headerFmt, s.componentName); err != nil {
		return err
	}

	randomIndex := rand.Intn(len(delimiters))
	delim := html.EscapeString(delimiters[randomIndex])

	statuszObj, ok := obj.(*v1alpha1.Statusz)
	if !ok {
		return fmt.Errorf("expected *v1alpha1.Statusz, got %T", obj)
	}

	startTime := html.EscapeString(statuszObj.StartTime.Time.Format(time.UnixDate))
	uptimeStr := html.EscapeString(uptime(statuszObj.StartTime.Time))
	goVersion := html.EscapeString(statuszObj.GoVersion)
	binaryVersion := html.EscapeString(statuszObj.BinaryVersion)

	var emulationVersion string
	if statuszObj.EmulationVersion != "" {
		emulationVersion = fmt.Sprintf(`Emulation version%s %s`, delim, html.EscapeString(statuszObj.EmulationVersion))
	}

	paths := strings.Join(statuszObj.Paths, " ")
	if paths != "" {
		paths = fmt.Sprintf(`Paths%s %s`, delim, html.EscapeString(paths))
	}

	status := fmt.Sprintf(`
Started%[1]s %[2]s
Up%[1]s %[3]s
Go version%[1]s %[4]s
Binary version%[1]s %[5]s
%[6]s
%[7]s
`, delim, startTime, uptimeStr, goVersion, binaryVersion, emulationVersion, paths)
	_, err := fmt.Fprint(w, status)
	return err
}

// Decode is not supported for text/plain serialization.
func (s statuszTextSerializer) Decode(data []byte, gvk *schema.GroupVersionKind, into runtime.Object) (runtime.Object, *schema.GroupVersionKind, error) {
	return nil, nil, fmt.Errorf("decode not supported for text/plain")
}

// Identifier returns a unique identifier for this serializer.
func (s statuszTextSerializer) Identifier() runtime.Identifier {
	return runtime.Identifier("statuszTextSerializer")
}
