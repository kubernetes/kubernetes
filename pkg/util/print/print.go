/*
Copyright 2015 The Kubernetes Authors.

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

package print

import (
	"bytes"
	"strings"

	"github.com/davecgh/go-spew/spew"
)

// PrettyPrintObject is wrapper of spew Sprintf returns object's value in string format.
// It can read a "expected" string as 1st arg and 2 objects args followed, returning formated comparasion string result.
// It can read a name of k8s object as 1st arg,  2 object args followed, returning structed object string value, like "ReplicaSet %q:\n%+v\n"
func PrettyPrintObject(arg ...interface{}) (result string) {
	switch len(arg) {
	case 3:
		if strings.Contains(arg[0].(string), "expected") {
			result = spew.Sprintf(arg[0].(string)+" %#v but got %#v", arg[1], arg[2])
		}
		result = spew.Sprintf(arg[0].(string)+"  %q:\n%+v\n", arg[1], arg[2])
	default:
		var buffer bytes.Buffer
		for i := 0; i < len(arg); i++ {
			buffer.WriteString("%#v ")
		}
		result = spew.Sprintf(buffer.String(), arg...)
	}
	return

}

// PrettyPrintQuotedObject returns a formated object's string value with information
func PrettyPrintQuotedObject(info string, arg interface{}) string {
	return spew.Sprintf(info+" %q ", arg)
}

// PrettyPrintStructObject returns a formated object's string value with information
func PrettyPrintStructObject(info string, arg interface{}) string {
	return spew.Sprintf(info+"%+v ", arg)
}
