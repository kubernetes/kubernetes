// Copyright 2016 The etcd Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package command

import (
	"encoding/json"
	"fmt"
	"os"
)

type jsonPrinter struct{ printer }

func newJSONPrinter() printer {
	return &jsonPrinter{
		&printerRPC{newPrinterUnsupported("json"), printJSON},
	}
}

func (p *jsonPrinter) EndpointStatus(r []epStatus) { printJSON(r) }
func (p *jsonPrinter) DBStatus(r dbstatus)         { printJSON(r) }

func printJSON(v interface{}) {
	b, err := json.Marshal(v)
	if err != nil {
		fmt.Fprintf(os.Stderr, "%v\n", err)
		return
	}
	fmt.Println(string(b))
}
