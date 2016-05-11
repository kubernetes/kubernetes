/*
Copyright (c) 2014-2015 VMware, Inc. All Rights Reserved.

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

package license

import (
	"fmt"
	"io"
	"os"
	"text/tabwriter"

	"github.com/vmware/govmomi/vim25/types"
)

type licenseOutput []types.LicenseManagerLicenseInfo

func (res licenseOutput) Write(w io.Writer) error {
	tw := tabwriter.NewWriter(os.Stdout, 4, 0, 2, ' ', 0)
	fmt.Fprintf(tw, "Key:\tEdition:\tUsed:\tTotal:\n")
	for _, v := range res {
		fmt.Fprintf(tw, "%s\t", v.LicenseKey)
		fmt.Fprintf(tw, "%s\t", v.EditionKey)
		fmt.Fprintf(tw, "%d\t", v.Used)
		fmt.Fprintf(tw, "%d\t", v.Total)
		fmt.Fprintf(tw, "\n")
	}
	return tw.Flush()
}
