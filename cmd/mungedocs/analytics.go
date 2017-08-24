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

package main

import (
	"fmt"
	"strings"
)

const analyticsMungeTag = "GENERATED_ANALYTICS"
const analyticsLinePrefix = "[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/"

func updateAnalytics(fileName string, mlines mungeLines) (mungeLines, error) {
	var out mungeLines
	fileName, err := makeRepoRelative(fileName, fileName)
	if err != nil {
		return mlines, err
	}

	link := fmt.Sprintf(analyticsLinePrefix+"%s?pixel)]()", fileName)
	insertLines := getMungeLines(link)
	mlines, err = removeMacroBlock(analyticsMungeTag, mlines)
	if err != nil {
		return mlines, err
	}

	// Remove floating analytics links not surrounded by the munge tags.
	for _, mline := range mlines {
		if mline.preformatted || mline.header || mline.beginTag || mline.endTag {
			out = append(out, mline)
			continue
		}
		if strings.HasPrefix(mline.data, analyticsLinePrefix) {
			continue
		}
		out = append(out, mline)
	}
	out = appendMacroBlock(out, analyticsMungeTag)
	out, err = updateMacroBlock(out, analyticsMungeTag, insertLines)
	if err != nil {
		return mlines, err
	}
	return out, nil
}
