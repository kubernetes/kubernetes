#!/bin/bash

# Copyright 2015 Google Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

GO_FILES=$(find . -not -wholename "*Godeps*" -name "*.go")

for FILE in ${GO_FILES}; do
	ERRS=`grep 'fmt.Errorf("[A-Z]' ${FILE}`
	if [ $? -eq 0 ]
	then
		echo Incorrect error format in file ${FILE}: $ERRS
		exit 1
	fi
done

exit 0
