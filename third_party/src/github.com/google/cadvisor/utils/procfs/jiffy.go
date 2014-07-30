// Copyright 2014 Google Inc. All Rights Reserved.
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

package procfs

/*
#include <unistd.h>
*/
import "C"
import "time"

var userHz uint64

func init() {
	userHzLong := C.sysconf(C._SC_CLK_TCK)
	userHz = uint64(userHzLong)
}

func JiffiesToDuration(jiffies uint64) time.Duration {
	d := jiffies * 1000000000 / userHz
	return time.Duration(d)
}
