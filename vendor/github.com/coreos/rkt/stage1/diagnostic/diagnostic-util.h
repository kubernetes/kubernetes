// Copyright 2014-2016 The rkt Authors
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

#ifndef _DIAGNOSTIC_UTIL_H
#define _DIAGNOSTIC_UTIL_H

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int exit_err;
#define exit_if(_cond, _fmt, _args...)				\
	exit_err++;						\
	if(_cond) {						\
		fprintf(stderr, "Error: " _fmt "\n", ##_args);	\
		exit(exit_err);					\
	}
#define pexit_if(_cond, _fmt, _args...)				\
	exit_if(_cond, _fmt ": %s", ##_args, strerror(errno))

#define MAX_DIAG_DEPTH 10
#define MIN(_a, _b) (((_a) < (_b)) ? (_a) : (_b))

void diag(const char *exe);

#endif /* _DIAGNOSTIC_UTIL_H */
