/*
Copyright 2016 The Kubernetes Authors.

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

#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#define STRINGIFY(x) #x
#define VERSION_STRING(x) STRINGIFY(x)

#ifndef VERSION
#define VERSION HEAD
#endif

static int ret_val;

static void sigdown(int signo) {
  psignal(signo, "Shutting down, got signal");
  exit(ret_val);
}

static void sigreap(int signo) {
  while (waitpid(-1, NULL, WNOHANG) > 0)
    ;
}

void set_ret_val() {
  const char* s = getenv("PAUSE_EXIT_CODE");
  char *ptr;
  if (s) {
    long tmp = strtol(s,&ptr, 10);
    if (*ptr != 0) {
      fprintf(stderr, "Error: could not parse '%s', non-parsable character '%c'\n", s, *ptr);
      exit(3);
    }
    if (tmp == 0 && errno != 0) {
      fprintf(stderr, "Error: could not parse '%s', errno: '%d'\n", s, errno);
      exit(4);
    }
    if (tmp > 255 || tmp < 0) {
      fprintf(stderr, "Error: '%ld' is outside int range: <0, 255>\n", tmp);
      exit(5);
    }
    ret_val = (int)tmp;
  } else {
    ret_val = 0;
  }
}

int main(int argc, char **argv) {
  int i;
  for (i = 1; i < argc; ++i) {
    if (!strcasecmp(argv[i], "-v")) {
      printf("pause.c %s\n", VERSION_STRING(VERSION));
      return 0;
    }
  }

  set_ret_val();

  if (getpid() != 1)
    /* Not an error because pause sees use outside of infra containers. */
    fprintf(stderr, "Warning: pause should be the first process\n");

  if (sigaction(SIGINT, &(struct sigaction){.sa_handler = sigdown}, NULL) < 0)
    return 1;
  if (sigaction(SIGTERM, &(struct sigaction){.sa_handler = sigdown}, NULL) < 0)
    return 2;
  if (sigaction(SIGCHLD, &(struct sigaction){.sa_handler = sigreap,
                                             .sa_flags = SA_NOCLDSTOP},
                NULL) < 0)
    return 3;

  for (;;)
    pause();
  fprintf(stderr, "Error: infinite loop terminated\n");
  return 42;
}
