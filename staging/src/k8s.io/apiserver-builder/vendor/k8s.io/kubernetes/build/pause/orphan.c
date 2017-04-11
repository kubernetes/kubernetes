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

/* Creates a zombie to be reaped by init. Useful for testing. */

#include <stdio.h>
#include <unistd.h>

int main() {
  pid_t pid;
  pid = fork();
  if (pid == 0) {
    while (getppid() > 1)
      ;
    printf("Child exiting: pid=%d ppid=%d\n", getpid(), getppid());
    return 0;
  } else if (pid > 0) {
    printf("Parent exiting: pid=%d ppid=%d\n", getpid(), getppid());
    return 0;
  }
  perror("Could not create child");
  return 1;
}
