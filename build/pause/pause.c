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

#include <assert.h>
#include <errno.h>
#include <getopt.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <dirent.h>
#include <ctype.h>

char debug = 1;
#define DEBUG(...)    if (debug > 0) { fprintf(stdout, "[DEBUG pause (%i)] ", getpid()); fprintf(stdout, __VA_ARGS__); fprintf(stdout, "\n"); fflush(stdout); }

int get_pid_max(void) {
	FILE *file = fopen("/proc/sys/kernel/pid_max", "r");
	char buf[1024];
	if (file && fgets(buf, 1024, file)) {
		return atoi(buf);
	}
	DEBUG("Unable to open /proc/sys/kernel/pid_max, using default value\n");
	return 32768;
}

int get_process_list(pid_t * array) {
	int count = 0;
	DIR *dp = opendir("/proc");
	if (dp != NULL) {
		struct dirent *ep;
		while ((ep = readdir(dp)) != NULL) {
			if (isdigit(*ep->d_name)) {
				array[count++] = strtol(ep->d_name, NULL, 10);
			}
		}
		closedir(dp);
	}
	return count;
}

int get_process_parent_id(const pid_t pid) {
    int ppid = -1;
    char path[BUFSIZ];
    sprintf(path, "/proc/%d/stat", pid);
    FILE* fp = fopen(path, "r");
    if (fp) {
        int  temp;
        char state, comm[BUFSIZ];

        /* http://man7.org/linux/man-pages/man5/proc.5.html */
        fscanf(fp, "%d %s %c %d ", &temp, comm, &state, &ppid);
        fclose(fp);
    }
    return ppid;
}

void forward_signal(int signum) {
	if (signum != -1) {
	    int current_pid = getpid();
		if (signum == SIGTSTP || signum == SIGTSTP || signum == SIGTSTP) {
			signum = SIGSTOP;
		}
		pid_t *array = (pid_t *) malloc(get_pid_max() * sizeof(pid_t));
		int count = get_process_list(array);
		int i;
		for (i = 0; i < count; i++) {
		    int pid = array[i];
		    int ppid = get_process_parent_id(pid);
			if (pid != current_pid && ppid == 0) {
				kill(pid, signum);
				DEBUG("Forwarded signal %d to PID %d(%d).\n", signum, pid, ppid);
			} else {
				DEBUG("Skipping signal %d to PID %d(%d).\n", signum, pid, ppid);
			}
		}
		free(array);
	} else {
		DEBUG("Not forwarding signal %d to children (ignored).\n", signum);
	}
}

void reap_zombies() {
	int status, exit_status;
	pid_t killed_pid;
	while ((killed_pid = waitpid(-1, &status, WNOHANG)) > 0) {
		if (WIFEXITED(status)) {
			exit_status = WEXITSTATUS(status);
			DEBUG("A child with PID %d exited with exit status %d.\n", killed_pid, exit_status);
		} else {
			assert(WIFSIGNALED(status));
			exit_status = 128 + WTERMSIG(status);
			DEBUG("A child with PID %d was terminated by signal %d.\n", killed_pid, exit_status - 128);
		}
	}
}

void forward_signals(sigset_t *all_signals) {
    struct timespec ts = { .tv_sec = 1, .tv_nsec = 0 };
	siginfo_t info;
	if (sigtimedwait(all_signals, &info, &ts) <= 0) {
		return;
	}
	int signum = info.si_signo;
	DEBUG("Received signal %d.\n", signum);
	if (signum == SIGCHLD) {
	    DEBUG("Skipping SIGCHLD");
		return;
	}
	forward_signal(signum);
	if (signum == SIGTSTP || signum == SIGTTOU || signum == SIGTTIN) {
		DEBUG("Suspending self due to TTY signal.\n");
		kill(getpid(), SIGSTOP);
	}
}

void initialize_signals(sigset_t * all_signals) {
	sigfillset(all_signals);
	sigprocmask(SIG_BLOCK, all_signals, NULL);
}

int main(int argc, char *argv[]) {
	sigset_t all_signals;
	initialize_signals(&all_signals);

	for (;;) {
		DEBUG("Forwarding signals\n");
		forward_signals(&all_signals);
		DEBUG("Reaping zombies\n");
		reap_zombies();
	}
}
