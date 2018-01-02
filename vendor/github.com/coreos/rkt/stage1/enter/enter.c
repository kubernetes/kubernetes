// Copyright 2014 The rkt Authors
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

#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <getopt.h>
#include <limits.h>
#include <sched.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#ifdef NO_SETNS_AVAILABLE

#include <linux/unistd.h>

static int setns(int fd, int nstype) {
	return syscall(__NR_setns, fd, nstype);
}

#endif /* NO_SETNS_AVAILABLE */

static int errornum;
#define exit_if(_cond, _fmt, _args...)				\
	errornum++;						\
	if(_cond) {						\
		fprintf(stderr, _fmt "\n", ##_args);		\
		exit(errornum);					\
	}
#define pexit_if(_cond, _fmt, _args...)				\
	exit_if(_cond, _fmt ": %s", ##_args, strerror(errno))

static int openpidfd(int pid, char *which) {
	char	path[PATH_MAX];
	int	fd;
	exit_if(snprintf(path, sizeof(path),
			 "/proc/%i/%s", pid, which) == sizeof(path),
		"Path overflow");
	pexit_if((fd = open(path, O_RDONLY|O_CLOEXEC)) == -1,
		"Unable to open \"%s\"", path);
	return fd;
}

int main(int argc, char *argv[])
{
	int	fd;
	int	pid = 0;
	char *appname = NULL;
	pid_t	child;
	int	status;
	int	root_fd;

	int c;

	/* The parameters list is specified in
	 * Documentation/devel/stage1-implementors-guide.md */
	while (1) {
		int option_index = 0;
		static struct option long_options[] = {
		   {"pid",     required_argument, 0,  'p' },
		   {"appname", optional_argument, 0,  'a' },
		   {0,         0,                 0,  0 }
		};

		c = getopt_long(argc, argv, "p:a:",
				long_options, &option_index);
		if (c == -1)
			break;

		switch (c) {
		case 'p':
			pid = atoi(optarg);
			break;
		case 'a':
			appname = optarg;
			break;
		case 0:
			break;
		case ':':   /* missing option argument */
		case '?':
		default:
			fprintf(stderr, "Usage: %s --pid=1234 "
					"[--appname=name] -- cmd [args...]",
					argv[0]);
			exit(1);
		}
	}

	root_fd = openpidfd(pid, "root");

#define ns(_typ, _nam)							\
	fd = openpidfd(pid, _nam);					\
	pexit_if(setns(fd, _typ), "Unable to enter " _nam " namespace");

#if 0
	/* TODO(vc): Nspawn isn't employing CLONE_NEWUSER, disabled for now */
	ns(CLONE_NEWUSER, "ns/user");
#endif
	ns(CLONE_NEWIPC,  "ns/ipc");
	ns(CLONE_NEWUTS,  "ns/uts");
	ns(CLONE_NEWNET,  "ns/net");
	ns(CLONE_NEWPID,  "ns/pid");
	ns(CLONE_NEWNS,	  "ns/mnt");

	pexit_if(fchdir(root_fd) < 0,
		"Unable to chdir to pod root");
	pexit_if(chroot(".") < 0,
		"Unable to chroot");
	pexit_if(close(root_fd) == -1,
		"Unable to close root_fd");

	/* Fork is required to realize consequence of CLONE_NEWPID */
	pexit_if(((child = fork()) == -1),
		"Unable to fork");

/* some stuff make the argv->args copy less cryptic */
#define ENTEREXEC_ARGV_FWD_OFFSET	8

	if(child == 0) {
		char		root[PATH_MAX];
		char		env[PATH_MAX];
		char		*args[ENTEREXEC_ARGV_FWD_OFFSET + argc - optind + 1 /* NULL terminator */];
		int		argsind;

		if (appname == NULL) {
			argsind = 0;
			while (optind < argc)
				args[argsind++] = argv[optind++];
		} else {
			/* Child goes on to execute /enterexec */

			exit_if(snprintf(root, sizeof(root),
				     "/opt/stage2/%s/rootfs", appname) == sizeof(root),
			    "Root path overflow");

			exit_if(snprintf(env, sizeof(env),
				     "/rkt/env/%s", appname) == sizeof(env),
			    "Env path overflow");

			args[0] = "/enterexec";
			args[1] = root;
			args[2] = "/";	/* TODO(vc): plumb this into app.WorkingDirectory */
			args[3] = env;
			args[4] = "0"; /* uid */
			args[5] = "0"; /* gid */
			args[6] = "-e"; /* entering phase */
			args[7] = "--";
			argsind = ENTEREXEC_ARGV_FWD_OFFSET;
			while (optind < argc)
				args[argsind++] = argv[optind++];
		}
		args[argsind] = NULL;

		pexit_if(execv(args[0], args) == -1,
			"Exec failed");
	}

	/* Wait for child, nsenter-like */
	for(;;) {
		if(waitpid(child, &status, WUNTRACED) == pid &&
		   (WIFSTOPPED(status))) {
			kill(getpid(), SIGSTOP);
			/* the above stops us, upon receiving SIGCONT we'll
			 * continue here and inform our child */
			kill(child, SIGCONT);
		} else {
			break;
		}
	}

	if(WIFEXITED(status)) {
		exit(WEXITSTATUS(status));
	} else if(WIFSIGNALED(status)) {
		kill(getpid(), WTERMSIG(status));
	}

	return EXIT_FAILURE;
}
