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
#include <grp.h>
#include <inttypes.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "diagnostic-util.h"

/* Create keep_env file from keep_env, if they're present in
 * current environment and file doesn't exist */
void initialize_keep_env(const char *keep_env_file, const char **keep_env)
{
	FILE		*f;
	const char	**p;
	char		*v;
	char		nul = '\0';

	if (!access(keep_env_file, F_OK)) return;
	pexit_if((f = fopen(keep_env_file, "a")) == NULL,
		"Unable to fopen \"%s\"", keep_env_file);

	p = keep_env;
	while (*p) {
		v = getenv(*p);
		if (v) {
			pexit_if(fprintf(f, "%s=%s%c", *p, v, nul) != (strlen(*p) + strlen(v) + 2),
				"Unable to write to \"%s\"", keep_env_file);
		}
		p++;
	}

	pexit_if(fclose(f) == EOF,
		"Unable to fclose \"%s\"", keep_env_file);
}

/* Try to set current env from keep_env and env file. */
static void set_env(const char *env_file)
{
	FILE 		*f;
	char 		*line = NULL;
	size_t 		len = 0;
	ssize_t 	read;
	char 		*v, *nl;

	pexit_if((f = fopen(env_file, "r")) == NULL,
             "Unable to fopen \"%s\"", env_file);
	while ((read = getline(&line, &len, f)) != -1) {
		pexit_if((v = strchr(line, '=')) == NULL,
				"Malformed environment entry: \"%s\"", line);
		*v = '\0';
		v++;
		/* remove new line character */
		if ((nl = strchr(v, '\n')) != NULL)
			*nl = '\0';
		pexit_if(setenv(line, v, 1) == -1,
				 "Unable to set env variable: \"%s\"=\"%s\"", line, v);
	}
	free(line);
	pexit_if(fclose(f) == EOF,
			 "Unable to fclose \"%s\"", env_file);
}

/* Read environment from env and keep_env files make it our own, keeping the env variables in
 * if they're present in the current environment.
 * The environment files must exist, may be empty, and are expected to be of the format:
 * key=value\nkey=value\n...
 */
static void load_env(const char *env_file, const char *keep_env_file, int entering)
{
	char *term = getenv("TERM"); /* useful to keep during entering. */
	pexit_if(clearenv() != 0,
		"Unable to clear environment");

	set_env(env_file);
	set_env(keep_env_file);
	if (entering) {
		// enter is typically interactive; ensure we always have a sane enough term
		// variable.
		if (term == NULL) {
			setenv("TERM", "vt100", 1);
		} else {
			setenv("TERM", term, 1);
		}
	}
}

/* Parse a comma-separated list of numeric gids from str, returns an malloc'd
 * array of gids in *gids_p with the number of elements in *n_gids_p.
 */
static void parse_gids(const char *str, size_t *n_gids_p, gid_t **gids_p)
{
	char	c = ',', last_c;
	int	i, n_gids = 0, done = 0;
	gid_t	gid = 0;
	gid_t	*gids = NULL;

	for(i = 0; !done; i++) {
		last_c = c;
		switch(c = str[i]) {
		case '0' ... '9':
			gid *= 10;
			gid += c - '0';
			break;

		case '\0':
			done = 1;
			/* fallthrough */
		case ',':
			exit_if(last_c == ',',
				"Gids contains an empty gid: \"%s\"", str);
			pexit_if((gids = realloc(gids, sizeof(*gids) * (n_gids + 1))) == NULL,
				"Unable to allocate gids: \"%s\"", str);
			gids[n_gids++] = gid;
			gid = 0;
			break;

		default:
			exit_if(1,
				"Gids contains invalid input (%c): \"%s\"",
				c, str);
		}
	}

	exit_if(!n_gids, "At least one gid is required, got: \"%s\"", str);

	*gids_p = gids;
	*n_gids_p = n_gids;
}

int main(int argc, char *argv[])
{
	int entering = 0;

	/* '-e' optional flag passed only during 'entering' phase from stage1.
	 */
	int c;
	while ((c = getopt(argc, argv, "e")) != -1)
		switch (c) {
			case 'e':
				entering = 1;
				break;
		}

	/* We need to keep these env variables since systemd uses them for socket
	 * activation
	 */
	static const char *keep_env[] = {
		"LISTEN_FDS",
		"LISTEN_PID",
		NULL
	};

	const char *keep_env_file = "/rkt/env/keep_env";
	const char	*root, *cwd, *env_file, *uid_str, *gid_str, *exe;

	char		**args;
	uid_t		uid;
	gid_t		*gids;
	size_t		n_gids;

	exit_if(argc < 7,
		"Usage: %s /path/to/root /work/directory /env/file uid gid[,gid...] [-e] /to/exec [args ...]", argv[0]);

	root = argv[optind];
	cwd = argv[optind+1];
	env_file = argv[optind+2];
	uid_str = argv[optind+3];
	uid = atoi(uid_str);
	gid_str = argv[optind+4];
	args = &argv[optind+5];
	exe = args[0];

	parse_gids(gid_str, &n_gids, &gids);

	initialize_keep_env(keep_env_file, keep_env);
	load_env(env_file, keep_env_file, entering);

	pexit_if(chroot(root) == -1, "Chroot \"%s\" failed", root);
	pexit_if(chdir(cwd) == -1, "Chdir \"%s\" failed", cwd);
	pexit_if(gids[0] > 0 && setresgid(gids[0], gids[0], gids[0]) == -1,
		"Setresgid \"%s\" failed", gid_str);
	pexit_if(n_gids > 1 && setgroups(n_gids - 1, &gids[1]) == -1,
		"Setgroups \"%s\" failed", gid_str);
	pexit_if(uid > 0 && setresuid(uid, uid, uid) == -1,
		"Setresuid \"%s\" failed", uid_str);

	/* XXX(vc): note that since execvp() is happening post-chroot, the
	 * app's environment settings correctly affect the PATH search.
	 * This is why execvpe() isn't being used, we manipulate the environment
	 * manually then let it potentially affect execvp().  execvpe() simply
	 * passes the environment to execve() _after_ performing the search, not
	 * what we want here. */
	pexit_if(execvp(exe, args) == -1 &&
		 errno != ENOENT && errno != EACCES,
		 "Exec of \"%s\" failed", exe);
	diag(exe);

	return EXIT_FAILURE;
}
