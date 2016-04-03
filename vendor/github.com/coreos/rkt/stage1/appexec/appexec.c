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

#include "elf.h"

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

static void map_file(const char *path, int prot, int flags, struct stat *st, void **map)
{
	int fd;

	pexit_if((fd = open(path, O_RDONLY)) == -1,
		"Unable to open \"%s\"", path);
	pexit_if(fstat(fd, st) == -1,
		"Cannot stat \"%s\"", path);
	exit_if(!S_ISREG(st->st_mode), "\"%s\" is not a regular file", path);
	pexit_if(!(*map = mmap(NULL, st->st_size, prot, flags, fd, 0)),
		"Mmap of \"%s\" failed", path);
	pexit_if(close(fd) == -1,
		"Close of %i [%s] failed", fd, path);
}

static void diag(const char *exe)
{
	static const uint8_t	elf[] = {0x7f, 'E', 'L', 'F'};
	static const uint8_t	shebang[] = {'#','!'};
	static int		diag_depth;
	struct stat		st;
	const uint8_t		*mm;
	const char		*itrp = NULL;

	map_file(exe, PROT_READ, MAP_SHARED, &st, (void **)&mm);
	exit_if(!((S_IXUSR|S_IXGRP|S_IXOTH) & st.st_mode),
		"\"%s\" is not executable", exe)

	if(st.st_size >= sizeof(shebang) &&
	   !memcmp(mm, shebang, sizeof(shebang))) {
		const uint8_t	*nl;
		int		maxlen = MIN(PATH_MAX, st.st_size - sizeof(shebang));
		/* TODO(vc): EOF-terminated shebang lines are technically possible */
		exit_if(!(nl = memchr(&mm[sizeof(shebang)], '\n', maxlen)),
			"Shebang line too long");
		pexit_if(!(itrp = strndup((char *)&mm[sizeof(shebang)], (nl - mm) - 2)),
			"Failed to dup interpreter path");
	} else if(st.st_size >= sizeof(elf) &&
		  !memcmp(mm, elf, sizeof(elf))) {
		uint64_t	(*lget)(const uint8_t *) = NULL;
		uint32_t	(*iget)(const uint8_t *) = NULL;
		uint16_t	(*sget)(const uint8_t *) = NULL;
		const void	*phoff = NULL, *phesz = NULL, *phecnt = NULL;
		const uint8_t	*ph = NULL;
		int		i, phreloff, phrelsz;

		exit_if(mm[ELF_VERSION] != 1,
			"Unsupported ELF version: %hhx", mm[ELF_VERSION]);

		/* determine which accessors to use and where */
		if(mm[ELF_BITS] == ELF_BITS_32) {
			if(mm[ELF_ENDIAN] == ELF_ENDIAN_LITL) {
				lget = le32_lget;
				sget = le_sget;
				iget = le_iget;
			} else if(mm[ELF_ENDIAN] == ELF_ENDIAN_BIG) {
				lget = be32_lget;
				sget = be_sget;
				iget = be_iget;
			}
			phoff = &mm[ELF32_PHT_OFF];
			phesz = &mm[ELF32_PHTE_SIZE];
			phecnt = &mm[ELF32_PHTE_CNT];
			phreloff = ELF32_PHE_OFF;
			phrelsz = ELF32_PHE_SIZE;
		} else if(mm[ELF_BITS] == ELF_BITS_64) {
			if(mm[ELF_ENDIAN] == ELF_ENDIAN_LITL) {
				lget = le64_lget;
				sget = le_sget;
				iget = le_iget;
			} else if(mm[ELF_ENDIAN] == ELF_ENDIAN_BIG) {
				lget = be64_lget;
				sget = be_sget;
				iget = be_iget;
			}
			phoff = &mm[ELF64_PHT_OFF];
			phesz = &mm[ELF64_PHTE_SIZE];
			phecnt = &mm[ELF64_PHTE_CNT];
			phreloff = ELF64_PHE_OFF;
			phrelsz = ELF64_PHE_SIZE;
		}

		exit_if(!lget, "Unsupported ELF format");

		if(!phoff) /* program header may be absent, don't make it an error */
			return;

		/* TODO(vc): sanity checks on values before using them */
		for(ph = &mm[lget(phoff)], i = 0; i < sget(phecnt); i++, ph += sget(phesz)) {
			if(iget(ph) == ELF_PT_INTERP) {
				itrp = strndup((char *)&mm[lget(&ph[phreloff])], lget(&ph[phrelsz]));
				break;
			}
		}
	} else {
		exit_if(1, "Unsupported file type");
	}

	exit_if(!itrp, "Unable to determine interpreter for \"%s\"", exe);
	exit_if(*itrp != '/', "Path must be absolute: \"%s\"", itrp);
	exit_if(++diag_depth > MAX_DIAG_DEPTH,
		"Excessive interpreter recursion, giving up");
	diag(itrp);
}

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
	struct stat st;
	char *map, *k, *v;
	typeof(st.st_size) i;

	map_file(env_file, PROT_READ|PROT_WRITE, MAP_PRIVATE, &st, (void **)&map);

	if(!st.st_size)
		return;

	map[st.st_size - 1] = '\0'; /* ensure the mapping is null-terminated */

	for(i = 0; i < st.st_size;) {
		k = &map[i];
		i += strlen(k) + 1;
		exit_if((v = strchr(k, '=')) == NULL,
			"Malformed environment entry: \"%s\"", k);
		/* a private writable map is used permitting s/=/\0/ */
		*v = '\0';
		v++;
		pexit_if(setenv(k, v, 1) == -1,
			"Unable to set env variable: \"%s\"=\"%s\"", k, v);
	}
}

/* Read environment from env and keep_env files make it our own, keeping the env variables in
 * if they're present in the current environment.
 * The environment files must exist, may be empty, and are expected to be of the format:
 * key=value\0key=value\0...
 */
static void load_env(const char *env_file, const char *keep_env_file, int entering)
{
	char *term = getenv("TERM"); /* useful to keep during entering. */
	pexit_if(clearenv() != 0,
		"Unable to clear environment");

	set_env(env_file);
	set_env(keep_env_file);
	if (entering) setenv("TERM", term, 1);
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
