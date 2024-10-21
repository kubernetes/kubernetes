#define _GNU_SOURCE
#include <fcntl.h>
#include <sched.h>
#include <stdio.h>
#include <unistd.h>
#include <stdarg.h>
#include <stdlib.h>

/*
 * All of the code here is run inside an aync-signal-safe context, so we need
 * to be careful to not call any functions that could cause issues. In theory,
 * since we are a Go program, there are fewer restrictions in practice, it's
 * better to be safe than sorry.
 *
 * The only exception is exit, which we need to call to make sure we don't
 * return into runc.
 */

void bail(int pipefd, const char *fmt, ...)
{
	va_list args;

	va_start(args, fmt);
	vdprintf(pipefd, fmt, args);
	va_end(args);

	exit(1);
}

int spawn_userns_cat(char *userns_path, char *path, int outfd, int errfd)
{
	char buffer[4096] = { 0 };

	pid_t child = fork();
	if (child != 0)
		return child;
	/* in child */

	/* Join the target userns. */
	int nsfd = open(userns_path, O_RDONLY);
	if (nsfd < 0)
		bail(errfd, "open userns path %s failed: %m", userns_path);

	int err = setns(nsfd, CLONE_NEWUSER);
	if (err < 0)
		bail(errfd, "setns %s failed: %m", userns_path);

	close(nsfd);

	/* Pipe the requested file contents. */
	int fd = open(path, O_RDONLY);
	if (fd < 0)
		bail(errfd, "open %s in userns %s failed: %m", path, userns_path);

	int nread, ntotal = 0;
	while ((nread = read(fd, buffer, sizeof(buffer))) != 0) {
		if (nread < 0)
			bail(errfd, "read bytes from %s failed (after %d total bytes read): %m", path, ntotal);
		ntotal += nread;

		int nwritten = 0;
		while (nwritten < nread) {
			int n = write(outfd, buffer, nread - nwritten);
			if (n < 0)
				bail(errfd, "write %d bytes from %s failed (after %d bytes written): %m",
				     nread - nwritten, path, nwritten);
			nwritten += n;
		}
		if (nread != nwritten)
			bail(errfd, "mismatch for bytes read and written: %d read != %d written", nread, nwritten);
	}

	close(fd);
	close(outfd);
	close(errfd);

	/* We must exit here, otherwise we would return into a forked runc. */
	exit(0);
}
