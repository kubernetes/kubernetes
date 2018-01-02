#define _GNU_SOURCE
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>

int main(int argc, char **argv)
{
	int err = acct("/tmp/t");
	if (err == -1) {
		fprintf(stderr, "acct failed: %s\n", strerror(errno));
		exit(EXIT_FAILURE);
	}
	exit(EXIT_SUCCESS);
}
