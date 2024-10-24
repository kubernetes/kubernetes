#ifdef RUNC_USE_STDLIB
#  include <linux/limits.h>
#  include <stdio.h>
#  include <string.h>
#  include <unistd.h>
#else
#  include "xstat.h"
#  include "nolibc/nolibc.h"
#endif

extern char **environ;

int main(int argc, char **argv)
{
	if (argc < 1)
		return 127;
	int ret = execve(argv[0], argv, environ);
	if (ret) {
		/* NOTE: This error message format MUST match Go's format. */
		char err_msg[5 + PATH_MAX + 1] = "exec ";	// "exec " + argv[0] + '\0'
		strncat(err_msg, argv[0], PATH_MAX);
		err_msg[sizeof(err_msg) - 1] = '\0';

		perror(err_msg);
	}
	return ret;
}
