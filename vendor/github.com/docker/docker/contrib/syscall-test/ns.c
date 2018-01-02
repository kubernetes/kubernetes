#define _GNU_SOURCE
#include <errno.h>
#include <sched.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

#define STACK_SIZE (1024 * 1024)	/* Stack size for cloned child */

struct clone_args {
	char **argv;
};

// child_exec is the func that will be executed as the result of clone
static int child_exec(void *stuff)
{
	struct clone_args *args = (struct clone_args *)stuff;
	if (execvp(args->argv[0], args->argv) != 0) {
		fprintf(stderr, "failed to execvp arguments %s\n",
			strerror(errno));
		exit(-1);
	}
	// we should never reach here!
	exit(EXIT_FAILURE);
}

int main(int argc, char **argv)
{
	struct clone_args args;
	args.argv = &argv[1];

	int clone_flags = CLONE_NEWNS | CLONE_NEWPID | SIGCHLD;

	// allocate stack for child
	char *stack;		/* Start of stack buffer */
	char *child_stack;	/* End of stack buffer */
	stack =
	    mmap(NULL, STACK_SIZE, PROT_READ | PROT_WRITE,
		 MAP_SHARED | MAP_ANON | MAP_STACK, -1, 0);
	if (stack == MAP_FAILED) {
		fprintf(stderr, "mmap failed: %s\n", strerror(errno));
		exit(EXIT_FAILURE);
	}
	child_stack = stack + STACK_SIZE;	/* Assume stack grows downward */

	// the result of this call is that our child_exec will be run in another
	// process returning its pid
	pid_t pid = clone(child_exec, child_stack, clone_flags, &args);
	if (pid < 0) {
		fprintf(stderr, "clone failed: %s\n", strerror(errno));
		exit(EXIT_FAILURE);
	}
	// lets wait on our child process here before we, the parent, exits
	if (waitpid(pid, NULL, 0) == -1) {
		fprintf(stderr, "failed to wait pid %d\n", pid);
		exit(EXIT_FAILURE);
	}
	exit(EXIT_SUCCESS);
}
