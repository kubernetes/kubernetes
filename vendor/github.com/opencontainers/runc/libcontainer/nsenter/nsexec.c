#define _GNU_SOURCE
#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>

#include <linux/limits.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <signal.h>
#include <setjmp.h>
#include <sched.h>
#include <signal.h>

#include <linux/netlink.h>
#include <linux/types.h>
#include <stdint.h>
#include <sys/socket.h>

/* All arguments should be above stack, because it grows down */
struct clone_arg {
	/*
	 * Reserve some space for clone() to locate arguments
	 * and retcode in this place
	 */
	char stack[4096] __attribute__ ((aligned(16)));
	char stack_ptr[0];
	jmp_buf *env;
};

#define pr_perror(fmt, ...) fprintf(stderr, "nsenter: " fmt ": %m\n", ##__VA_ARGS__)

static int child_func(void *_arg)
{
	struct clone_arg *arg = (struct clone_arg *)_arg;
	longjmp(*arg->env, 1);
}

// Use raw setns syscall for versions of glibc that don't include it (namely glibc-2.12)
#if __GLIBC__ == 2 && __GLIBC_MINOR__ < 14
#define _GNU_SOURCE
#include "syscall.h"
#if defined(__NR_setns) && !defined(SYS_setns)
#define SYS_setns __NR_setns
#endif
#ifdef SYS_setns
int setns(int fd, int nstype)
{
	return syscall(SYS_setns, fd, nstype);
}
#endif
#endif

static int clone_parent(jmp_buf * env) __attribute__ ((noinline));
static int clone_parent(jmp_buf * env)
{
	struct clone_arg ca;
	int child;

	ca.env = env;
	child = clone(child_func, ca.stack_ptr, CLONE_PARENT | SIGCHLD, &ca);

	return child;
}

static uint32_t readint32(char *buf)
{
	return *(uint32_t *) buf;
}

// list of known message types we want to send to bootstrap program
// These are defined in libcontainer/message_linux.go
#define INIT_MSG 62000
#define PID_ATTR 27281
#define CONSOLE_PATH_ATTR 27282

void nsexec()
{
	char *namespaces[] = { "ipc", "uts", "net", "pid", "mnt", "user" };
	const int num = sizeof(namespaces) / sizeof(char *);
	jmp_buf env;
	char buf[PATH_MAX], *val;
	int i, tfd, self_tfd, child, n, len, pipenum, consolefd = -1;
	pid_t pid = 0;

	// if we dont have INITTYPE or this is the init process, skip the bootstrap process
	val = getenv("_LIBCONTAINER_INITTYPE");
	if (val == NULL || strcmp(val, "standard") == 0) {
		return;
	}
	if (strcmp(val, "setns") != 0) {
		pr_perror("Invalid inittype %s", val);
		exit(1);
	}

	val = getenv("_LIBCONTAINER_INITPIPE");
	if (val == NULL) {
		pr_perror("Child pipe not found");
		exit(1);
	}
	pipenum = atoi(val);
	snprintf(buf, sizeof(buf), "%d", pipenum);
	if (strcmp(val, buf)) {
		pr_perror("Unable to parse _LIBCONTAINER_INITPIPE");
		exit(1);
	}

	char nlbuf[NLMSG_HDRLEN];
	struct nlmsghdr *nh;
	if ((n = read(pipenum, nlbuf, NLMSG_HDRLEN)) != NLMSG_HDRLEN) {
		pr_perror("Failed to read netlink header, got %d", n);
		exit(1);
	}

	nh = (struct nlmsghdr *)nlbuf;
	if (nh->nlmsg_type == NLMSG_ERROR) {
		pr_perror("Invalid netlink header message");
		exit(1);
	}
	if (nh->nlmsg_type != INIT_MSG) {
		pr_perror("Unexpected netlink message type %d", nh->nlmsg_type);
		exit(1);
	}
	// read the netlink payload
	len = NLMSG_PAYLOAD(nh, 0);
	char data[len];
	if ((n = read(pipenum, data, len)) != len) {
		pr_perror("Failed to read netlink payload, got %d", n);
		exit(1);
	}

	int start = 0;
	struct nlattr *attr;
	while (start < len) {
		int payload_len;
		attr = (struct nlattr *)((void *)data + start);
		start += NLA_HDRLEN;
		payload_len = attr->nla_len - NLA_HDRLEN;
		switch (attr->nla_type) {
		case PID_ATTR:
			pid = (pid_t) readint32(data + start);
			break;
		case CONSOLE_PATH_ATTR:
			consolefd = open((char *)data + start, O_RDWR);
			if (consolefd < 0) {
				pr_perror("Failed to open console %s", (char *)data + start);
				exit(1);
			}
			break;
		}
		start += NLA_ALIGN(payload_len);
	}

	// required pid to be passed
	if (pid == 0) {
		pr_perror("missing pid");
		exit(1);
	}

	/* Check that the specified process exists */
	snprintf(buf, PATH_MAX - 1, "/proc/%d/ns", pid);
	tfd = open(buf, O_DIRECTORY | O_RDONLY);
	if (tfd == -1) {
		pr_perror("Failed to open \"%s\"", buf);
		exit(1);
	}

	self_tfd = open("/proc/self/ns", O_DIRECTORY | O_RDONLY);
	if (self_tfd == -1) {
		pr_perror("Failed to open /proc/self/ns");
		exit(1);
	}

	for (i = 0; i < num; i++) {
		struct stat st;
		struct stat self_st;
		int fd;

		/* Symlinks on all namespaces exist for dead processes, but they can't be opened */
		if (fstatat(tfd, namespaces[i], &st, 0) == -1) {
			// Ignore nonexistent namespaces.
			if (errno == ENOENT)
				continue;
		}

		/* Skip namespaces we're already part of */
		if (fstatat(self_tfd, namespaces[i], &self_st, 0) != -1 && st.st_ino == self_st.st_ino) {
			continue;
		}

		fd = openat(tfd, namespaces[i], O_RDONLY);
		if (fd == -1) {
			pr_perror("Failed to open ns file %s for ns %s", buf, namespaces[i]);
			exit(1);
		}
		// Set the namespace.
		if (setns(fd, 0) == -1) {
			pr_perror("Failed to setns for %s", namespaces[i]);
			exit(1);
		}
		close(fd);
	}

	close(self_tfd);
	close(tfd);

	if (setjmp(env) == 1) {
		// Child

		if (setsid() == -1) {
			pr_perror("setsid failed");
			exit(1);
		}
		if (consolefd != -1) {
			if (ioctl(consolefd, TIOCSCTTY, 0) == -1) {
				pr_perror("ioctl TIOCSCTTY failed");
				exit(1);
			}
			if (dup3(consolefd, STDIN_FILENO, 0) != STDIN_FILENO) {
				pr_perror("Failed to dup 0");
				exit(1);
			}
			if (dup3(consolefd, STDOUT_FILENO, 0) != STDOUT_FILENO) {
				pr_perror("Failed to dup 1");
				exit(1);
			}
			if (dup3(consolefd, STDERR_FILENO, 0) != STDERR_FILENO) {
				pr_perror("Failed to dup 2");
				exit(1);
			}
		}
		// Finish executing, let the Go runtime take over.
		return;
	}
	// Parent

	// We must fork to actually enter the PID namespace, use CLONE_PARENT
	// so the child can have the right parent, and we don't need to forward
	// the child's exit code or resend its death signal.
	child = clone_parent(&env);
	if (child < 0) {
		pr_perror("Unable to fork");
		exit(1);
	}

	len = snprintf(buf, sizeof(buf), "{ \"pid\" : %d }\n", child);

	if (write(pipenum, buf, len) != len) {
		pr_perror("Unable to send a child pid");
		kill(child, SIGKILL);
		exit(1);
	}

	exit(0);
}
