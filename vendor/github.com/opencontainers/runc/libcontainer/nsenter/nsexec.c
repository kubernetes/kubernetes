#define _GNU_SOURCE
#include <endian.h>
#include <errno.h>
#include <fcntl.h>
#include <grp.h>
#include <sched.h>
#include <setjmp.h>
#include <signal.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <unistd.h>

#include <sys/ioctl.h>
#include <sys/prctl.h>
#include <sys/socket.h>
#include <sys/types.h>

#include <linux/limits.h>
#include <linux/netlink.h>
#include <linux/types.h>

/* Get all of the CLONE_NEW* flags. */
#include "namespace.h"

/* Synchronisation values. */
enum sync_t {
	SYNC_USERMAP_PLS = 0x40, /* Request parent to map our users. */
	SYNC_USERMAP_ACK = 0x41, /* Mapping finished by the parent. */
	SYNC_RECVPID_PLS = 0x42, /* Tell parent we're sending the PID. */
	SYNC_RECVPID_ACK = 0x43, /* PID was correctly received by parent. */
	SYNC_GRANDCHILD  = 0x44, /* The grandchild is ready to run. */
	SYNC_CHILD_READY = 0x45, /* The child or grandchild is ready to return. */

	/* XXX: This doesn't help with segfaults and other such issues. */
	SYNC_ERR = 0xFF, /* Fatal error, no turning back. The error code follows. */
};

/* longjmp() arguments. */
#define JUMP_PARENT 0x00
#define JUMP_CHILD  0xA0
#define JUMP_INIT   0xA1

/* JSON buffer. */
#define JSON_MAX 4096

/* Assume the stack grows down, so arguments should be above it. */
struct clone_t {
	/*
	 * Reserve some space for clone() to locate arguments
	 * and retcode in this place
	 */
	char stack[4096] __attribute__ ((aligned(16)));
	char stack_ptr[0];

	/* There's two children. This is used to execute the different code. */
	jmp_buf *env;
	int jmpval;
};

struct nlconfig_t {
	char *data;
	uint32_t cloneflags;
	char *uidmap;
	size_t uidmap_len;
	char *gidmap;
	size_t gidmap_len;
	char *namespaces;
	size_t namespaces_len;
	uint8_t is_setgroup;
	uint8_t is_rootless;
	char *oom_score_adj;
	size_t oom_score_adj_len;
};

/*
 * List of netlink message types sent to us as part of bootstrapping the init.
 * These constants are defined in libcontainer/message_linux.go.
 */
#define INIT_MSG			62000
#define CLONE_FLAGS_ATTR	27281
#define NS_PATHS_ATTR		27282
#define UIDMAP_ATTR			27283
#define GIDMAP_ATTR			27284
#define SETGROUP_ATTR		27285
#define OOM_SCORE_ADJ_ATTR	27286
#define ROOTLESS_ATTR	    27287

/*
 * Use the raw syscall for versions of glibc which don't include a function for
 * it, namely (glibc 2.12).
 */
#if __GLIBC__ == 2 && __GLIBC_MINOR__ < 14
#	define _GNU_SOURCE
#	include "syscall.h"
#	if !defined(SYS_setns) && defined(__NR_setns)
#		define SYS_setns __NR_setns
#	endif

#ifndef SYS_setns
#	error "setns(2) syscall not supported by glibc version"
#endif

int setns(int fd, int nstype)
{
	return syscall(SYS_setns, fd, nstype);
}
#endif

/* XXX: This is ugly. */
static int syncfd = -1;

/* TODO(cyphar): Fix this so it correctly deals with syncT. */
#define bail(fmt, ...)								\
	do {									\
		int ret = __COUNTER__ + 1;					\
		fprintf(stderr, "nsenter: " fmt ": %m\n", ##__VA_ARGS__);	\
		if (syncfd >= 0) {						\
			enum sync_t s = SYNC_ERR;				\
			if (write(syncfd, &s, sizeof(s)) != sizeof(s))		\
				fprintf(stderr, "nsenter: failed: write(s)");	\
			if (write(syncfd, &ret, sizeof(ret)) != sizeof(ret))	\
				fprintf(stderr, "nsenter: failed: write(ret)");	\
		}								\
		exit(ret);							\
	} while(0)

static int write_file(char *data, size_t data_len, char *pathfmt, ...)
{
	int fd, len, ret = 0;
	char path[PATH_MAX];

	va_list ap;
	va_start(ap, pathfmt);
	len = vsnprintf(path, PATH_MAX, pathfmt, ap);
	va_end(ap);
	if (len < 0)
		return -1;

	fd = open(path, O_RDWR);
	if (fd < 0) {
		return -1;
	}

	len = write(fd, data, data_len);
	if (len != data_len) {
		ret = -1;
		goto out;
	}

out:
	close(fd);
	return ret;
}

enum policy_t {
	SETGROUPS_DEFAULT = 0,
	SETGROUPS_ALLOW,
	SETGROUPS_DENY,
};

/* This *must* be called before we touch gid_map. */
static void update_setgroups(int pid, enum policy_t setgroup)
{
	char *policy;

	switch (setgroup) {
		case SETGROUPS_ALLOW:
			policy = "allow";
			break;
		case SETGROUPS_DENY:
			policy = "deny";
			break;
		case SETGROUPS_DEFAULT:
		default:
			/* Nothing to do. */
			return;
	}

	if (write_file(policy, strlen(policy), "/proc/%d/setgroups", pid) < 0) {
		/*
		 * If the kernel is too old to support /proc/pid/setgroups,
		 * open(2) or write(2) will return ENOENT. This is fine.
		 */
		if (errno != ENOENT)
			bail("failed to write '%s' to /proc/%d/setgroups", policy, pid);
	}
}

static void update_uidmap(int pid, char *map, size_t map_len)
{
	if (map == NULL || map_len <= 0)
		return;

	if (write_file(map, map_len, "/proc/%d/uid_map", pid) < 0)
		bail("failed to update /proc/%d/uid_map", pid);
}

static void update_gidmap(int pid, char *map, size_t map_len)
{
	if (map == NULL || map_len <= 0)
		return;

	if (write_file(map, map_len, "/proc/%d/gid_map", pid) < 0)
		bail("failed to update /proc/%d/gid_map", pid);
}

static void update_oom_score_adj(char *data, size_t len)
{
	if (data == NULL || len <= 0)
		return;

	if (write_file(data, len, "/proc/self/oom_score_adj") < 0)
		bail("failed to update /proc/self/oom_score_adj");
}

/* A dummy function that just jumps to the given jumpval. */
static int child_func(void *arg) __attribute__ ((noinline));
static int child_func(void *arg)
{
	struct clone_t *ca = (struct clone_t *)arg;
	longjmp(*ca->env, ca->jmpval);
}

static int clone_parent(jmp_buf *env, int jmpval) __attribute__ ((noinline));
static int clone_parent(jmp_buf *env, int jmpval)
{
	struct clone_t ca = {
		.env    = env,
		.jmpval = jmpval,
	};

	return clone(child_func, ca.stack_ptr, CLONE_PARENT | SIGCHLD, &ca);
}

/*
 * Gets the init pipe fd from the environment, which is used to read the
 * bootstrap data and tell the parent what the new pid is after we finish
 * setting up the environment.
 */
static int initpipe(void)
{
	int pipenum;
	char *initpipe, *endptr;

	initpipe = getenv("_LIBCONTAINER_INITPIPE");
	if (initpipe == NULL || *initpipe == '\0')
		return -1;

	pipenum = strtol(initpipe, &endptr, 10);
	if (*endptr != '\0')
		bail("unable to parse _LIBCONTAINER_INITPIPE");

	return pipenum;
}

/* Returns the clone(2) flag for a namespace, given the name of a namespace. */
static int nsflag(char *name)
{
	if (!strcmp(name, "cgroup"))
		return CLONE_NEWCGROUP;
	else if (!strcmp(name, "ipc"))
		return CLONE_NEWIPC;
	else if (!strcmp(name, "mnt"))
		return CLONE_NEWNS;
	else if (!strcmp(name, "net"))
		return CLONE_NEWNET;
	else if (!strcmp(name, "pid"))
		return CLONE_NEWPID;
	else if (!strcmp(name, "user"))
		return CLONE_NEWUSER;
	else if (!strcmp(name, "uts"))
		return CLONE_NEWUTS;

	/* If we don't recognise a name, fallback to 0. */
	return 0;
}

static uint32_t readint32(char *buf)
{
	return *(uint32_t *) buf;
}

static uint8_t readint8(char *buf)
{
	return *(uint8_t *) buf;
}

static void nl_parse(int fd, struct nlconfig_t *config)
{
	size_t len, size;
	struct nlmsghdr hdr;
	char *data, *current;

	/* Retrieve the netlink header. */
	len = read(fd, &hdr, NLMSG_HDRLEN);
	if (len != NLMSG_HDRLEN)
		bail("invalid netlink header length %zu", len);

	if (hdr.nlmsg_type == NLMSG_ERROR)
		bail("failed to read netlink message");

	if (hdr.nlmsg_type != INIT_MSG)
		bail("unexpected msg type %d", hdr.nlmsg_type);

	/* Retrieve data. */
	size = NLMSG_PAYLOAD(&hdr, 0);
	current = data = malloc(size);
	if (!data)
		bail("failed to allocate %zu bytes of memory for nl_payload", size);

	len = read(fd, data, size);
	if (len != size)
		bail("failed to read netlink payload, %zu != %zu", len, size);

	/* Parse the netlink payload. */
	config->data = data;
	while (current < data + size) {
		struct nlattr *nlattr = (struct nlattr *)current;
		size_t payload_len = nlattr->nla_len - NLA_HDRLEN;

		/* Advance to payload. */
		current += NLA_HDRLEN;

		/* Handle payload. */
		switch (nlattr->nla_type) {
		case CLONE_FLAGS_ATTR:
			config->cloneflags = readint32(current);
			break;
		case ROOTLESS_ATTR:
			config->is_rootless = readint8(current);
			break;
		case OOM_SCORE_ADJ_ATTR:
			config->oom_score_adj = current;
			config->oom_score_adj_len = payload_len;
			break;
		case NS_PATHS_ATTR:
			config->namespaces = current;
			config->namespaces_len = payload_len;
			break;
		case UIDMAP_ATTR:
			config->uidmap = current;
			config->uidmap_len = payload_len;
			break;
		case GIDMAP_ATTR:
			config->gidmap = current;
			config->gidmap_len = payload_len;
			break;
		case SETGROUP_ATTR:
			config->is_setgroup = readint8(current);
			break;
		default:
			bail("unknown netlink message type %d", nlattr->nla_type);
		}

		current += NLA_ALIGN(payload_len);
	}
}

void nl_free(struct nlconfig_t *config)
{
	free(config->data);
}

void join_namespaces(char *nslist)
{
	int num = 0, i;
	char *saveptr = NULL;
	char *namespace = strtok_r(nslist, ",", &saveptr);
	struct namespace_t {
		int fd;
		int ns;
		char type[PATH_MAX];
		char path[PATH_MAX];
	} *namespaces = NULL;

	if (!namespace || !strlen(namespace) || !strlen(nslist))
		bail("ns paths are empty");

	/*
	 * We have to open the file descriptors first, since after
	 * we join the mnt namespace we might no longer be able to
	 * access the paths.
	 */
	do {
		int fd;
		char *path;
		struct namespace_t *ns;

		/* Resize the namespace array. */
		namespaces = realloc(namespaces, ++num * sizeof(struct namespace_t));
		if (!namespaces)
			bail("failed to reallocate namespace array");
		ns = &namespaces[num - 1];

		/* Split 'ns:path'. */
		path = strstr(namespace, ":");
		if (!path)
			bail("failed to parse %s", namespace);
		*path++ = '\0';

		fd = open(path, O_RDONLY);
		if (fd < 0)
			bail("failed to open %s", path);

		ns->fd = fd;
		ns->ns = nsflag(namespace);
		strncpy(ns->path, path, PATH_MAX);
	} while ((namespace = strtok_r(NULL, ",", &saveptr)) != NULL);

	/*
	 * The ordering in which we join namespaces is important. We should
	 * always join the user namespace *first*. This is all guaranteed
	 * from the container_linux.go side of this, so we're just going to
	 * follow the order given to us.
	 */

	for (i = 0; i < num; i++) {
		struct namespace_t ns = namespaces[i];

		if (setns(ns.fd, ns.ns) < 0)
			bail("failed to setns to %s", ns.path);

		close(ns.fd);
	}

	free(namespaces);
}

void nsexec(void)
{
	int pipenum;
	jmp_buf env;
	int sync_child_pipe[2], sync_grandchild_pipe[2];
	struct nlconfig_t config = {0};

	/*
	 * If we don't have an init pipe, just return to the go routine.
	 * We'll only get an init pipe for start or exec.
	 */
	pipenum = initpipe();
	if (pipenum == -1)
		return;

	/* Parse all of the netlink configuration. */
	nl_parse(pipenum, &config);

	/* Set oom_score_adj. This has to be done before !dumpable because
	 * /proc/self/oom_score_adj is not writeable unless you're an privileged
	 * user (if !dumpable is set). All children inherit their parent's
	 * oom_score_adj value on fork(2) so this will always be propagated
	 * properly.
	 */
	update_oom_score_adj(config.oom_score_adj, config.oom_score_adj_len);

	/*
	 * Make the process non-dumpable, to avoid various race conditions that
	 * could cause processes in namespaces we're joining to access host
	 * resources (or potentially execute code).
	 *
	 * However, if the number of namespaces we are joining is 0, we are not
	 * going to be switching to a different security context. Thus setting
	 * ourselves to be non-dumpable only breaks things (like rootless
	 * containers), which is the recommendation from the kernel folks.
	 */
	if (config.namespaces) {
		if (prctl(PR_SET_DUMPABLE, 0, 0, 0, 0) < 0)
			bail("failed to set process as non-dumpable");
	}

	/* Pipe so we can tell the child when we've finished setting up. */
	if (socketpair(AF_LOCAL, SOCK_STREAM, 0, sync_child_pipe) < 0)
		bail("failed to setup sync pipe between parent and child");

	/*
	 * We need a new socketpair to sync with grandchild so we don't have
	 * race condition with child.
	 */
	if (socketpair(AF_LOCAL, SOCK_STREAM, 0, sync_grandchild_pipe) < 0)
		bail("failed to setup sync pipe between parent and grandchild");

	/* TODO: Currently we aren't dealing with child deaths properly. */

	/*
	 * Okay, so this is quite annoying.
	 *
	 * In order for this unsharing code to be more extensible we need to split
	 * up unshare(CLONE_NEWUSER) and clone() in various ways. The ideal case
	 * would be if we did clone(CLONE_NEWUSER) and the other namespaces
	 * separately, but because of SELinux issues we cannot really do that. But
	 * we cannot just dump the namespace flags into clone(...) because several
	 * usecases (such as rootless containers) require more granularity around
	 * the namespace setup. In addition, some older kernels had issues where
	 * CLONE_NEWUSER wasn't handled before other namespaces (but we cannot
	 * handle this while also dealing with SELinux so we choose SELinux support
	 * over broken kernel support).
	 *
	 * However, if we unshare(2) the user namespace *before* we clone(2), then
	 * all hell breaks loose.
	 *
	 * The parent no longer has permissions to do many things (unshare(2) drops
	 * all capabilities in your old namespace), and the container cannot be set
	 * up to have more than one {uid,gid} mapping. This is obviously less than
	 * ideal. In order to fix this, we have to first clone(2) and then unshare.
	 *
	 * Unfortunately, it's not as simple as that. We have to fork to enter the
	 * PID namespace (the PID namespace only applies to children). Since we'll
	 * have to double-fork, this clone_parent() call won't be able to get the
	 * PID of the _actual_ init process (without doing more synchronisation than
	 * I can deal with at the moment). So we'll just get the parent to send it
	 * for us, the only job of this process is to update
	 * /proc/pid/{setgroups,uid_map,gid_map}.
	 *
	 * And as a result of the above, we also need to setns(2) in the first child
	 * because if we join a PID namespace in the topmost parent then our child
	 * will be in that namespace (and it will not be able to give us a PID value
	 * that makes sense without resorting to sending things with cmsg).
	 *
	 * This also deals with an older issue caused by dumping cloneflags into
	 * clone(2): On old kernels, CLONE_PARENT didn't work with CLONE_NEWPID, so
	 * we have to unshare(2) before clone(2) in order to do this. This was fixed
	 * in upstream commit 1f7f4dde5c945f41a7abc2285be43d918029ecc5, and was
	 * introduced by 40a0d32d1eaffe6aac7324ca92604b6b3977eb0e. As far as we're
	 * aware, the last mainline kernel which had this bug was Linux 3.12.
	 * However, we cannot comment on which kernels the broken patch was
	 * backported to.
	 *
	 * -- Aleksa "what has my life come to?" Sarai
	 */

	switch (setjmp(env)) {
	/*
	 * Stage 0: We're in the parent. Our job is just to create a new child
	 *          (stage 1: JUMP_CHILD) process and write its uid_map and
	 *          gid_map. That process will go on to create a new process, then
	 *          it will send us its PID which we will send to the bootstrap
	 *          process.
	 */
	case JUMP_PARENT: {
			int len;
			pid_t child, first_child = -1;
			char buf[JSON_MAX];
			bool ready = false;

			/* For debugging. */
			prctl(PR_SET_NAME, (unsigned long) "runc:[0:PARENT]", 0, 0, 0);

			/* Start the process of getting a container. */
			child = clone_parent(&env, JUMP_CHILD);
			if (child < 0)
				bail("unable to fork: child_func");

			/*
			 * State machine for synchronisation with the children.
			 *
			 * Father only return when both child and grandchild are
			 * ready, so we can receive all possible error codes
			 * generated by children.
			 */
			while (!ready) {
				enum sync_t s;
				int ret;

				syncfd = sync_child_pipe[1];
				close(sync_child_pipe[0]);

				if (read(syncfd, &s, sizeof(s)) != sizeof(s))
					bail("failed to sync with child: next state");

				switch (s) {
				case SYNC_ERR:
					/* We have to mirror the error code of the child. */
					if (read(syncfd, &ret, sizeof(ret)) != sizeof(ret))
						bail("failed to sync with child: read(error code)");

					exit(ret);
				case SYNC_USERMAP_PLS:
					/*
					 * Enable setgroups(2) if we've been asked to. But we also
					 * have to explicitly disable setgroups(2) if we're
					 * creating a rootless container (this is required since
					 * Linux 3.19).
					 */
					if (config.is_rootless && config.is_setgroup) {
						kill(child, SIGKILL);
						bail("cannot allow setgroup in an unprivileged user namespace setup");
					}

					if (config.is_setgroup)
						update_setgroups(child, SETGROUPS_ALLOW);
					if (config.is_rootless)
						update_setgroups(child, SETGROUPS_DENY);

					/* Set up mappings. */
					update_uidmap(child, config.uidmap, config.uidmap_len);
					update_gidmap(child, config.gidmap, config.gidmap_len);

					s = SYNC_USERMAP_ACK;
					if (write(syncfd, &s, sizeof(s)) != sizeof(s)) {
						kill(child, SIGKILL);
						bail("failed to sync with child: write(SYNC_USERMAP_ACK)");
					}
					break;
				case SYNC_RECVPID_PLS: {
						first_child = child;

						/* Get the init_func pid. */
						if (read(syncfd, &child, sizeof(child)) != sizeof(child)) {
							kill(first_child, SIGKILL);
							bail("failed to sync with child: read(childpid)");
						}

						/* Send ACK. */
						s = SYNC_RECVPID_ACK;
						if (write(syncfd, &s, sizeof(s)) != sizeof(s)) {
							kill(first_child, SIGKILL);
							kill(child, SIGKILL);
							bail("failed to sync with child: write(SYNC_RECVPID_ACK)");
						}
					}
					break;
				case SYNC_CHILD_READY:
					ready = true;
					break;
				default:
					bail("unexpected sync value: %u", s);
				}
			}

			/* Now sync with grandchild. */

			ready = false;
			while (!ready) {
				enum sync_t s;
				int ret;

				syncfd = sync_grandchild_pipe[1];
				close(sync_grandchild_pipe[0]);

				s = SYNC_GRANDCHILD;
				if (write(syncfd, &s, sizeof(s)) != sizeof(s)) {
					kill(child, SIGKILL);
					bail("failed to sync with child: write(SYNC_GRANDCHILD)");
				}

				if (read(syncfd, &s, sizeof(s)) != sizeof(s))
					bail("failed to sync with child: next state");

				switch (s) {
				case SYNC_ERR:
					/* We have to mirror the error code of the child. */
					if (read(syncfd, &ret, sizeof(ret)) != sizeof(ret))
						bail("failed to sync with child: read(error code)");

					exit(ret);
				case SYNC_CHILD_READY:
					ready = true;
					break;
				default:
					bail("unexpected sync value: %u", s);
				}
			}

			/*
			 * Send the init_func pid and the pid of the first child back to our parent.
			 *
			 * We need to send both back because we can't reap the first child we created (CLONE_PARENT).
			 * It becomes the responsibility of our parent to reap the first child.
			 */
			len = snprintf(buf, JSON_MAX, "{\"pid\": %d, \"pid_first\": %d}\n", child, first_child);
			if (len < 0) {
				kill(child, SIGKILL);
				bail("unable to generate JSON for child pid");
			}
			if (write(pipenum, buf, len) != len) {
				kill(child, SIGKILL);
				bail("unable to send child pid to bootstrapper");
			}

			exit(0);
		}

	/*
	 * Stage 1: We're in the first child process. Our job is to join any
	 *          provided namespaces in the netlink payload and unshare all
	 *          of the requested namespaces. If we've been asked to
	 *          CLONE_NEWUSER, we will ask our parent (stage 0) to set up
	 *          our user mappings for us. Then, we create a new child
	 *          (stage 2: JUMP_INIT) for PID namespace. We then send the
	 *          child's PID to our parent (stage 0).
	 */
	case JUMP_CHILD: {
			pid_t child;
			enum sync_t s;

			/* We're in a child and thus need to tell the parent if we die. */
			syncfd = sync_child_pipe[0];
			close(sync_child_pipe[1]);

			/* For debugging. */
			prctl(PR_SET_NAME, (unsigned long) "runc:[1:CHILD]", 0, 0, 0);

			/*
			 * We need to setns first. We cannot do this earlier (in stage 0)
			 * because of the fact that we forked to get here (the PID of
			 * [stage 2: JUMP_INIT]) would be meaningless). We could send it
			 * using cmsg(3) but that's just annoying.
			 */
			if (config.namespaces)
				join_namespaces(config.namespaces);

			/*
			 * Unshare all of the namespaces. Now, it should be noted that this
			 * ordering might break in the future (especially with rootless
			 * containers). But for now, it's not possible to split this into
			 * CLONE_NEWUSER + [the rest] because of some RHEL SELinux issues.
			 *
			 * Note that we don't merge this with clone() because there were
			 * some old kernel versions where clone(CLONE_PARENT | CLONE_NEWPID)
			 * was broken, so we'll just do it the long way anyway.
			 */
			if (unshare(config.cloneflags) < 0)
				bail("failed to unshare namespaces");

			/*
			 * Deal with user namespaces first. They are quite special, as they
			 * affect our ability to unshare other namespaces and are used as
			 * context for privilege checks.
			 */
			if (config.cloneflags & CLONE_NEWUSER) {
				/*
				 * We don't have the privileges to do any mapping here (see the
				 * clone_parent rant). So signal our parent to hook us up.
				 */

				/* Switching is only necessary if we joined namespaces. */
				if (config.namespaces) {
					if (prctl(PR_SET_DUMPABLE, 1, 0, 0, 0) < 0)
						bail("failed to set process as dumpable");
				}
				s = SYNC_USERMAP_PLS;
				if (write(syncfd, &s, sizeof(s)) != sizeof(s))
					bail("failed to sync with parent: write(SYNC_USERMAP_PLS)");

				/* ... wait for mapping ... */

				if (read(syncfd, &s, sizeof(s)) != sizeof(s))
					bail("failed to sync with parent: read(SYNC_USERMAP_ACK)");
				if (s != SYNC_USERMAP_ACK)
					bail("failed to sync with parent: SYNC_USERMAP_ACK: got %u", s);
				/* Switching is only necessary if we joined namespaces. */
				if (config.namespaces) {
					if (prctl(PR_SET_DUMPABLE, 0, 0, 0, 0) < 0)
						bail("failed to set process as dumpable");
				}
			}

			/*
			 * TODO: What about non-namespace clone flags that we're dropping here?
			 *
			 * We fork again because of PID namespace, setns(2) or unshare(2) don't
			 * change the PID namespace of the calling process, because doing so
			 * would change the caller's idea of its own PID (as reported by getpid()),
			 * which would break many applications and libraries, so we must fork
			 * to actually enter the new PID namespace.
			 */
			child = clone_parent(&env, JUMP_INIT);
			if (child < 0)
				bail("unable to fork: init_func");

			/* Send the child to our parent, which knows what it's doing. */
			s = SYNC_RECVPID_PLS;
			if (write(syncfd, &s, sizeof(s)) != sizeof(s)) {
				kill(child, SIGKILL);
				bail("failed to sync with parent: write(SYNC_RECVPID_PLS)");
			}
			if (write(syncfd, &child, sizeof(child)) != sizeof(child)) {
				kill(child, SIGKILL);
				bail("failed to sync with parent: write(childpid)");
			}

			/* ... wait for parent to get the pid ... */

			if (read(syncfd, &s, sizeof(s)) != sizeof(s)) {
				kill(child, SIGKILL);
				bail("failed to sync with parent: read(SYNC_RECVPID_ACK)");
			}
			if (s != SYNC_RECVPID_ACK) {
				kill(child, SIGKILL);
				bail("failed to sync with parent: SYNC_RECVPID_ACK: got %u", s);
			}

			s = SYNC_CHILD_READY;
			if (write(syncfd, &s, sizeof(s)) != sizeof(s)) {
				kill(child, SIGKILL);
				bail("failed to sync with parent: write(SYNC_CHILD_READY)");
			}

			/* Our work is done. [Stage 2: JUMP_INIT] is doing the rest of the work. */
			exit(0);
		}

	/*
	 * Stage 2: We're the final child process, and the only process that will
	 *          actually return to the Go runtime. Our job is to just do the
	 *          final cleanup steps and then return to the Go runtime to allow
	 *          init_linux.go to run.
	 */
	case JUMP_INIT: {
			/*
			 * We're inside the child now, having jumped from the
			 * start_child() code after forking in the parent.
			 */
			enum sync_t s;

			/* We're in a child and thus need to tell the parent if we die. */
			syncfd = sync_grandchild_pipe[0];
			close(sync_grandchild_pipe[1]);
			close(sync_child_pipe[0]);
			close(sync_child_pipe[1]);

			/* For debugging. */
			prctl(PR_SET_NAME, (unsigned long) "runc:[2:INIT]", 0, 0, 0);

			if (read(syncfd, &s, sizeof(s)) != sizeof(s))
				bail("failed to sync with parent: read(SYNC_GRANDCHILD)");
			if (s != SYNC_GRANDCHILD)
				bail("failed to sync with parent: SYNC_GRANDCHILD: got %u", s);

			if (setsid() < 0)
				bail("setsid failed");

			if (setuid(0) < 0)
				bail("setuid failed");

			if (setgid(0) < 0)
				bail("setgid failed");

			if (!config.is_rootless && config.is_setgroup) {
				if (setgroups(0, NULL) < 0)
					bail("setgroups failed");
			}

			s = SYNC_CHILD_READY;
			if (write(syncfd, &s, sizeof(s)) != sizeof(s))
				bail("failed to sync with patent: write(SYNC_CHILD_READY)");

			/* Close sync pipes. */
			close(sync_grandchild_pipe[0]);

			/* Free netlink data. */
			nl_free(&config);

			/* Finish executing, let the Go runtime take over. */
			return;
		}
	default:
		bail("unexpected jump value");
	}

	/* Should never be reached. */
	bail("should never be reached");
}
