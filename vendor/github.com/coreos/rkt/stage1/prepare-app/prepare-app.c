// Copyright 2015 The rkt Authors
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
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mount.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/vfs.h>
#include <dirent.h>
#include <inttypes.h>
#include <stdbool.h>

#define err_out(_fmt, _args...)						\
		fprintf(stderr, "Error: " _fmt "\n", ##_args);
static int exit_err;
#define exit_if(_cond, _fmt, _args...)					\
	exit_err++;							\
	if(_cond) {							\
		err_out(_fmt, ##_args);					\
		exit(exit_err);						\
	}
#define pexit_if(_cond, _fmt, _args...)					\
	exit_if(_cond, _fmt ": %s", ##_args, strerror(errno))

#define goto_if(_cond, _lbl, _fmt, _args...)				\
	if(_cond) {							\
		err_out(_fmt, ##_args);					\
		goto _lbl;						\
	}
#define pgoto_if(_cond, _lbl, _fmt, _args...)				\
	goto_if(_cond, _lbl, _fmt ": %s", ##_args, strerror(errno));

#define nelems(_array) \
	(sizeof(_array) / sizeof(_array[0]))
#define lenof(_str) \
	(sizeof(_str) - 1)

#define MACHINE_ID_LEN		lenof("0123456789abcdef0123456789abcdef")
#define MACHINE_NAME_LEN	lenof("rkt-01234567-89ab-cdef-0123-456789abcdef")

#define UNMAPPED ((uid_t) -1)

#ifndef CGROUP2_SUPER_MAGIC
#define CGROUP2_SUPER_MAGIC 0x63677270
#endif

/* permission masks */
#define WORLD_READABLE          0444
#define WORLD_WRITABLE          0222

typedef struct _dir_op_t {
	const char	*name;
	mode_t		mode;
} dir_op_t;

typedef struct _mount_point_t {
	const char	*source;
	const char	*target;
	const char	*type;
	const char	*options;
	unsigned long	flags;
	const bool	skip_if_dst_exists; // Only respected for files_mount_table
} mount_point;

#define dir(_name, _mode) \
	{ .name = _name, .mode = _mode }

static void mount_at(const char *root, const mount_point *mnt)
{
	char to[4096];
	exit_if(snprintf(to, sizeof(to), "%s/%s", root, mnt->target) >= sizeof(to),
		"Path too long: \"%s\"", to);
	pexit_if(mount(mnt->source, to, mnt->type,
		       mnt->flags, mnt->options) == -1,
		 "Mounting \"%s\" on \"%s\" failed", mnt->source, to);
}

static int mount_sys_required(const char *root)
{
	FILE *f;
	char *line = NULL;
	size_t len = 0;
	ssize_t read;

	pexit_if((f = fopen("/proc/self/mountinfo", "re")) == NULL,
		 "Unable to open /proc/self/mountinfo");

	while ((read = getline(&line, &len, f)) != -1) {
		char *sys_dir;
		char *sys_subdir;
		char *mountpoint;

		exit_if(asprintf(&sys_dir, "%s/sys", root) == -1,
			"Calling asprintf failed");
		exit_if(asprintf(&sys_subdir, "%s/sys/", root) == -1,
			"Calling asprintf failed");
		sscanf(line, "%*s %*s %*s %*s %ms", &mountpoint);

		// The mount point is exactly $ROOTFS/sys
		if (strcmp(sys_dir, mountpoint) == 0) {
			free(mountpoint);
			return 0;
		}
		// The mount point is a subdirectory of $ROOTFS/sys
		if (strncmp(sys_subdir, mountpoint, strlen(sys_subdir)) == 0) {
			free(mountpoint);
			return 0;
		}

		free(mountpoint);
	}

	pexit_if(fclose(f) != 0, "Unable to close /proc/self/mountinfo");

	return 1;
}
static void mount_sys(const char *root)
{
	struct statfs fs;
	const mount_point sys_bind_rec = { "/sys", "sys", "bind", NULL, MS_BIND|MS_REC, false };
	const mount_point sys_bind = { "/sys", "sys", "bind", NULL, MS_BIND, false };

	pexit_if(statfs("/sys/fs/cgroup", &fs) != 0,
	         "Cannot statfs /sys/fs/cgroup");
	if (fs.f_type == (typeof(fs.f_type)) CGROUP2_SUPER_MAGIC) {
		/* With the unified cgroup hierarchy, recursive bind mounts
		 * are fine. */
		mount_at(root, &sys_bind_rec);
		return;
	}

	// For security reasons recent Linux kernels do not allow to bind-mount non-recursively
	// if it would give read-write access to other subdirectories mounted as read-only.
	// Hence we have to check if we are in a user namespaced environment and bind mount recursively instead.
	if (access("/proc/1/uid_map", F_OK) == 0) {
		FILE *f;
		int k;
		uid_t uid_base, uid_shift, uid_range;

		pexit_if((f = fopen("/proc/1/uid_map", "re")) == NULL,
			 "Unable to open /proc/1/uid_map");

		if (sizeof(uid_t) == 4) {
			k = fscanf(f, "%"PRIu32" %"PRIu32" %"PRIu32,
				   &uid_base, &uid_shift, &uid_range);
		} else {
			k = fscanf(f, "%"PRIu16" %"PRIu16" %"PRIu16,
				   &uid_base, &uid_shift, &uid_range);
		}
		pexit_if(fclose(f) != 0, "Unable to close /proc/1/uid_map");
		pexit_if(k != 3, "Invalid uid_map format");

		// do a recursive bind mount if we are in a user namespace having a parent namespace set,
		// i.e. either one of uid base, shift, or the range is set, see user_namespaces(7).
		if (uid_base != 0 || uid_shift != 0 || uid_range != UNMAPPED) {
			mount_at(root, &sys_bind_rec);
			return;
		}
	}

	/* With cgroup-v1, rkt and systemd-nspawn add more cgroup
	 * bind-mounts to control which files are read-only. To avoid
	 * a quadratic progression, prepare-app does not bind mount
	 * /sys recursively. See:
	 * https://github.com/coreos/rkt/issues/2351 */
	mount_at(root, &sys_bind);
}

static void copy_volume_symlinks()
{
	DIR *volumes_dir;
	struct dirent *de;
	const char *rkt_volume_links_path = "/rkt/volumes";
	const char *dev_rkt_path = "/dev/.rkt";

	pexit_if(mkdir(dev_rkt_path, 0700) == -1 && errno != EEXIST,
		"Failed to create directory \"%s\"", dev_rkt_path);

	pexit_if((volumes_dir = opendir(rkt_volume_links_path)) == NULL && errno != ENOENT,
                 "Failed to open directory \"%s\"", rkt_volume_links_path);
	while (volumes_dir) {
		errno = 0;
		if ((de = readdir(volumes_dir)) != NULL) {
			char *link_path;
			char *new_link;
			char target[4096] = {0,};

			if (!strcmp(de->d_name, ".") || !strcmp(de->d_name, ".."))
			  continue;

			exit_if(asprintf(&link_path, "%s/%s", rkt_volume_links_path, de->d_name) == -1,
				"Calling asprintf failed");
			exit_if(asprintf(&new_link, "%s/%s", dev_rkt_path, de->d_name) == -1,
				"Calling asprintf failed");

			pexit_if(readlink(link_path, target, sizeof(target)) == -1,
				 "Error reading \"%s\" link", link_path);
			pexit_if(symlink(target, new_link) == -1 && errno != EEXIST,
				"Failed to create volume symlink \"%s\"", new_link);
		} else {
			pexit_if(errno != 0,
				"Error reading \"%s\" directory", rkt_volume_links_path);
			pexit_if(closedir(volumes_dir),
				 "Error closing \"%s\" directory", rkt_volume_links_path);
			return;
		}
	}
}

/* Determine if the specified ptmx device (or symlink to a device)
 * is usable by all users.
 *
 * dirfd: Open file descriptor of a root directory.
 * path: Relative path to ptmx device below the path specified by dirfd.
 *
 * Returns true on success, else false.
 */
bool
ptmx_device_usable (int dirfd, const char *path)
{
	struct stat st;
	int perms;
	bool world_readable, world_writable;
	bool is_char;
	bool dev_type;
	dev_t expected_dev;

	if (dirfd < 0 || ! path) {
		return false;
	}

	expected_dev = makedev(5, 2);

	if (fstatat (dirfd, path, &st, 0) < 0) {
		return false;
	}

	is_char = S_ISCHR (st.st_mode);
	dev_type = (expected_dev == st.st_rdev);

	perms = (st.st_mode & ACCESSPERMS);

	world_readable = (perms & WORLD_READABLE) == WORLD_READABLE;
	world_writable = (perms & WORLD_WRITABLE) == WORLD_WRITABLE;

	return (is_char && dev_type && world_readable && world_writable);
}

int main(int argc, char *argv[])
{
	static const char *unlink_paths[] = {
		"dev/shm",
		 NULL
	};
	static const dir_op_t dirs[] = {
		dir("dev",	0755),
		dir("dev/net",	0755),
		dir("dev/shm",	0755),
		dir("etc",	0755),
		dir("proc",	0755),
		dir("sys",	0755),
		dir("tmp",	01777),
		dir("dev/pts",	0755),
		dir("run",			0755),
		dir("run/systemd",		0755),
		dir("run/systemd/journal",	0755),
	};
	static const char *devnodes[] = {
		"/dev/null",
		"/dev/zero",
		"/dev/full",
		"/dev/random",
		"/dev/urandom",
		"/dev/tty",
		"/dev/net/tun",
		"/dev/console",
		NULL
	};
	static const mount_point dirs_mount_table[] = {
		{ "/proc", "/proc", "bind", NULL, MS_BIND|MS_REC, false },
		{ "/dev/shm", "/dev/shm", "bind", NULL, MS_BIND, false },
		{ "/dev/pts", "/dev/pts", "bind", NULL, MS_BIND, false },
		{ "/run/systemd/journal", "/run/systemd/journal", "bind", NULL, MS_BIND, false },
		/* /sys is handled separately */
	};
	static const mount_point files_mount_table[] = {
		{ "/etc/rkt-resolv.conf", "/etc/resolv.conf", "bind", NULL, MS_BIND, false },
		{ "/etc/rkt-hosts", "/etc/hosts", "bind", NULL, MS_BIND, false },
		{ "/etc/hosts-fallback", "/etc/hosts", "bind", NULL, MS_BIND, true }, // only create as fallback
		{ "/proc/sys/kernel/hostname", "/etc/hostname", "bind", NULL, MS_BIND, false },
		// TODO @alepuccetti this could be removed when https://github.com/systemd/systemd/issues/3544 is solved
		{ "/run/systemd/notify", "/run/systemd/notify", "bind", NULL, MS_BIND, false },
	};
	const char *root;
	int rootfd;
	char to[4096];
	int i;
	bool ptmx_usable, pts_ptmx_usable;

	exit_if(argc < 2,
		"Usage: %s /path/to/root", argv[0]);

	root = argv[1];

	/* Make stage2's root a mount point. Chrooting an application in a
	 * directory which is not a mount point is not nice because the
	 * application would not be able to remount "/" it as private mount.
	 * This allows Docker to run inside rkt.
	 * The recursive flag is to preserve volumes mounted previously by
	 * systemd-nspawn via "rkt run -volume".
	 * */
	pexit_if(mount(root, root, "bind", MS_BIND | MS_REC, NULL) == -1,
			"Make / a mount point failed");

	rootfd = open(root, O_DIRECTORY | O_CLOEXEC);
	pexit_if(rootfd < 0,
		"Failed to open directory \"%s\"", root);

	/* Some images have annoying symlinks that are resolved as dangling
	 * links before the chroot in stage1. E.g. "/dev/shm" -> "/run/shm"
	 * Just remove the symlinks.
         */
	for (i = 0; unlink_paths[i]; i++) {
		pexit_if(unlinkat(rootfd, unlink_paths[i], 0) != 0
			 && errno != ENOENT && errno != EISDIR,
			 "Failed to unlink \"%s\"", unlink_paths[i])
	}

	/* Create the directories */
	umask(0);
	for (i = 0; i < nelems(dirs); i++) {
		const dir_op_t *d = &dirs[i];
		pexit_if(mkdirat(rootfd, d->name, d->mode) == -1 &&
			 errno != EEXIST,
			"Failed to create directory \"%s/%s\"", root, d->name);
	}

	close(rootfd);

	/* systemd-nspawn already creates few /dev entries in the container
	 * namespace: copy_devnodes()
	 * http://cgit.freedesktop.org/systemd/systemd/tree/src/nspawn/nspawn.c?h=v219#n1345
	 *
	 * But they are not visible by the apps because they are "protected" by
	 * the chroot.
	 *
	 * Bind mount them individually over the chroot border.
	 *
	 * Do NOT bind mount the whole directory /dev because it would shadow
	 * potential individual bind mount by stage0 ("rkt run --volume...").
	 *
	 * Do NOT use mknod, it would not work for /dev/console because it is
	 * a bind mount to a pts and pts device nodes only work when they live
	 * on a devpts filesystem.
	 */
	for (i = 0; devnodes[i]; i++) {
		const char *from = devnodes[i];
		int fd;

		/* If the file does not exist, skip it. It might be because
		 * the kernel does not provide it (e.g. kernel compiled without
		 * CONFIG_TUN) or because systemd-nspawn does not provide it
		 * (/dev/net/tun is not available with systemd-nspawn < v217
		 */
		if (access(from, F_OK) != 0)
			continue;

		exit_if(snprintf(to, sizeof(to), "%s%s", root, from) >= sizeof(to),
			"Path too long: \"%s\"", to);

		/* The mode does not matter: it will be bind-mounted over.
		 */
		fd = open(to, O_WRONLY|O_CREAT|O_CLOEXEC|O_NOCTTY, 0644);
		if (fd != -1)
			close(fd);

		pexit_if(mount(from, to, "bind", MS_BIND, NULL) == -1,
				"Mounting \"%s\" on \"%s\" failed", from, to);
	}

	/* Bind mount directories */
	for (i = 0; i < nelems(dirs_mount_table); i++) {
		mount_at(root, &dirs_mount_table[i]);
	}

	/* Bind mount /sys: handled differently, depending on cgroups */
	if (mount_sys_required(root))
		mount_sys(root);

	/* Bind mount files, if the source exists.
	 * By default, overwrite dst unless skip_if_dst_exists is true. */
	for (i = 0; i < nelems(files_mount_table); i++) {
		const mount_point *mnt = &files_mount_table[i];
		int fd;

		exit_if(snprintf(to, sizeof(to), "%s/%s", root, mnt->target) >= sizeof(to),
			"Path too long: \"%s\"", to);
		if (access(mnt->source, F_OK) != 0)
			continue;
		if( mnt->skip_if_dst_exists && access(to, F_OK) == 0)
			continue;
		if (access(to, F_OK) != 0) {
			pexit_if((fd = creat(to, 0644)) == -1,
				"Cannot create file: \"%s\"", to);
			pexit_if(close(fd) == -1,
				"Cannot close file: \"%s\"", to);
		}
		pexit_if(mount(mnt->source, to, mnt->type,
			       mnt->flags, mnt->options) == -1,
				"Mounting \"%s\" on \"%s\" failed", mnt->source, to);
	}

	/* Now that all mounts have been handled, reopen the root
	 * directory to special-case the handling of ptmx devices.
	 */
	rootfd = open(root, O_DIRECTORY | O_CLOEXEC);
	pexit_if(rootfd < 0,
		"Failed to open directory \"%s\"", root);

	ptmx_usable = ptmx_device_usable (rootfd, "dev/ptmx");
	pts_ptmx_usable = ptmx_device_usable (rootfd, "dev/pts/ptmx");

	if (pts_ptmx_usable) {
		if (! ptmx_usable) {
			pexit_if(unlinkat(rootfd, "dev/ptmx", 0) != 0
					&& errno != ENOENT,
					"Failed to unlink \"%s\"", "dev/ptmx");
			pexit_if(symlinkat("/dev/pts/ptmx", rootfd, "dev/ptmx") == -1,
					"Failed to create /dev/ptmx symlink");
		}
	} else {
		if (! ptmx_usable) {
			int perms = (WORLD_READABLE + WORLD_WRITABLE);

			pexit_if(unlinkat(rootfd, "dev/ptmx", 0) != 0
					&& errno != ENOENT,
					"Failed to unlink \"%s\"", "dev/ptmx");
			pexit_if(mknodat (rootfd, "dev/ptmx", (S_IFCHR|perms), makedev (5, 2)) < 0,
					"Failed to create device: \"%s\"", "dev/ptmx");
		}
	}

	close(rootfd);

	/* Copy symlinks to device node volumes to "/dev/.rkt" so they can be
	 * used in the DeviceAllow= option of the app's unit file (systemd
	 * needs the path to start with "/dev". */
	copy_volume_symlinks();

	/* /dev/log -> /run/systemd/journal/dev-log */
	exit_if(snprintf(to, sizeof(to), "%s/dev/log", root) >= sizeof(to),
		"Path too long: \"%s\"", to);
	pexit_if(symlink("/run/systemd/journal/dev-log", to) == -1 && errno != EEXIST,
		"Failed to create /dev/log symlink");

	return EXIT_SUCCESS;
}
