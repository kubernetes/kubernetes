// +build cgo

#include <sys/statvfs.h>

int getBytesFree(const char *path, unsigned long long *bytes) {
	struct statvfs buf;
	int res;
	if ((res = statvfs(path, &buf)) && res != 0) {
		return -1;
	}
	*bytes = buf.f_frsize * buf.f_bfree;
	return 0;
}

int getBytesTotal(const char *path, unsigned long long *bytes) {
	struct statvfs buf;
	int res;
	if ((res = statvfs(path, &buf)) && res != 0) {
		return -1;
	}
	*bytes = buf.f_frsize * buf.f_blocks;
	return 0;
}

// Bytes available to non-root.
int getBytesAvail(const char *path, unsigned long long *bytes) {
	struct statvfs buf;
	int res;
	if ((res = statvfs(path, &buf)) && res != 0) {
		return -1;
	}
	*bytes = buf.f_frsize * buf.f_bavail;
	return 0;
}
