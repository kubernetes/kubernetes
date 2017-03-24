// +build linux,arm64

package sys

// #include <sys/epoll.h>
/*
int EpollCreate1(int flag) {
	return epoll_create1(flag);
}

int EpollCtl(int efd, int op,int sfd, int events, int fd) {
	struct epoll_event event;
	event.events = events;
	event.data.fd = fd;

	return epoll_ctl(efd, op, sfd, &event);
}

struct event_t {
	uint32_t events;
	int fd;
};

struct epoll_event events[128];
int run_epoll_wait(int fd, struct event_t *event) {
	int n, i;
	n = epoll_wait(fd, events, 128, -1);
	for (i = 0; i < n; i++) {
		event[i].events = events[i].events;
		event[i].fd = events[i].data.fd;
	}
	return n;
}
*/
import "C"

import (
	"fmt"
	"syscall"
	"unsafe"
)

// EpollCreate1 calls a C implementation
func EpollCreate1(flag int) (int, error) {
	fd := int(C.EpollCreate1(C.int(flag)))
	if fd < 0 {
		return fd, fmt.Errorf("failed to create epoll, errno is %d", fd)
	}
	return fd, nil
}

// EpollCtl calls a C implementation
func EpollCtl(epfd int, op int, fd int, event *syscall.EpollEvent) error {
	errno := C.EpollCtl(C.int(epfd), C.int(syscall.EPOLL_CTL_ADD), C.int(fd), C.int(event.Events), C.int(event.Fd))
	if errno < 0 {
		return fmt.Errorf("Failed to ctl epoll")
	}
	return nil
}

// EpollWait calls a C implementation
func EpollWait(epfd int, events []syscall.EpollEvent, msec int) (int, error) {
	var c_events [128]C.struct_event_t
	n := int(C.run_epoll_wait(C.int(epfd), (*C.struct_event_t)(unsafe.Pointer(&c_events))))
	if n < 0 {
		return int(n), fmt.Errorf("Failed to wait epoll")
	}
	for i := 0; i < n; i++ {
		events[i].Fd = int32(c_events[i].fd)
		events[i].Events = uint32(c_events[i].events)
	}
	return int(n), nil
}
