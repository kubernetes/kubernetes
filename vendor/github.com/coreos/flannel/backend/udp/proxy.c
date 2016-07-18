// Copyright 2015 CoreOS, Inc.
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

#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <memory.h>
#include <assert.h>

#include <errno.h>
#include <poll.h>
#include <unistd.h>
#include <sys/types.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <linux/ip.h>
#include <linux/icmp.h>
#include <fcntl.h>

#define CMD_DEFINE
#include "proxy.h"

struct ip_net {
	in_addr_t ip;
	in_addr_t mask;
};

struct route_entry {
	struct ip_net      dst;
	struct sockaddr_in next_hop;
};

typedef struct icmp_pkt {
	struct iphdr   iph;
	struct icmphdr icmph;
	/* dest unreachable must include IP hdr 8 bytes of upper layer proto
	 * of the original packet. */
	char    data[sizeof(struct iphdr) + MAX_IPOPTLEN + 8];
} __attribute__ ((aligned (4))) icmp_pkt;

/* we calc hdr checksums using 32bit uints that can alias other types */
typedef uint32_t __attribute__((__may_alias__)) aliasing_uint32_t;

struct route_entry *routes;
size_t routes_alloc;
size_t routes_cnt;

in_addr_t tun_addr;

int log_enabled;
int exit_flag;

static inline in_addr_t netmask(int prefix_len) {
	return htonl(~((uint32_t)0) << (32 - prefix_len));
}

static inline int contains(struct ip_net net, in_addr_t ip) {
	return net.ip == (ip & net.mask);
}

static void log_error(const char *fmt, ...) {
	va_list ap;

	if( log_enabled ) {
		va_start(ap, fmt);
		vfprintf(stderr, fmt, ap);
		va_end(ap);
	}
}

/* fast version -- only works with mults of 4 bytes */
static uint16_t cksum(aliasing_uint32_t *buf, int len) {
	uint32_t sum = 0;
	uint16_t t1, t2;

	for( ; len > 0; len-- ) {
		uint32_t s = *buf++;
		sum += s;
		if( sum < s )
			sum++;
	}

	/* Fold down to 16 bits */
	t1 = sum;
	t2 = sum >> 16;
	t1 += t2;
	if( t1 < t2 )
		t1++;

	return ~t1;
}

static void send_net_unreachable(int tun, char *offender) {
	icmp_pkt pkt;
	int off_iph_len;
	struct iphdr *off_iph = (struct iphdr *)offender;
	size_t pktlen, nsent;

	off_iph_len = off_iph->ihl * 4;
	if( off_iph_len >= sizeof(struct iphdr) + MAX_IPOPTLEN ) {
		log_error("not sending net unreachable: mulformed ip pkt: iph=%d\n", (int)off_iph_len);
		return; /* ip pkt mulformed */
	}

	if( off_iph->protocol == IPPROTO_ICMP ) {
		/* To avoid infinite loops, RFC 792 instructs not to send ICMPs
		 * about ICMPs */
		return;
	}

	/* Lower 3 bits (in network order) of frag_off is actually flags */
	if( (off_iph->frag_off & htons(0x1FFF)) != 0 ) {
		/* ICMP messages are only sent for first fragemnt */
		return;
	}

	pktlen = sizeof(struct iphdr) + sizeof(struct icmphdr) + off_iph_len + 8;

	memset(&pkt, 0, sizeof(pkt));

	/* Fill in the IP header */
	pkt.iph.ihl = sizeof(struct iphdr) / 4;
	pkt.iph.version = IPVERSION;
	pkt.iph.tot_len = htons(pktlen);
	pkt.iph.ttl = 8;
	pkt.iph.protocol = IPPROTO_ICMP;
	pkt.iph.saddr = tun_addr;
	pkt.iph.daddr = off_iph->saddr;
	pkt.iph.check = cksum((aliasing_uint32_t*) &pkt.iph, sizeof(struct iphdr) / sizeof(aliasing_uint32_t));

	/* Fill in the ICMP header */
	pkt.icmph.type = ICMP_DEST_UNREACH;
	pkt.icmph.code = ICMP_NET_UNREACH;

	/* Copy the offenders IP hdr + first 8 bytes of IP payload */
	memcpy(pkt.data, offender, off_iph_len + 8);

	/* Compute the checksum over the ICMP header and data */
	pkt.icmph.checksum = cksum((aliasing_uint32_t*) &pkt.icmph,
			(sizeof(struct icmphdr) + off_iph_len + 8) / sizeof(aliasing_uint32_t));

	/* Kick it back */
	nsent = write(tun, &pkt, pktlen);

	if( nsent < 0 ) {
		log_error("failed to send ICMP net unreachable: %s\n", strerror(errno));
	} else if( nsent != pktlen ) {
		log_error("failed to send ICMP net unreachable: only %d out of %d byte sent\n", (int)nsent, (int)pktlen);
	}
}

static int set_route(struct ip_net dst, struct sockaddr_in *next_hop) {
	size_t i;

	for( i = 0; i < routes_cnt; i++ ) {
		if( dst.ip == routes[i].dst.ip && dst.mask == routes[i].dst.mask ) {
			routes[i].next_hop = *next_hop;
			return 0;
		}
	}

	if( routes_alloc == routes_cnt ) {
		int new_alloc = (routes_alloc ? 2*routes_alloc : 8);
		struct route_entry *new_routes = (struct route_entry *) realloc(routes, new_alloc*sizeof(struct route_entry));
		if( !new_routes )
			return ENOMEM;

		routes = new_routes;
		routes_alloc = new_alloc;
	}

	routes[routes_cnt].dst = dst;
	routes[routes_cnt].next_hop = *next_hop;
	routes_cnt++;

	return 0;
}

static int del_route(struct ip_net dst) {
	size_t i;

	for( i = 0; i < routes_cnt; i++ ) {
		if( dst.ip == routes[i].dst.ip && dst.mask == routes[i].dst.mask ) {
			routes[i] = routes[routes_cnt-1];
			routes_cnt--;
			return 0;
		}
	}

	return ENOENT;
}

static struct sockaddr_in *find_route(in_addr_t dst) {
	size_t i;

	for( i = 0; i < routes_cnt; i++ ) {
		if( contains(routes[i].dst, dst) ) {
			// packets for same dest tend to come in bursts. swap to front make it faster for subsequent ones
			if( i != 0 ) {
				struct route_entry tmp = routes[i];
				routes[i] = routes[0];
				routes[0] = tmp;
			}

			return &routes[0].next_hop;
		}
	}

	return NULL;
}

static char *inaddr_str(in_addr_t a, char *buf, size_t len) {
	struct in_addr addr;
	addr.s_addr = a;

	strncpy(buf, inet_ntoa(addr), len);
	buf[len-1] = '\0';

	return buf;
}

static ssize_t tun_recv_packet(int tun, char *buf, size_t buflen) {
	ssize_t nread = read(tun, buf, buflen);

	if( nread < sizeof(struct iphdr) ) {
		if( nread < 0 ) {
			if( errno != EAGAIN && errno != EWOULDBLOCK )
				log_error("TUN recv failed: %s\n", strerror(errno));
		} else {
			log_error("TUN recv packet too small: %d bytes\n", (int)nread);
		}
		return -1;
	}

	return nread;
}

static ssize_t sock_recv_packet(int sock, char *buf, size_t buflen) {
	ssize_t nread = recv(sock, buf, buflen, MSG_DONTWAIT);

	if( nread < sizeof(struct iphdr) ) {
		if( nread < 0 ) {
			if( errno != EAGAIN && errno != EWOULDBLOCK )
				log_error("UDP recv failed: %s\n", strerror(errno));
		} else {
			log_error("UDP recv packet too small: %d bytes\n", (int)nread);
		}
		return -1;
	}

	return nread;
}

static void sock_send_packet(int sock, char *pkt, size_t pktlen, struct sockaddr_in *dst) {
	ssize_t nsent = sendto(sock, pkt, pktlen, 0, (struct sockaddr *)dst, sizeof(struct sockaddr_in));

	if( nsent != pktlen ) {
		if( nsent < 0 ) {
			log_error("UDP send to %s:%hu failed: %s\n",
					inet_ntoa(dst->sin_addr), ntohs(dst->sin_port), strerror(errno));
		} else {
			log_error("Was only able to send %d out of %d bytes to %s:%hu\n",
					(int)nsent, (int)pktlen, inet_ntoa(dst->sin_addr), ntohs(dst->sin_port));
		}
	}
}

static void tun_send_packet(int tun, char *pkt, size_t pktlen) {
	ssize_t nsent;
_retry:
	nsent = write(tun, pkt, pktlen);

	if( nsent != pktlen ) {
		if( nsent < 0 ) {
			if( errno == EAGAIN || errno == EWOULDBLOCK)
				goto _retry;

			log_error("TUN send failed: %s\n", strerror(errno));
		} else {
			log_error("Was only able to send %d out of %d bytes to TUN\n", (int)nsent, (int)pktlen);
		}
	}
}

inline static int decrement_ttl(struct iphdr *iph) {
	if( --(iph->ttl) == 0 ) {
		char saddr[32], daddr[32];
		log_error("Discarding IP fragment %s -> %s due to zero TTL\n",
				inaddr_str(iph->saddr, saddr, sizeof(saddr)),
				inaddr_str(iph->daddr, daddr, sizeof(daddr)));
		return 0;
	}

	/* patch up IP checksum (see RFC 1624) */
	if( iph->check >= htons(0xFFFFu - 0x100) ) {
		iph->check += htons(0x100) + 1;
	} else {
		iph->check += htons(0x100);
	}

	return 1;
}

static int tun_to_udp(int tun, int sock, char *buf, size_t buflen) {
	struct iphdr *iph;
	struct sockaddr_in *next_hop;

	ssize_t pktlen = tun_recv_packet(tun, buf, buflen);
	if( pktlen < 0 )
		return 0;
	
	iph = (struct iphdr *)buf;

	next_hop = find_route((in_addr_t) iph->daddr);
	if( !next_hop ) {
		send_net_unreachable(tun, buf);
		goto _active;
	}

	if( !decrement_ttl(iph) ) {
		/* TTL went to 0, discard.
		 * TODO: send back ICMP Time Exceeded
		 */
		goto _active;
	}

	sock_send_packet(sock, buf, pktlen, next_hop);
_active:
	return 1;
}

static int udp_to_tun(int sock, int tun, char *buf, size_t buflen) {
	struct iphdr *iph;

	ssize_t pktlen = sock_recv_packet(sock, buf, buflen);
	if( pktlen < 0 )
		return 0;

	iph = (struct iphdr *)buf;

	if( !decrement_ttl(iph) ) {
		/* TTL went to 0, discard.
		 * TODO: send back ICMP Time Exceeded
		 */
		goto _active;
	}

	tun_send_packet(tun, buf, pktlen);
_active:
	return 1;
}

static void process_cmd(int ctl) {
	struct command cmd;
	struct ip_net ipn;
	struct sockaddr_in sa = {
		.sin_family = AF_INET
	};

	ssize_t nrecv = recv(ctl, (char *) &cmd, sizeof(cmd), 0);
	if( nrecv < 0 ) {
		log_error("CTL recv failed: %s\n", strerror(errno));
		return;
	}

	if( cmd.cmd == CMD_SET_ROUTE ) {
		ipn.mask = netmask(cmd.dest_net_len);
		ipn.ip = cmd.dest_net & ipn.mask;

		sa.sin_addr.s_addr = cmd.next_hop_ip;
		sa.sin_port = htons(cmd.next_hop_port);

		set_route(ipn, &sa);

	} else if( cmd.cmd == CMD_DEL_ROUTE ) {
		ipn.mask = netmask(cmd.dest_net_len);
		ipn.ip = cmd.dest_net & ipn.mask;

		del_route(ipn);

	} else if( cmd.cmd == CMD_STOP ) {
		exit_flag = 1;
	}
}

enum PFD {
	PFD_TUN = 0,
	PFD_SOCK,
	PFD_CTL,
	PFD_CNT
};

void run_proxy(int tun, int sock, int ctl, in_addr_t tun_ip, size_t tun_mtu, int log_errors) {
	char *buf;
	struct pollfd fds[PFD_CNT] = {
		{
			.fd = tun,
			.events = POLLIN
		},
		{
			.fd = sock,
			.events = POLLIN
		},
		{
			.fd = ctl,
			.events = POLLIN
		},
	};

	exit_flag = 0;
	tun_addr = tun_ip;
	log_enabled = log_errors;

	buf = (char *) malloc(tun_mtu);
	if( !buf ) {
		log_error("Failed to allocate %d byte buffer\n", tun_mtu);
		exit(1);
	}

	fcntl(tun, F_SETFL, O_NONBLOCK);

	while( !exit_flag ) {
		int nfds = poll(fds, PFD_CNT, -1), activity;
		if( nfds < 0 ) {
			if( errno == EINTR )
				continue;

			log_error("Poll failed: %s\n", strerror(errno));
			exit(1);
		}

		if( fds[PFD_CTL].revents & POLLIN )
			process_cmd(ctl);

		if( fds[PFD_TUN].revents & POLLIN || fds[PFD_SOCK].revents & POLLIN )
			do {
				activity = 0;
				activity += tun_to_udp(tun, sock, buf, tun_mtu);
				activity += udp_to_tun(sock, tun, buf, tun_mtu);

				/* As long as tun or udp is readable bypass poll().
				 * We'll just occasionally get EAGAIN on an unreadable fd which
				 * is cheaper than the poll() call, the rest of the time the
				 * read/recvfrom call moves data which poll() never does for us.
				 *
				 * This is at the expense of the ctl socket, a counter could be
				 * used to place an upper bound on how long we may neglect ctl.
				 */
			} while( activity );
	}

	free(buf);
}

