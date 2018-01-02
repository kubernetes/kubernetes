#include <stdio.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/ip.h>
#include <netinet/udp.h>

int main() {
	if (socket(PF_INET, SOCK_RAW, IPPROTO_UDP) == -1) {
		perror("socket");
		return 1;
	}

	return 0;
}
