#include <sys/types.h>
#include <unistd.h>
#include <stdio.h>

int main() {
	if (setgid(1) == -1) {
		perror("setgid");
		return 1;
	}
	return 0;
}
