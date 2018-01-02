#include <sys/types.h>
#include <unistd.h>
#include <stdio.h>

int main() {
	if (setuid(1) == -1) {
		perror("setuid");
		return 1;
	}
	return 0;
}
