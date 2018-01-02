FROM buildpack-deps:jessie

COPY . /usr/src/

WORKDIR /usr/src/

RUN gcc -g -Wall -static userns.c -o /usr/bin/userns-test \
	&& gcc -g -Wall -static ns.c -o /usr/bin/ns-test \
	&& gcc -g -Wall -static acct.c -o /usr/bin/acct-test \
	&& gcc -g -Wall -static setuid.c -o /usr/bin/setuid-test \
	&& gcc -g -Wall -static setgid.c -o /usr/bin/setgid-test \
	&& gcc -g -Wall -static socket.c -o /usr/bin/socket-test \
	&& gcc -g -Wall -static raw.c -o /usr/bin/raw-test

RUN [ "$(uname -m)" = "x86_64" ] && gcc -s -m32 -nostdlib exit32.s -o /usr/bin/exit32-test || true
