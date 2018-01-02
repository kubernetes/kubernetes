TARGETS?=docker
MODULES?=${TARGETS:=.pp.bz2}
SHAREDIR?=/usr/share

all: ${TARGETS:=.pp.bz2}

%.pp.bz2: %.pp
	@echo Compressing $^ -\> $@
	bzip2 -9 $^

%.pp: %.te
	make -f ${SHAREDIR}/selinux/devel/Makefile $@

clean:
	rm -f *~  *.tc *.pp *.pp.bz2
	rm -rf tmp *.tar.gz

man: install
	sepolicy manpage --domain ${TARGETS}_t

install:
	semodule -i ${TARGETS}

