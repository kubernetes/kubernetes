prefix=/usr/local/bin

default: build

build:
	@echo "Nothing to build. Use make install"

install: intemp.sh
	install intemp.sh $(DESTDIR)$(prefix)

uninstall:
	-rm $(DESTDIR)$(prefix)/intemp.sh
