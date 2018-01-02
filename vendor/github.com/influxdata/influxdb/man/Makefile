#!/usr/bin/make -f

DESTDIR = /usr/local

MAN1_TXT =
MAN1_TXT += influxd.txt
MAN1_TXT += influxd-backup.txt
MAN1_TXT += influxd-config.txt
MAN1_TXT += influxd-restore.txt
MAN1_TXT += influxd-run.txt
MAN1_TXT += influxd-version.txt
MAN1_TXT += influx.txt
MAN1_TXT += influx_inspect.txt
MAN1_TXT += influx_stress.txt
MAN1_TXT += influx_tsm.txt

MAN_TXT = $(MAN1_TXT)
MAN_XML = $(patsubst %.txt,%.xml,$(MAN_TXT))

DOC_MAN1 = $(patsubst %.txt,%.1,$(MAN1_TXT))

build: $(DOC_MAN1)

install: build
	@echo '  INSTALL $(DOC_MAN1)' && \
	mkdir -p $(DESTDIR)/share/man/man1 && \
	install -m 0644 $(DOC_MAN1) $(DESTDIR)/share/man/man1

clean:
	rm -f $(MAN_XML) $(DOC_MAN1)

%.xml : %.txt
	@echo '  ASCIIDOC $@' && rm -f $@+ && \
	asciidoc -d manpage -b docbook -o $@+ $< && \
	mv $@+ $@

%.1 : %.xml
	@echo '  XMLTO $@' && \
	xmlto man $< 2> /dev/null

.PHONY: build install clean
