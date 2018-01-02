CASK = cask
EMACS_BIN ?= emacs
EMACS_FLAGS =
EMACS_EXEC = $(CASK) exec $(EMACS_BIN) --no-site-file --no-site-lisp --batch $(EMACS_FLAGS)

OBJECTS = govc.elc

elpa:
	$(CASK) install
	$(CASK) update
	touch $@

.PHONY: build test docs clean

build: elpa $(OBJECTS)

test: build docs
	$(EMACS_EXEC) -l test/make.el -f make-test

docs: build
	$(EMACS_EXEC) -l test/make.el -f make-docs
clean:
	rm -f $(OBJECTS) elpa
	rm -rf .cask

%.elc: %.el
	$(EMACS_EXEC) -f batch-byte-compile $<
