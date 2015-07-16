FONT := DroidSansMono.ttf

PNGS := $(patsubst %.seqdiag,%.png,$(wildcard *.seqdiag))

.PHONY: all
all: $(PNGS)

.PHONY: watch
watch:
	fswatch *.seqdiag | xargs -n 1 sh -c "make || true"

$(FONT):
	curl -sLo $@ https://googlefontdirectory.googlecode.com/hg/apache/droidsansmono/$(FONT)

%.png: %.seqdiag $(FONT)
	seqdiag --no-transparency -a -f '$(FONT)' $<

# Build the stuff via a docker image
.PHONY: docker
docker:
	docker build -t clustering-seqdiag .
	docker run --rm clustering-seqdiag | tar xvf -

docker-clean:
	docker rmi clustering-seqdiag || true
	docker images -q --filter "dangling=true" | xargs docker rmi

fix-clock-skew:
	boot2docker ssh sudo date -u -D "%Y%m%d%H%M.%S" --set "$(shell date -u +%Y%m%d%H%M.%S)"
