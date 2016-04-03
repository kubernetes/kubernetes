FROM docs/base:latest
MAINTAINER Mary Anthony <mary@docker.com> (@moxiegirl)

# To get the git info for this repo
COPY . /src

COPY . /docs/content/

WORKDIR /docs/content

RUN /docs/content/touch-up.sh

WORKDIR /docs
