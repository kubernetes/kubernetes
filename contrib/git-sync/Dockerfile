FROM golang:1.4-onbuild
VOLUME ["/git"]
ENV GIT_SYNC_DEST /git
ENTRYPOINT ["/go/bin/git-sync"]
