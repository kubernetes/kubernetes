FROM busybox
MAINTAINER Muhammed Uluyol "uluyol@google.com"

ADD dc /diurnal

RUN chown root:users /diurnal && chmod 755 /diurnal

ENTRYPOINT ["/diurnal"]
