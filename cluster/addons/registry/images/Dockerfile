FROM haproxy:1.5
MAINTAINER Muhammed Uluyol <uluyol@google.com>

RUN apt-get update && apt-get install -y dnsutils

ADD proxy.conf.insecure.in /proxy.conf.in
ADD run_proxy.sh /usr/bin/run_proxy

RUN chown root:users /usr/bin/run_proxy
RUN chmod 755 /usr/bin/run_proxy

CMD ["/usr/bin/run_proxy"]
