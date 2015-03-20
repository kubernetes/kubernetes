FROM google/cloud-sdk

RUN apt-get update && apt-get install -y curl

ADD run.sh /run.sh
RUN chmod a+x /*.sh

CMD ["/run.sh"]
