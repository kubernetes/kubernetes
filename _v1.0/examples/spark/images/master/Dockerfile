FROM gcr.io/google_containers/spark-base

ADD start.sh /
ADD log4j.properties /opt/spark/conf/log4j.properties
EXPOSE 7077

ENTRYPOINT ["/start.sh"]
