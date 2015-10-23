FROM gcr.io/google_containers/spark-base:1.4.0_v1

ADD start.sh /
ADD log4j.properties /opt/spark/conf/log4j.properties
EXPOSE 7077

ENTRYPOINT ["/start.sh"]
