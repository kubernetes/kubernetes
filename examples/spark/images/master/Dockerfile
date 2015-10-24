FROM gcr.io/google_containers/spark-base:1.5.1_v1

ADD start.sh /
ADD log4j.properties /opt/spark/conf/log4j.properties
EXPOSE 7077

ENTRYPOINT ["/start.sh"]
