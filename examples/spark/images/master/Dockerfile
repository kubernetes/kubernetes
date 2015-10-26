FROM gcr.io/google_containers/spark-base:latest

ADD start.sh /
ADD log4j.properties /opt/spark/conf/log4j.properties
EXPOSE 7077 8080

ENTRYPOINT ["/start.sh"]
