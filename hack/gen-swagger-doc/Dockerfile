FROM java:7-jre

RUN apt-get update
RUN apt-get install -qq -y asciidoctor
RUN apt-get install -qq -y unzip
RUN wget https://services.gradle.org/distributions/gradle-2.5-bin.zip
RUN mkdir build/
RUN unzip gradle-2.5-bin.zip -d build/

COPY build.gradle build/
COPY gen-swagger-docs.sh build/

#run the script once to download the dependent java libraries into the image
RUN mkdir /output
RUN build/gen-swagger-docs.sh /dev/null /dev/null || true

ENTRYPOINT ["build/gen-swagger-docs.sh"]
