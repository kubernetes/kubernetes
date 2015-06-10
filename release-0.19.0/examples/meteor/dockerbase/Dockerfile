FROM node:0.10
MAINTAINER Christiaan Hees <christiaan@q42.nl>

ONBUILD WORKDIR /appsrc
ONBUILD COPY . /appsrc

ONBUILD RUN curl https://install.meteor.com/ | sh && \
    meteor build ../app --directory --architecture os.linux.x86_64 && \
    rm -rf /appsrc
# TODO rm meteor so it doesn't take space in the image?

ONBUILD WORKDIR /app/bundle

ONBUILD RUN (cd programs/server && npm install)
EXPOSE 8080
CMD []
ENV PORT 8080
ENTRYPOINT MONGO_URL=mongodb://$MONGO_SERVICE_HOST:$MONGO_SERVICE_PORT /usr/local/bin/node main.js
