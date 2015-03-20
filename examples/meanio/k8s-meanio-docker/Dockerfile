FROM google/nodejs

WORKDIR /app
ADD package.json /app/
ADD . /app
RUN npm install -g bower node-gyp grunt
RUN npm install
RUN bower --allow-root install

EXPOSE 3000
CMD []
ENTRYPOINT ["node", "server.js"]
