FROM google/nodejs

WORKDIR /app/
RUN git clone https://github.com/linnovate/mean
RUN npm install -g bower node-gyp grunt
RUN npm install
RUN bower --allow-root install

EXPOSE 3000
CMD []
ENTRYPOINT ["node", "server.js"]
