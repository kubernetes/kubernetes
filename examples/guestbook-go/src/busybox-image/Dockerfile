FROM busybox:ubuntu-14.04

ADD ./bin/guestbook-example /app/guestbook-example
ADD ./src/github.com/Gurpartap/guestbook-example/public/index.html /app/public/index.html
ADD ./src/github.com/Gurpartap/guestbook-example/public/script.js /app/public/script.js
ADD ./src/github.com/Gurpartap/guestbook-example/public/style.css /app/public/style.css

WORKDIR /app
CMD ["./guestbook-example"]
EXPOSE 3000
