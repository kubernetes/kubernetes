FROM busybox:ubuntu-14.04

ADD ./bin/guestbook /app/guestbook
ADD ./src/public/index.html /app/public/index.html
ADD ./src/public/script.js /app/public/script.js
ADD ./src/public/style.css /app/public/style.css

WORKDIR /app
CMD ["./guestbook"]
EXPOSE 3000
