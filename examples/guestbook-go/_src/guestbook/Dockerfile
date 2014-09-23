FROM busybox:ubuntu-14.04

ADD ./bin/guestbook /app/guestbook
ADD ./_src/public/index.html /app/public/index.html
ADD ./_src/public/script.js /app/public/script.js
ADD ./_src/public/style.css /app/public/style.css

WORKDIR /app
CMD ["./guestbook"]
EXPOSE 3000
